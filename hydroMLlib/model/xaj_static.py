"""
XAJ hydrological model implementation module.

This module provides a PyTorch implementation of the Xin'anjiang (XAJ) hydrological
model with multi-component parameterization and deep-learning-based parameter inversion.

Main components:
1. XAJ hydrological model class:
   - XAJMul: multi-component XAJ model with static parameters

2. Deep learning + XAJ hybrid model class:
   - MultiInv_XAJModel: deep-learning parameter inversion + static XAJ model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hydroRouting import UH_conv, UH_gamma
from .dLmodels import (
    CudnnLstmModel, CudnnGruModel, CudnnBiLstmModel,
    CudnnBiGruModel, CudnnRnnModel, CudnnCnnLstmModel, CudnnCnnBiLstmModel
)


class XAJMul(nn.Module):
    """
    Multi-component Xin'anjiang model (vectorized, HBVMul-style conditional logic).
    ------------------------------------------------------------------
    Goals:
    - Under the same meteorological forcing, allow each grid cell to carry `mu`
      sets of XAJ parameters to represent heterogeneity / uncertainty.
    - Keep the whole computation in a tensorized style (no Python `if`, use
      `clamp` / `where` for conditional behavior).
    - Explicitly separate the daily process into:
      three-layer evapotranspiration → saturation-excess runoff generation →
      soil moisture update → free water reservoir routing → three-flow
      separation → linear reservoirs → (optional) river routing.
    """

    def __init__(self):
        super(XAJMul, self).__init__()

    def forward(self, x, parameters, mu,
                muwts=None, rtwts=None, bufftime=0,
                outstate=False, routOpt=False, comprout=False,
                corrwts=None, pcorr=None):
        """
        Argument description
        --------------------
        x: [T, B, 2], ordering (P, PET)
           - P: mm/d, basin-averaged precipitation
           - PET: mm/d, potential evapotranspiration
        parameters: [B, 12, mu], 12 dimensionless parameters, `mu` sets for each grid
        mu: int, number of parameter components
        muwts: [B, mu] or None, weights for the `mu` components
        rtwts: [B, 2] or [B*mu, 2], two routing control parameters for the unit hydrograph
        bufftime: int, warm-up length (time steps)
        outstate: bool, if True, also return final model states (for segmented runs)
        routOpt: bool, if True, perform river routing
        comprout: bool, if True, route each component before aggregation
        corrwts: tensor or None, weights for precipitation correction
        pcorr: list or None, precipitation correction parameter range
        """

        # Small numerical protection used for divisions and exponent bases
        PRECS = 1e-6

        # ============================================================
        # 1. Initial model states: either from warm-up or small positive values
        # ============================================================
        if bufftime > 0:
            # With warm-up: reuse the same-structure model, run the first `bufftime`
            # steps and take the final states.
            with torch.no_grad():
                x_init = x[:bufftime, :, :]
                init_model = XAJMul()
                Qs_init, WU, WL, WD, S, FR, QI, QG = init_model(
                    x_init, parameters, mu,
                    muwts=muwts, rtwts=rtwts,
                    bufftime=0, outstate=True,
                    routOpt=False, comprout=False,
                    corrwts=corrwts, pcorr=pcorr,
                )
        else:
            # Without warm-up: initialize all storages with a small positive value
            # to avoid division by zero later.
            Ngrid = x.shape[1]
            device = x.device if hasattr(x, "device") else torch.device("cpu")
            # Initialization following HBVMul-style: zeros + small positive offset
            WU = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # upper tension water
            WL = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # middle tension water
            WD = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # deep tension water
            S  = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # free water storage
            QI = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # interflow linear reservoir state
            QG = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # groundwater linear reservoir state
            FR = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # contributing area of last time step

        # ============================================================
        # 2. Remove warm-up segment and extract P / PET
        # ============================================================
        P   = x[bufftime:, :, 0]     # [T, B]
        PET = x[bufftime:, :, 2]     # [T, B]
        Nstep, Ngrid = P.shape
        device = P.device

        # ============================================================
        # 3. Optional precipitation correction: P_corr = P * α(basin)
        #    This can pull systematic bias (e.g., from CFS/ECMWF) back toward observations.
        # ============================================================
        if pcorr is not None:
            parPCORR = pcorr[0] + corrwts[:, 0] * (pcorr[1] - pcorr[0])
            P = parPCORR.unsqueeze(0).repeat(Nstep, 1) * P

        # Expand time–space forcing into the `mu` dimension to run multiple parameter
        # sets in one shot.
        Pm   = P.unsqueeze(-1).repeat(1, 1, mu)     # [T, B, mu]
        PETm = PET.unsqueeze(-1).repeat(1, 1, mu)

        # ============================================================
        # 4. Parameter de-normalization: map [0, 1] to hydrologically
        #    meaningful ranges
        # ============================================================
        # Parameter order:
        # 0 ke   evapotranspiration scaling factor
        # 1 b    storage capacity curve exponent
        # 2 wum  upper tension water capacity
        # 3 wlm  middle tension water capacity
        # 4 wm   total tension water capacity
        # 5 c    evapotranspiration reduction factor for lower layers
        # 6 sm   maximum water depth of free water reservoir
        # 7 ex   exponent of free water reservoir outflow curve
        # 8 ki   recharge coefficient from free water to interflow reservoir
        # 9 kg   recharge coefficient from free water to groundwater reservoir
        # 10 ci  linear reservoir coefficient for interflow
        # 11 cg  linear reservoir coefficient for groundwater
        parascaLst = [
            [0.3, 2.0],     # 0 ke
            [0.0, 5.0],     # 1 b
            [0.01, 100.0],  # 2 wum
            [0.01, 100.0],  # 3 wlm
            [0.01, 200.0],  # 4 wm
            [0.09, 1.0],    # 5 c
            [0.01, 100.0],  # 6 sm
            [0.0, 10.0],    # 7 ex
            [0.01, 0.7],    # 8 ki
            [0.01, 0.7],    # 9 kg
            [0.0, 0.998],   # 10 ci
            [0.0, 0.998],   # 11 cg
        ]
        # Value ranges for the two routing-unit-hydrograph control parameters
        routscaLst = [[0.0, 2.9], [0.0, 6.5]]

        sc = lambda arr, i: parascaLst[i][0] + arr * (parascaLst[i][1] - parascaLst[i][0])

        parKE, parB, parWUM, parWLM, parWM, parC, parSM, parEX, parKI_, parKG_, parCI, parCG = [
            sc(parameters[:, i, :], i) for i in range(len(parascaLst))
        ]

        # Deep tension water capacity = total capacity - upper - middle.
        # If user-specified ranges overlap, clamp to ≥ 0 to keep physical meaning.
        wdm = torch.clamp(parWM - parWUM - parWLM, min=0.0)

        # KI + KG must be < 1; otherwise the free water storage would be fully
        # depleted in a single time step.
        sum_k = parKI_ + parKG_
        parKI = torch.where(sum_k < 1.0, parKI_, (1 - PRECS) * parKI_ / (sum_k + PRECS))
        parKG = torch.where(sum_k < 1.0, parKG_, (1 - PRECS) * parKG_ / (sum_k + PRECS))

        # ============================================================
        # 5. Pre-allocate time series fluxes for later `mu` aggregation
        # ============================================================
        QS_inflow = torch.zeros((Nstep, Ngrid, mu), device=device)  # surface / quick flow
        QI_inflow = torch.zeros_like(QS_inflow)                     # interflow
        QG_inflow = torch.zeros_like(QS_inflow)                     # groundwater flow
        ET_series = torch.zeros_like(QS_inflow)                     # actual evapotranspiration
        FR0 = FR.clone()                                            # contributing area of previous step

        # ============================================================
        # 6. Main time-stepping loop: traverse the XAJ daily process
        # ============================================================
        for t in range(Nstep):
            # --------------------------------------------------------
            # 6.1 Three-layer evapotranspiration
            #     Logic: first consume water from upper layer + today's rainfall,
            #     then request from middle layer, finally from deep layer.
            #     Upper layer is energy-controlled; the lower two are limited by
            #     available water and reduction factor.
            # --------------------------------------------------------
            P_t   = Pm[t, :, :]               # 当前时段降水
            PET_t = PETm[t, :, :] * parKE     # scaled PET: actual evaporative capacity of this surface

            # Upper layer ET: if energy is enough, evaporate PET; otherwise evaporate
            # all water available in the upper layer plus today's rainfall.
            EU = torch.min(PET_t, WU + P_t)

            # ET demand passed to lower layers
            D = torch.clamp(PET_t - EU, min=0.0)

            # Middle-layer ET: use a "wetness factor" to smoothly transition between
            # the wet and dry formulations instead of writing `if WL > c*WLM ...`,
            # so the tensor computation stays differentiable.
            wet_excess = torch.clamp(WL - parC * parWLM, min=0.0)   # >0 means the middle layer is wet
            wet_ratio  = wet_excess / (wet_excess + 1.0)            # maps to (0, 1)

            # Wet regime: supply `D * WL / WLM`
            EL_wet = D * WL / (parWLM + PRECS)
            # Dry regime: limited by `c * D` and by current WL
            EL_dry = torch.min(parC * D, WL)

            # Final middle-layer ET: weighted sum of the two formulas
            EL = wet_ratio * EL_wet + (1 - wet_ratio) * EL_dry
            EL = torch.clamp(EL, min=0.0)
            EL = torch.min(EL, WL)  # cannot evaporate more than stored

            # Deep-layer ET: use remaining demand, scaled by `c`, but not exceeding
            # current deep-layer storage.
            D_res = torch.clamp(D - EL, min=0.0)
            ED = torch.min(parC * D_res, WD)

            # Actual ET for this day
            E = EU + EL + ED
            ET_series[t, :, :] = E

            # --------------------------------------------------------
            # 6.2 Saturation-excess runoff generation
            #     Classic XAJ idea: wetter soil leads to easier runoff generation;
            #     use a storage capacity curve to map "soil wetness + net rainfall"
            #     to runoff.
            # --------------------------------------------------------
            PE = torch.clamp(P_t - E, min=0.0)   # net rainfall after ET, entering soil/free water
            W  = WU + WL + WD                    # total soil water in three layers, representing antecedent wetness

            # Maximum equivalent depth of the storage capacity curve
            WMM = parWM * (1.0 + parB)

            # 1 - W/WM is the "unfilled fraction"; the smaller it is, the wetter the soil.
            # It appears as a power base, so we clamp it to a positive number.
            base_W = torch.clamp(1.0 - W / (parWM + PRECS), min=PRECS)

            # A is the equivalent water depth already contributed by the curve; wetter soil → larger A.
            A = WMM * (1.0 - base_W ** (1.0 / (1.0 + parB)))

            # (PE + A) / WMM expresses today's water plus antecedent wetness
            # relative to maximum storage.
            ratio = torch.clamp((PE + A) / (WMM + PRECS), max=1.0)

            # Piecewise runoff generation:
            # - Unsaturated: nonlinear runoff from a power function
            # - Saturated: pure excess runoff
            R_per = torch.where(
                PE + A < WMM,
                PE - (parWM - W) + parWM * (1.0 - ratio) ** (1.0 + parB),
                PE - (parWM - W)
            )
            R_per = torch.clamp(R_per, min=0.0)  # non-negative runoff only

            # --------------------------------------------------------
            # 6.3 Three-layer soil tension water update (mass balance in
            #     the permeable zone)
            #     This step controls antecedent soil wetness for the next time
            #     step and is coupled with everything that follows.
            # --------------------------------------------------------
            dW = P_t - E - R_per  # net water stored in soil (can be positive or negative)

            # Positive dW: surplus water → refill upper, then middle, then deep layer
            dpos = torch.clamp(dW, min=0.0)

            # Fill upper layer; cannot exceed parWUM
            fill_u = torch.min(dpos, torch.clamp(parWUM - WU, min=0.0))
            WU = WU + fill_u
            dpos = dpos - fill_u

            # Fill middle layer
            fill_l = torch.min(dpos, torch.clamp(parWLM - WL, min=0.0))
            WL = WL + fill_l
            dpos = dpos - fill_l

            # Fill deep layer
            fill_d = torch.min(dpos, torch.clamp(wdm - WD, min=0.0))
            WD = WD + fill_d

            # Negative dW: water deficit → withdraw in order from upper to middle to deep,
            # not letting any storage go negative.
            dneg = torch.clamp(-dW, min=0.0)

            sub_u = torch.min(dneg, WU)
            WU = WU - sub_u
            dneg = dneg - sub_u

            sub_l = torch.min(dneg, WL)
            WL = WL - sub_l
            dneg = dneg - sub_l

            sub_d = torch.min(dneg, WD)
            WD = WD - sub_d

            # --------------------------------------------------------
            # 6.4 Free water reservoir routing + quick flow generation
            #     This corresponds to the "free water reservoir" in XAJ with four steps:
            #     (1) Define current contributing area FR (R/PE when raining, else
            #         inherit from previous step);
            #     (2) Convert previous S from area FR0 to FR to preserve volume;
            #     (3) Compute non-linear releasable water AU;
            #     (4) Compute quick flow RS with a two-segment formula, capped by R_per.
            # --------------------------------------------------------
            # (1) Contributing area FR: R/PE when PE > 0; retain previous FR0 otherwise.
            FR_new = R_per / (PE + PRECS)
            FR = FR_new * (PE > 0.0).float() + FR0 * (PE <= 0.0).float()
            FR = torch.clamp(FR, 0.0, 1.0)

            # (2) Area conversion: previous S is on area FR0; convert to area FR to
            #     keep total volume unchanged.
            S_eq = S * FR0 / (FR + PRECS)
            S_eq = torch.clamp(S_eq, max=parSM)

            # (3) Non-linear outflow capacity AU of the free water reservoir
            SMM = parSM * (1.0 + parEX)
            ratio_s = torch.clamp(1.0 - S_eq / (parSM + PRECS), min=PRECS)
            AU = SMM * (1.0 - ratio_s ** (1.0 / (1.0 + parEX)))

            # (4) Quick flow RS: two-segment runoff formula of XAJ
            ratio_pa = torch.clamp((PE + AU) / (SMM + PRECS), max=1.0)
            RS_part1 = (PE + S_eq - parSM + parSM * (1.0 - ratio_pa) ** (1.0 + parEX)) * FR
            RS_part2 = (PE + S_eq - parSM) * FR
            RS = torch.where(PE + AU < SMM, RS_part1, RS_part2)
            RS = torch.clamp(RS, min=0.0)
            RS = torch.min(RS, R_per)  # quick flow cannot exceed runoff from permeable zone

            # Refill: water not converted to quick flow returns to the free water storage
            S = S_eq + (R_per - RS) / (FR + PRECS)
            S = torch.clamp(S, max=parSM)

            # Update FR0 for the next time step
            FR0 = FR

            # --------------------------------------------------------
            # 6.5 Three-flow separation + linear reservoirs before hillslope outlet
            #     In XAJ, the free water reservoir releases water as:
            #     direct quick flow RS and two delayed components RI / RG, each
            #     entering a first-order linear reservoir.
            # --------------------------------------------------------
            # Inflow from free water to interflow and groundwater
            RI = parKI * S * FR
            RG = parKG * S * FR
            # Remaining free water storage
            S = S * (1.0 - parKI - parKG)

            # First-order linear reservoirs:
            # current outflow = previous outflow * decay + (1 - decay) * current inflow
            QI = parCI * QI + (1.0 - parCI) * RI
            QG = parCG * QG + (1.0 - parCG) * RG

            # Save fluxes for this time step
            QS_inflow[t, :, :] = RS
            QI_inflow[t, :, :] = QI
            QG_inflow[t, :, :] = QG

        # ============================================================
        # 7. Component aggregation: combine `mu` parameter sets into a basin response
        # ============================================================
        # First, average each component separately for analysis.
        QSave = QS_inflow.mean(-1, keepdim=True)
        QIave = QI_inflow.mean(-1, keepdim=True)
        QGave = QG_inflow.mean(-1, keepdim=True)
        ETave = ET_series.mean(-1, keepdim=True)

        # The actual inflow to the river is the sum of the three components
        river_inflow = QS_inflow + QI_inflow + QG_inflow  # [T, B, mu]

        if muwts is None:
            inflow_mean = river_inflow.mean(-1)           # [T, B]
        else:
            inflow_mean = (river_inflow * muwts).sum(-1)

        # ============================================================
        # 8. Optional river routing: unit hydrograph convolution
        # ============================================================
        if routOpt:
            # Choose routing input: route each component separately or route after aggregation.
            if comprout:
                Qsim = river_inflow.view(Nstep, Ngrid * mu)
            else:
                Qsim = inflow_mean

            # Map routing parameters from [0, 1] back to physical ranges
            tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])

            # Expand to time dimension
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)

            # Generate Gamma unit hydrograph and perform 1D convolution
            UH = UH_gamma(routa, routb, lenF=15).permute(1, 2, 0)
            rf = torch.unsqueeze(Qsim, -1).permute(1, 2, 0)
            Qsrout = UH_conv(rf, UH).permute(2, 0, 1)

            if comprout:
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout
        else:
            # Without routing, use inflow as outlet discharge directly
            Qs = inflow_mean.unsqueeze(-1)

        # ============================================================
        # 9. Output control
        # ============================================================
        if outstate:
            # Return final states for continued segmented runs
            return Qs, WU, WL, WD, S, FR, QI, QG
        else:
            # Standard output: main discharge + three components + ET
            Qall = torch.cat((Qs, QSave, QIave, QGave, ETave), dim=-1)
            return Qall



class MultiInv_XAJModel(nn.Module):
    """
    Deep-learning-based parameter inversion + static XAJ model.

    This is the core implementation of the dPLXAJ model, combining deep learning
    parameter inversion with the XAJ hydrological model.

    The model uses an RNN (LSTM / GRU / BiLSTM / BiGRU / RNN / CNN-LSTM /
    CNN-BiLSTM) to invert XAJ parameters from static attributes, and then uses
    the XAJ model for hydrological simulation.

    Model architecture:
    ----------
    1. RNN-based parameter inversion: static attributes -> XAJ parameters
    2. XAJ hydrological model: meteorological forcing + XAJ parameters -> runoff simulation

    Parameters:
    ----------
    ninv : int
        Input feature dimension (number of static attributes)
    nfea : int
        Number of XAJ parameter features (fixed to 12)
    nmul : int
        Number of components, used for parameter uncertainty quantification
    hiddeninv : int
        Hidden size of the RNN
    drinv : float, default=0.5
        Dropout rate for the RNN
    inittime : int, default=0
        Warm-up time steps for the XAJ model
    routOpt : bool, default=False
        Whether to enable river routing
    comprout : bool, default=False
        Whether to route each component separately
    compwts : bool, default=False
        Whether to learn multi-component weights
    pcorr : list or None, default=None
        Precipitation correction parameter range
    rnn_type : str, default='lstm'
        RNN type: 'lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'

    Usage:
    ----------
    Mainly used for deep-learning-based parameter inversion of the static-parameter
    XAJ model for basin hydrological simulation and prediction.
    """
    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
                 compwts=False, pcorr=None, rnn_type='lstm'):
        """
        Initialize the deep-learning-based parameter inversion + static XAJ model.

        See the class docstring for a full description of the arguments.
        """
        super(MultiInv_XAJModel, self).__init__()
        self.ninv = ninv                    # input feature dimension (number of static attributes)
        self.nfea = nfea                    # number of XAJ parameter features (fixed to 12)
        self.hiddeninv = hiddeninv          # RNN hidden size
        self.nmul = nmul                    # number of components
        self.rnn_type = rnn_type.lower()    # RNN type: 'lstm', 'gru', 'bilstm', 'bigru', 'rnn'
        
        # =============================================================================
        # Parameter counting
        # =============================================================================
        # Total number of parameters the RNN needs to output
        nxajpm = nfea * nmul  # total XAJ parameters = number of features × number of components

        # Routing parameter count
        if comprout is False:
            nroutpm = 2  # simple routing: two parameters per basin
        else:
            nroutpm = nmul * 2  # component-wise routing: two parameters for each component

        # Component weight parameter count
        if compwts is False:
            nwtspm = 0  # no weights: simple average
        else:
            nwtspm = nmul  # with weights: one weight per component

        # Precipitation correction parameter count
        if pcorr is None:
            ntp = nxajpm + nroutpm + nwtspm  # total number of parameters
        else:
            ntp = nxajpm + nroutpm + nwtspm + 1  # total + one precipitation correction parameter
        
        # =============================================================================
        # RNN model selection
        # =============================================================================
        # Choose the RNN model according to `rnn_type`.
        if self.rnn_type == 'lstm':
            self.lstminv = CudnnLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'gru':
            self.lstminv = CudnnGruModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bilstm':
            self.lstminv = CudnnBiLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bigru':
            self.lstminv = CudnnBiGruModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'rnn':
            self.lstminv = CudnnRnnModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnlstm':
            self.lstminv = CudnnCnnLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnbilstm':
            self.lstminv = CudnnCnnBiLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        else:
            raise ValueError("rnn_type must be one of: 'lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'")

        # XAJ hydrological model instance
        self.XAJ = XAJMul()

        # =============================================================================
        # Store model attributes
        # =============================================================================
        self.gpu = 1                    # GPU flag (kept for compatibility)
        self.inittime = inittime        # warm-up time for the XAJ model
        self.routOpt = routOpt          # routing option
        self.comprout = comprout        # component-wise routing option
        self.nxajpm = nxajpm            # number of XAJ parameters
        self.nwtspm = nwtspm            # number of weight parameters
        self.nroutpm = nroutpm          # number of routing parameters
        self.pcorr = pcorr              # precipitation correction parameters

    def forward(self, x, z, doDropMC=False):
        """
        Forward pass of the deep-learning-based parameter inversion + static XAJ model.

        Parameters:
        ----------
        x : torch.Tensor
            Meteorological forcing, shape [time, basin, var].
            `var` includes: P (mm/d, precipitation), PET (mm/d, potential ET).
        z : torch.Tensor
            Static attributes, shape [time, basin, ninv], used as RNN input for parameter inversion.
        doDropMC : bool, default=False
            Whether to use Monte Carlo dropout at inference time (currently unused).

        Returns:
        ----------
        torch.Tensor
            XAJ model output containing simulated runoff and related variables.
        """
        # =============================================================================
        # RNN-based parameter inversion
        # =============================================================================
        # Use RNN to invert XAJ parameters from static attributes
        Gen = self.lstminv(z)  # RNN output, shape [time, basin, ntp]

        # Take parameters at the last time step as XAJ parameters
        Params0 = Gen[-1, :, :]  # [basin, ntp]
        ngage = Params0.shape[0]  # number of basins

        # =============================================================================
        # Parameter parsing and transformation
        # =============================================================================
        # XAJ parameters: take the corresponding slice from the RNN output
        xajpara0 = Params0[:, 0:self.nxajpm]  # XAJ parameter part
        # Use sigmoid to constrain parameters to [0, 1]
        xajpara = torch.sigmoid(xajpara0).view(ngage, self.nfea, self.nmul)

        # Routing parameters
        routpara0 = Params0[:, self.nxajpm:self.nxajpm+self.nroutpm]
        if self.comprout is False:
            # Simple routing: two parameters per basin
            routpara = torch.sigmoid(routpara0)
        else:
            # Component-wise routing: two parameters per component
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)

        # Component weights
        if self.nwtspm == 0:
            # No weights: simple averaging
            wts = None
        else:
            wtspara = Params0[:, self.nxajpm+self.nroutpm:self.nxajpm+self.nroutpm+self.nwtspm]
            wts = F.softmax(wtspara, dim=-1)  # ensure weights sum to 1

        # Precipitation correction parameters
        if self.pcorr is None:
            corrpara = None
        else:
            corrpara0 = Params0[:, self.nxajpm+self.nroutpm+self.nwtspm:self.nxajpm+self.nroutpm+self.nwtspm+1]
            corrpara = torch.sigmoid(corrpara0)

        # =============================================================================
        # XAJ hydrological simulation
        # =============================================================================
        # Run the XAJ model with the inverted parameters
        out = self.XAJ(
            x,
            parameters=xajpara,
            mu=self.nmul,
            muwts=wts,
            rtwts=routpara,
            bufftime=self.inittime,
            routOpt=self.routOpt,
            comprout=self.comprout,
            corrwts=corrpara,
            pcorr=self.pcorr
        )
        return out
