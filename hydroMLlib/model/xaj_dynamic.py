import torch  
import torch.nn as nn
from .hydroRouting import UH_conv, UH_gamma
from .xaj_static import XAJMul
from .dLmodels import (
    CudnnLstmModel, CudnnGruModel, CudnnBiLstmModel,
    CudnnBiGruModel, CudnnRnnModel, CudnnCnnLstmModel, CudnnCnnBiLstmModel,
)

import torch
import torch.nn as nn

class XAJMulTD(nn.Module):
    """
    Multi-component XAJ model with time-varying parameters (XAJMulTD).

    Core ideas:
    1. Keep the physical structure of the static XAJ model unchanged:
       three-layer evapotranspiration → saturation-excess runoff generation
       → three-layer tension water update → free water reservoir routing
       → three-way runoff partitioning → linear recession → (optional) river routing.
    2. Allow a subset of parameters (e.g., KE, WM, SM, EX, etc.) to vary with time
       so the model can respond to non-stationary climate, land surface and
       human activities.
    3. Use `tdlst` to specify which parameters are time dependent; the others
       use the parameter frame at `staind` as a static baseline.
    4. Use `dydrop` as a dropout/annealing mechanism for dynamic parameters:
       with some probability they fall back to the static baseline, which
       reduces overfitting at each single time step.
    5. Numerical stability follows the static XAJ + HBVMul style:
       state initialization uses (zeros + eps), and all denominators/bases
       for powers are clamped.
    """

    def __init__(self):
        super(XAJMulTD, self).__init__()

    def forward(self, x,
                parameters,  # [T, B, 12, mu] time-varying parameters (normalized)
                staind,      # int, time index of the static baseline parameters
                tdlst,       # list[int] indices (1-based) of parameters to be dynamic
                mu,
                muwts=None,
                rtwts=None,
                bufftime=0,
                outstate=False,
                routOpt=False,
                comprout=False,
                corrwts=None,
                pcorr=None,
                dydrop=0.0):
        """
        x: [time, basin, var] input forcings, with at least P and PET
           (here x[..., 0] is P and x[..., 2] is PET).
        parameters: [time, basin, 12, mu] sequence of dynamic parameters in [0, 1].
        staind: which time index to take as the "static baseline".
        tdlst: which parameters are dynamic (1-based, following `parascaLst` order).
        mu: number of components.
        dydrop: dropout probability for dynamic parameters,
                0 = always use dynamic values, 1 = always revert to static.
        """

        # ------------------------------------------------------------
        # 0. Numerical protection constant used in all divisions and bases for powers
        # ------------------------------------------------------------
        PRECS = 1e-6

        # ============================================================
        # 1. Spin-up / initial hydrological states
        #    Goal: make WU/WL/WD/S/QI/QG/FR start from a reasonable wetness
        #          instead of exact zeros.
        #    Two options:
        #    (1) bufftime > 0: actually run a static XAJ over the warm-up
        #        period and take its final states;
        #    (2) otherwise: follow HBVMul and use very small positive initial values.
        # ============================================================
        if bufftime > 0:
            # 1.1 Use static XAJ to run the warm-up segment
            with torch.no_grad():
                x_init = x[:bufftime, :, :]                 # [buff, B, var]
                # Use the last-frame dynamic parameters in warm-up as "static" parameters
                par_init = parameters[bufftime - 1, :, :, :]  # [B, 12, mu]
                init_model = XAJMul()                       # static-parameter XAJ model
                Qs_init, WU, WL, WD, S, FR, QI, QG = init_model(
                    x_init,
                    par_init,
                    mu,
                    muwts=muwts,
                    rtwts=rtwts,
                    bufftime=0,
                    outstate=True,
                    routOpt=False,
                    comprout=False,
                    corrwts=corrwts,
                    pcorr=pcorr,
                )
        else:
            # 1.2 No warm-up: follow HBVMul initialization
            Ngrid = x.shape[1]
            device = x.device if hasattr(x, "device") else torch.device("cpu")
            # All reservoirs start from a small positive number (almost dry
            # but avoid division by zero).
            WU = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # upper-layer tension water
            WL = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # middle-layer tension water
            WD = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # deep-layer tension water
            S  = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # free water storage
            QI = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # interflow linear reservoir
            QG = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # groundwater linear reservoir
            FR = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # runoff area from last time step

        # ============================================================
        # 2. Remove warm-up forcings and split into P / PET
        # ============================================================
        P   = x[bufftime:, :, 0]   # [T', B]
        PET = x[bufftime:, :, 2]   # [T', B]
        Nstep, Ngrid = P.shape
        device = P.device

        # ============================================================
        # 3. Optional precipitation correction: P ← P * α(basin)
        #    Typical use: external precipitation has a systematic bias
        #    and needs to be rescaled.
        # ============================================================
        if pcorr is not None:
            # parPCORR: [B], one correction factor per basin/grid
            parPCORR = pcorr[0] + corrwts[:, 0] * (pcorr[1] - pcorr[0])
            # Apply scaling for all time steps
            P = parPCORR.unsqueeze(0).repeat(Nstep, 1) * P

        # Expand P/PET to mu dimension so that all components are processed in parallel.
        Pm   = P.unsqueeze(-1).repeat(1, 1, mu)    # [T', B, mu]
        PETm = PET.unsqueeze(-1).repeat(1, 1, mu)  # [T', B, mu]

        # ============================================================
        # 4. Mechanism of "static baseline + dynamic overlay + dropout"
        # ============================================================
        # 4.1 Physical ranges for the 12 parameters, consistent with static XAJ
        parascaLst = [
            [0.3, 2.0],     # 0 ke   evapotranspiration correction factor
            [0.0, 5.0],     # 1 b    exponent of soil storage capacity curve
            [0.01, 100.0],  # 2 wum  upper-layer tension water capacity
            [0.01, 100.0],  # 3 wlm  middle-layer tension water capacity
            [0.01, 200.0],  # 4 wm   total tension water capacity
            [0.09, 1.0],    # 5 c    reduction factor for middle/deep layer ET
            [0.01, 100.0],  # 6 sm   maximum free water storage depth
            [0.0, 10.0],    # 7 ex   nonlinearity exponent of free water storage
            [0.01, 0.7],    # 8 ki   fraction from free water to interflow
            [0.01, 0.7],    # 9 kg   fraction from free water to groundwater
            [0.0, 0.998],   # 10 ci  interflow linear recession coefficient
            [0.0, 0.998],   # 11 cg  groundwater linear recession coefficient
        ]

        # 4.2 Extract parameters for the simulation period (warm-up removed)
        #     parAll: [T', B, 12, mu]
        parAll = parameters[bufftime:, :, :, :]

        # 4.3 Map all dynamic parameters from [0, 1] back to physical ranges
        parAllTrans = torch.zeros_like(parAll)
        for ip, (pmin, pmax) in enumerate(parascaLst):
            parAllTrans[:, :, ip, :] = pmin + parAll[:, :, ip, :] * (pmax - pmin)

        # 4.4 Build a "static" version by replicating the `staind` frame
        #     parStaFull: [T', B, 12, mu]
        total_steps = parAllTrans.shape[0]
        # In ALL/PUB/PUR modes, dynamic parameter length may differ, so `staind`
        # is clipped here to avoid out-of-range indices (e.g., when TestBuff
        # refers to the training-period length).
        if total_steps == 0:
            raise ValueError("parAllTrans has zero time length; cannot obtain static baseline frame.")
        staind = 0 if staind is None else staind
        staind = max(0, min(staind, total_steps - 1))
        parStaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat(Nstep, 1, 1, 1)

        # 4.5 Start from the static parameters; later we overwrite entries in `tdlst`
        parUseFull = parStaFull.clone()

        # 4.6 Dropout mask for dynamic parameters:
        #     - dydrop = 0   → always use dynamic values
        #     - dydrop = 0.3 → 30% of (time, basin) points are forced to static
        if dydrop > 0.0:
            # Shape [T', B, 1, 1]: same mask for all parameter components
            # at a given (time, basin)
            drop_mask = torch.bernoulli(
                torch.full((Nstep, Ngrid, 1, 1), fill_value=dydrop, device=device)
            )
        else:
            drop_mask = torch.zeros((Nstep, Ngrid, 1, 1), device=device)

        # 4.7 Combine static and dynamic parameters for those in `tdlst`
        #     Note: `tdlst` is 1-based; convert to 0-based indices.
        for idx1 in tdlst:
            i = idx1 - 1
            dyn_val = parAllTrans[:, :, i, :]               # dynamic value
            sta_val = parStaFull[:, :, i, :]                # static baseline
            # drop_mask=1 → use static; 0 → use dynamic
            mixed = dyn_val * (1 - drop_mask.squeeze(-1)) + sta_val * drop_mask.squeeze(-1)
            parUseFull[:, :, i, :] = mixed

        # ============================================================
        # 5. Time series buffers for fluxes (same as HBVMul/XAJMul)
        #    All have shape [T', B, mu]; aggregation is by mean(-1) or muwts.
        # ============================================================
        QS_inflow = torch.zeros([Nstep, Ngrid, mu], dtype=torch.float32, device=device)  # surface/quick flow
        QI_inflow = torch.zeros_like(QS_inflow)                                          # interflow
        QG_inflow = torch.zeros_like(QS_inflow)                                          # groundwater flow
        ET_series = torch.zeros_like(QS_inflow)                                          # actual evapotranspiration
        FR0 = FR.clone()                                                                 # runoff area from last time step

        # ============================================================
        # 6. Main time loop: core hydrological processes
        #    Same as the static version, but with time-varying parameters parUseFull[t]
        # ============================================================
        for t in range(Nstep):
            # 6.1 Extract the 12 physical parameters at this time step
            par_t = parUseFull[t]  # [B, 12, mu]
            (parKE, parB, parWUM, parWLM, parWM,
             parC, parSM, parEX, parKI_, parKG_, parCI, parCG) = [
                par_t[:, i, :] for i in range(par_t.shape[1])
            ]

            # Because WM, WUM and WLM may be dynamic, deep-layer capacity
            # must be recomputed at each time step.
            wdm = torch.clamp(parWM - parWUM - parWLM, min=0.0)

            # Ensure KI + KG < 1 even with dynamic parameters; otherwise
            # free water would be fully emptied in one step.
            sum_k = parKI_ + parKG_
            parKI = torch.where(sum_k < 1.0,
                                parKI_,
                                (1 - PRECS) * parKI_ / (sum_k + PRECS))
            parKG = torch.where(sum_k < 1.0,
                                parKG_,
                                (1 - PRECS) * parKG_ / (sum_k + PRECS))

            # 6.2 Forcings at this time step
            P_t   = Pm[t, :, :]                    # [B, mu]
            PET_t = PETm[t, :, :] * parKE          # corrected PET

            # --------------------------------------------------------
            # 6.3 Three-layer evapotranspiration (energy → upper → middle → deep)
            # --------------------------------------------------------
            # Upper layer: EU = min(PET, WU + P)
            EU = torch.min(PET_t, WU + P_t)

            # Remaining demand
            D = torch.clamp(PET_t - EU, min=0.0)

            # Middle layer: use "wetness excess" (WL - c*WLM) to smoothly switch
            wet_excess = torch.clamp(WL - parC * parWLM, min=0.0)
            wet_ratio  = wet_excess / (wet_excess + 1.0)  # in (0, 1)

            # Wet formula: supply water according to relative soil moisture
            EL_wet = D * WL / (parWLM + PRECS)
            # Dry formula: limited by c, cannot exceed middle-layer storage
            EL_dry = torch.min(parC * D, WL)

            # Smoothly mixed middle-layer ET
            EL = wet_ratio * EL_wet + (1 - wet_ratio) * EL_dry
            EL = torch.clamp(EL, min=0.0)
            EL = torch.min(EL, WL)

            # Deep-layer ET: take the remaining demand and apply factor c
            D_res = torch.clamp(D - EL, min=0.0)
            ED = torch.min(parC * D_res, WD)

            # Actual ET
            E = EU + EL + ED
            ET_series[t, :, :] = E

            # --------------------------------------------------------
            # 6.4 Saturation-excess runoff (core soil storage curve in XAJ)
            # --------------------------------------------------------
            PE = torch.clamp(P_t - E, min=0.0)  # effective water after ET
            W  = WU + WL + WD                   # current soil moisture

            # Maximum equivalent water depth on the storage curve
            WMM = parWM * (1.0 + parB)
            # Base for the power; 1 - W/WM is remaining storage ratio and must be positive
            base_W = torch.clamp(1.0 - W / (parWM + PRECS), min=PRECS)
            # A is the equivalent water depth already filled by soil moisture
            A = WMM * (1.0 - base_W ** (1.0 / (1.0 + parB)))

            # Piecewise runoff generation
            ratio = torch.clamp((PE + A) / (WMM + PRECS), max=1.0)
            R_per = torch.where(
                PE + A < WMM,
                # Nonlinear part: soil is not fully saturated
                PE - (parWM - W) + parWM * (1.0 - ratio) ** (1.0 + parB),
                # Linear part: surplus runoff when storage is full
                PE - (parWM - W)
            )
            R_per = torch.clamp(R_per, min=0.0)

            # --------------------------------------------------------
            # 6.5 Three-layer tension water update
            #     Positive dW fills soil; negative dW drains soil; use clamp everywhere.
            # --------------------------------------------------------
            dW = P_t - E - R_per  # net infiltrated water; can be positive or negative

            # Positive part: fill upper → middle → deep
            dpos = torch.clamp(dW, min=0.0)
            fill_u = torch.min(dpos, torch.clamp(parWUM - WU, min=0.0))
            WU = WU + fill_u
            dpos = dpos - fill_u

            fill_l = torch.min(dpos, torch.clamp(parWLM - WL, min=0.0))
            WL = WL + fill_l
            dpos = dpos - fill_l

            fill_d = torch.min(dpos, torch.clamp(wdm - WD, min=0.0))
            WD = WD + fill_d

            # Negative part: drain upper → middle → deep, preventing negative storage
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
            # 6.6 Free water reservoir routing + two-part quick flow
            # --------------------------------------------------------
            # 6.6.1 Runoff area FR: when PE>0 use R/PE, otherwise keep FR0
            FR = torch.where(PE > 0.0, R_per / (PE + PRECS), FR0)
            FR = torch.clamp(FR, 0.0, 1.0)

            # 6.6.2 Area conversion: convert storage S from previous FR0 to current FR
            S_eq = S * FR0 / (FR + PRECS)
            S_eq = torch.clamp(S_eq, max=parSM)

            # 6.6.3 Outflow capacity AU of free water reservoir (nonlinear XAJ formula)
            SMM = parSM * (1.0 + parEX)
            ratio_s = torch.clamp(1.0 - S_eq / (parSM + PRECS), min=PRECS)
            AU = SMM * (1.0 - ratio_s ** (1.0 / (1.0 + parEX)))

            # 6.6.4 Surface quick flow: still two-part and constrained by RS ≤ R_per
            ratio_pa = torch.clamp((PE + AU) / (SMM + PRECS), max=1.0)
            RS_part1 = (PE + S_eq - parSM + parSM * (1.0 - ratio_pa) ** (1.0 + parEX)) * FR
            RS_part2 = (PE + S_eq - parSM) * FR
            RS = torch.where(PE + AU < SMM, RS_part1, RS_part2)
            RS = torch.clamp(RS, min=0.0)
            RS = torch.min(RS, R_per)

            # 6.6.5 Free water storage update: part not contributing to quick flow
            S = S_eq + (R_per - RS) / (FR + PRECS)
            S = torch.clamp(S, max=parSM)

            # 6.6.6 Update runoff area for the next time step
            FR0 = FR

            # --------------------------------------------------------
            # 6.7 Partition into three runoff sources + linear recession
            # --------------------------------------------------------
            # Remaining free water is split into interflow and groundwater recharge
            RI = parKI * S * FR
            RG = parKG * S * FR
            S = S * (1.0 - parKI - parKG)  # final free water storage

            # Linear recession: classic first-order recursive form
            QI = parCI * QI + (1.0 - parCI) * RI
            QG = parCG * QG + (1.0 - parCG) * RG

            # Record components
            QS_inflow[t, :, :] = RS
            QI_inflow[t, :, :] = QI
            QG_inflow[t, :, :] = QG

        # ============================================================
        # 7. Multi-component aggregation
        #    First aggregate along mu (mean or weighted),
        #    then optionally perform river routing.
        # ============================================================
        QSave = QS_inflow.mean(-1, keepdim=True)
        QIave = QI_inflow.mean(-1, keepdim=True)
        QGave = QG_inflow.mean(-1, keepdim=True)
        ETave = ET_series.mean(-1, keepdim=True)

        river_inflow = QS_inflow + QI_inflow + QG_inflow  # [T', B, mu]

        if muwts is None:
            inflow_mean = river_inflow.mean(-1)  # [T', B]
        else:
            inflow_mean = (river_inflow * muwts).sum(-1)

        # ============================================================
        # 8. Optional routing: same Gamma unit hydrograph as the static model
        # ============================================================
        if routOpt:
            # Decide whether to combine components before routing or route each separately
            if comprout:
                Qsim = river_inflow.view(Nstep, Ngrid * mu)
            else:
                Qsim = inflow_mean

            # Routing parameter ranges follow the static version
            tempa = 0.0 + rtwts[:, 0] * (2.9 - 0.0)
            tempb = 0.0 + rtwts[:, 1] * (6.5 - 0.0)

            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)

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
            # Without routing, directly output area-averaged inflow to river
            Qs = inflow_mean.unsqueeze(-1)

        # ============================================================
        # 9. Output control
        # ============================================================
        if outstate:
            # Return final states for warm start or segmented runs
            return Qs, WU, WL, WD, S, FR, QI, QG
        else:
            # Same as static model: outlet flow + three components + actual ET
            Qall = torch.cat((Qs, QSave, QIave, QGave, ETave), dim=-1)
            return Qall


class MultiInv_XAJTDModel(nn.Module):
    """
    Deep-learning parameter inversion + dynamic-parameter XAJ model.

    Structure similar to `MultiInv_HBVTDModel`:
    - An RNN inverts a time series of XAJ parameters:
      [Time, Basin, (nfea * nmul)]
    - Optional learned component weights and optional routing parameters
      (shape depends on `comprout`).
    - Optional precipitation correction parameter (1 value), whose range
      is scaled inside the physical model by `pcorr`.
    - Calls the dynamic physical model `XAJMulTD` to perform hydrological simulation.
    """

    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5,
                 inittime=0, routOpt=False, comprout=False,
                 compwts=False, staind=-1, tdlst=None, dydrop=0.0,
                 pcorr=None, rnn_type='lstm'):
        super(MultiInv_XAJTDModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea            # number of XAJ physical parameters (12)
        self.nmul = nmul
        self.hiddeninv = hiddeninv
        self.rnn_type = rnn_type.lower()

        # Total number of parameters:
        # dynamic XAJ parameter time series + routing + component weights
        # + optional precipitation correction
        nxajpm = nfea * nmul
        if comprout is False:
            nroutpm = 2
        else:
            nroutpm = nmul * 2
        nwtspm = (nmul if compwts else 0)
        ntp = nxajpm + nroutpm + nwtspm + (0 if pcorr is None else 1)

        # RNN backbone selection
        if self.rnn_type == 'lstm':
            self.lstminv = CudnnLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'gru':
            self.lstminv = CudnnGruModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bilstm':
            self.lstminv = CudnnBiLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bigru':
            self.lstminv = CudnnBiGruModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'rnn':
            self.lstminv = CudnnRnnModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnlstm':
            self.lstminv = CudnnCnnLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnbilstm':
            self.lstminv = CudnnCnnBiLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        else:
            raise ValueError("rnn_type must be one of: 'lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'")

        # Physical model
        self.XAJ = XAJMulTD()

        # Runtime configuration / attributes
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.nxajpm = nxajpm
        self.nwtspm = nwtspm
        self.nroutpm = nroutpm
        self.staind = staind if staind is not None else -1
        self.tdlst = tdlst or []
        self.dydrop = dydrop
        self.pcorr = pcorr

    def forward(self, x, z, doDropMC=False):
        # Invert parameter time series
        Params0 = self.lstminv(z)              # [Time, Basin, ntp]
        ntstep = Params0.shape[0]
        ngage = Params0.shape[1]

        # Dynamic XAJ parameters for the whole period
        xajpara0 = Params0[:, :, 0:self.nxajpm]
        xajpara = torch.sigmoid(xajpara0).view(ntstep, ngage, self.nfea, self.nmul)

        # Routing parameters (taken from the last time step)
        routpara0 = Params0[-1, :, self.nxajpm:self.nxajpm + self.nroutpm]
        if self.comprout is False:
            routpara = torch.sigmoid(routpara0)                 # [B, 2]
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)  # [B*mu, 2]

        # Component weights (optional, taken from the last time step)
        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = Params0[-1, :, self.nxajpm + self.nroutpm:self.nxajpm + self.nroutpm + self.nwtspm]
            wts = torch.softmax(wtspara, dim=-1)

        # Precipitation correction (optional, taken from the last time step)
        if self.pcorr is None:
            corrpara = None
        else:
            start = self.nxajpm + self.nroutpm + self.nwtspm
            corrpara0 = Params0[-1, :, start:start + 1]
            corrpara = torch.sigmoid(corrpara0)

        # Physical simulation
        out = self.XAJ(
            x,
            parameters=xajpara,
            staind=self.staind,
            tdlst=self.tdlst,
            mu=self.nmul,
            muwts=wts,
            rtwts=routpara,
            bufftime=self.inittime,
            outstate=False,
            routOpt=self.routOpt,
            comprout=self.comprout,
            corrwts=corrpara,
            pcorr=self.pcorr,
            dydrop=self.dydrop,
        )
        return out
