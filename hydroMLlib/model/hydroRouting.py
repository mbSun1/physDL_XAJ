import torch
import torch.nn.functional as F

# =============================================================================
# Unit hydrograph convolution
# -----------------------------------------------------------------------------
# Given a unit hydrograph (UH) for each basin, convolve the inflow series that
# reaches the channel to obtain the routed / dispersed outlet flow.
#
# Notes:
# - Conceptually, UH routing assumes a unit runoff input is distributed in time
#   when arriving at the outlet; UH is that discrete time distribution.
# - Discrete convolution: Q_out[t] = sum_k Q_in[t-k] * UH[k]
# - PyTorch conv1d applies kernels in the correlation form w * x(t+τ), so we
#   flip UH along the time dimension to match hydrologic "past-to-present" sum.
# - Padding keeps the output length aligned with the original time axis.
# - We use grouped convolution (groups=nb) so each basin uses its own UH.
# =============================================================================
def UH_conv(x, UH, viewmode=1):
    """
    Args:
        x: [batch, 1, time] 1D inflow series for each basin (or basin*component).
        UH: [batch, 1, uhLen] Unit hydrograph corresponding to x.

    Returns:
        y: [batch, 1, time] Routed outflow series after convolution.
    """
    mm = x.shape
    nb = mm[0]           # number of basins (or basin*components)
    m = UH.shape[-1]     # UH length
    padd = m - 1         # padding to keep output length

    if viewmode == 1:
        # conv1d expects [N, C, L]; pack basins into channel dimension
        xx = x.view([1, nb, mm[-1]])    # [1, batch, time]
        w = UH.view([nb, 1, m])         # [batch, 1, uhLen]
        groups = nb                     # grouped conv: one group per basin

    # Flip UH to match hydrologic convolution direction
    y = F.conv1d(xx,
                 torch.flip(w, [2]),
                 groups=groups,
                 padding=padd,
                 stride=1,
                 bias=None)
    # Remove padded tail to restore original length
    y = y[:, :, 0:-padd]
    # Restore original shape
    return y.view(mm)


# =============================================================================
# Gamma-distribution unit hydrograph
# -----------------------------------------------------------------------------
# Many routing UHs can be approximated by a Gamma distribution (fast rise,
# slow recession). Parameters:
# - a (shape): larger -> later peak, smoother curve
# - b (scale): larger -> more dispersion (wider UH)
#
# The model typically outputs dimensionless parameters (often in [0, 1]); they
# should be mapped to reasonable ranges before generating UH of length lenF.
# After generation, normalize over time so sum(UH)=1.
# =============================================================================
def UH_gamma(a, b, lenF=15):
    """
    Args:
        a, b: [time, basin, 1] Gamma parameters per basin (often repeated over time).
        lenF: Unit hydrograph length.

    Returns:
        UH: [lenF, basin, 1]
    """
    m = a.shape
    # For numerical stability: enforce non-negativity and add lower bounds
    aa = F.relu(a[0:lenF, :, :]) + 0.1
    theta = F.relu(b[0:lenF, :, :]) + 0.5

    # Time axis: start from 0.5 to avoid t=0 causing t^(a-1) blow-up
    t = torch.arange(0.5, lenF * 1.0, device=a.device).view(lenF, 1, 1).repeat(1, m[1], m[2])

    # Gamma PDF: f(t) = t^(a-1) * exp(-t/theta) / (Gamma(a) * theta^a)
    denom = (aa.lgamma().exp()) * (theta ** aa)
    mid = t ** (aa - 1)
    right = torch.exp(-t / theta)
    w = 1.0 / denom * mid * right

    # Normalize over time for each basin (sum to 1)
    w = w / w.sum(0)
    return w
