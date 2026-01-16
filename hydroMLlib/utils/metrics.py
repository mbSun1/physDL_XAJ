import numpy as np
import scipy.stats



keyLst = ['NSE', 'KGE']


def statError(pred, target):
    """
    Compute evaluation metrics NSE and KGE.

    This helper computes the Nash–Sutcliffe Efficiency (NSE) and
    Kling–Gupta Efficiency (KGE) for each grid/catchment, given
    predicted and observed time series.

    Args:
        pred: predictions, 2D array of shape (ngrid, nt)
        target: observations, 2D array of shape (ngrid, nt)

    Returns:
        dict: dictionary with keys 'NSE' and 'KGE', each a 1D array
              of length `ngrid`.
    """
    ngrid, nt = pred.shape
    # compute only NSE and KGE
    NSE = np.full(ngrid, np.nan)
    KGE = np.full(ngrid, np.nan)
    
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]

        if ind.shape[0] > 1:  # need at least two points to compute metrics
            xx = x[ind]
            yy = y[ind]

            # NSE (Nash–Sutcliffe Efficiency)
            yymean = yy.mean()
            SST = np.sum((yy-yymean)**2)
            SSRes = np.sum((yy-xx)**2)
            NSE[k] = 1-SSRes/SST

            # KGE (Kling–Gupta Efficiency)
            yystd = np.std(yy)
            xxmean = xx.mean()
            xxstd = np.std(xx)
            Corr = scipy.stats.pearsonr(xx, yy)[0]
            KGE[k] = 1 - np.sqrt((Corr-1)**2 + (xxstd/yystd-1)**2 + (xxmean/yymean-1)**2)

    outDict = dict(NSE=NSE, KGE=KGE)
    return outDict
