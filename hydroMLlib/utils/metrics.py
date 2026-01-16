import numpy as np
import scipy.stats



keyLst = ['NSE', 'KGE']


def statError(pred, target):
    """
    计算dPLHBV主线任务实际使用的评估指标：NSE和KGE
    
    Args:
        pred: 预测值 (ngrid, nt)
        target: 观测值 (ngrid, nt)
    
    Returns:
        dict: 包含NSE和KGE指标的字典
    """
    ngrid, nt = pred.shape
    # 只计算dPLHBV主线任务实际使用的指标：NSE和KGE
    NSE = np.full(ngrid, np.nan)
    KGE = np.full(ngrid, np.nan)
    
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        
        if ind.shape[0] > 1:  # 至少需要两个点来计算NSE和KGE
            xx = x[ind]
            yy = y[ind]
            
            # 计算NSE (Nash-Sutcliffe Efficiency)
            yymean = yy.mean()
            SST = np.sum((yy-yymean)**2)
            SSRes = np.sum((yy-xx)**2)
            NSE[k] = 1-SSRes/SST
            
            # 计算KGE (Kling-Gupta Efficiency)
            yystd = np.std(yy)
            xxmean = xx.mean()
            xxstd = np.std(xx)
            Corr = scipy.stats.pearsonr(xx, yy)[0]
            KGE[k] = 1 - np.sqrt((Corr-1)**2 + (xxstd/yystd-1)**2 + (xxmean/yymean-1)**2)

    # 只返回dPLHBV主线任务实际使用的指标
    outDict = dict(NSE=NSE, KGE=KGE)
    return outDict
