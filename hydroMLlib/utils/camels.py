"""
CAMELS数据集处理模块

本模块提供CAMELS（Catchment Attributes and Meteorology for Large-sample Studies）
数据集的读取、处理和管理功能，专门为dPLHBV主线任务优化。

主要功能：
- 读取CAMELS数据集的站点信息、气象强迫数据、流域属性数据
- 提供数据标准化和预处理功能
- 支持多种数据源（NLDAS、Daymet、Maurer等）
- 为dPLHBV模型提供标准化的数据接口

模块结构（阅读指引）：
- 基础依赖与静态配置：get_camels_config
- 内部通用辅助：_resolve_dir_gage, _resolve_attr_dir
- 基础读取函数：readGageInfo, readUsgsGage, readUsgs, readForcingGage, readForcing, readcsvGage
- 属性读取函数：readAttrAll, readAttr
- 统计与标准化：calStat, getStatDic, transNormbyDic, basinNorm
- 数据框封装：DataframeCamels
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
import time as time_module  # Python内置time模块
from hydroMLlib.utils import time
from pandas.api.types import is_numeric_dtype, is_string_dtype
import json
 



# =============================================================================
# 静态配置访问函数（避免模块级变量暴露）
# =============================================================================

def get_camels_config():
    """
    获取CAMELS数据集的静态配置参数
    
    返回dPLHBV主线任务使用的所有静态配置，包括观测时间范围、时间序列数组、
    时间步数、气象强迫变量列表和流域属性变量列表。
    
    Returns:
        dict: 包含以下键值的配置字典：
            - tRangeobs: 观测数据时间范围 [19790101, 20150101]
            - tLstobs: 观测时间序列数组
            - ntobs: 观测数据时间步数
            - forcingLst: 气象强迫变量列表
            - attrLstSel: 流域属性变量列表
            
    Note:
        - 此函数被dPLHBV主线任务调用，用于获取所有静态配置
        - 避免模块导入时执行语句，实现纯函数式设计
        - 每次调用返回新的配置字典，避免状态污染
    """
    config = {}
    
    # 观测数据时间范围（径流观测数据）
    config['tRangeobs'] = [19790101, 20150101]  # 径流观测数据时间范围
    config['tLstobs'] = time.tRange2Array(config['tRangeobs'])  # 观测时间序列数组
    config['ntobs'] = len(config['tLstobs'])  # 观测数据时间步数
    
    # 气象强迫数据变量列表（dPLHBV主线任务使用的变量）
    config['forcingLst'] = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'vp']
    # dayl: 日长 (hours)
    # prcp: 降水量 (mm/day)
    # srad: 太阳辐射 (W/m2)
    # tmax: 最高温度 (C)
    # tmin: 最低温度 (C)
    # vp: 水汽压 (kPa)
    
    # 流域属性变量列表（dPLHBV主线任务使用的属性）
    config['attrLstSel'] = [
        'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
        'lai_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
        'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
        'max_water_content', 'geol_1st_class', 'geol_2nd_class', 'geol_porostiy',
        'geol_permeability'
    ]
    # elev_mean: 平均海拔 (m)
    # slope_mean: 平均坡度 (degrees)
    # area_gages2: 流域面积 (km2)
    # frac_forest: 森林覆盖率
    # lai_max: 最大叶面积指数
    # lai_diff: 叶面积指数差异
    # dom_land_cover_frac: 主要土地覆盖类型比例
    # dom_land_cover: 主要土地覆盖类型
    # root_depth_50: 50%根深 (mm)
    # soil_depth_statsgo: 土壤深度 (mm)
    # soil_porosity: 土壤孔隙度
    # soil_conductivity: 土壤导水率 (mm/day)
    # max_water_content: 最大含水量 (mm)
    # geol_1st_class: 主要地质类型
    # geol_2nd_class: 次要地质类型
    # geol_porostiy: 地质孔隙度
    # geol_permeability: 地质渗透率 (mm/day)
    
    return config

 

def readGageInfo(dirDB):
    """
    读取CAMELS数据集的站点基本信息
    
    从CAMELS数据集的gauge_information.txt文件中读取所有站点的基本信息，
    包括HUC代码、站点ID、站点名称、经纬度坐标和流域面积等。
    
    Args:
        dirDB (str): CAMELS数据集根目录路径
        
    Returns:
        dict: 包含站点信息的字典，键包括：
            - 'huc': HUC代码数组
            - 'id': 站点ID数组  
            - 'name': 站点名称列表
            - 'lat': 纬度数组
            - 'lon': 经度数组
            - 'area': 流域面积数组 (km2)
            

    """
    gageFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
                            'basin_dataset_public_v1p2', 'basin_metadata',
                            'gauge_information.txt')

    data = pd.read_csv(gageFile, sep='\t', header=None, skiprows=1)
    # header gives some troubles. Skip and hardcode
    fieldLst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
    out = dict()
    for s in fieldLst:
        if s == 'name':
            out[s] = data[fieldLst.index(s)].values.tolist()
        else:
            out[s] = data[fieldLst.index(s)].values
    return out

def readUsgsGage(usgsId, *, readQc=False, dirDB_opt=None, gageDict_opt=None):
    """
    读取单个站点的USGS径流观测数据
    
    从CAMELS数据集中读取指定站点的USGS径流观测数据，支持数据质量控制。
    
    Args:
        usgsId (int): USGS站点ID
        readQc (bool, optional): 是否读取数据质量控制信息。默认为False
        
    Returns:
        numpy.ndarray or tuple: 
            - 如果readQc=False: 返回径流观测数据数组 (get_camels_config()['ntobs'],)
            - 如果readQc=True: 返回(径流数据, 质量控制)元组
            
    Note:
        - 径流数据单位：ft3/s
        - 质量控制代码：1=A(优秀), 2=A:e(优秀但估计), 3=M(修正)
        - 负值被转换为NaN
        - 此函数被readUsgs()调用
    """
    _dirDB, _gageDict = dirDB_opt, gageDict_opt
    ind = np.argwhere(_gageDict['id'] == usgsId)[0][0]
    huc = _gageDict['huc'][ind]
    usgsFile = os.path.join(_dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
                            'basin_dataset_public_v1p2', 'usgs_streamflow',
                            str(huc).zfill(2),
                            '%08d_streamflow_qc.txt' % (usgsId))
    try:
        dataTemp = pd.read_csv(usgsFile, sep=r'\s+', header=None)
    except pd.errors.ParserError:
        try:
            dataTemp = pd.read_csv(usgsFile, sep=r'\s+', header=None, engine='python')
        except Exception as e:
            print(f"警告: 无法读取文件 {usgsFile}: {e}")
            obs = np.full(get_camels_config()['ntobs'], np.nan)
            if readQc is True:
                qc = np.full(get_camels_config()['ntobs'], 3)  # 默认质量为M
            return obs, qc if readQc else obs
    obs = dataTemp[4].values
    obs[obs < 0] = np.nan
    if readQc is True:
        qcDict = {'A': 1, 'A:e': 2, 'M': 3}
        qc = np.array([qcDict[x] for x in dataTemp[5]])
    if len(obs) != get_camels_config()['ntobs']:
        out = np.full([get_camels_config()['ntobs']], np.nan)
        dfDate = dataTemp[[1, 2, 3]]
        dfDate.columns = ['year', 'month', 'day']
        date = pd.to_datetime(dfDate).values.astype('datetime64[D]')
        [C, ind1, ind2] = np.intersect1d(date, get_camels_config()['tLstobs'], return_indices=True)
        out[ind2] = obs
        if readQc is True:
            outQc = np.full([get_camels_config()['ntobs']], np.nan)
            outQc[ind2] = qc
    else:
        out = obs
        if readQc is True:
            outQc = qc

    if readQc is True:
        return out, outQc
    else:
        return out


def readUsgs(usgsIdLst, *, dirDB_opt=None, gageDict_opt=None):
    """
    批量读取多个站点的USGS径流观测数据
    
    并行读取多个站点的USGS径流观测数据，返回标准化的数据矩阵。
    
    Args:
        usgsIdLst (list or numpy.ndarray): USGS站点ID列表
        
    Returns:
        numpy.ndarray: 径流观测数据矩阵，形状为 (n_sites, get_camels_config()['ntobs'])
                      - n_sites: 站点数量
                      - get_camels_config()['ntobs']: 观测时间步数
                      
    Note:
        - 此函数被DataframeCamels.getDataObs()调用
        - 数据单位：ft3/s
        - 缺失数据用NaN表示
    """
    t0 = time_module.time()
    y = np.empty([len(usgsIdLst), get_camels_config()['ntobs']])
    for k in range(len(usgsIdLst)):
        dataObs = readUsgsGage(usgsIdLst[k], dirDB_opt=dirDB_opt, gageDict_opt=gageDict_opt)
        y[k, :] = dataObs
    # 移除调试输出
    # print("read usgs streamflow", time_module.time() - t0)
    return y


def readForcingGage(usgsId, varLst, *, dataset, nt, dirDB_opt=None, gageDict_opt=None):
    """
    读取单个站点的气象强迫数据
    
    从CAMELS数据集中读取指定站点的气象强迫数据，支持多种数据源。
    
    Args:
        usgsId (int): USGS站点ID
        varLst (list): 需要读取的气象变量列表
        dataset (str): 数据源名称，支持：
            - 'daymet': Daymet数据集
            - 'nldas': NLDAS数据集  
            - 'nldas_extended': 扩展NLDAS数据集
            - 'maurer': Maurer数据集
            - 'maurer_extended': 扩展Maurer数据集
        nt (int): 时间步数
        
    Returns:
        numpy.ndarray: 气象强迫数据矩阵，形状为 (nt, n_vars)
                      - nt: 时间步数
                      - n_vars: 变量数量
                      
    Note:
        - 此函数被readForcing()调用
        - 支持的气象变量：dayl, prcp, srad, swe, tmax, tmin, vp
        - 数据文件格式：空格分隔的文本文件
        - 缺失数据用NaN表示
    """
    # dataset = daymet or maurer or nldas or nldas_extedned with tmaxtmin
    forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    _dirDB, _gageDict = dirDB_opt, gageDict_opt
    ind = np.argwhere(_gageDict['id'] == usgsId)[0][0]
    huc = _gageDict['huc'][ind]

    dataFolder = os.path.join(
        _dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
        'basin_dataset_public_v1p2', 'basin_mean_forcing')
    if dataset == 'daymet':
        tempS = 'cida'
    elif dataset == 'nldas_extended':
        tempS = 'nldas'
    elif dataset == 'maurer_extended':
        tempS = 'maurer'
    else:
        tempS = dataset
    dataFile = os.path.join(dataFolder, dataset,
                            str(huc).zfill(2),
                            '%08d_lump_%s_forcing_leap.txt' % (usgsId, tempS))
    try:
        dataTemp = pd.read_csv(dataFile, sep=r'\s+', header=None, skiprows=4)
    except pd.errors.ParserError:
        # 如果C引擎失败，尝试使用Python引擎
        try:
            dataTemp = pd.read_csv(dataFile, sep=r'\s+', header=None, skiprows=4, engine='python')
        except Exception as e:
            print(f"警告: 无法读取文件 {dataFile}: {e}")
            # 返回空数据或默认值
            nf = len(varLst)
            out = np.full([nt, nf], np.nan)
            return out
    nf = len(varLst)
    out = np.empty([nt, nf])
    for k in range(nf):
        # assume all files are of same columns. May check later.
        ind = get_camels_config()['forcingLst'].index(varLst[k])
        out[:, k] = dataTemp[ind + 4].values
    return out


def readForcing(usgsIdLst, varLst, fordata, nt, *, dirDB_opt=None, gageDict_opt=None):
    """
    批量读取多个站点的气象强迫数据
    
    并行读取多个站点的气象强迫数据，返回标准化的数据矩阵。
    
    Args:
        usgsIdLst (list or numpy.ndarray): USGS站点ID列表
        varLst (list): 需要读取的气象变量列表
        fordata (str): 数据源名称（同readForcingGage的dataset参数）
        nt (int): 时间步数
        
    Returns:
        numpy.ndarray: 气象强迫数据矩阵，形状为 (n_sites, nt, n_vars)
                      - n_sites: 站点数量
                      - nt: 时间步数
                      - n_vars: 变量数量
                      
    Note:
        - 此函数被DataframeCamels.getDataTs()调用
        - 数据单位取决于具体变量（见forcingLst注释）
        - 缺失数据用NaN表示
    """
    t0 = time_module.time()
    x = np.empty([len(usgsIdLst), nt, len(varLst)])
    for k in range(len(usgsIdLst)):
        data = readForcingGage(usgsIdLst[k], varLst, dataset=fordata, nt=nt,
                               dirDB_opt=dirDB_opt, gageDict_opt=gageDict_opt)
        x[k, :, :] = data
    # 移除调试输出
    # print("read usgs streamflow", time_module.time() - t0)
    return x


def _resolve_attr_dir(root_dir):
    """
    解析CAMELS属性数据目录路径
    
    自动检测CAMELS属性数据文件的实际存储位置，支持多种目录结构布局。
    
    Args:
        root_dir (str): CAMELS数据集根目录路径
        
    Returns:
        str: 属性数据文件的实际目录路径
        
    Note:
        - 此函数为内部辅助函数，被readAttrAll()调用
        - 支持三种目录结构：
            1. root_dir/camels_attributes_v2.0/camels_attributes_v2.0/
            2. root_dir/camels_attributes_v2.0/
            3. root_dir/ (属性文件直接放在根目录)
        - 通过检查camels_topo.txt文件的存在来验证目录结构
        - 如果所有候选路径都不存在，返回cand2作为默认值
    """
    # Try standard CAMELS v2 layout first
    cand1 = os.path.join(root_dir, 'camels_attributes_v2.0', 'camels_attributes_v2.0')
    cand2 = os.path.join(root_dir, 'camels_attributes_v2.0')
    # Some datasets place attribute txt files directly under root
    cand3 = root_dir
    for p in [cand1, cand2, cand3]:
        topo_file = os.path.join(p, 'camels_topo.txt')
        if os.path.isdir(p) and os.path.isfile(topo_file):
            return p
    # Fallback to cand2 even if files missing (will error later with clear path)
    return cand2


def readAttrAll(*, saveDict=False, dirDB_opt=None, gageDict_opt=None):
    """
    读取所有站点的流域属性数据
    
    从CAMELS数据集中读取所有站点的完整流域属性数据，包括地形、气候、水文、
    植被、土壤和地质等六大类属性。
    
    Args:
        saveDict (bool, optional): 是否保存变量字典到文件。默认为False
        
    Returns:
        tuple: (属性数据矩阵, 变量列表)
            - 属性数据矩阵 (numpy.ndarray): 形状为 (n_sites, n_attrs)
              - n_sites: 站点数量
              - n_attrs: 属性变量总数
            - 变量列表 (list): 所有属性变量的名称列表
            
    Note:
        - 此函数被readAttr()调用，用于获取完整的属性数据
        - 支持六大类属性：topo(地形), clim(气候), hydro(水文), vege(植被), soil(土壤), geol(地质)
        - 自动处理字符串类型变量（如地质类型）的因子化编码
        - 数值类型变量直接使用原始值
        - 如果saveDict=True，会保存因子化字典和变量字典到JSON文件
        - 数据文件格式：分号分隔的文本文件 (camels_*.txt)
    """
    _dirDB, _gageDict = dirDB_opt, gageDict_opt
    dataFolder = _resolve_attr_dir(_dirDB)
    fDict = dict()  # factorize dict
    varDict = dict()
    varLst = list()
    outLst = list()
    keyLst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']

    for key in keyLst:
        dataFile = os.path.join(dataFolder, 'camels_' + key + '.txt')
        dataTemp = pd.read_csv(dataFile, sep=';')
        varLstTemp = list(dataTemp.columns[1:])
        varDict[key] = varLstTemp
        varLst.extend(varLstTemp)
        k = 0
        nGage = len(_gageDict['id'])
        outTemp = np.full([nGage, len(varLstTemp)], np.nan)
        for field in varLstTemp:
            if is_string_dtype(dataTemp[field]):
                value, ref = pd.factorize(dataTemp[field], sort=True)
                outTemp[:, k] = value
                fDict[field] = ref.tolist()
            elif is_numeric_dtype(dataTemp[field]):
                outTemp[:, k] = dataTemp[field].values
            k = k + 1
        outLst.append(outTemp)
    out = np.concatenate(outLst, 1)
    if saveDict is True:
        fileName = os.path.join(dataFolder, 'dictFactorize.json')
        with open(fileName, 'w') as fp:
            json.dump(fDict, fp, indent=4)
        fileName = os.path.join(dataFolder, 'dictAttribute.json')
        with open(fileName, 'w') as fp:
            json.dump(varDict, fp, indent=4)
    return out, varLst


def readAttr(usgsIdLst, varLst, *, dirDB_opt=None, gageDict_opt=None):
    """
    读取指定站点的流域属性数据
    
    从完整的属性数据中提取指定站点和指定变量的属性数据。
    
    Args:
        usgsIdLst (list or numpy.ndarray): 需要读取的站点ID列表
        varLst (list): 需要读取的属性变量列表
        
    Returns:
        numpy.ndarray: 属性数据矩阵，形状为 (n_sites, n_vars)
                      - n_sites: 站点数量
                      - n_vars: 变量数量
                      
    Note:
        - 此函数被DataframeCamels.getDataConst()和basinNorm()调用
        - 内部调用readAttrAll()获取完整的属性数据
        - 支持按站点ID和变量名进行精确提取
        - 保持输入站点ID列表的顺序
        - 如果某个站点ID不存在，对应的行将包含NaN值
        - 如果某个变量名不存在，会抛出KeyError异常
    """
    _dirDB, _gageDict = dirDB_opt, gageDict_opt
    attrAll, varLstAll = readAttrAll(dirDB_opt=_dirDB, gageDict_opt=_gageDict)
    indVar = list()
    for var in varLst:
        indVar.append(varLstAll.index(var))
    idLstAll = _gageDict['id']
    indGrid = np.full(usgsIdLst.size, -1, dtype=int)
    for ii in range(usgsIdLst.size):
        tempind = np.where(idLstAll==usgsIdLst[ii])
        indGrid[ii] = tempind[0][0]
    temp = attrAll[indGrid, :]
    out = temp[:, indVar]
    return out



def readcsvGage(dataDir, usgsId, varLst, ntime):
    """
    读取CSV格式的站点数据
    
    从CSV文件中读取指定站点的数据，主要用于读取PET（潜在蒸散发）等额外数据。
    
    Args:
        dataDir (str): CSV数据文件目录路径
        usgsId (int): USGS站点ID
        varLst (list): 需要读取的变量列表
        ntime (int): 时间步数
        
    Returns:
        numpy.ndarray: 数据矩阵，形状为 (ntime, n_vars)
                      - ntime: 时间步数
                      - n_vars: 变量数量
                      
    Note:
        - 此函数被dPLHBV主线任务直接调用，用于读取PET数据
        - 文件格式：CSV格式，包含时间序列数据
        - 文件命名：{usgsId}.csv
    """
    dataFile = os.path.join(dataDir, str(usgsId)+'.csv')
    dataTemp = pd.read_csv(dataFile)
    nf = len(varLst)
    out = np.empty([ntime, nf])
    for k in range(nf):
        # assume all files are of same columns. May check later.
        out[:, k] = dataTemp[varLst[k]].values
    return out



def calStat(x):
    """
    计算数据的基本统计信息
    
    计算输入数据的基本统计信息，包括10%分位数、90%分位数、均值和标准差。
    
    Args:
        x (numpy.ndarray): 输入数据，可以是任意维度的数组
        
    Returns:
        list: 统计信息列表 [p10, p90, mean, std]
            - p10: 10%分位数
            - p90: 90%分位数  
            - mean: 均值
            - std: 标准差
            
    Note:
        - 自动处理NaN值（忽略NaN值进行计算）
        - 如果所有值都是NaN，返回默认值[0.0, 1.0, 0.0, 1.0]
        - 如果标准差小于0.001，设置为1.0
        - 此函数被getStatDic()调用
    """
    a = x.flatten()
    b = a[~np.isnan(a)] # kick out Nan
    
    # Check if array is empty after removing NaN values
    if len(b) == 0:
        return [0.0, 1.0, 0.0, 1.0]  # [p10, p90, mean, std]
    
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]



def getStatDic(attrLst = None, attrdata=None, seriesLst = None, seriesdata=None):
    """
    生成统计信息字典
    
    为指定的属性数据和时间序列数据生成统计信息字典，用于数据标准化。
    
    Args:
        attrLst (list, optional): 属性变量列表
        attrdata (numpy.ndarray, optional): 属性数据矩阵，形状为 (n_sites, n_attrs)
        seriesLst (list, optional): 时间序列变量列表
        seriesdata (numpy.ndarray, optional): 时间序列数据矩阵，形状为 (n_sites, n_time, n_vars)
        
    Returns:
        dict: 统计信息字典，键为变量名，值为统计信息列表 [p10, p90, mean, std]
        
    Note:
        - 此函数被dPLHBV主线任务直接调用，用于生成标准化所需的统计信息
        - 统计信息用于transNormbyDic()函数进行数据标准化
        - 支持同时处理属性数据和时间序列数据
    """
    statDict = dict()
    # series data
    if seriesLst is not None:
        for k in range(len(seriesLst)):
            var = seriesLst[k]
            # 简化统计计算，统一使用calStat函数
            statDict[var] = calStat(seriesdata[:, :, k])
    # const attribute
    if attrLst is not None:
        for k in range(len(attrLst)):
            var = attrLst[k]
            statDict[var] = calStat(attrdata[:, k])
    return statDict



def transNormbyDic(x, varLst, staDic, *, toNorm):
    """
    基于统计字典进行数据标准化或反标准化
    
    使用预计算的统计信息对数据进行标准化（Z-score标准化）或反标准化处理。
    对于径流和降水等变量，使用特殊的对数变换处理。
    
    Args:
        x (numpy.ndarray): 输入数据矩阵，形状为 (n_sites, n_time, n_vars) 或 (n_sites, n_vars)
        varLst (list or str): 变量列表或单个变量名
        staDic (dict): 统计信息字典，键为变量名，值为 [p10, p90, mean, std]
        toNorm (bool): 是否进行标准化
            - True: 标准化（原始数据 -> 标准化数据）
            - False: 反标准化（标准化数据 -> 原始数据）
            
    Returns:
        numpy.ndarray: 处理后的数据矩阵，形状与输入相同
        
    Note:
        - 此函数被dPLHBV主线任务直接调用，用于数据标准化
        - 对于径流和降水变量，使用log10(sqrt(x)+0.1)变换
        - 支持2D和3D数据矩阵
        - 特殊变量：prcp, usgsFlow, Precip, runoff, Runoff, Runofferror
    """
    if type(varLst) is str:
        varLst = [varLst]
    out = np.zeros(x.shape)
    for k in range(len(varLst)):
        var = varLst[k]
        stat = staDic[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if var in ['prcp', 'usgsFlow', 'Precip', 'runoff', 'Runoff', 'Runofferror']:
                    temp = np.log10(np.sqrt(x[:, :, k])+0.1)
                    # temp = np.sqrt(x[:, :, k])
                    out[:, :, k] = (temp - stat[2]) / stat[3]
                else:
                    out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var in ['prcp', 'usgsFlow', 'Precip', 'runoff', 'Runoff', 'Runofferror']:
                    temp = np.log10(np.sqrt(x[:, k])+0.1)
                    # temp = np.sqrt(x[:, :, k])
                    out[:, k] = (temp - stat[2]) / stat[3]
                else:
                    out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var in ['prcp', 'usgsFlow', 'Precip', 'runoff', 'Runoff', 'Runofferror']:
                    temptrans = np.power(10,out[:, :, k])-0.1
                    # temptrans = out[:, :, k]
                    temptrans[temptrans<0] = 0 # set negative as zero
                    out[:, :, k] = (temptrans)**2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var in ['prcp', 'usgsFlow', 'Precip', 'runoff', 'Runoff', 'Runofferror']:
                    temptrans = np.power(10,out[:, k])-0.1
                    # temptrans = out[:, :, k]
                    temptrans[temptrans < 0] = 0
                    out[:, k] = (temptrans)**2
    return out

def basinNorm(x, gageid, toNorm, *, dirDB_opt=None, gageDict_opt=None):
    # for regional training, gageid should be numpyarray
    if type(gageid) is str:
        if gageid == 'All':
            if dirDB_opt is None or gageDict_opt is None:
                raise ValueError('dirDB_opt 与 gageDict_opt 必须显式传入')
            gageid = gageDict_opt['id']
    nd = len(x.shape)
    basinarea = readAttr(gageid, ['area_gages2'], dirDB_opt=dirDB_opt, gageDict_opt=gageDict_opt)
    meanprep = readAttr(gageid, ['p_mean'], dirDB_opt=dirDB_opt, gageDict_opt=gageDict_opt)
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:,:,0] # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    if toNorm is True:
        flow = (x * 0.0283168 * 3600 * 24) / ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3))) # (m^3/day)/(m^3/day)
    else:

        flow = x * ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))/(0.0283168 * 3600 * 24)
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow





class DataframeCamels:
    """
    CAMELS数据集框架类
    
    提供CAMELS数据集的统一访问接口，支持读取气象强迫数据、径流观测数据和流域属性数据。
    
    Attributes:
        subset (str or list): 站点子集，'All'表示所有站点，或指定站点ID列表
        tRange (list): 时间范围 [start_date, end_date]，格式为YYYYMMDD
        forType (str): 气象数据源类型，支持'nldas', 'daymet', 'maurer'等
        usgsId (numpy.ndarray): 站点ID数组
        crd (numpy.ndarray): 站点坐标数组，形状为 (n_sites, 2)，[lat, lon]
        time (numpy.ndarray): 时间序列数组
    """
    
    def __init__(self, *, subset='All', tRange, forType='nldas', dirDB_opt=None, gageDict_opt=None):
        """
        初始化CAMELS数据框架
        
        Args:
            subset (str or list, optional): 站点子集
                - 'All': 使用所有可用站点
                - list: 指定站点ID列表
            tRange (list): 时间范围 [start_date, end_date]，格式为YYYYMMDD
            forType (str, optional): 气象数据源类型。默认为'nldas'
                - 'nldas': NLDAS数据集
                - 'daymet': Daymet数据集
                - 'maurer': Maurer数据集
                
        Raises:
            Exception: 当subset格式不正确时抛出异常
            
        Note:
            - 此类被dPLHBV主线任务直接使用
            - 需在外部通过 readGageInfo(rootDB) 显式获取 gageDict 并传入
            - 支持部分站点缺失的情况（坐标设为0）
        """
        self.subset = subset        
        self._dirDB = dirDB_opt
        self._gageDict = gageDict_opt
        if subset == 'All':  # change to read subset later
            self.usgsId = self._gageDict['id']
            crd = np.zeros([len(self.usgsId), 2])
            crd[:, 0] = self._gageDict['lat']
            crd[:, 1] = self._gageDict['lon']
            self.crd = crd
        elif type(subset) is list:
            self.usgsId = np.array(subset)
            crd = np.zeros([len(self.usgsId), 2])
            ind = np.full(len(self.usgsId), -1, dtype=int)  # 使用-1代替np.nan
            for ii in range(len(self.usgsId)):
                tempind = np.where(self._gageDict['id'] == self.usgsId[ii])
                if len(tempind[0]) > 0:
                    ind[ii] = tempind[0][0]
            # 只处理找到的索引
            valid_indices = ind >= 0
            if np.any(valid_indices):
                crd[valid_indices, 0] = self._gageDict['lat'][ind[valid_indices]]
                crd[valid_indices, 1] = self._gageDict['lon'][ind[valid_indices]]
            self.crd = crd
        else:
            raise Exception('The format of subset is not correct!')
        self.time = time.tRange2Array(tRange)
        self.forType = forType

    def getGeo(self):
        """
        获取站点地理坐标信息
        
        Returns:
            numpy.ndarray: 站点坐标数组，形状为 (n_sites, 2)
                          - 第0列：纬度 (lat)
                          - 第1列：经度 (lon)
        """
        return self.crd

    def getT(self):
        """
        获取时间序列信息
        
        Returns:
            numpy.ndarray: 时间序列数组，包含所有时间步的日期信息
        """
        return self.time

    def getDataObs(self, *, doNorm=True, rmNan=True, basinnorm = True):
        """
        获取径流观测数据
        
        读取指定站点的USGS径流观测数据，支持流域标准化、数据标准化和NaN处理。
        
        Args:
            doNorm (bool, optional): 是否进行数据标准化。默认为True
            rmNan (bool, optional): 是否处理NaN值。默认为True
            basinnorm (bool, optional): 是否进行流域标准化。默认为True
            
        Returns:
            numpy.ndarray: 径流观测数据矩阵，形状为 (n_sites, n_time, 1)
                          - n_sites: 站点数量
                          - n_time: 时间步数
                          - 1: 径流变量维度
                          
        Note:
            - 此方法被dPLHBV主线任务调用，通常使用doNorm=False和rmNan=False
            - 数据单位：ft3/s（原始）或无量纲（流域标准化后）
            - 流域标准化：将径流转换为径流系数（径流/降水）
            - 标准化在外部使用transNormbyDic()处理
        """
        data = readUsgs(self.usgsId, dirDB_opt=self._dirDB, gageDict_opt=self._gageDict)
        if basinnorm is True:
            data = basinNorm(data, gageid=self.usgsId, toNorm=True,
                             dirDB_opt=self._dirDB, gageDict_opt=self._gageDict)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, get_camels_config()['tLstobs'], return_indices=True)
        data = data[:, ind2, :]
        # dPLHBV主线任务使用doNorm=False和rmNan=False，简化处理
        if doNorm is True:
            pass  # 标准化在外部使用transNormbyDic处理
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def getDataTs(self, *, varLst=None, doNorm=True, rmNan=True):
        """
        获取气象强迫时间序列数据
        
        读取指定站点的气象强迫数据，支持多种数据源和变量组合。
        
        Args:
            varLst (list or str, optional): 气象变量列表。默认为forcingLst
                - 支持变量：dayl, prcp, srad, tmax, tmin, vp, tmean
                - 单个变量名或变量列表
            doNorm (bool, optional): 是否进行数据标准化。默认为True
            rmNan (bool, optional): 是否处理NaN值。默认为True
            
        Returns:
            numpy.ndarray: 气象强迫数据矩阵，形状为 (n_sites, n_time, n_vars)
                          - n_sites: 站点数量
                          - n_time: 时间步数
                          - n_vars: 变量数量
                          
        Note:
            - 此方法被dPLHBV主线任务调用，通常使用doNorm=False和rmNan=False
            - 支持特殊处理：Daymet数据源的tmean变量由tmax和tmin计算得出
            - 不同数据源的时间范围可能不同（Maurer: 1980-2009, 其他: 1980-2015）
            - 数据单位取决于具体变量（见forcingLst注释）
            - 标准化在外部使用transNormbyDic()处理
        """
        if varLst is None:
            varLst = get_camels_config()['forcingLst']
        if type(varLst) is str:
            varLst = [varLst]
        if self.forType in ['maurer', 'maurer_extended']:
            tRange = [19800101, 20090101]
        else:
            tRange = [19800101, 20150101]
        tLst = time.tRange2Array(tRange)
        nt = len(tLst)
        # read ts forcing
        if self.forType in ['daymet'] and 'tmean' in varLst:
            # 移除调试输出
            # print('daymet tmean was used!')
            tmeanind = varLst.index('tmean')
            varLstex = [ivar for ivar in varLst if ivar != 'tmean']
            data = readForcing(self.usgsId, varLstex, fordata=self.forType, nt=nt,
                               dirDB_opt=self._dirDB, gageDict_opt=self._gageDict)
            tmaxmin = readForcing(self.usgsId, ['tmax','tmin'], fordata=self.forType, nt=nt,
                                  dirDB_opt=self._dirDB, gageDict_opt=self._gageDict)
            tmeandata = np.mean(tmaxmin, axis=2, keepdims=True)
            data = np.concatenate((data[:,:,0:tmeanind], tmeandata, data[:,:,tmeanind:]), axis=2)
        else:
            data = readForcing(self.usgsId, varLst, fordata=self.forType, nt=nt,
                               dirDB_opt=self._dirDB, gageDict_opt=self._gageDict)
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        # dPLHBV主线任务使用doNorm=False和rmNan=False，简化处理
        if doNorm is True:
            pass  # 标准化在外部使用transNormbyDic处理
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def getDataConst(self, *, varLst=None, doNorm=True, rmNan=True, SAOpt=None):
        """
        获取流域属性数据
        
        读取指定站点的流域属性数据，支持敏感性分析（SA）和数据处理选项。
        
        Args:
            varLst (list or str, optional): 属性变量列表。默认为attrLstSel
                - 支持变量：elev_mean, slope_mean, area_gages2, frac_forest等
                - 单个变量名或变量列表
            doNorm (bool, optional): 是否进行数据标准化。默认为True
            rmNan (bool, optional): 是否处理NaN值。默认为True
            SAOpt (tuple, optional): 敏感性分析选项。默认为None
                - 格式：(变量名, 变化因子)
                - 例如：('area_gages2', 0.1) 表示面积增加10%
                
        Returns:
            numpy.ndarray: 流域属性数据矩阵，形状为 (n_sites, n_attrs)
                          - n_sites: 站点数量
                          - n_attrs: 属性变量数量
                          
        Note:
            - 此方法被dPLHBV主线任务调用，通常使用doNorm=False和rmNan=False
            - 支持敏感性分析：可以按指定因子调整特定属性值
            - 数据单位取决于具体属性（见attrLstSel注释）
            - 标准化在外部使用transNormbyDic()处理
        """
        if varLst is None:
            varLst = get_camels_config()['attrLstSel']
        if type(varLst) is str:
            varLst = [varLst]
        data = readAttr(self.usgsId, varLst, dirDB_opt=self._dirDB, gageDict_opt=self._gageDict)
        if SAOpt is not None:
            SAname, SAfac = SAOpt
            # find the index of target constant
            indVar = varLst.index(SAname)
            data[:, indVar] = data[:, indVar] * (1 + SAfac)
        # dPLHBV主线任务使用doNorm=False和rmNan=False，简化处理
        if doNorm is True:
            pass  # 标准化在外部使用transNormbyDic处理
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data


