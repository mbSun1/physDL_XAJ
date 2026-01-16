"""
默认配置设置模块

本模块提供dPLHBV主线任务的默认配置参数，采用纯函数式设计。
所有配置通过函数返回，避免模块级语句暴露。

主要功能：
- 提供CAMELS数据集的默认配置参数
- 提供模型训练的默认配置参数
- 提供损失函数的默认配置参数
- 支持配置参数的更新和强制更新
"""

from collections import OrderedDict
from hydroMLlib.utils import camels


def get_data_config():
    """
    获取CAMELS数据集的默认配置参数
    
    返回dPLHBV主线任务使用的CAMELS数据集默认配置，包括数据源、变量选择、
    时间范围、标准化选项等。
    
    Returns:
        OrderedDict: CAMELS数据配置字典，包含以下键值：
            - name: 数据框架类名
            - subset: 站点子集（'All'表示所有站点）
            - varT: 时间序列变量列表（气象强迫数据）
            - varC: 常量变量列表（流域属性数据）
            - target: 目标变量列表（径流）
            - tRange: 时间范围 [开始日期, 结束日期]
            - doNorm: 标准化选项 [时间序列标准化, 常量标准化]
            - rmNan: NaN处理选项 [时间序列NaN处理, 常量NaN处理]
            - basinNorm: 是否进行流域标准化
            - forType: 气象数据源类型
            
    Note:
        - 此函数被dPLHBV主线任务调用，用于获取默认数据配置
        - 时间范围默认为1990-1995年（训练期）
        - 气象数据源默认为NLDAS
        - 支持流域标准化（将径流转换为径流系数）
    """
    data_config = OrderedDict()
    data_config['name'] = 'hydroMLlib.utils.camels.DataframeCamels'
    data_config['subset'] = 'All'
    data_config['varT'] = camels.get_camels_config()['forcingLst']  # 气象强迫变量列表
    data_config['varC'] = camels.get_camels_config()['attrLstSel']  # 流域属性变量列表
    data_config['target'] = ['Streamflow']    # 目标变量（径流）
    data_config['tRange'] = [19900101, 19950101]  # 默认时间范围
    data_config['doNorm'] = [True, True]      # 标准化选项
    data_config['rmNan'] = [True, False]     # NaN处理选项
    data_config['basinNorm'] = True          # 流域标准化
    data_config['forType'] = 'nldas'          # 气象数据源
    return data_config


def get_train_config():
    """
    获取模型训练的默认配置参数
    
    返回dPLHBV主线任务使用的模型训练默认配置，包括批次大小、训练轮数、
    保存频率、随机种子等。
    
    Returns:
        OrderedDict: 训练配置字典，包含以下键值：
            - miniBatch: 列表 [batch_size, 序列长度rho]
            - nEpoch: 训练轮数
            - saveEpoch: 模型保存频率（每多少轮保存一次）
            - seed: 随机种子（None表示不设置）
            - trainBuff: 训练缓冲区大小
            
    Note:
        - 此函数被dPLHBV主线任务调用，用于获取默认训练配置
        - 默认训练100轮，每50轮保存一次模型
        - 支持不同的训练和验证批次大小
        - 随机种子为None，使用系统默认随机状态
    """
    train_config = OrderedDict()
    train_config['miniBatch'] = [100, 200]  # [batch_size, 序列长度rho]
    train_config['nEpoch'] = 100            # 训练轮数
    train_config['saveEpoch'] = 50         # 模型保存频率
    train_config['seed'] = None             # 随机种子
    train_config['trainBuff'] = 0           # 训练缓冲区大小
    return train_config


def get_loss_config():
    """
    获取损失函数的默认配置参数
    
    返回dPLHBV主线任务使用的损失函数默认配置，包括损失函数类型、
    先验分布、权重等。
    
    Returns:
        OrderedDict: 损失函数配置字典，包含以下键值：
            - name: 损失函数类名
            - prior: 先验分布类型
            - weight: 权重参数
            
    Note:
        - 此函数被dPLHBV主线任务调用，用于获取默认损失函数配置
        - 使用RmseLossComb损失函数（RMSE和log-sqrt RMSE的组合）
        - 先验分布为高斯分布
        - 权重为0.0（不使用先验项）
    """
    loss_config = OrderedDict()
    loss_config['name'] = 'hydroMLlib.utils.criterion.RmseLossComb'  # 损失函数类名
    loss_config['prior'] = 'gauss'                                   # 先验分布类型
    loss_config['weight'] = 0.0                                     # 权重参数
    return loss_config


def update_config(opt, **kw):
    """
    更新配置参数
    
    根据提供的关键字参数更新配置字典，支持类型检查和错误处理。
    
    Args:
        opt (OrderedDict): 要更新的配置字典
        **kw: 要更新的配置参数，键值对形式
        
    Returns:
        OrderedDict: 更新后的配置字典
        
    Note:
        - 此函数被dPLHBV主线任务调用，用于更新默认配置
        - 支持类型检查，确保参数类型与原始配置一致
        - 对于特殊参数（如subset、seed），直接赋值不进行类型转换
        - 如果参数不存在或类型不匹配，会跳过并打印警告信息
        - 保持原始配置字典的顺序（OrderedDict）
    """
    for key in kw:
        if key in opt:
            try:
                # 特殊参数直接赋值，不进行类型转换
                if key in ['subset', 'seed']:
                    opt[key] = kw[key]
                else:
                    # 尝试类型转换，保持与原始配置相同的类型
                    opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print(f'跳过参数 {key}: 类型错误')
        else:
            print(f'跳过参数 {key}: 不在配置字典中')
    return opt


def force_update_config(opt, **kw):
    """
    强制更新配置参数
    
    直接更新配置字典，不进行类型检查，用于特殊情况下的配置覆盖。
    
    Args:
        opt (OrderedDict): 要更新的配置字典
        **kw: 要更新的配置参数，键值对形式
        
    Returns:
        OrderedDict: 更新后的配置字典
        
    Note:
        - 此函数被dPLHBV主线任务调用，用于强制更新配置
        - 不进行类型检查，直接覆盖原有值
        - 如果参数不存在，会添加新的键值对
        - 适用于需要完全覆盖配置的场景
        - 保持原始配置字典的顺序（OrderedDict）
    """
    for key in kw:
        opt[key] = kw[key]
    return opt

