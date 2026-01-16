"""
文件管理工具模块

本模块提供模型训练和预测过程中的文件管理功能，包括配置文件读写、
模型加载和预测文件命名等。主要用于dPLHBV模型的输出文件管理。

主要功能：
- 配置文件管理：读写master.json配置文件，保存训练参数和模型配置
- 模型加载：根据配置文件和训练轮次加载训练好的模型
- 预测文件命名：为预测结果生成标准化的文件名和路径
- 支持多种损失函数和不确定性量化的文件命名规则
"""

import os
import hydroMLlib
from collections import OrderedDict
import json


def wrapMaster(out, optData, optModel, optLoss, optTrain):
    """
    包装主配置文件函数
    
    将训练过程中的各种配置参数包装成有序字典格式，用于保存到master.json文件。
    这个函数是配置文件创建的核心，确保所有训练参数都被正确记录。
    
    参数:
        out (str): 输出目录路径，用于保存模型和配置文件
        optData (dict): 数据配置参数，包含数据集路径、变量列表等
        optModel (dict): 模型配置参数，包含网络结构、超参数等
        optLoss (dict): 损失函数配置参数，包含损失类型、权重等
        optTrain (dict): 训练配置参数，包含学习率、批次大小、训练轮次等
    
    返回:
        OrderedDict: 包含所有配置参数的有序字典
                    - out: 输出路径
                    - data: 数据配置
                    - model: 模型配置
                    - loss: 损失函数配置
                    - train: 训练配置
    
    示例:
        >>> config = wrapMaster('/path/to/output', dataOpt, modelOpt, lossOpt, trainOpt)
        >>> writeMasterFile(config)  # 保存配置文件
    """
    # 使用OrderedDict保持参数顺序，便于配置文件的可读性
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain)
    return mDict


def readMasterFile(out):
    """
    读取主配置文件函数
    
    从指定输出目录读取master.json配置文件，恢复训练参数和模型配置。
    用于模型加载、预测和结果分析时的参数恢复。
    
    参数:
        out (str): 输出目录路径，包含master.json文件的目录
    
    返回:
        OrderedDict: 从配置文件读取的参数字典
                    - out: 输出路径
                    - data: 数据配置参数
                    - model: 模型配置参数
                    - loss: 损失函数配置参数
                    - train: 训练配置参数
    
    异常:
        FileNotFoundError: 当master.json文件不存在时抛出
        json.JSONDecodeError: 当JSON文件格式错误时抛出
    
    示例:
        >>> config = readMasterFile('/path/to/output')
        >>> print(config['train']['nEpoch'])  # 获取训练轮次
    """
    # 构建master.json文件的完整路径
    mFile = os.path.join(out, 'master.json')
    
    # 读取JSON文件，使用OrderedDict保持参数顺序
    with open(mFile, 'r') as fp:
        mDict = json.load(fp, object_pairs_hook=OrderedDict)
    
    print('read master file ' + mFile)
    return mDict


def writeMasterFile(mDict):
    """
    写入主配置文件函数
    
    将配置参数字典保存为master.json文件到指定输出目录。
    如果输出目录不存在，会自动创建。配置文件采用缩进格式便于阅读。
    
    参数:
        mDict (OrderedDict): 包含所有配置参数的有序字典
                            - 必须包含'out'键，指定输出目录
                            - 其他键包含data、model、loss、train等配置
    
    返回:
        str: 输出目录路径
    
    示例:
        >>> config = wrapMaster('/path/to/output', dataOpt, modelOpt, lossOpt, trainOpt)
        >>> outputDir = writeMasterFile(config)
    """
    # 从配置字典中获取输出目录路径
    out = mDict['out']
    
    # 如果输出目录不存在，创建目录
    if not os.path.isdir(out):
        os.makedirs(out)
    
    # 构建master.json文件的完整路径
    mFile = os.path.join(out, 'master.json')
    
    # 将配置字典写入JSON文件，使用缩进格式便于阅读
    with open(mFile, 'w') as fp:
        json.dump(mDict, fp, indent=4)
    
    print('write master file ' + mFile)
    return out


def loadModel(out, epoch=None):
    """
    加载训练好的模型函数
    
    根据输出目录和训练轮次加载训练好的模型。如果未指定轮次，
    则自动从配置文件中读取最后一个训练轮次。
    
    参数:
        out (str): 输出目录路径，包含模型文件和配置文件
        epoch (int, optional): 指定加载的训练轮次
                              - None: 自动从配置文件读取最后轮次（默认）
                              - int: 指定具体的训练轮次
    
    返回:
        torch.nn.Module: 加载的PyTorch模型对象
                        - 模型已加载训练好的权重参数
                        - 可以直接用于预测或继续训练
    
    异常:
        FileNotFoundError: 当模型文件不存在时抛出
        KeyError: 当配置文件中缺少必要信息时抛出
    
    示例:
        >>> model = loadModel('/path/to/output')  # 加载最后轮次的模型
        >>> model = loadModel('/path/to/output', epoch=100)  # 加载第100轮次的模型
    """
    # 如果未指定训练轮次，从配置文件中读取
    if epoch is None:
        mDict = readMasterFile(out)
        epoch = mDict['train']['nEpoch']  # 获取最后训练轮次
    
    # 调用训练模块的loadModel函数加载模型
    model = hydroMLlib.model.train.loadModel(out, epoch)
    return model


def namePred(out, tRange, subset, epoch=None, targLst=None):
    """
    生成预测文件名函数
    
    根据训练配置、时间范围、数据集子集等信息生成预测结果文件的标准化名称。
    支持多种损失函数和不确定性量化的文件命名规则，确保文件名的唯一性和可读性。
    
    参数:
        out (str): 输出目录路径，预测文件将保存到此目录
        tRange (list): 时间范围，包含两个元素 [开始时间, 结束时间]
                      - 格式: [YYYYMMDD, YYYYMMDD]
                      - 用于标识预测的时间段
        subset (str/list): 数据集子集标识
                          - str: 直接使用字符串作为标识
                          - list: 使用列表长度作为标识
        epoch (int, optional): 模型训练轮次
                              - None: 从配置文件读取最后轮次（默认）
                              - int: 指定具体的训练轮次
        targLst (list, optional): 目标变量列表
                                 - None: 从配置文件读取（默认）
                                 - list: 自定义目标变量列表
    
    返回:
        list: 预测文件的完整路径列表
              - 每个元素是一个文件的完整路径
              - 文件格式为CSV
              - 路径包含输出目录和文件名
    
    文件命名规则:
        基础格式: {subset}_{startTime}_{endTime}_ep{epoch}_{target}.csv
        不确定性: {subset}_{startTime}_{endTime}_ep{epoch}_{target}_SigmaX.csv
        
    
    示例:
        >>> files = namePred('/output', [20200101, 20201231], 'test', epoch=100)
        >>> # 返回: ['/output/test_20200101_20201231_ep100_Streamflow.csv']
        
        
    """
    # 读取配置文件获取训练参数
    mDict = readMasterFile(out)
    
    # 确定目标变量列表
    if targLst is not None:
        # 使用用户指定的目标变量列表
        target = targLst
    else:
        # 从配置文件读取目标变量
        if 'name' in mDict['data'].keys() and mDict['data']['name'] == 'hydroMLlib.utils.camels.DataframeCamels':
            # CAMELS数据集默认目标变量为径流
            target = ['Streamflow']
        else:
            # 其他数据集使用配置文件中指定的目标变量
            target = mDict['data']['target']
    
    # 确保target是列表格式
    if type(target) is not list:
        target = [target]
    
    # 获取目标变量数量
    nt = len(target)
    
    # 获取损失函数名称
    lossName = mDict['loss']['name']
    
    # 确定训练轮次
    if epoch is None:
        epoch = mDict['train']['nEpoch']
    
    # 处理子集标识
    if type(subset) is list:
        # 如果子集是列表，使用列表长度作为标识
        subset = str(len(subset))
    
    # 生成基础文件名列表
    fileNameLst = list()
    for k in range(nt):
        # 构建基础测试名称: subset_startTime_endTime_ep{epoch}
        testName = '_'.join(
            [subset, str(tRange[0]),
             str(tRange[1]), 'ep' + str(epoch)])
        
        # 添加目标变量名称
        fileName = '_'.join([testName, target[k]])
        fileNameLst.append(fileName)
        
        # 如果使用SigmaLoss损失函数，添加不确定性量化文件
        if lossName == 'hydroMLlib.utils.loss.SigmaLoss':
            fileName = '_'.join([testName, target[k], 'SigmaX'])
            fileNameLst.append(fileName)
  
    # 生成完整的文件路径列表
    filePathLst = list()
    for fileName in fileNameLst:
        # 构建完整的文件路径
        filePath = os.path.join(out, fileName + '.csv')
        filePathLst.append(filePath)
    
    return filePathLst
