"""
dPLXAJ Dynamic Parameter Training Script
======================================
训练动态参数 XAJ 模型的脚本
- 使用 LSTM/GRU 等作为参数预测骨干网络
- 部分 XAJ 参数为动态（随时间变化）
- 支持多种 RNN 类型和配置选项
"""

# =============================================================================
# 导入库和基础设置
# =============================================================================
from hydroMLlib.utils import camels, defaultSetting, fileManager, time, criterion
from hydroMLlib.model import xaj_static, xaj_dynamic, train

import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
import datetime as dt
import argparse


def parse_args():
    """
    解析命令行参数
    
    该函数定义了所有动态参数训练脚本需要的命令行参数，包括：
    - 实验模式配置：流域选择、预热选项、参数模式、气象数据源
    - 模型架构配置：RNN类型、批次大小、序列长度、隐藏层大小、XAJ参数个数
    - 时间配置：训练时间、反演时间、预热时间
    - XAJ模型配置：汇流选项、多组件数量、组件汇流、组件权重
    - 训练配置：训练轮数、保存间隔、随机种子、GPU设置
    - 动态参数配置：动态参数索引、蒸散发模块、dropout、静态参数时间步
    
    注意：动态参数模式下，td_opt 默认为 True；XAJ 参数个数固定为 12
    
    Returns:
        argparse.Namespace: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description='dPLXAJ Dynamic Parameter Training')

    # =============================================================================
    # 实验模式配置参数
    # =============================================================================
    # 这些参数控制实验的整体设置，包括数据分割方式和数据源选择
    # 动态参数模式与静态参数模式的主要区别在于td_opt默认为True
    # =============================================================================
    
    parser.add_argument('--pu_opt', type=int, default=0, choices=[0, 1, 2],
                       help='流域选择模式: 0=ALL(所有流域训练测试), 1=PUB(随机保留流域空间泛化), 2=PUR(连续区域保留空间泛化)')
    parser.add_argument('--buff_opt', type=int, default=0, choices=[0, 1, 2],
                       help='预热选项: 0=第一年数据仅用于预热下一年, 1=重复第一年数据预热第一年, 2=加载额外一年数据预热第一年')
    parser.add_argument('--td_opt', action='store_true', default=True,
                       help='参数模式: False=静态参数(所有XAJ参数不随时间变化), True=动态参数(部分XAJ参数随时间变化) - 动态模式默认为True')
    parser.add_argument('--for_type', type=str, default='daymet',
                       choices=['daymet', 'nldas', 'maurer'],
                       help='气象数据源: daymet=Daymet气象数据, nldas=NLDAS气象数据, maurer=Maurer气象数据')

    # =============================================================================
    # 模型架构配置参数
    # =============================================================================
    # 这些参数控制深度学习模型的架构和训练设置
    # XAJ 参数固定为 12（不包含 ETMod）
    # =============================================================================
    
    parser.add_argument('--rnn_type', type=str, default='cnnlstm',
                       choices=['lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'],
                       help='RNN类型: lstm=LSTM网络, gru=GRU网络, bilstm=双向LSTM, bigru=双向GRU, rnn=简单RNN, cnnlstm=CNN-LSTM混合网络, cnnbilstm=CNN-BiLSTM混合网络')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='批次大小: 每个训练批次包含的流域数量，影响训练速度和内存使用')
    parser.add_argument('--rho', type=int, default=365,
                       help='序列长度（天数）: 输入时间序列的长度，通常设置为一年（365天）')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='隐藏层大小: RNN隐藏层的神经元数量，影响模型容量和表达能力')
    parser.add_argument('--nfea', type=int, default=12,
                       help='XAJ参数个数: XAJ水文模型的参数数量，固定为12')

    # 时间配置
    parser.add_argument('--train_start', type=str, default='19801001',
                       help='训练开始日期 (YYYYMMDD)')
    parser.add_argument('--train_end', type=str, default='20041001',
                       help='训练结束日期 (YYYYMMDD)')
    parser.add_argument('--inv_start', type=str, default='19801001',
                       help='反演开始日期 (YYYYMMDD)')
    parser.add_argument('--inv_end', type=str, default='20041001',
                       help='反演结束日期 (YYYYMMDD)')
    parser.add_argument('--buff_time', type=int, default=365,
                       help='预热时间（天）')

    # XAJ模型配置
    parser.add_argument('--routing', action='store_true', default=True,
                       help='汇流选项')
    parser.add_argument('--nmul', type=int, default=10,
                       help='多组件数量')
    parser.add_argument('--comprout', action='store_true', default=False,
                       help='组件汇流')
    parser.add_argument('--compwts', action='store_true', default=False,
                       help='组件权重')

    # 训练配置
    parser.add_argument('--epoch', type=int, default=30,
                       help='总训练轮数')
    parser.add_argument('--save_epoch', type=int, default=1,
                       help='模型保存间隔')
    parser.add_argument('--random_seed', type=int, default=111111,
                       help='随机种子')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--test_fold', type=int, default=1,
                       help='测试折数索引')

    # =============================================================================
    # 动态参数配置参数（动态参数模式的核心配置）
    # =============================================================================
    # 这些参数是动态参数模式的核心配置，控制哪些 XAJ 参数为动态参数
    # 以及动态参数的行为和训练策略
    # =============================================================================
    # 动态参数 = ke, wm, c, sm, ki（编号 1, 5, 6, 7, 9）
    parser.add_argument('--td_rep', type=int, nargs='+', default=[1, 5, 6, 7, 9],
                       help='动态参数索引列表: 指定哪些XAJ参数为动态参数（范围: 1-12）')
    parser.add_argument('--dy_drop', type=float, default=0.0,
                       help='动态参数dropout: 动态参数的dropout率，0.0=始终动态，1.0=始终静态，中间值=随机选择动态/静态')
    parser.add_argument('--sta_ind', type=int, default=-1,
                       help='静态参数时间步: 静态参数使用的时间步，-1=使用最后一个时间步，其他值=使用指定时间步的静态参数')

    return parser.parse_args()

def setup_environment(args):
    """
    设置环境和随机种子
    
    该函数负责初始化训练环境，包括：
    1. 设置所有随机数生成器的种子，确保实验可重现
    2. 配置PyTorch的确定性算法设置
    3. 设置GPU设备
    
    Args:
        args: 命令行参数对象，包含random_seed和gpu_id等设置
    """
    # =============================================================================
    # 随机种子设置 - 确保结果可重现
    # =============================================================================
    # 设置所有随机数生成器的种子，包括Python内置random、NumPy、PyTorch等
    # 这对于确保实验的可重现性至关重要
    # =============================================================================
    
    random.seed(args.random_seed)                    # Python内置random模块
    torch.manual_seed(args.random_seed)              # PyTorch CPU随机数生成器
    np.random.seed(args.random_seed)                 # NumPy随机数生成器
    torch.cuda.manual_seed(args.random_seed)         # PyTorch GPU随机数生成器（当前GPU）
    torch.cuda.manual_seed_all(args.random_seed)     # PyTorch GPU随机数生成器（所有GPU）
    
    # =============================================================================
    # PyTorch确定性算法设置
    # =============================================================================
    # 这些设置确保PyTorch使用确定性算法，进一步保证结果的可重现性
    # =============================================================================
    
    torch.backends.cudnn.deterministic = True        # 使用确定性CUDNN算法
    torch.backends.cudnn.benchmark = False           # 禁用CUDNN自动优化（可能引入随机性）
    torch.use_deterministic_algorithms(True, warn_only=True)  # 使用确定性算法，仅警告不报错

    # =============================================================================
    # GPU设备设置
    # =============================================================================
    # 根据参数设置使用的GPU设备
    # =============================================================================
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)           # 设置当前使用的GPU设备
        print(f"使用GPU设备: {args.gpu_id}")
    else:
        print("CUDA不可用，使用CPU进行计算")

# =============================================================================
# 路径配置函数
# =============================================================================
def init_camels_path():
    """
    初始化 CAMELS/输出目录路径（带详细路径推断与优先级说明）

    路径推断规则（从高到低优先级）：
    1) 工程根目录下的本地数据与输出目录（推荐放置）：
       - 数据:  <project_root>/Camels 或 <project_root>/CAMELS（任选其一存在即可）
       - 输出:  <project_root>/outputs（若不存在则自动创建）
    2) 若工程内未找到 Camels/CAMELS，则回退到通用默认路径（适配集群/服务器）：
       - 数据:  /scratch/Camels
       - 输出:  /data/rnnStreamflow

    变量说明：
    - project_root: 当前脚本（experiments 目录）上一级目录，即工程根目录 PhysDL
    - pathCamels['DB']: CAMELS 数据根目录（包含 basin_timeseries_v1p2_metForcing_obsFlow 等）
    - pathCamels['Out']: 训练与测试产生的模型与结果输出目录
    """
    # 推断工程根目录：从当前脚本（experiments 目录内）回到其上一级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录（绝对路径）
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))  # 工程根目录
    
    # 本地优先：工程内 Camels/CAMELS 和 outputs
    candidate_camels = [
        os.path.join(project_root, 'Camels'),   # 常见目录名：Camels
        os.path.join(project_root, 'CAMELS')    # 某些环境使用全大写：CAMELS
    ]
    # 依序查找首个存在的候选目录；若都不存在则返回 None
    local_camels = next((p for p in candidate_camels if os.path.isdir(p)), None)
    # 统一将输出目录置于工程根目录下的 outputs（便于打包与清理）
    local_outputs = os.path.join(project_root, 'outputs')
    
    # 确保输出目录存在
    if not os.path.isdir(local_outputs):
        try:
            # 尝试创建 outputs 目录；忽略竞争条件（exist_ok=True）
            os.makedirs(local_outputs, exist_ok=True)
        except Exception:
            # 部分受限环境（只读文件系统）可能创建失败，此处静默跳过
            pass

    # 若工程内存在 Camels/CAMELS，则优先使用本地；否则回退到默认路径
    if local_camels is not None:
        # 情况1：工程内提供了 Camels/CAMELS 目录 -> 使用本地数据与输出
        pathCamels = OrderedDict(
            DB=local_camels,   # 数据根目录（包含 basin_timeseries_v1p2_metForcing_obsFlow 等）
            Out=local_outputs  # 输出根目录（模型、日志、评估结果等）
        )
    else:
        # 情况2：工程内未找到数据目录 -> 回退到通用默认路径（常见于服务器/集群）
        # 注意：/scratch 与 /data 是示例路径，必要时可自行替换为环境实际挂载点
        pathCamels = OrderedDict(
            DB=os.path.join(os.path.sep, 'scratch', 'Camels'),      # /scratch/Camels
            Out=os.path.join(os.path.sep, 'data', 'rnnStreamflow')  # /data/rnnStreamflow
        )

    # 返回统一的路径配置字典：{'DB': <数据根目录>, 'Out': <输出根目录>}
    return pathCamels


def setup_basin_split(args, pathCamels):
    """
    设置流域数据分割
    
    该函数负责根据实验模式（ALL/PUB/PUR）设置流域数据的训练和测试分割：
    1. 初始化CAMELS数据集路径和配置
    2. 根据pu_opt参数选择不同的流域分割策略
    3. 返回训练和测试流域列表及其索引
    
    Args:
        args: 命令行参数对象，包含pu_opt和test_fold等设置
        pathCamels: 路径配置字典，包含DB和Out路径
        
    Returns:
        tuple: 包含以下元素的元组
            - rootDatabase: CAMELS数据集根目录路径
            - rootOut: 模型输出根目录路径
            - gageinfo: 流域信息字典
            - puN: 流域模式名称（'ALL'/'PUB'/'PUR'）
            - TrainLS: 训练流域ID列表
            - TrainInd: 训练流域索引列表
            - TestLS: 测试流域ID列表
            - TestInd: 测试流域索引列表
            - gageDic: 流域字典，包含训练和测试流域ID
    """
    # =============================================================================
    # CAMELS数据集路径配置说明
    # =============================================================================
    # pathCamels 是传入的路径字典，包含：
    # - pathCamels['DB']: CAMELS数据集根目录路径
    # - pathCamels['Out']: 模型输出根目录路径
    #
    # 路径自动检测逻辑（在 hydroMLlib/__init__.py 的 initPath() 函数中）：
    # 1. 首先检查项目根目录下是否存在 'Camels' 或 'CAMELS' 文件夹
    # 2. 如果存在，则使用项目内的本地路径：
    #    - DB: 项目根目录/Camels 或 项目根目录/CAMELS
    #    - Out: 项目根目录/outputs
    # 3. 如果不存在，则使用默认路径：
    #    - DB: /scratch/Camels
    #    - Out: /data/rnnStreamflow
    #
    # 当前项目的路径结构应该是：
    # PhysDL/
    # ├── Camels/          # CAMELS数据集目录
    # │   ├── basin_timeseries_v1p2_metForcing_obsFlow/
    # │   ├── basin_metadata/
    # │   ├── pet_harg/
    # │   └── ...
    # ├── outputs/         # 模型输出目录
    # │   └── CAMELSDemo/
    # └── hydroMLlib/
    # =============================================================================

    # 获取CAMELS数据集根目录路径
    rootDatabase = pathCamels['DB']  # 例如: D:/01_学术研究/.../PhysDL/Camels
    # 获取模型输出根目录路径
    rootOut = pathCamels['Out']      # 例如: D:/01_学术研究/.../PhysDL/outputs

    # 读取站点信息（不依赖全局状态）
    gageinfo = camels.readGageInfo(rootDatabase)
    hucinfo = gageinfo['huc']
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()

    if args.pu_opt == 0:  # ALL模式
        puN = 'ALL'
        TrainLS = gageidLst
        TrainInd = [gageidLst.index(j) for j in TrainLS]
        TestLS = gageidLst
        TestInd = [gageidLst.index(j) for j in TestLS]
        gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

    elif args.pu_opt == 1:  # PUB模式
        puN = 'PUB'
        splitPath = 'PUBsplitLst.txt'
        with open(splitPath, 'r') as fp:
            testIDLst = json.load(fp)
        TestLS = testIDLst[args.test_fold - 1]
        TestInd = [gageidLst.index(j) for j in TestLS]
        TrainLS = list(set(gageid.tolist()) - set(TestLS))
        TrainInd = [gageidLst.index(j) for j in TrainLS]
        gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

    elif args.pu_opt == 2:  # PUR模式
        puN = 'PUR'
        regionID = list()
        regionDivide = [[1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17]]

        for ii in range(len(regionDivide)):
            tempcomb = regionDivide[ii]
            tempregid = list()
            for ih in tempcomb:
                tempid = gageid[hucinfo == ih].tolist()
                tempregid = tempregid + tempid
            regionID.append(tempregid)

        iexp = args.test_fold - 1
        TestLS = regionID[iexp]
        TestInd = [gageidLst.index(j) for j in TestLS]
        TrainLS = list(set(gageid.tolist()) - set(TestLS))
        TrainInd = [gageidLst.index(j) for j in TrainLS]
        gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

    return rootDatabase, rootOut, gageinfo, puN, TrainLS, TrainInd, TestLS, TestInd, gageDic

def load_and_preprocess_data(args, rootDatabase, gageinfo, TrainLS, TrainInd):
    """
    加载和预处理数据
    
    该函数负责加载CAMELS数据集并进行预处理，包括：
    1. 加载气象强迫数据（降水、温度等）
    2. 加载观测流量数据
    3. 加载流域属性数据
    4. 加载潜在蒸散发（PET）数据
    5. 数据单位转换和标准化处理
    6. 根据预热选项处理输入数据
    
    Args:
        args: 命令行参数对象
        rootDatabase: CAMELS数据集根目录路径
        gageinfo: 流域信息字典
        TrainLS: 训练流域ID列表
        TrainInd: 训练流域索引列表
        
    Returns:
        tuple: 包含以下元素的元组
            - optData: 数据加载选项配置
            - statDict: 数据标准化统计信息字典
            - forcTuple: 输入数据元组 (xTrainIn, zTrainIn)
            - yTrainIn: 目标数据（观测流量）
            - attrs: 标准化的流域属性数据
    """
    # =============================================================================
    # 时间范围处理
    # =============================================================================
    # 设置训练和反演的时间范围
    # =============================================================================
    
    Ttrain = [int(args.train_start), int(args.train_end)]  # XAJ模型训练时间范围
    Tinv = [int(args.inv_start), int(args.inv_end)]        # 参数反演时间范围
    print(f"训练时间范围: {Ttrain[0]} - {Ttrain[1]}")
    print(f"反演时间范围: {Tinv[0]} - {Tinv[1]}")

    # =============================================================================
    # 应用预热选项
    # =============================================================================
    # 根据buff_opt参数决定是否加载额外数据用于模型预热
    # =============================================================================
    
    if args.buff_opt == 2:  # 加载额外数据用于预热
        print(f"预热选项2: 加载额外{args.buff_time}天数据用于预热")
        sd = time.t2dt(Ttrain[0]) - dt.timedelta(days=args.buff_time)
        sdint = int(sd.strftime("%Y%m%d"))
        TtrainLoad = [sdint, Ttrain[1]]  # 扩展训练时间范围
        TinvLoad = [sdint, Tinv[1]]      # 扩展反演时间范围
        print(f"扩展后训练时间: {TtrainLoad[0]} - {TtrainLoad[1]}")
    else:
        TtrainLoad = Ttrain  # 使用原始训练时间范围
        TinvLoad = Tinv      # 使用原始反演时间范围
        print(f"预热选项{args.buff_opt}: 使用原始时间范围")

    # =============================================================================
    # 气象变量配置
    # =============================================================================
    # 根据气象数据源类型选择相应的气象变量
    # =============================================================================
    
    if args.for_type == 'daymet':
        varF = ['prcp', 'tmean']      # Daymet数据：降水、平均温度
        varFInv = ['prcp', 'tmean']   # 反演期使用相同变量
        print("气象数据源: Daymet (降水、平均温度)")
    else:
        varF = ['prcp', 'tmax']       # Maurer和NLDAS数据：降水、最高温度（实际为平均温度）
        varFInv = ['prcp', 'tmax']    # 反演期使用相同变量
        print(f"气象数据源: {args.for_type.upper()} (降水、最高温度)")

    # =============================================================================
    # 流域属性变量列表
    # =============================================================================
    # 定义用于模型训练的流域属性变量，包括气候、地形、土壤、地质等特征
    # 这些属性用于参数学习和流域特征描述
    # =============================================================================
    
    attrnewLst = [
        'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
        'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
        'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
        'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
        'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class',
        'glim_1st_class_frac', 'geol_2nd_class', 'glim_2nd_class_frac',
        'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'
    ]
    print(f"流域属性变量数量: {len(attrnewLst)}个")

    # =============================================================================
    # 数据加载配置和实际数据读取
    # =============================================================================
    # CAMELS数据集文件结构：
    # Camels/
    # ├── basin_timeseries_v1p2_metForcing_obsFlow/  # 气象和流量时间序列数据
    # │   ├── daymet/                                # Daymet气象数据
    # │   │   ├── 01030500_lump_cida_forcing_leap.txt
    # │   │   ├── 01030500_lump_cida_forcing_leap.txt
    # │   │   └── ...
    # │   ├── nldas/                                 # NLDAS气象数据
    # │   └── maurer/                                # Maurer气象数据
    # ├── basin_metadata/                            # 流域属性数据
    # │   ├── camels_attributes_v2.0/
    # │   │   ├── camels_clim.txt
    # │   │   ├── camels_geol.txt
    # │   │   ├── camels_hydro.txt
    # │   │   ├── camels_name.txt
    # │   │   ├── camels_soil.txt
    # │   │   └── camels_topo.txt
    # │   └── camels_attributes_v2.0.csv
    # └── pet_harg/                                  # 潜在蒸散发数据
    #     ├── daymet/
    #     ├── nldas/
    #     └── maurer/
    # =============================================================================

    # 配置数据加载选项
    optData = defaultSetting.get_data_config()  # 获取数据配置
    optData = defaultSetting.update_config(optData,
                            tRange=TtrainLoad,      # 时间范围
                            varT=varFInv,           # 时间序列变量（气象数据）
                            varC=attrnewLst,        # 常量变量（流域属性）
                            subset=TrainLS,         # 流域子集
                            forType=args.for_type)  # 气象数据源类型

    # 加载 XAJ 模型训练数据（气象强迫和观测流量）
    # 数据文件路径：Camels/basin_timeseries_v1p2_metForcing_obsFlow/{for_type}/{basin_id}_lump_cida_forcing_leap.txt
    dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=args.for_type,
                                     dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)      # 气象强迫数据
    obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)  # 观测流量数据

    # 加载dPL反演数据（用于参数学习的输入数据）
    # 包括气象数据和流域属性数据
    dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=args.for_type,
                                   dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)  # 反演期气象数据
    attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)  # 流域属性数据

    # 单位转换：流量从ft3/s转换为mm/day
    areas = gageinfo['area'][TrainInd]
    temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1], 1))
    obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3

    # =============================================================================
    # 加载潜在蒸散发数据 (PET - Potential Evapotranspiration)
    # =============================================================================
    # PET数据文件路径结构：
    # rootDatabase/pet_harg/{for_type}/{basin_id}.csv
    # 例如: Camels/pet_harg/daymet/01030500.csv
    #
    # 数据来源：使用Hargreaves方法计算的潜在蒸散发
    # 时间范围：根据气象数据源类型确定
    #   - maurer: 1980-2009年
    #   - daymet/nldas: 1980-2015年
    # =============================================================================

    varLstNL = ['PEVAP']  # 变量名：潜在蒸散发
    usgsIdLst = gageinfo['id']  # 流域ID列表

    # 根据气象数据源确定PET数据的时间范围
    if args.for_type == 'maurer':
        tPETRange = [19800101, 20090101]  # maurer数据：1980-2009
    else:
        tPETRange = [19800101, 20150101]  # daymet/nldas数据：1980-2015

    tPETLst = time.tRange2Array(tPETRange)  # 生成时间序列数组

    # 构建PET数据目录路径
    # 格式：Camels/pet_harg/daymet/ 或 Camels/pet_harg/maurer/ 等
    PETDir = rootDatabase + '/pet_harg/' + args.for_type + '/'

    ntime = len(tPETLst)  # 时间步数
    PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])  # 初始化PET数据数组

    # 逐个加载每个流域的PET数据
    for k in range(len(usgsIdLst)):
        # 读取单个流域的PET数据文件
        # 文件路径：PETDir + basin_id + '.csv'
        # 例如：Camels/pet_harg/daymet/01030500.csv
        dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
        PETfull[k, :, :] = dataTemp

    # 提取对应时间段的PET数据
    TtrainLst = time.tRange2Array(TtrainLoad)
    TinvLst = time.tRange2Array(TinvLoad)
    _, _, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
    PETUN = PETfull[:, ind2, :]
    PETUN = PETUN[TrainInd, :, :]  # 选择训练流域
    _, _, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    PETInvUN = PETfull[:, ind2inv, :]
    PETInvUN = PETInvUN[TrainInd, :, :]

    # 数据预处理：归一化和NaN处理
    series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
    seriesvarLst = varFInv + ['pet']

    # 计算归一化统计信息
    statDict = camels.getStatDic(attrLst=attrnewLst, attrdata=attrsUN,
                                seriesLst=seriesvarLst, seriesdata=series_inv)

    # 归一化数据
    attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    # 准备输入数据
    zTrain = series_norm  # dPL反演输入
    xTrain = np.concatenate([forcUN, PETUN], axis=2)  # XAJ 强迫输入
    xTrain[np.isnan(xTrain)] = 0.0

    # 应用预热选项
    if args.buff_opt == 1:  # 重复第一年数据预热
        zTrainIn = np.concatenate([zTrain[:, 0:args.buff_time, :], zTrain], axis=1)
        xTrainIn = np.concatenate([xTrain[:, 0:args.buff_time, :], xTrain], axis=1)
        yTrainIn = np.concatenate([obsUN[:, 0:args.buff_time, :], obsUN], axis=1)
    else:  # 不重复，第一年数据仅用于预热下一年
        zTrainIn = zTrain
        xTrainIn = xTrain
        yTrainIn = obsUN

    forcTuple = (xTrainIn, zTrainIn)
    attrs = attr_norm

    return optData, statDict, forcTuple, yTrainIn, attrs

def train_model(args, rootOut, puN, optData, statDict, forcTuple, yTrainIn, attrs):
    """
    训练模型
    
    该函数负责构建和训练 dPLXAJ 动态参数模型，包括：
    1. 根据参数模式（静态/动态）配置模型参数
    2. 设置损失函数和训练选项
    3. 构建模型输出路径
    4. 定义模型架构（动态参数模型）
    5. 保存配置和统计信息
    6. 执行模型训练
    
    Args:
        args: 命令行参数对象
        rootOut: 模型输出根目录路径
        puN: 流域模式名称（'ALL'/'PUB'/'PUR'）
        optData: 数据加载选项配置
        statDict: 数据标准化统计信息字典
        forcTuple: 输入数据元组 (xTrainIn, zTrainIn)
        yTrainIn: 目标数据（观测流量）
        attrs: 标准化的流域属性数据
        
    Returns:
        tuple: 包含以下元素的元组
            - trainedModel: 训练完成的模型对象
            - out: 模型输出路径
    """
    # =============================================================================
    # 动态参数配置
    # =============================================================================
    # 根据 td_opt 参数决定是否使用动态参数模式
    # 动态参数模式下，部分 XAJ 参数会随时间变化
    # =============================================================================
    
    if args.td_opt:
        # 动态参数模式配置
        tdRep = args.td_rep                    # 动态参数索引列表，例如[1, 12]
        tdRepS = [str(ix) for ix in tdRep]     # 转换为字符串列表，用于路径构建
        Nfea = args.nfea                       # XAJ 参数个数：固定为 12
        dydrop = args.dy_drop                  # 动态参数dropout率
        staind = args.sta_ind                  # 静态参数时间步
        TDN = '/TDTestforc/' + 'TD' + "_".join(tdRepS) + '/'  # 动态参数路径标识
        print(f"动态参数模式: 参数{tdRep}为动态参数，参数个数={Nfea}")
    else:
        # 静态参数模式配置
        TDN = '/Testforc/'                     # 静态参数路径标识
        Nfea = args.nfea                       # XAJ 参数个数
        print(f"静态参数模式: 所有参数为静态，参数个数={Nfea}")

    # =============================================================================
    # 损失函数配置
    # =============================================================================
    # 配置模型训练使用的损失函数
    # 使用RMSE损失函数，权重为0.25
    # =============================================================================
    
    alpha = 0.25  # RMSE损失权重，控制损失函数中RMSE项的权重
    optLoss = defaultSetting.update_config(defaultSetting.get_loss_config(), name='hydroMLlib.utils.criterion.RmseLossComb', weight=alpha)
    lossFun = criterion.RmseLossComb(alpha=alpha)  # 创建损失函数对象
    print(f"损失函数配置: RMSE权重={alpha}")

    # =============================================================================
    # 训练选项配置
    # =============================================================================
    # 配置模型训练的各种选项，包括批次大小、训练轮数、保存间隔等
    # =============================================================================
    
    optTrain = defaultSetting.update_config(defaultSetting.get_train_config(),
                             miniBatch=[args.batch_size, args.rho],  # 批次大小和序列长度
                             nEpoch=args.epoch,                      # 训练轮数
                             saveEpoch=args.save_epoch)              # 模型保存间隔
    print(f"训练选项: 批次大小={args.batch_size}, 序列长度={args.rho}, 训练轮数={args.epoch}")

    # =============================================================================
    # 模型输出路径设置
    # =============================================================================
    # 输出路径结构：
    # rootOut/exp_name/exp_disp/exp_info/
    # 例如：outputs/CAMELSDemo/dPLXAJ/ALL/TDTestforc/TD1_12/daymet/BuffOpt0/RMSE_para0.25/111111/Fold1/LSTM/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_12_Buff_365_Mul_16_LSTM/
    #
    # 路径组成部分：
    # - exp_name: 实验名称 (CAMELSDemo)
    # - exp_disp: 实验配置路径 (dPLXAJ/{puN}{TDN}{for_type}/BuffOpt{buff_opt}/RMSE_para{alpha}/{random_seed}/Fold{test_fold}/{rnn_type})
    # - exp_info: 模型参数信息 (T_{train_start}_{train_end}_BS_{batch_size}_HS_{hidden_size}_RHO_{rho}_NF_{nfea}_Buff_{buff_time}_Mul_{nmul}_{rnn_type})
    # =============================================================================

    exp_name = 'CAMELSDemo'  # 实验名称
    rnn_type_str = args.rnn_type.upper()  # RNN类型（大写）

    # 构建实验配置路径
    # 格式：dPLXAJ/{流域模式}{参数模式}{气象数据源}/BuffOpt{预热选项}/RMSE_para{损失权重}/{随机种子}/Fold{折数}/{RNN类型}
    exp_disp = (f'dPLXAJ/{puN}{TDN}{args.for_type}/BuffOpt{args.buff_opt}/RMSE_para{alpha}/'
               f'{args.random_seed}/Fold{args.test_fold}/{rnn_type_str}')

    # 构建模型参数信息路径
    # 格式：T_{训练开始}_{训练结束}_BS_{批次大小}_HS_{隐藏层大小}_RHO_{序列长度}_NF_{参数个数}_Buff_{预热时间}_Mul_{组件数}_{RNN类型}
    exp_info = (f'T_{args.train_start}_{args.train_end}_BS_{args.batch_size}_HS_{args.hidden_size}_RHO_{args.rho}_'
               f'NF_{Nfea}_Buff_{args.buff_time}_Mul_{args.nmul}_{rnn_type_str}')

    # 组合完整输出路径
    save_path = os.path.join(exp_name, exp_disp)
    out = os.path.join(rootOut, save_path, exp_info)

    # =============================================================================
    # 模型定义
    # =============================================================================
    # 根据参数模式定义不同的模型架构
    # 动态参数模式使用 MultiInv_XAJTDModel，静态参数模式使用 MultiInv_XAJModel
    # =============================================================================
    
    Ninv = forcTuple[1].shape[-1] + attrs.shape[-1]  # 计算输入维度：时间序列维度 + 流域属性维度
    print(f"模型输入维度: {Ninv} (时间序列: {forcTuple[1].shape[-1]}, 流域属性: {attrs.shape[-1]})")

    if not args.td_opt:  # 静态参数模型（XAJ）
        print("构建静态参数 XAJ 模型...")
        model = xaj_static.MultiInv_XAJModel(
            ninv=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddeninv=args.hidden_size, inittime=args.buff_time,
            routOpt=args.routing, comprout=args.comprout,
            compwts=args.compwts, pcorr=None,
            rnn_type=args.rnn_type)

        optModel = OrderedDict(
            name='dPLXAJ', nx=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddenSize=args.hidden_size, doReLU=True,
            Tinv=[int(args.inv_start), int(args.inv_end)],
            Trainbuff=args.buff_time, routOpt=args.routing,
            comprout=args.comprout, compwts=args.compwts,
            pcorr=None, buffOpt=args.buff_opt, TDOpt=args.td_opt,
            rnn_type=args.rnn_type)
    else:  # 动态参数模型（XAJ）
        print("构建动态参数 XAJ 模型...")
        print(f"动态参数配置: 参数{tdRep}为动态, dropout={dydrop}")
        model = xaj_dynamic.MultiInv_XAJTDModel(
            ninv=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddeninv=args.hidden_size, inittime=args.buff_time,
            routOpt=args.routing, comprout=args.comprout,
            compwts=args.compwts, staind=staind, tdlst=tdRep,
            dydrop=dydrop, rnn_type=args.rnn_type)

        optModel = OrderedDict(
            name='dPLXAJDP', nx=Ninv, nfea=Nfea, nmul=args.nmul,
            hiddenSize=args.hidden_size, doReLU=True,
            Tinv=[int(args.inv_start), int(args.inv_end)],
            Trainbuff=args.buff_time, routOpt=args.routing,
            comprout=args.comprout, compwts=args.compwts,
            pcorr=None, staind=staind, tdlst=tdRep, dydrop=dydrop,
            buffOpt=args.buff_opt, TDOpt=args.td_opt,
            rnn_type=args.rnn_type)

    # 显示模型信息
    print("\n" + "=" * 50)
    print(f"使用{args.rnn_type.upper()}作为参数预测骨干网络 (Using {args.rnn_type.upper()} as parameter prediction backbone)")
    print("=" * 50 + "\n")

    # =============================================================================
    # 保存配置和统计信息
    # =============================================================================
    # 保存模型配置、数据配置、损失函数配置、训练配置等信息
    # 这些信息对于后续的模型加载和测试至关重要
    # =============================================================================
    
    print("保存模型配置信息...")
    masterDict = fileManager.wrapMaster(out, optData, optModel, optLoss, optTrain)
    fileManager.writeMasterFile(masterDict)
    print("模型配置已保存")

    # 保存标准化统计信息
    print("保存数据标准化统计信息...")
    statFile = os.path.join(out, 'statDict.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)
    print("标准化统计信息已保存")

    # =============================================================================
    # 开始模型训练
    # =============================================================================
    # 执行模型训练过程，包括前向传播、损失计算、反向传播、参数更新等
    # 动态参数模式下，模型会学习参数的时间变化规律
    # =============================================================================
    
    print("=" * 50)
    print("开始模型训练 (Starting Model Training)")
    print("=" * 50)
    if args.td_opt:
        print(f"动态参数训练: 参数{tdRep}将学习时间变化规律")
    else:
        print("静态参数训练: 所有参数保持恒定")

    trainedModel = train.trainModel(
        model,                                    # 模型对象
        forcTuple,                               # 输入数据元组 (xTrainIn, zTrainIn)
        yTrainIn,                                # 目标数据（观测流量）
        attrs,                                   # 流域属性数据
        lossFun,                                 # 损失函数
        nEpoch=args.epoch,                       # 训练轮数
        miniBatch=[args.batch_size, args.rho],   # 批次大小和序列长度
        saveEpoch=args.save_epoch,               # 模型保存间隔
        saveFolder=out,                          # 模型保存路径
        bufftime=args.buff_time                  # 预热时间
    )

    print("=" * 50)
    print("模型训练完成! (Model Training Completed!)")
    print("=" * 50)
    if args.td_opt:
        print(f"动态参数训练完成: 参数{tdRep}的时间变化规律已学习")
    else:
        print("静态参数训练完成: 所有参数已优化")

    return trainedModel, out

def main():
    """
    主函数
    
    该函数是动态参数训练脚本的入口点，负责协调整个训练流程：
    1. 解析命令行参数
    2. 设置训练环境（随机种子、GPU等）
    3. 初始化路径配置
    4. 设置流域数据分割
    5. 加载和预处理数据
    6. 训练动态参数模型
    7. 输出训练结果
    
    动态参数训练流程说明：
    - 首先解析用户提供的命令行参数（注意td_opt默认为True）
    - 设置随机种子确保实验可重现
    - 初始化CAMELS数据集路径配置
    - 根据实验模式（ALL/PUB/PUR）分割流域数据
    - 加载CAMELS数据集并进行预处理
    - 构建 dPLXAJ 动态参数模型并开始训练
    - 训练过程中，指定的 XAJ 参数会随时间动态变化
    - 训练完成后保存模型和配置信息
    
    动态参数模式特点：
    - 部分 XAJ 参数（由 td_rep 指定）会随时间变化
    - 模型需要学习参数的时间变化规律
    - 训练复杂度比静态参数模式更高
    - 可以更好地捕捉水文过程的时间变异性
    
    Returns:
        tuple: 包含训练完成的模型对象和输出路径
    """
    # =============================================================================
    # 1. 解析命令行参数
    # =============================================================================
    # 解析用户提供的所有训练参数，包括动态参数配置
    # =============================================================================
    args = parse_args()
    print("=" * 60)
    print("dPLXAJ 动态参数模型训练开始")
    print("=" * 60)
    print(f"实验配置: {args.pu_opt}模式, {args.rnn_type.upper()}网络, {args.for_type}数据源")
    print(f"训练时间: {args.train_start} - {args.train_end}")
    print(f"模型参数: 批次大小={args.batch_size}, 隐藏层={args.hidden_size}, 序列长度={args.rho}")
    print(f"动态参数: {args.td_rep} (参数{', '.join(map(str, args.td_rep))}为动态参数)")

    # =============================================================================
    # 2. 设置训练环境
    # =============================================================================
    # 设置随机种子、GPU设备等环境配置，确保实验可重现
    # =============================================================================
    setup_environment(args)

    # =============================================================================
    # 3. 初始化路径配置
    # =============================================================================
    # 初始化CAMELS数据集路径配置，包括数据路径和输出路径
    # =============================================================================
    pathCamels = init_camels_path()

    # =============================================================================
    # 4. 设置流域数据分割
    # =============================================================================
    # 根据实验模式（ALL/PUB/PUR）设置训练和测试流域
    # =============================================================================
    rootDatabase, rootOut, gageinfo, puN, TrainLS, TrainInd, TestLS, TestInd, gageDic = setup_basin_split(args, pathCamels)
    print(f"流域分割完成: 训练流域{len(TrainLS)}个, 测试流域{len(TestLS)}个")

    # =============================================================================
    # 5. 加载和预处理数据
    # =============================================================================
    # 加载CAMELS数据集，包括气象数据、流量数据、流域属性、PET数据等
    # 进行数据标准化和预处理
    # =============================================================================
    print("开始加载和预处理数据...")
    optData, statDict, forcTuple, yTrainIn, attrs = load_and_preprocess_data(
        args, rootDatabase, gageinfo, TrainLS, TrainInd)
    print("数据加载和预处理完成")

    # =============================================================================
    # 6. 训练动态参数模型
    # =============================================================================
    # 构建 dPLXAJ 动态参数模型并开始训练
    # 训练过程包括动态参数学习、前向传播、损失计算、反向传播、参数更新等
    # =============================================================================
    print("开始动态参数模型训练...")
    print(f"动态参数配置: 参数{', '.join(map(str, args.td_rep))}将随时间变化")
    trainedModel, out = train_model(
        args, rootOut, puN, optData, statDict, forcTuple, yTrainIn, attrs)

    # =============================================================================
    # 7. 输出训练结果
    # =============================================================================
    # 显示训练完成信息和模型保存路径
    # =============================================================================
    print("=" * 60)
    print("动态参数模型训练完成！")
    print("=" * 60)
    print(f"模型保存在: {out}")
    print(f"动态参数配置已保存，可用于后续测试")
    print(f"动态参数: {args.td_rep} 已学习时间变化规律")
    return trainedModel, out

if __name__ == "__main__":
    main()