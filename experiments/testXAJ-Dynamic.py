"""
dPLXAJ Dynamic Parameter Testing Script
=====================================
测试动态参数 XAJ 模型的脚本
- 加载训练好的动态参数 XAJ 模型
- 在测试数据集上进行预测和评估
- 生成结果图表和统计指标
"""

# =============================================================================
# 导入库和基础设置
# =============================================================================
from hydroMLlib.utils import camels, fileManager, time, metrics
from hydroMLlib.model import train

import os
import numpy as np
import torch
import pandas as pd
import random
import json
import datetime as dt
import argparse
from collections import OrderedDict
# 导入结果保存和绘图模块
import save_results_plot

def parse_args():
    """
    解析命令行参数
    
    该函数定义了所有动态参数测试脚本需要的命令行参数，包括：
    - 实验模式配置：流域选择、预热选项、气象数据源
    - 模型架构配置：RNN类型、批次大小、序列长度、隐藏层大小、XAJ参数个数
    - 时间配置：训练时间、反演时间、测试时间、预热时间
    - XAJ模型配置：汇流选项、多组件数量、组件汇流、组件权重
    - 测试配置：测试批次大小、模型epoch选择、随机种子、GPU设置
    - 动态参数配置：动态参数索引、蒸散发模块、dropout、静态参数时间步
    
    注意：测试脚本的参数必须与训练脚本完全一致，以确保模型正确加载
    动态参数模式下，td_opt 默认为 True；XAJ 参数个数固定为 12
    
    Returns:
        argparse.Namespace: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description='dPLXAJ Dynamic Parameter Testing')

    # =============================================================================
    # 实验模式配置参数
    # =============================================================================
    # 这些参数控制实验的整体设置，必须与训练脚本完全一致
    # =============================================================================
    
    parser.add_argument('--pu_opt', type=int, default=2, choices=[0, 1, 2],
                       help='流域选择模式: 0=ALL(所有流域训练测试), 1=PUB(随机保留流域空间泛化), 2=PUR(连续区域保留空间泛化)')
    parser.add_argument('--buff_opt', type=int, default=0, choices=[0, 1, 2],
                       help='预热选项: 0=第一年数据仅用于预热下一年, 1=重复第一年数据预热第一年, 2=加载额外一年数据预热第一年')
    parser.add_argument('--for_type', type=str, default='daymet',
                       choices=['daymet', 'nldas', 'maurer'],
                       help='气象数据源: daymet=Daymet气象数据, nldas=NLDAS气象数据, maurer=Maurer气象数据')

    # =============================================================================
    # 模型架构配置参数
    # =============================================================================
    # 这些参数控制深度学习模型的架构和训练设置（需与训练脚本一致）
    # XAJ 参数固定为 12（不包含 ETMod）
    # =============================================================================
    
    parser.add_argument('--rnn_type', type=str, default='bigru',
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

    # =============================================================================
    # 时间配置参数
    # =============================================================================
    # 这些参数控制训练、反演和测试的时间范围，以及模型预热设置
    # 必须与训练脚本完全一致，以确保模型正确加载
    # =============================================================================
    
    parser.add_argument('--train_start', type=str, default='19801001',
                       help='训练开始日期 (YYYYMMDD): XAJ模型训练的时间段开始日期，格式为YYYYMMDD')
    parser.add_argument('--train_end', type=str, default='20041001',
                       help='训练结束日期 (YYYYMMDD): XAJ模型训练的时间段结束日期，格式为YYYYMMDD')
    parser.add_argument('--test_start', type=str, default='20041001',
                       help='测试开始日期 (YYYYMMDD): 模型测试的时间段开始日期，格式为YYYYMMDD')
    parser.add_argument('--test_end', type=str, default='20101001',
                       help='测试结束日期 (YYYYMMDD): 模型测试的时间段结束日期，格式为YYYYMMDD')
    parser.add_argument('--inv_start', type=str, default='19801001',
                       help='反演开始日期 (YYYYMMDD): 参数学习（反演）的时间段开始日期，通常与训练期相同')
    parser.add_argument('--inv_end', type=str, default='20041001',
                       help='反演结束日期 (YYYYMMDD): 参数学习（反演）的时间段结束日期，通常与训练期相同')
    parser.add_argument('--buff_time', type=int, default=365,
                       help='预热时间（天）: 模型预热的天数，通常设置为365天（一年），用于模型状态初始化')

    # =============================================================================
    # XAJ模型配置参数
    # =============================================================================
    # 这些参数控制 XAJ 水文模型的具体配置和组件设置
    # 必须与训练脚本完全一致，以确保模型正确加载
    # =============================================================================
    
    parser.add_argument('--routing', action='store_true', default=True,
                       help='汇流选项: 是否启用汇流模块，True=使用汇流模块进行径流汇合，False=不使用汇流模块')
    parser.add_argument('--nmul', type=int, default=10,
                       help='多组件数量: 并行 XAJ 组件的数量，1表示单组件，>1表示多组件')
    parser.add_argument('--comprout', action='store_true', default=False,
                       help='组件汇流: 是否对每个组件单独进行汇流，True=每个组件单独汇流，False=组件间不汇流')
    parser.add_argument('--compwts', action='store_true', default=False,
                       help='组件权重: 是否使用加权平均组合多组件结果，True=使用加权平均，False=使用简单平均')

    # =============================================================================
    # 测试配置参数
    # =============================================================================
    # 这些参数控制模型测试过程和计算环境设置
    # =============================================================================
    
    parser.add_argument('--test_batch', type=int, default=30,
                       help='测试批次大小: 测试时每个批次包含的流域数量，影响测试速度和内存使用')
    parser.add_argument('--test_epoch', type=int, default=5,
                       help='模型epoch选择: 选择训练好的第几个epoch的模型进行测试')
    parser.add_argument('--random_seed', type=int, default=111111,
                       help='随机种子: 用于确保实验的可重现性，训练和测试脚本必须使用相同的随机种子')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID: 使用的GPU设备编号，0表示第一块GPU，-1表示使用CPU')
    parser.add_argument( '--test_fold',type=int,default=1,
        help=('test in PUB/PUR 模式；1 表示 PUBsplitLst.txt 中的第一个列表，依此类推；设为 <=0 时遍历全部折。')
    )

    # =============================================================================
    # 动态参数配置参数（动态参数模式的核心配置）
    # =============================================================================
    # 这些参数是动态参数模式的核心配置，控制哪些 XAJ 参数为动态参数
    # 以及动态参数的行为和测试策略
    # 动态参数模式下，td_opt默认为True
    # =============================================================================
    
    parser.add_argument('--td_opt', action='store_true', default=True,
                       help='参数模式: False=静态参数(所有XAJ参数不随时间变化), True=动态参数(部分XAJ参数随时间变化) - 动态模式默认为True')
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
    
    该函数负责初始化测试环境，包括：
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
    - pathCamels['Out']: 测试产生的结果与图表输出目录
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
            Out=local_outputs  # 输出根目录（评估结果、图表等）
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
            - tarIDLst: 测试流域ID列表（每个fold一个列表）
            - gageidLst: 所有流域ID列表
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
        tarIDLst = [gageidLst]

    elif args.pu_opt == 1:  # PUB模式
        puN = 'PUB'
        # load the subset ID
        # splitPath saves the basin ID of random groups
        splitPath = 'PUBsplitLst.txt'
        with open(splitPath, 'r') as fp:
            testIDLst = json.load(fp)
        tarIDLst = testIDLst

    elif args.pu_opt == 2:  # PUR模式
        puN = 'PUR'
        # Divide CAMELS dataset into 7 PUR regions
        # get the id list of each region
        regionID = list()
        regionNum = list()
        regionDivide = [[1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17]] # seven regions
        for ii in range(len(regionDivide)):
            tempcomb = regionDivide[ii]
            tempregid = list()
            for ih in tempcomb:
                tempid = gageid[hucinfo==ih].tolist()
                tempregid = tempregid + tempid
            regionID.append(tempregid)
            regionNum.append(len(tempregid))
        tarIDLst = regionID

    return rootDatabase, rootOut, gageinfo, puN, tarIDLst, gageidLst

def load_test_data(args, rootDatabase, gageinfo):
    """
    加载测试数据
    
    该函数负责加载CAMELS测试数据集并进行预处理，包括：
    1. 设置时间范围（训练、反演、测试）
    2. 配置气象变量和流域属性变量
    3. 加载潜在蒸散发（PET）数据
    4. 处理预热选项
    
    Args:
        args: 命令行参数对象
        rootDatabase: CAMELS数据集根目录路径
        gageinfo: 流域信息字典
        
    Returns:
        tuple: 包含以下元素的元组
            - Ttrain: 训练时间范围
            - Tinv: 反演时间范围
            - Ttest: 测试时间范围
            - TtestLst: 测试时间序列数组
            - TtestLoad: 测试数据加载时间范围
            - TtrainLoad: 训练数据加载时间范围
            - TinvLoad: 反演数据加载时间范围
            - varF: 气象变量列表
            - varFInv: 反演期气象变量列表
            - attrnewLst: 流域属性变量列表
            - PETfull: 完整PET数据数组
            - tPETLst: PET时间序列数组
    """
    # =============================================================================
    # 时间范围处理
    # =============================================================================
    # 设置训练、反演和测试的时间范围
    # =============================================================================
    
    Ttrain = [int(args.train_start), int(args.train_end)]  # XAJ模型训练时间范围
    Tinv = [int(args.inv_start), int(args.inv_end)]        # 参数反演时间范围
    Ttest = [int(args.test_start), int(args.test_end)]     # 模型测试时间范围
    TtestLst = time.tRange2Array(Ttest)              # 测试时间序列数组
    TtestLoad = [int(args.test_start), int(args.test_end)] # 测试数据加载时间范围，可用于预热
    print(f"训练时间范围: {Ttrain[0]} - {Ttrain[1]}")
    print(f"反演时间范围: {Tinv[0]} - {Tinv[1]}")
    print(f"测试时间范围: {Ttest[0]} - {Ttest[1]}")

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
        'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity',
        'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
        'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
        'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover',
        'root_depth_50', 'soil_depth_pelletier', 'soil_depth_statsgo',
        'soil_porosity', 'soil_conductivity', 'max_water_content',
        'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class',
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

    # 显示数据加载开始提示
    print("\n" + "=" * 50)
    print("正在加载数据... (Loading data...)")
    print("=" * 50)

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

    return Ttrain, Tinv, Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad, varF, varFInv, attrnewLst, PETfull, tPETLst

def test_model_fold(args, rootDatabase, rootOut, puN, gageinfo, TestLS, TestInd, TrainLS, TrainInd, 
                   Ttest, TtestLoad, TtrainLoad, TinvLoad, 
                   varF, varFInv, attrnewLst, PETfull, tPETLst, testfold):
    """
    测试单个fold的模型（动态参数模式）
    
    该函数负责测试单个fold的动态参数 XAJ 模型，包括：
    1. 加载训练好的动态参数模型
    2. 加载训练和测试数据
    3. 数据预处理和标准化
    4. 动态参数模型预热和预测
    5. 保存预测结果
    
    Args:
        args: 命令行参数对象
        rootOut: 模型输出根目录路径
        puN: 流域模式名称
        gageinfo: 流域信息字典
        TestLS: 测试流域ID列表
        TestInd: 测试流域索引列表
        TrainLS: 训练流域ID列表
        TrainInd: 训练流域索引列表
        Ttest: 测试时间范围
        TtestLoad: 测试数据加载时间范围
        TtrainLoad: 训练数据加载时间范围
        TinvLoad: 反演数据加载时间范围
        varF: 气象变量列表
        varFInv: 反演期气象变量列表
        attrnewLst: 流域属性变量列表
        PETfull: 完整PET数据数组
        tPETLst: PET时间序列数组
        testfold: 测试fold编号
        
    Returns:
        tuple: 包含以下元素的元组
            - dataPred: 预测结果数组
            - obs: 观测数据数组
            - TestLS: 测试流域ID列表
            - TestBuff: 测试缓冲时间
            - testout: 测试输出路径
            - filePathLst: 预测结果文件路径列表
            - testmodel: 测试模型对象
            - forcTestUN: 测试期气象数据
            - PETTestUN: 测试期PET数据
    """
    # =============================================================================
    # 模型路径设置
    # =============================================================================
    # 测试模型路径结构：
    # rootOut/exp_name/exp_disp/exp_info/
    # 例如：outputs/CAMELSDemo/dPLXAJ/ALL/TDTestforc/TD_1_12/daymet/BuffOpt0/RMSE_para0.25/111111/Fold1/LSTM/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_12_Buff_365_Mul_16_LSTM/
    #
    # 路径组成部分：
    # - exp_name: 实验名称 (CAMELSDemo)
    # - exp_disp: 实验配置路径 (dPLXAJ/{puN}/TDTestforc/{TD参数}/{for_type}/BuffOpt{buff_opt}/RMSE_para{alpha}/{random_seed}/Fold{test_fold}/{rnn_type})
    # - exp_info: 模型参数信息 (T_{train_start}_{train_end}_BS_{batch_size}_HS_{hidden_size}_RHO_{rho}_NF_{nfea}_Buff_{buff_time}_Mul_{nmul}_{rnn_type})
    # =============================================================================

    rnn_type_str = args.rnn_type.upper()  # RNN类型（大写）
    tdRepS = [str(ix) for ix in args.td_rep]  # 动态参数索引字符串列表
    TDN = 'TD' + "_".join(tdRepS)  # 动态模式下的TD参数
    
    testsave_path = (f'CAMELSDemo/dPLXAJ/{puN}/TDTestforc/{TDN}/{args.for_type}/BuffOpt{args.buff_opt}/'
                    f'RMSE_para0.25/{args.random_seed}')
    
    foldstr = 'Fold' + str(testfold)
    exp_info = (f'T_{args.train_start}_{args.train_end}_BS_{args.batch_size}_HS_{args.hidden_size}_'
               f'RHO_{args.rho}_NF_{args.nfea}_Buff_{args.buff_time}_Mul_{args.nmul}_{rnn_type_str}')
    
    foldpath = os.path.join(rootOut, testsave_path, foldstr)
    testout = os.path.join(foldpath, rnn_type_str, exp_info)
    
    # 打印模型路径，方便调试
    print("\n" + "=" * 50)
    print(f"尝试加载模型路径: {testout}")
    print("=" * 50 + "\n")
    
    if not os.path.isdir(testout):
        raise FileNotFoundError(f'未找到训练好的模型目录: {testout}')
    
    model_file = os.path.join(testout, f'model_Ep{args.test_epoch}.pt')
    if not os.path.isfile(model_file):
        print(f"警告: 找不到指定epoch {args.test_epoch}的模型文件")
        print(f"Warning: Cannot find model file for epoch {args.test_epoch}")
    
    testmodel = fileManager.loadModel(testout, epoch=args.test_epoch)

    # =============================================================================
    # 加载训练数据（用于模型预热）
    # =============================================================================
    # 训练数据包括：
    # 1. 气象强迫数据：降水、温度等气象变量
    # 2. 观测流量数据：用于模型训练的目标变量
    # 3. 流域属性数据：用于参数学习的静态特征
    # =============================================================================
    
    # 加载 XAJ 模型训练数据（气象强迫和观测流量）
    # 数据文件路径：Camels/basin_timeseries_v1p2_metForcing_obsFlow/{for_type}/{basin_id}_lump_cida_forcing_leap.txt
    dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=args.for_type,
                                     dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)      # 气象强迫数据
    print("- 已加载训练强迫数据 (Training forcing data loaded)")

    # 加载dPL反演数据（用于参数学习的输入数据）
    # 包括气象数据和流域属性数据
    dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=args.for_type,
                                   dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)  # 反演期气象数据
    attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)  # 流域属性数据
    print("- 已加载反演训练数据 (Inversion training data loaded)")

    # 加载测试数据
    # 测试数据用于模型评估和结果生成
    dfTest = camels.DataframeCamels(tRange=TtestLoad, subset=TestLS, forType=args.for_type,
                                    dirDB_opt=rootDatabase, gageDict_opt=gageinfo)
    forcTestUN = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False)      # 测试期气象数据
    obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False)  # 测试期观测流量
    attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)  # 测试期流域属性
    print("- 已加载测试数据 (Testing data loaded)")

    # =============================================================================
    # 数据单位转换和预处理
    # =============================================================================
    # 单位转换：流量从ft3/s转换为mm/day
    # 转换公式：(ft3/s * 0.0283168 * 3600 * 24) / (km2 * 10^6) * 1000 = mm/day
    # 其中：0.0283168是ft3到m3的转换系数，3600*24是秒到天的转换
    # =============================================================================
    
    areas = gageinfo['area'][TestInd] # 流域面积，单位：km2
    temparea = np.tile(areas[:, None, None], (1, obsTestUN.shape[1],1))  # 扩展面积维度以匹配时间序列
    obsTestUN = (obsTestUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3 # 转换为mm/day

    # =============================================================================
    # 提取对应时间段的PET数据
    # =============================================================================
    # 根据训练、反演和测试的时间范围，从完整的PET数据中提取对应时间段的数据
    # =============================================================================
    
    TtrainLst = time.tRange2Array(TtrainLoad)    # 训练期时间序列
    TinvLst = time.tRange2Array(TinvLoad)        # 反演期时间序列
    TtestLoadLst = time.tRange2Array(TtestLoad)  # 测试期时间序列
    
    # 提取训练期PET数据
    _, _, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
    PETUN = PETfull[:, ind2, :]
    PETUN = PETUN[TrainInd, :, :] # 选择训练流域
    
    # 提取反演期PET数据
    _, _, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    PETInvUN = PETfull[:, ind2inv, :]
    PETInvUN = PETInvUN[TrainInd, :, :]
    
    # 提取测试期PET数据
    _, _, ind2test = np.intersect1d(TtestLoadLst, tPETLst, return_indices=True)
    PETTestUN = PETfull[:, ind2test, :]
    PETTestUN = PETTestUN[TestInd, :, :]

    print("- 已加载蒸散发数据 (Evapotranspiration data loaded)")
    
    # =============================================================================
    # 数据标准化处理
    # =============================================================================
    # 使用训练时保存的统计信息对数据进行标准化，确保测试数据与训练数据使用相同的标准化参数
    # 标准化包括：时间序列数据（气象数据+PET）和流域属性数据
    # =============================================================================
    
    # 加载训练时保存的统计信息
    statFile = os.path.join(testout, 'statDict.json')
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)
    print("- 已加载统计信息 (Statistics loaded)")

    # 准备反演数据（用于参数学习）
    series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)  # 合并气象数据和PET数据
    seriesvarLst = varFInv + ['pet']  # 变量列表：气象变量 + PET
    
    # 标准化训练期数据
    attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0  # 处理NaN值
    series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    # 标准化测试期数据
    attrtest_norm = camels.transNormbyDic(attrsTestUN, attrnewLst, statDict, toNorm=True)
    attrtest_norm[np.isnan(attrtest_norm)] = 0.0
    seriestest_inv = np.concatenate([forcTestUN, PETTestUN], axis=2)  # 合并测试期气象数据和PET数据
    seriestest_norm = camels.transNormbyDic(seriestest_inv, seriesvarLst, statDict, toNorm=True)
    seriestest_norm[np.isnan(seriestest_norm)] = 0.0
    
    print("- 已完成数据标准化 (Data normalization completed)")
    print("\n" + "=" * 50)
    print("数据加载完毕! (Data loading completed!)")
    print("=" * 50 + "\n")

    # =============================================================================
    # 准备模型输入数据
    # =============================================================================
    # 根据预热选项处理输入数据：
    # buff_opt=0: 第一年数据仅用于预热下一年
    # buff_opt=1: 重复第一年数据预热第一年
    # buff_opt=2: 加载额外一年数据预热第一年
    # =============================================================================
    
    # 准备反演输入数据（用于参数学习）
    zTrain = series_norm  # 标准化的时间序列数据（气象+PET）
    # 准备 XAJ 强迫输入数据
    xTrain = np.concatenate([forcUN, PETUN], axis=2) # XAJ 强迫：气象数据 + PET数据
    xTrain[np.isnan(xTrain)] = 0.0  # 处理NaN值

 
    # =============================================================================
    # 准备测试数据和模型预热（动态参数模式）
    # =============================================================================
    # 动态参数测试数据准备包括：
    # 1. 设置模型预热时间和动态参数配置
    # 2. 准备测试输入数据
    # 3. 处理不同测试模式（ALL/PUB/PUR）的数据
    # 4. 构建最终的测试输入元组
    # =============================================================================
    
    # 设置测试缓冲时间（用于模型预热）
    TestBuff = xTrain.shape[1]  # 使用整个训练期来预热模型进行测试
    runBUFF = 0  # 动态参数模式下的运行缓冲时间
    # 设置动态参数模型的特殊配置
    testmodel.inittime = runBUFF
    testmodel.dydrop = args.dy_drop  # 动态参数dropout

    # 准备预测结果保存路径
    # 预测结果包括：Qr（总径流）、Q0（快速径流）、Q1（慢速径流）、Q2（基流）、ET（蒸散发）
    filePathLst = fileManager.namePred(
          testout, Ttest, 'All_Buff'+str(TestBuff), epoch=args.test_epoch, targLst=['Qr', 'Q0', 'Q1', 'Q2', 'ET'])

    # =============================================================================
    # 根据测试模式准备测试输入数据（动态参数模式）
    # =============================================================================
    # ALL模式：时间泛化测试，使用训练期数据预热
    # PUB/PUR模式：空间泛化测试，使用测试期前部分数据预热
    # 动态参数模式需要特殊处理静态参数时间步
    # =============================================================================
    
    if args.pu_opt == 0: # ALL模式：时间泛化测试
        testmodel.staind = TestBuff-1  # 设置静态参数时间步
        zTest = np.concatenate([series_norm[:, -TestBuff:, :], seriestest_norm], axis=1)  # 拼接训练期和测试期数据
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)  # 测试期 XAJ 强迫数据
        # 使用训练期的强迫数据来预热模型
        xTestBuff = xTrain[:, -TestBuff:, :]  # 训练期最后TestBuff天的数据
        xTest = np.concatenate([xTestBuff, xTest], axis=1)  # 拼接预热数据和测试数据
        obs = obsTestUN[:, 0:, :]  # 测试期观测数据（从第0天开始）

    else:  # PUB/PUR模式：空间泛化测试
        testmodel.staind = TestBuff-1  # 设置静态参数时间步
        zTest = seriestest_norm  # 使用测试期的反演数据
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)  # 测试期 XAJ 强迫数据
        obs = obsTestUN  # PUB/PUR 无额外预热段，直接使用测试期观测数据

    # =============================================================================
    # 构建最终的测试输入（动态参数模式）
    # =============================================================================
    # 将流域属性数据添加到时间序列数据中，作为参数学习的输入
    # 动态参数模式需要处理时间维度的属性扩展
    # =============================================================================
    
    # 处理NaN值
    xTest[np.isnan(xTest)] = 0.0
    attrtest = attrtest_norm
    
    # 将流域属性扩展到时间维度
    cTemp = np.repeat(
        np.reshape(attrtest, [attrtest.shape[0], 1, attrtest.shape[-1]]), zTest.shape[1], axis=1)
    zTest = np.concatenate([zTest, cTemp], 2) # 将属性添加到历史强迫数据中作为反演部分
    
    # 构建测试输入元组
    testTuple = (xTest, zTest) # xTest: XAJ 强迫输入; zTest: 参数学习LSTM的输入

    # 显示模型信息
    print("\n" + "=" * 50)
    print(f"使用{args.rnn_type.upper()}作为参数预测骨干网络 (Using {args.rnn_type.upper()} as parameter prediction backbone)")
    print(f"动态参数配置: {args.td_rep} (Dynamic parameters: {args.td_rep})")
    print("=" * 50 + "\n")

    # 模型前向传播
    print("=" * 50)
    print("开始模型测试 (Starting Model Testing)")
    print("=" * 50)

    train.testModel(
        testmodel, testTuple, c=None, batchSize=args.test_batch, filePathLst=filePathLst)

    # 读取预测结果
    pred_time_len = obs.shape[1] + TestBuff - runBUFF
    dataPred = np.full(
        (obs.shape[0], pred_time_len, len(filePathLst)),
        np.nan,
        dtype=float
    )
    for k, filePath in enumerate(filePathLst):
        pred_df = pd.read_csv(
            filePath,
            header=None,
            comment="(",
            engine="python",
            dtype=str
        )
        pred_arr = pd.to_numeric(pred_df.stack(), errors="coerce").unstack().to_numpy()
        if pred_arr.shape[1] > pred_time_len:
            raise ValueError(
                f"预测文件 {os.path.basename(filePath)} 的时间长度 {pred_arr.shape[1]} "
                f"超过预期长度 {pred_time_len}"
            )
        dataPred[:, -pred_arr.shape[1]:, k] = pred_arr

    return dataPred, obs, TestLS, TestBuff, runBUFF, testout, filePathLst, testmodel, forcTestUN, PETTestUN

def save_test_results(args, rootOut, puN, Ttrain, Ttest, TestBuff, testmodel, evaDict, obstestALL, predtestALL):
    """
    保存测试结果（动态参数模式）
    
    该函数负责保存动态参数模型测试的结果，包括：
    1. 创建输出目录
    2. 保存评估指标
    3. 保存观测数据
    4. 保存预测数据
    
    Args:
        args: 命令行参数对象
        rootOut: 模型输出根目录路径
        puN: 流域模式名称
        Ttrain: 训练时间范围
        Ttest: 测试时间范围
        TestBuff: 测试缓冲时间
        testmodel: 测试模型对象
        evaDict: 评估指标字典
        obstestALL: 所有测试流域的观测数据
        predtestALL: 所有测试流域的预测数据
        
    Returns:
        str: 输出路径
    """
    # 保存测试结果
    rnn_type_str = args.rnn_type.upper()
    seStr = (f'Train{Ttrain[0]}_{Ttrain[1]}Test{Ttest[0]}_{Ttest[1]}'
             f'Buff{TestBuff}Staind{testmodel.staind}_{rnn_type_str}')
    
    tdRepS = [str(ix) for ix in args.td_rep]
    TDN = 'TD' + "_".join(tdRepS)
    testsave_path = (f'CAMELSDemo/dPLXAJ/{puN}/TDTestforc/{TDN}/{args.for_type}/BuffOpt{args.buff_opt}/'
                    f'RMSE_para0.25/{args.random_seed}')
    outpath = os.path.join(rootOut, testsave_path, seStr)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    # 保存评估结果、观测数据和预测数据
    EvaFile = os.path.join(outpath, f'Eva{args.test_epoch}.npy')
    np.save(EvaFile, evaDict)

    obsFile = os.path.join(outpath, 'obs.npy')
    np.save(obsFile, obstestALL)

    predFile = os.path.join(outpath, f'pred{args.test_epoch}.npy')
    np.save(predFile, predtestALL)

    return outpath

def process_xaj_parameters(outpath, logtestIDLst, filePathLst, test_start_date=19951001, test_end_date=20101001):
    """
    处理XAJ参数（动态参数模式）
    
    该函数负责处理动态XAJ参数，包括：
    1. 查找XAJ参数文件
    2. 加载和合并所有批次的参数（沿站点维度合并）
    3. 保存合并后的参数
    4. 生成参数统计信息
    5. 保存参数到CSV文件
    6. 生成动态参数时间序列图
    
    Args:
        outpath: 输出路径
        logtestIDLst: 测试流域ID列表
        filePathLst: 预测结果文件路径列表
        test_start_date: 测试开始日期
        test_end_date: 测试结束日期
    """
    print("\n" + "=" * 50)
    print("XAJ参数信息 (XAJ Parameters Info)")
    print("=" * 50)

    # 查找XAJ参数文件（动态：XAJTD）
    xaj_params_files = []
    for file in os.listdir(os.path.dirname(filePathLst[0])):
        if file.startswith('phy_params_XAJTD') and file.endswith('.npy'):
            xaj_params_files.append(file)

    if xaj_params_files:
        print(f"找到 {len(xaj_params_files)} 个XAJ参数文件")
        print(f"Found {len(xaj_params_files)} XAJ parameter files")
        
        # 加载所有批次的XAJ参数
        all_xaj_params = []
        for file in sorted(xaj_params_files):
            file_path = os.path.join(os.path.dirname(filePathLst[0]), file)
            xaj_params = np.load(file_path)
            all_xaj_params.append(xaj_params)
            print(f"- {file}: 形状 {xaj_params.shape}")
            print(f"  {file}: shape {xaj_params.shape}")
        
        # 合并所有批次的参数（动态模型沿站点维度合并）
        if len(all_xaj_params) > 1:
            combined_xaj_params = np.concatenate(all_xaj_params, axis=1)
        else:
            combined_xaj_params = all_xaj_params[0]
        
        print(f"\n合并后的XAJ参数形状: {combined_xaj_params.shape}")
        print(f"Combined XAJ parameters shape: {combined_xaj_params.shape}")
        
        # 保存合并后的参数
        combined_params_file = os.path.join(outpath, 'xaj_parameters_dynamic.npy')
        np.save(combined_params_file, combined_xaj_params)
        print(f"XAJ参数已保存到: {combined_params_file}")
        print(f"XAJ parameters saved to: {combined_params_file}")
        
        # 打印参数统计信息
        print(f"\nXAJ参数统计信息:")
        print(f"XAJ Parameters Statistics:")
        print(f"- 时间步数 (Number of time steps): {combined_xaj_params.shape[0]}")
        print(f"- 站点数 (Number of basins): {combined_xaj_params.shape[1]}")
        print(f"- 参数数 (Number of parameters): {combined_xaj_params.shape[2]}")
        print(f"- 组件数 (Number of components): {combined_xaj_params.shape[3]}")
        print(f"- 参数范围 (Parameter range): [{combined_xaj_params.min():.4f}, {combined_xaj_params.max():.4f}]")
        print(f"- 参数均值 (Parameter mean): {combined_xaj_params.mean():.4f}")
        print(f"- 参数标准差 (Parameter std): {combined_xaj_params.std():.4f}")
        
        # 保存XAJ参数到CSV文件
        xaj_csv_dir = save_results_plot.save_xaj_parameters(outpath, logtestIDLst, combined_xaj_params, model_type='dynamic')
        print(f"XAJ参数CSV文件已保存到: {xaj_csv_dir}")
        print(f"XAJ parameters CSV files saved to: {xaj_csv_dir}")
        
        # 生成动态参数时间序列图
        # 可以选择绘制全部站点或指定数量
        # max_basins=None: 绘制全部站点
        # max_basins=20: 只绘制前20个站点
        # 使用传入的测试时间参数
        param_plots_dir = save_results_plot.plot_dynamic_parameters(outpath, logtestIDLst, combined_xaj_params, model_type='dynamic', max_basins=20, test_start_date=test_start_date, test_end_date=test_end_date)
        print(f"动态参数图表已保存到: {param_plots_dir}")
        print(f"Dynamic parameter plots saved to: {param_plots_dir}")
        
    else:
        print("未找到XAJ参数文件")
        print("No XAJ parameter files found")

def run_all_folds_test(args, rootDatabase, rootOut, puN, gageinfo, gageidLst, tarIDLst, 
                      Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad,
                      varF, varFInv, attrnewLst, PETfull, tPETLst):
    """
    运行所有fold的测试（动态参数模式）
    
    该函数负责运行所有fold的动态参数模型测试，包括：
    1. 初始化结果矩阵
    2. 循环测试所有fold
    3. 对每个fold进行动态参数模型测试
    4. 合并所有fold的测试结果
    
    Args:
        args: 命令行参数对象
        rootOut: 模型输出根目录路径
        puN: 流域模式名称
        gageinfo: 流域信息字典
        gageidLst: 所有流域ID列表
        tarIDLst: 测试流域ID列表（每个fold一个列表）
        Ttest: 测试时间范围
        TtestLst: 测试时间序列数组
        TtestLoad: 测试数据加载时间范围
        TtrainLoad: 训练数据加载时间范围
        TinvLoad: 反演数据加载时间范围
        varF: 气象变量列表
        varFInv: 反演期气象变量列表
        attrnewLst: 流域属性变量列表
        PETfull: 完整PET数据数组
        tPETLst: PET时间序列数组
        
    Returns:
        tuple: 包含以下元素的元组
            - predtestALL: 所有测试流域的预测结果
            - obstestALL: 所有测试流域的观测数据
            - logtestIDLst: 测试流域ID列表
            - TestBuff: 测试缓冲时间
            - testmodel: 测试模型对象
            - filePathLst: 预测结果文件路径列表
            - forcTestUN_all: 所有测试流域的测试期气象数据
            - PETTestUN_all: 所有测试流域的测试期PET数据
    """
    # 定义结果矩阵
    predtestALL = np.full([len(gageidLst), len(TtestLst), 5], np.nan)
    obstestALL = np.full([len(gageidLst), len(TtestLst), 1], np.nan)
    
    # 存储测试期气象数据（用于后续绘图）
    forcTestUN_all = None
    PETTestUN_all = None

    # 根据参数决定要测试的 fold：>0 表示只测试指定折，否则遍历全部
    if args.test_fold is not None and args.test_fold > 0:
        fold_indices = [args.test_fold]
    else:
        fold_indices = list(range(1, len(tarIDLst) + 1))

    nstart = 0
    logtestIDLst = []
    for testfold in fold_indices:
        if testfold < 1 or testfold > len(tarIDLst):
            raise ValueError(f"test_fold={testfold} 超出可用范围 1~{len(tarIDLst)}")
        TestLS = tarIDLst[testfold - 1]
        TestInd = [gageidLst.index(j) for j in TestLS]
        if args.pu_opt == 0:  # Train and test on ALL basins
            TrainLS = gageidLst
            TrainInd = [gageidLst.index(j) for j in TrainLS]
        else:
            TrainLS = list(set(gageinfo['id'].tolist()) - set(TestLS))
            TrainInd = [gageidLst.index(j) for j in TrainLS]

        nbasin = len(TestLS) # number of basins for testing

        # 测试模型
        dataPred, obs, TestLS, TestBuff, runBUFF, testout, filePathLst, testmodel, forcTestUN, PETTestUN = test_model_fold(
            args, rootDatabase, rootOut, puN, gageinfo, TestLS, TestInd, TrainLS, TrainInd,
            Ttest, TtestLoad, TtrainLoad, TinvLoad,
            varF, varFInv, attrnewLst, PETfull, tPETLst, testfold)

        # 保存预测结果到总矩阵
        predtestALL[nstart:nstart + nbasin, :, :] = dataPred[:, TestBuff-runBUFF:, :]
        obstestALL[nstart:nstart + nbasin, :, :] = obs
        
        # 保存测试期气象数据（只保存第一个fold的数据，因为所有fold的测试期数据相同）
        if forcTestUN_all is None:
            forcTestUN_all = forcTestUN
            PETTestUN_all = PETTestUN
        
        nstart = nstart + nbasin
        logtestIDLst = logtestIDLst + TestLS

    return predtestALL, obstestALL, logtestIDLst, TestBuff, runBUFF, testmodel, filePathLst, forcTestUN_all, PETTestUN_all

def compute_evaluation_metrics(predtestALL, obstestALL, logtestIDLst):
    """
    计算评估指标
    
    该函数负责计算模型性能评估指标，包括：
    1. 计算NSE、RMSE、MAE等基本指标
    2. 计算KGE指标（如果缺失）
    3. 返回评估指标字典
    
    Args:
        predtestALL: 所有测试流域的预测结果
        obstestALL: 所有测试流域的观测数据
        logtestIDLst: 测试流域ID列表
        
    Returns:
        list: 包含评估指标字典的列表
    """
    # 计算评估指标
    evaDict = [metrics.statError(predtestALL[:,:,0], obstestALL.squeeze())]  # Q0: 径流

    # 计算KGE指标（如果evaDict中没有）
    if 'KGE' not in evaDict[0]:
        print("Computing KGE metrics for all basins...")
        kge_values = []
        for i in range(len(logtestIDLst)):
            obs = obstestALL[i, :, 0]
            sim = predtestALL[i, :, 0]
            kge = save_results_plot.calculate_kge(obs, sim)
            kge_values.append(kge)
        evaDict[0]['KGE'] = np.array(kge_values)

    return evaDict

def generate_results_and_plots(args, outpath, logtestIDLst, evaDict, obstestALL, predtestALL, 
                              TestBuff, Ttest, filePathLst, forcTestUN_all):
    """
    生成结果和图表（动态参数模式）
    
    该函数负责生成动态参数测试结果的可视化图表，包括：
    1. 输出测试结果摘要
    2. 生成评估指标箱线图
    3. 处理动态XAJ参数
    4. 保存站点数据并生成径流对比图
    5. 生成动态参数时间序列图
    
    Args:
        args: 命令行参数对象
        outpath: 输出路径
        logtestIDLst: 测试流域ID列表
        evaDict: 评估指标字典
        obstestALL: 所有测试流域的观测数据
        predtestALL: 所有测试流域的预测数据
        TestBuff: 测试缓冲时间
        Ttest: 测试时间范围
        filePathLst: 预测结果文件路径列表
        forcTestUN_all: 所有测试流域的测试期气象数据
    """
    # 输出测试结果
    print('=' * 50)
    print('测试完成! (Testing Completed!)')
    print('=' * 50)
    print('评估结果保存在 (Evaluation results saved in):\n', outpath)
    print('所有流域NSE中位数 (For all basins, NSE median):', np.nanmedian(evaDict[0]['NSE']))

    # 使用新函数绘制全部站点的箱线图
    print(f"Generating boxplot for all {len(logtestIDLst)} basins...")
    save_results_plot.plot_metrics_boxplot(outpath, evaDict[0], args.rnn_type)

    # 处理XAJ参数
    process_xaj_parameters(outpath, logtestIDLst, filePathLst, Ttest[0], Ttest[1])

    # 保存测试结果并生成径流对比图
    print("\n正在保存站点数据和生成径流对比图...")

    # 调用函数保存结果并生成图表
    save_results_plot.save_results_and_plot(
        outpath,             # 输出路径
        args.test_epoch,     # 测试使用的epoch
        logtestIDLst,        # 测试站点ID列表
        forcTestUN_all,      # 测试期的气象数据
        obstestALL,          # 测试期的观测流量
        predtestALL,         # 测试期的预测结果
        max_plots=None,      # 生成图表的数量，None表示全部，0表示不生成，正整数表示生成指定数量
        test_start_date=Ttest[0]  # 测试开始日期
    )

    print("结果保存和图表生成完成！")
    print(f"- CSV数据保存在: {os.path.join(outpath, 'csv_results')}")
    print(f"- 径流对比图保存在: {os.path.join(outpath, 'plots')}")

def main():
    """
    主函数
    
    该函数是动态参数测试脚本的入口点，负责协调整个测试流程：
    1. 解析命令行参数
    2. 设置测试环境（随机种子、GPU等）
    3. 设置流域数据分割
    4. 加载测试数据
    5. 运行所有fold的模型测试
    6. 计算评估指标
    7. 保存测试结果
    8. 生成结果图表和统计信息
    
    动态参数测试流程说明：
    - 首先解析用户提供的命令行参数（必须与训练脚本一致）
    - 设置随机种子确保实验可重现
    - 根据实验模式（ALL/PUB/PUR）设置流域数据分割
    - 加载CAMELS测试数据集并进行预处理
    - 加载训练好的动态参数模型
    - 对所有fold进行模型测试和预测
    - 计算NSE、KGE等水文评估指标
    - 保存测试结果和生成可视化图表
    - 分析动态参数的时间变化规律
    
    动态参数模式特点：
    - 部分 XAJ 参数（由td_rep指定）会随时间变化
    - 模型需要处理参数的时间序列特性
    - 测试过程比静态参数模式更复杂
    - 可以分析参数的时间变化规律和趋势
    - 提供更丰富的水文过程理解
    
    Returns:
        None: 测试结果保存到指定路径，不返回值
    """
    # =============================================================================
    # 1. 解析命令行参数
    # =============================================================================
    # 解析用户提供的所有测试参数，必须与训练脚本参数完全一致
    # =============================================================================
    args = parse_args()
    print("=" * 60)
    print("dPLXAJ 动态参数模型测试开始")
    print("=" * 60)
    print(f"实验配置: {args.pu_opt}模式, {args.rnn_type.upper()}网络, {args.for_type}数据源")
    print(f"测试时间: {args.test_start} - {args.test_end}")
    print(f"模型参数: 批次大小={args.batch_size}, 隐藏层={args.hidden_size}, 序列长度={args.rho}")
    print(f"测试配置: 测试批次={args.test_batch}, 模型epoch={args.test_epoch}")
    print(f"动态参数: {args.td_rep} (参数{', '.join(map(str, args.td_rep))}为动态参数)")

    # =============================================================================
    # 2. 设置测试环境
    # =============================================================================
    # 设置随机种子、GPU设备等环境配置，确保实验可重现
    # =============================================================================
    setup_environment(args)

    # =============================================================================
    # 3. 初始化路径配置
    # =============================================================================
    # 初始化CAMELS数据集路径配置，优先使用本地路径
    # =============================================================================
    pathCamels = init_camels_path()

    # =============================================================================
    # 4. 设置流域数据分割
    # =============================================================================
    # 根据实验模式（ALL/PUB/PUR）设置流域数据分割
    # =============================================================================
    rootDatabase, rootOut, gageinfo, puN, tarIDLst, gageidLst = setup_basin_split(args, pathCamels)
    print(f"流域分割完成: 共{len(tarIDLst)}个fold, 总流域数{len(gageidLst)}个")

    # =============================================================================
    # 5. 加载测试数据
    # =============================================================================
    # 加载CAMELS测试数据集，包括气象数据、流量数据、流域属性、PET数据等
    # 进行数据标准化和预处理
    # =============================================================================
    print("开始加载测试数据...")
    Ttrain, Tinv, Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad, varF, varFInv, attrnewLst, PETfull, tPETLst = load_test_data(
        args, rootDatabase, gageinfo)
    print("测试数据加载完成")

    # =============================================================================
    # 6. 运行所有fold的模型测试
    # =============================================================================
    # 对所有fold进行模型测试，包括模型加载、数据预处理、模型预测等
    # 动态参数模式下需要处理参数的时间序列特性
    # =============================================================================
    print("开始运行所有fold的模型测试...")
    print(f"动态参数配置: 参数{', '.join(map(str, args.td_rep))}将随时间变化")
    predtestALL, obstestALL, logtestIDLst, TestBuff, runBUFF, testmodel, filePathLst, forcTestUN_all, PETTestUN_all = run_all_folds_test(
        args, rootDatabase, rootOut, puN, gageinfo, gageidLst, tarIDLst,
        Ttest, TtestLst, TtestLoad, TtrainLoad, TinvLoad,
        varF, varFInv, attrnewLst, PETfull, tPETLst)
    print("所有fold测试完成")

    # =============================================================================
    # 7. 计算评估指标
    # =============================================================================
    # 计算NSE、KGE等水文评估指标，评估模型性能
    # =============================================================================
    print("开始计算评估指标...")
    evaDict = compute_evaluation_metrics(predtestALL, obstestALL, logtestIDLst)
    print("评估指标计算完成")

    # =============================================================================
    # 8. 保存测试结果
    # =============================================================================
    # 保存评估结果、观测数据、预测数据到指定路径
    # =============================================================================
    print("开始保存测试结果...")
    outpath = save_test_results(args, rootOut, puN, Ttrain, Ttest, TestBuff, testmodel, evaDict, obstestALL, predtestALL)
    print("测试结果保存完成")

    # =============================================================================
    # 9. 生成结果图表和统计信息
    # =============================================================================
    # 生成箱线图、径流对比图、XAJ参数分析、动态参数时间序列图等可视化结果
    # =============================================================================
    print("开始生成结果图表...")
    generate_results_and_plots(args, outpath, logtestIDLst, evaDict, obstestALL, predtestALL, TestBuff, Ttest, filePathLst, forcTestUN_all)
    
    # =============================================================================
    # 10. 输出测试完成信息
    # =============================================================================
    print("=" * 60)
    print("动态参数模型测试完成！")
    print("=" * 60)
    print(f"测试结果保存在: {outpath}")
    print(f"测试流域数: {len(logtestIDLst)}个")
    print(f"NSE中位数: {np.nanmedian(evaDict[0]['NSE']):.4f}")
    print(f"KGE中位数: {np.nanmedian(evaDict[0]['KGE']):.4f}")
    print("所有结果图表已生成完成")

if __name__ == "__main__":
    main()
