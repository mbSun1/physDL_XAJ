"""
dPLXAJ Static/Dynamic Parameter Training Script
==============================================
训练 XAJ（新安江）模型（静态/动态参数）的脚本
- 使用 LSTM/GRU 等作为参数反演骨干网络
- 支持 XAJ 静态参数与动态参数两种模式
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
    
    该函数定义了所有训练脚本需要的命令行参数，包括：
    - 实验模式配置：流域选择、预热选项、参数模式、气象数据源
    - 模型架构配置：RNN类型、批次大小、序列长度、隐藏层大小、XAJ参数个数
    - 时间配置：训练时间、反演时间、预热时间
    - XAJ模型配置：汇流选项、多组件数量、组件汇流、组件权重
    - 训练配置：训练轮数、保存间隔、随机种子、GPU设置
    - 动态参数配置：动态参数索引、蒸散发模块、dropout、静态参数时间步
    
    Returns:
        argparse.Namespace: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description='dPLXAJ Static/Dynamic Parameter Training')

    # =============================================================================
    # 实验模式配置参数
    # =============================================================================
    # 这些参数控制实验的整体设置，包括数据分割方式和数据源选择
    # =============================================================================
    
    parser.add_argument('--pu_opt', type=int, default=0, choices=[0, 1, 2],
                       help='流域选择模式: 0=ALL(所有流域训练测试), 1=PUB(随机保留流域空间泛化), 2=PUR(连续区域保留空间泛化)')
    parser.add_argument('--buff_opt', type=int, default=0, choices=[0, 1, 2],
                       help='预热选项: 0=第一年数据仅用于预热下一年, 1=重复第一年数据预热第一年, 2=加载额外一年数据预热第一年')
    parser.add_argument('--td_opt', action='store_true', default=False,
                       help='参数模式: False=静态参数(所有XAJ参数不随时间变化), True=动态参数(部分XAJ参数随时间变化)')
    parser.add_argument('--for_type', type=str, default='daymet',
                       choices=['daymet', 'nldas', 'maurer'],
                       help='气象数据源: daymet=Daymet气象数据, nldas=NLDAS气象数据, maurer=Maurer气象数据')

    # =============================================================================
    # 模型架构配置参数
    # =============================================================================
    # 这些参数控制深度学习模型的架构和训练设置
    # =============================================================================
    #已经训练lstm10_10、bilstm20_10、gru10_10、rnn10_10、cnnbilstm20_10、cnnlstm19_10、bigru10_10，正在训练bigru20_10
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
    # 这些参数控制训练和反演的时间范围，以及模型预热设置
    # =============================================================================
    
    parser.add_argument('--train_start', type=str, default='19801001',
                       help='训练开始日期 (YYYYMMDD): XAJ模型训练的时间段开始日期，格式为YYYYMMDD')
    parser.add_argument('--train_end', type=str, default='20041001',
                       help='训练结束日期 (YYYYMMDD): XAJ模型训练的时间段结束日期，格式为YYYYMMDD')
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
    # =============================================================================
    
    parser.add_argument('--routing', action='store_true', default=True,
                       help='汇流选项: 是否启用汇流模块，True=使用汇流模块进行径流汇合，False=不使用汇流模块')
    parser.add_argument('--nmul', type=int, default=10,
                       help='多组件数量: 并行XAJ组件数量，1表示单组件，>1表示多组件')
    parser.add_argument('--comprout', action='store_true', default=False,
                       help='组件汇流: 是否对每个组件单独进行汇流，True=每个组件单独汇流，False=组件间不汇流')
    parser.add_argument('--compwts', action='store_true', default=False,
                       help='组件权重: 是否使用加权平均组合多组件结果，True=使用加权平均，False=使用简单平均')

    # =============================================================================
    # 训练配置参数
    # =============================================================================
    # 这些参数控制模型训练过程和计算环境设置
    # =============================================================================
    
    parser.add_argument('--epoch', type=int, default=10,
                       help='总训练轮数: 模型训练的完整轮数，每轮遍历一次完整的训练数据集')
    parser.add_argument('--save_epoch', type=int, default=1,
                       help='模型保存间隔: 每隔多少个epoch保存一次模型，用于模型检查点和早停')
    parser.add_argument('--random_seed', type=int, default=111111,
                       help='随机种子: 用于确保实验的可重现性，训练和测试脚本必须使用相同的随机种子')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID: 使用的GPU设备编号，0表示第一块GPU，-1表示使用CPU')
    parser.add_argument( '--test_fold',type=int,default=1,
        help=('test in PUB/PUR 模式；1 表示 PUBsplitLst.txt 中的第一个列表，依此类推；设为 <=0 时遍历全部折。'))

    # =============================================================================
    # 动态参数配置参数（仅当 td_opt=True 时生效）
    # =============================================================================
    # 这些参数控制动态参数模式下的特殊设置，在静态参数模式下不使用
    # =============================================================================
    
    parser.add_argument('--td_rep', type=int, nargs='+', default=[1, 12],
                       help='动态参数索引列表: 指定哪些XAJ参数为动态参数（范围: 1-12）')
    parser.add_argument('--dy_drop', type=float, default=0.0,
                       help='动态参数dropout: 动态参数的dropout率，0.0=始终动态，1.0=始终静态，中间值=随机选择')
    parser.add_argument('--sta_ind', type=int, default=-1,
                       help='静态参数时间步: 静态参数使用的时间步，-1=使用最后一个时间步，其他值=使用指定时间步')

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
    # GPU设备设置和显存监控
    # =============================================================================
    # 根据参数设置使用的GPU设备，并添加正确的显存监控
    # =============================================================================
    
    if torch.cuda.is_available():
        # 检查指定的GPU是否存在
        if args.gpu_id >= torch.cuda.device_count():
            print(f"警告: GPU {args.gpu_id} 不存在，系统只有 {torch.cuda.device_count()} 块GPU")
            print(f"回退到GPU 0")
            args.gpu_id = 0
        
        torch.cuda.set_device(args.gpu_id)           # 设置当前使用的GPU设备
        print(f"使用GPU设备: {args.gpu_id}")
        print(f"GPU名称: {torch.cuda.get_device_name(args.gpu_id)}")
        
        # 正确的显存信息
        total_memory = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3
        print(f"GPU总显存: {total_memory:.2f}GB")
        
        # 清理显存缓存
        torch.cuda.empty_cache()
        
        # 显示初始显存使用
        allocated = torch.cuda.memory_allocated(args.gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(args.gpu_id) / 1024**3
        print(f"初始显存使用: {allocated:.2f}GB / {total_memory:.2f}GB")
        print(f"显存缓存: {reserved:.2f}GB")
        print("-" * 50)
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

    典型目录结构示例：
    PhysDL/
      ├─ Camels/                      # 数据（或使用 CAMELS/ 同义）
      │   ├─ basin_timeseries_v1p2_metForcing_obsFlow/
      │   ├─ basin_metadata/
      │   └─ pet_harg/
      ├─ outputs/                     # 训练/测试输出
      └─ experiments/

    注意事项：
    - Windows 与 Linux 路径均可支持；默认回退路径使用绝对根（/scratch, /data）。
    - 若使用自定义路径，可在此函数外部重载或将返回的 pathCamels 替换。
    """
    # 推断工程根目录：从当前脚本（experiments 目录内）回到其上一级目录
    # 示例：.../PhysDL/experiments/<this_script>.py -> .../PhysDL
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
    # 时间范围处理
    Ttrain = [int(args.train_start), int(args.train_end)]
    Tinv = [int(args.inv_start), int(args.inv_end)]

    # 应用预热选项
    if args.buff_opt == 2:  # 加载额外数据用于预热
        sd = time.t2dt(Ttrain[0]) - dt.timedelta(days=args.buff_time)
        sdint = int(sd.strftime("%Y%m%d"))
        TtrainLoad = [sdint, Ttrain[1]]
        TinvLoad = [sdint, Tinv[1]]
    else:
        TtrainLoad = Ttrain
        TinvLoad = Tinv

    # 气象变量配置
    if args.for_type == 'daymet':
        varF = ['prcp', 'tmean']
        varFInv = ['prcp', 'tmean']
    else:
        varF = ['prcp', 'tmax']  # maurer和nldas数据中tmax实际是tmean
        varFInv = ['prcp', 'tmax']

    # 流域属性变量列表
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

    # 加载XAJ模型训练数据（气象强迫和观测流量）
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
    
    该函数负责构建和训练 dPLXAJ 模型，包括：
    1. 根据参数模式（静态/动态）配置模型参数
    2. 设置损失函数和训练选项
    3. 构建模型输出路径
    4. 定义模型架构（静态参数模型或动态参数模型）
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
    # 动态参数配置
    if args.td_opt:
        tdRep = args.td_rep
        tdRepS = [str(ix) for ix in tdRep]
        Nfea = args.nfea
        dydrop = args.dy_drop
        staind = args.sta_ind
        TDN = '/TDTestforc/' + 'TD' + "_".join(tdRepS) + '/'
    else:
        TDN = '/Testforc/'
        Nfea = args.nfea

    # 损失函数配置
    alpha = 0.25  # RMSE损失权重
    optLoss = defaultSetting.update_config(defaultSetting.get_loss_config(), name='hydroMLlib.utils.criterion.RmseLossComb', weight=alpha)
    lossFun = criterion.RmseLossComb(alpha=alpha)

    # 训练选项
    optTrain = defaultSetting.update_config(defaultSetting.get_train_config(),
                             miniBatch=[args.batch_size, args.rho],
                             nEpoch=args.epoch,
                             saveEpoch=args.save_epoch)

    # =============================================================================
    # 模型输出路径设置
    # =============================================================================
    # 输出路径结构：
    # rootOut/exp_name/exp_disp/exp_info/
    # 例如：outputs/CAMELSDemo/dPLXAJ/ALL/Testforc/daymet/BuffOpt0/RMSE_para0.25/111111/Fold1/LSTM/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_12_Buff_365_Mul_16_LSTM/
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

    # 模型定义
    Ninv = forcTuple[1].shape[-1] + attrs.shape[-1]

    if not args.td_opt:  # 静态参数模型（XAJ）
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

    # 保存配置和统计信息
    masterDict = fileManager.wrapMaster(out, optData, optModel, optLoss, optTrain)
    fileManager.writeMasterFile(masterDict)

    # 保存标准化统计信息
    statFile = os.path.join(out, 'statDict.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)

    # 开始模型训练
    print("=" * 50)
    print("开始模型训练 (Starting Model Training)")
    print("=" * 50)

    trainedModel = train.trainModel(
        model,
        forcTuple,
        yTrainIn,
        attrs,
        lossFun,
        nEpoch=args.epoch,
        miniBatch=[args.batch_size, args.rho],
        saveEpoch=args.save_epoch,
        saveFolder=out,
        bufftime=args.buff_time
    )

    print("=" * 50)
    print("模型训练完成! (Model Training Completed!)")
    print("=" * 50)

    return trainedModel, out

def main():
    """
    主函数
    
    该函数是训练脚本的入口点，负责协调整个训练流程：
    1. 解析命令行参数
    2. 设置训练环境（随机种子、GPU等）
    3. 初始化路径配置
    4. 设置流域数据分割
    5. 加载和预处理数据
    6. 训练模型
    7. 输出训练结果
    
    训练流程说明：
    - 首先解析用户提供的命令行参数
    - 设置随机种子确保实验可重现
    - 初始化CAMELS数据集路径配置
    - 根据实验模式（ALL/PUB/PUR）分割流域数据
    - 加载CAMELS数据集并进行预处理
    - 构建 dPLXAJ 模型并开始训练
    - 训练完成后保存模型和配置信息
    
    Returns:
        tuple: 包含训练完成的模型对象和输出路径
    """
    # =============================================================================
    # 1. 解析命令行参数
    # =============================================================================
    # 解析用户提供的所有训练参数，包括模型配置、数据设置等
    # =============================================================================
    args = parse_args()
    print("=" * 60)
    print("dPLXAJ 参数模型训练开始（静态/动态）")
    print("=" * 60)
    print(f"实验配置: {args.pu_opt}模式, {args.rnn_type.upper()}网络, {args.for_type}数据源")
    print(f"训练时间: {args.train_start} - {args.train_end}")
    print(f"模型参数: 批次大小={args.batch_size}, 隐藏层={args.hidden_size}, 序列长度={args.rho}")

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
    # 6 训练模型
    # =============================================================================
    # 构建 dPLXAJ 模型并开始训练
    # 训练过程包括前向传播、损失计算、反向传播、参数更新等
    # =============================================================================
    print("开始模型训练...")
    trainedModel, out = train_model(args, rootOut, puN, optData, statDict, forcTuple, yTrainIn, attrs)

    # =============================================================================
    # 7. 输出训练结果
    # =============================================================================
    # 显示训练完成信息和模型保存路径
    # =============================================================================
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"模型保存在: {out}")
    print(f"训练配置已保存，可用于后续测试")
    return trainedModel, out

if __name__ == "__main__":
    main()