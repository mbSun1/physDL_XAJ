# -*- coding: utf-8 -*-
"""
XAJ水文模型实现模块 (XAJ Hydrological Model Implementation Module)

本模块包含新安江(XAJ)水文模型的PyTorch实现，支持多组件参数化和深度学习参数反演。

主要功能包括：

1. XAJ水文模型类：
   - XAJMul: 多组件静态参数XAJ模型

2. 深度学习+XAJ模型组合类：
   - MultiInv_XAJModel: 深度学习参数反演+静态XAJ模型

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hydroRouting import UH_conv, UH_gamma
from .dLmodels import (
    CudnnLstmModel, CudnnGruModel, CudnnBiLstmModel, 
    CudnnBiGruModel, CudnnRnnModel, CudnnCnnLstmModel, CudnnCnnBiLstmModel
)



class XAJMul(nn.Module):
    """
    多组件新安江模型（向量化版，HBVMul 风格条件写法）
    ------------------------------------------------------------------
    目标：
    - 在同一批气象强迫下，让每个格点支持 mu 组新安江参数（表达流域异质性/不确定性）
    - 全过程保持“无 Python if、用 clamp/where 做条件触发”的张量化风格
    - 明确分成：三层蒸散 → 蓄满产流 → 土壤含水更新 → 自由水库调蓄 → 三水源分配 → 线性退水 → (可选)河网汇流
    """

    def __init__(self):
        super(XAJMul, self).__init__()

    def forward(self, x, parameters, mu,
                muwts=None, rtwts=None, bufftime=0,
                outstate=False, routOpt=False, comprout=False,
                corrwts=None, pcorr=None):
        """
        参数说明
        --------
        x: [T, B, 2]，依次为 (P, PET)
           - P: mm/d，流域平均降水
           - PET: mm/d，潜在蒸散
        parameters: [B, 12, mu]，12 个无量纲参数，每个格点有 mu 组
        mu: int，多组件个数
        muwts: [B, mu] or None，多组件权重
        rtwts: [B, 2] or [B*mu, 2]，汇流的单位线两个控制参数
        bufftime: 预热长度
        outstate: 是否返回末态（便于分段计算）
        routOpt: 是否做河网汇流
        comprout: 是否“每个组件先汇流再合成”
        """

        # 小数值保护，所有除法、幂次底数都用它兜底
        PRECS = 1e-6

        # ============================================================
        # 1. 初始状态获取：要么预热，要么造一个小正数初值
        # ============================================================
        if bufftime > 0:
            # 有预热：复用同结构模型，把前 bufftime 步推一遍，取末态
            with torch.no_grad():
                x_init = x[:bufftime, :, :]
                init_model = XAJMul()
                Qs_init, WU, WL, WD, S, FR, QI, QG = init_model(
                    x_init, parameters, mu,
                    muwts=muwts, rtwts=rtwts,
                    bufftime=0, outstate=True,
                    routOpt=False, comprout=False,
                    corrwts=corrwts, pcorr=pcorr,
                )
        else:
            # 无预热：直接给各个水库/张力水一个很小的正数，防止后面出现除 0
            Ngrid = x.shape[1]
            device = x.device if hasattr(x, "device") else torch.device("cpu")
            # 参考 HBVMul 方式一初始化（全零张量 + 小正数）
            WU = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 上层张力水
            WL = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 中层张力水
            WD = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 深层张力水
            S  = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 自由水库水深
            QI = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 壤中线性库状态
            QG = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 地下线性库状态
            FR = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 上一时段产流面积


        # ============================================================
        # 2. 去掉预热段，拆出 P / PET
        # ============================================================
        P   = x[bufftime:, :, 0]     # [T, B]
        PET = x[bufftime:, :, 2]     # [T, B]
        Nstep, Ngrid = P.shape
        device = P.device

        # ============================================================
        # 3. 降水订正（可选）：P_corr = P * α(basin)
        #    这样可以把 CFS/ECMWF 之类的系统性偏差拉回观测量级
        # ============================================================
        if pcorr is not None:
            parPCORR = pcorr[0] + corrwts[:, 0] * (pcorr[1] - pcorr[0])
            P = parPCORR.unsqueeze(0).repeat(Nstep, 1) * P

        # 把时间-格点的强迫扩展出 mu 这个维度，后面就能一次性算多个参数集了
        Pm   = P.unsqueeze(-1).repeat(1, 1, mu)     # [T, B, mu]
        PETm = PET.unsqueeze(-1).repeat(1, 1, mu)

        # ============================================================
        # 4. 参数反归一化：把 [0,1] 还原到水文意义范围
        # ============================================================
        # 参数顺序对应：
        # 0 ke   蒸散折算系数
        # 1 b    蓄水容量曲线指数
        # 2 wum  上层张力水容量
        # 3 wlm  中层张力水容量
        # 4 wm   总张力水容量
        # 5 c    中下层蒸散折减系数
        # 6 sm   自由水库最大水深
        # 7 ex   自由水库出流曲线指数
        # 8 ki   自由水库→壤中补给系数
        # 9 kg   自由水库→地下补给系数
        # 10 ci  壤中线性退水系数
        # 11 cg  地下线性退水系数
        parascaLst = [
            [0.3, 2.0],     # 0 ke
            [0.0, 5.0],     # 1 b
            [0.01, 100.0],  # 2 wum
            [0.01, 100.0],  # 3 wlm
            [0.01, 200.0],  # 4 wm
            [0.09, 1.0],    # 5 c
            [0.01, 100.0],  # 6 sm
            [0.0, 10.0],    # 7 ex
            [0.01, 0.7],    # 8 ki
            [0.01, 0.7],    # 9 kg
            [0.0, 0.998],   # 10 ci
            [0.0, 0.998],   # 11 cg
        ]
        routscaLst = [[0.0, 2.9], [0.0, 6.5]]  # 汇流单位线两个控制量的范围

        sc = lambda arr, i: parascaLst[i][0] + arr * (parascaLst[i][1] - parascaLst[i][0])

        parKE, parB, parWUM, parWLM, parWM, parC, parSM, parEX, parKI_, parKG_, parCI, parCG = [
            sc(parameters[:, i, :], i) for i in range(len(parascaLst))
        ]

        # 深层张力水容量 = 总容量 - 上层 - 中层，若用户给的范围有重叠，这里强行截成 ≥0，保证有物理意义
        wdm = torch.clamp(parWM - parWUM - parWLM, min=0.0)

        # KI + KG 必须 < 1，否则会把自由水在同一时段抽光
        sum_k = parKI_ + parKG_
        parKI = torch.where(sum_k < 1.0, parKI_, (1 - PRECS) * parKI_ / (sum_k + PRECS))
        parKG = torch.where(sum_k < 1.0, parKG_, (1 - PRECS) * parKG_ / (sum_k + PRECS))

        # ============================================================
        # 5. 给时间序列通量预分配空间，便于最后做 mu 聚合
        # ============================================================
        QS_inflow = torch.zeros((Nstep, Ngrid, mu), device=device)  # 坡面/快流
        QI_inflow = torch.zeros_like(QS_inflow)                     # 壤中流
        QG_inflow = torch.zeros_like(QS_inflow)                     # 地下流
        ET_series = torch.zeros_like(QS_inflow)                     # 实际蒸散
        FR0 = FR.clone()                                            # 上一时段产流面积

        # ============================================================
        # 6. 主时间推进：一层一层走完整个新安江日过程
        # ============================================================
        for t in range(Nstep):
            # --------------------------------------------------------
            # 6.1 三层蒸散模块
            #    逻辑：先消耗上层 + 当日雨能提供的水 → 再向中层索取 → 再向深层索取
            #    上层是能量受控，下面两层是水量受限 + 折减受限
            # --------------------------------------------------------
            P_t   = Pm[t, :, :]               # 当前时段降水
            PET_t = PETm[t, :, :] * parKE     # 折算后的 PET，表示“这片下垫面真正能蒸掉的上限”

            # 上层蒸散：能量足够时把 PET 全蒸掉，否则把手上能拿到的水蒸掉（WU + 当日雨）
            EU = torch.min(PET_t, WU + P_t)

            # 向下层传的蒸散需求量
            D = torch.clamp(PET_t - EU, min=0.0)

            # 中层蒸散：我们用“湿润度因子”来平滑地在“湿润公式”和“干旱公式”之间切换，
            # 而不是写 if WL > c*WLM ... 这样可以保持张量可导。
            wet_excess = torch.clamp(WL - parC * parWLM, min=0.0)   # >0 说明中层比较湿
            wet_ratio  = wet_excess / (wet_excess + 1.0)            # 映射到 (0,1)

            # 湿润时：按“剩余需水 × 相对含水量”供水
            EL_wet = D * WL / (parWLM + PRECS)
            # 干旱时：只能按 c 倍供一点，且不能超过 WL 自己的水
            EL_dry = torch.min(parC * D, WL)

            # 最终中层蒸散 = 两个公式按湿润度加权
            EL = wet_ratio * EL_wet + (1 - wet_ratio) * EL_dry
            EL = torch.clamp(EL, min=0.0)
            EL = torch.min(EL, WL)  # 不能蒸出负数

            # 深层蒸散：上一层没供够的部分，再按 c 吸一点，不能超过深层现有水
            D_res = torch.clamp(D - EL, min=0.0)
            ED = torch.min(parC * D_res, WD)

            # 当日实际蒸散
            E = EU + EL + ED
            ET_series[t, :, :] = E

            # --------------------------------------------------------
            # 6.2 蓄满产流模块
            #    经典新安江思路：土壤越湿，产流越容易；用容量曲线把“湿度+净雨”映射成产流量
            # --------------------------------------------------------
            PE = torch.clamp(P_t - E, min=0.0)   # 降水扣掉蒸散后的“能进张力水/自由水的净雨”
            W  = WU + WL + WD                    # 当前三层土壤总含水量，代表“土壤预湿状况”

            # 容量曲线的“最大等效水深” WMM
            WMM = parWM * (1.0 + parB)

            # 1 - W/WM 是“还没装满的比例”，越小越湿，这个值要当幂次底数，所以 clamp 到正数
            base_W = torch.clamp(1.0 - W / (parWM + PRECS), min=PRECS)

            # A 是“容量曲线已经贡献出来的等效水深”，越湿 A 越大
            A = WMM * (1.0 - base_W ** (1.0 / (1.0 + parB)))

            # (PE + A) / WMM 表征“今天的水+已有湿度”相对于能装的上限的比例
            ratio = torch.clamp((PE + A) / (WMM + PRECS), max=1.0)

            # 分段产流：
            # - 没装满：有个幂次形式的非线性产流
            # - 装满了：就是超量产流
            R_per = torch.where(
                PE + A < WMM,
                PE - (parWM - W) + parWM * (1.0 - ratio) ** (1.0 + parB),
                PE - (parWM - W)
            )
            R_per = torch.clamp(R_per, min=0.0)  # 不允许负产流

            # --------------------------------------------------------
            # 6.3 三层张力水更新（透水区质量守恒）
            #    这一步决定了下个时段的“土壤预湿状况”，跟后面每一步都耦合
            # --------------------------------------------------------
            dW = P_t - E - R_per  # 真正留在土壤里的水量（可能正也可能负）

            # 正的：说明今天水多 → 往上中下三层依次回填
            dpos = torch.clamp(dW, min=0.0)

            # 填上层：不能超过 parWUM
            fill_u = torch.min(dpos, torch.clamp(parWUM - WU, min=0.0))
            WU = WU + fill_u
            dpos = dpos - fill_u

            # 填中层
            fill_l = torch.min(dpos, torch.clamp(parWLM - WL, min=0.0))
            WL = WL + fill_l
            dpos = dpos - fill_l

            # 填深层
            fill_d = torch.min(dpos, torch.clamp(wdm - WD, min=0.0))
            WD = WD + fill_d

            # 负的：说明今天亏水 → 按上→中→下的顺序抽水，不能抽到负数
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
            # 6.4 自由水库调蓄 + 快流生成
            #    这里对应新安江的“自由水库”部分，核心有四步：
            #    (1) 定义当前产流面积 FR（有雨就按 R/PE，没雨就承袭上一步）；
            #    (2) 把上一时段 S 按面积换算，保证体积守恒；
            #    (3) 算自由水库的“可释水量” AU，非线性函数；
            #    (4) 按两段式公式算快流 RS，并不超过 R_per
            # --------------------------------------------------------
            # (1) 产流面积 FR：有雨时 R/PE，PE=0 时沿用上时段
            FR_new = R_per / (PE + PRECS)
            FR = FR_new * (PE > 0.0).float() + FR0 * (PE <= 0.0).float()
            FR = torch.clamp(FR, 0.0, 1.0)

            # (2) 面积换算：上时段 S 是 FR0 面积上的水深，现在要换到 FR 面积上，保持体积不变
            S_eq = S * FR0 / (FR + PRECS)
            S_eq = torch.clamp(S_eq, max=parSM)

            # (3) 自由水库非线性出流能力 AU
            SMM = parSM * (1.0 + parEX)
            ratio_s = torch.clamp(1.0 - S_eq / (parSM + PRECS), min=PRECS)
            AU = SMM * (1.0 - ratio_s ** (1.0 / (1.0 + parEX)))

            # (4) 快流 RS：对应新安江的“两段式产流”写法
            ratio_pa = torch.clamp((PE + AU) / (SMM + PRECS), max=1.0)
            RS_part1 = (PE + S_eq - parSM + parSM * (1.0 - ratio_pa) ** (1.0 + parEX)) * FR
            RS_part2 = (PE + S_eq - parSM) * FR
            RS = torch.where(PE + AU < SMM, RS_part1, RS_part2)
            RS = torch.clamp(RS, min=0.0)
            RS = torch.min(RS, R_per)  # 坡面快流不能超过透水区当期产流

            # 回蓄：没变成快流的就回到自由水库
            S = S_eq + (R_per - RS) / (FR + PRECS)
            S = torch.clamp(S, max=parSM)

            # 更新 FR0，给下一时段做面积换算用
            FR0 = FR

            # --------------------------------------------------------
            # 6.5 三水源分配 + 坡面前的线性水库退水
            #    新安江里自由水库不是一次把水都放出去，而是分成：直接出流的 RS，
            #    还有两条滞后项 RI / RG，再各自走一阶线性水库
            # --------------------------------------------------------
            # 自由水库→壤中、地下的入流
            RI = parKI * S * FR
            RG = parKG * S * FR
            # 自由水库剩余水量
            S = S * (1.0 - parKI - parKG)

            # 一阶线性退水：当前出流 = 上一时刻出流 * 衰减 + (1-衰减)*当前入库
            QI = parCI * QI + (1.0 - parCI) * RI
            QG = parCG * QG + (1.0 - parCG) * RG

            # 记录本时段通量
            QS_inflow[t, :, :] = RS
            QI_inflow[t, :, :] = QI
            QG_inflow[t, :, :] = QG

        # ============================================================
        # 7. 组件聚合：把 mu 个参数集的结果合成一个流域响应
        # ============================================================
        # 先把三个分量各自做组件平均，方便分析
        QSave = QS_inflow.mean(-1, keepdim=True)
        QIave = QI_inflow.mean(-1, keepdim=True)
        QGave = QG_inflow.mean(-1, keepdim=True)
        ETave = ET_series.mean(-1, keepdim=True)

        # 真正要进河的，是三部分的和
        river_inflow = QS_inflow + QI_inflow + QG_inflow  # [T, B, mu]

        if muwts is None:
            inflow_mean = river_inflow.mean(-1)           # [T, B]
        else:
            inflow_mean = (river_inflow * muwts).sum(-1)

        # ============================================================
        # 8. 河网汇流（可选）：单位线卷积
        # ============================================================
        if routOpt:
            # 选谁做汇流输入：每个组件单独路由 or 合成后再路由
            if comprout:
                Qsim = river_inflow.view(Nstep, Ngrid * mu)
            else:
                Qsim = inflow_mean

            # 把 [0,1] 的路由参数映射回物理区间
            tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])

            # 展开到时间维
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)

            # 生成 Gamma 单位线并做 1D 卷积
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
            # 不做汇流就直接把“入河量”当出口
            Qs = inflow_mean.unsqueeze(-1)

        # ============================================================
        # 9. 输出控制
        # ============================================================
        if outstate:
            # 返回末态：给后面分段接着跑
            return Qs, WU, WL, WD, S, FR, QI, QG
        else:
            # 标准输出：主流量 + 三分量 + 蒸散
            Qall = torch.cat((Qs, QSave, QIave, QGave, ETave), dim=-1)
            return Qall



class MultiInv_XAJModel(nn.Module):
    """
    深度学习参数反演+静态XAJ模型类 (Deep Learning Parameter Inversion + Static XAJ Model Class)
    
    这是dPLXAJ模型的核心实现，结合了深度学习参数反演和XAJ水文模型。
    模型使用RNN（LSTM/GRU/BiLSTM/BiGRU/RNN/CNN-LSTM/CNN-BiLSTM）从静态属性反演XAJ模型参数，
    然后使用XAJ模型进行水文模拟。
    
    模型架构 (Model Architecture):
    ----------
    1. RNN参数反演模块：静态属性 -> XAJ参数
    2. XAJ水文模型：气象数据 + XAJ参数 -> 径流模拟
    
    参数 (Parameters):
    ----------
    ninv : int
        输入特征维度（静态属性数量）
    nfea : int
        XAJ参数特征数量（固定为12）
    nmul : int
        多组件数量，用于参数不确定性量化
    hiddeninv : int
        RNN隐藏层大小
    drinv : float, default=0.5
        RNN Dropout比率
    inittime : int, default=0
        XAJ模型预热时间步长
    routOpt : bool, default=False
        是否启用汇流计算
    comprout : bool, default=False
        是否对每个组件分别进行汇流
    compwts : bool, default=False
        是否学习多组件权重
    pcorr : list or None, default=None
        降水校正参数范围
    rnn_type : str, default='lstm'
        RNN类型：'lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'
        
    用途 (Usage):
    ----------
    主要用于静态参数XAJ模型的深度学习参数反演，适用于流域水文模拟和预测。
    """
    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
                 compwts=False, pcorr=None, rnn_type='lstm'):
        """
        初始化深度学习参数反演+静态XAJ模型
        
        参数说明见类文档
        """
        super(MultiInv_XAJModel, self).__init__()
        self.ninv = ninv                    # 输入特征维度（静态属性数量）
        self.nfea = nfea                    # XAJ参数特征数量（固定为12）
        self.hiddeninv = hiddeninv          # RNN隐藏层大小
        self.nmul = nmul                    # 多组件数量
        self.rnn_type = rnn_type.lower()    # RNN类型：'lstm', 'gru', 'bilstm', 'bigru', 'rnn'
        
        # =============================================================================
        # 参数计算 (Parameter Calculation)
        # =============================================================================
        # 计算RNN需要输出的总参数数量
        nxajpm = nfea * nmul  # XAJ参数总数 = 特征数 × 多组件数
        
        # 汇流参数数量计算
        if comprout is False:
            nroutpm = 2  # 简单汇流：每个流域2个参数
        else:
            nroutpm = nmul * 2  # 组件汇流：每个组件2个参数
        
        # 多组件权重参数数量计算
        if compwts is False:
            nwtspm = 0  # 不使用权重：简单平均
        else:
            nwtspm = nmul  # 使用权重：每个组件一个权重
        
        # 降水校正参数数量计算
        if pcorr is None:
            ntp = nxajpm + nroutpm + nwtspm  # 总参数数
        else:
            ntp = nxajpm + nroutpm + nwtspm + 1  # 总参数数 + 1个降水校正参数
        
        # =============================================================================
        # RNN模型选择 (RNN Model Selection)
        # =============================================================================
        # 根据rnn_type参数选择相应的RNN模型
        if self.rnn_type == 'lstm':
            self.lstminv = CudnnLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'gru':
            self.lstminv = CudnnGruModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bilstm':
            self.lstminv = CudnnBiLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bigru':
            self.lstminv = CudnnBiGruModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'rnn':
            self.lstminv = CudnnRnnModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnlstm':
            self.lstminv = CudnnCnnLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnbilstm':
            self.lstminv = CudnnCnnBiLstmModel(
                nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        else:
            raise ValueError("rnn_type must be one of: 'lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'")

        # XAJ水文模型实例
        self.XAJ = XAJMul()

        # =============================================================================
        # 模型属性保存 (Model Attributes Storage)
        # =============================================================================
        self.gpu = 1                    # GPU标志（保留用于兼容性）
        self.inittime = inittime        # XAJ模型预热时间
        self.routOpt = routOpt          # 汇流选项
        self.comprout = comprout        # 组件汇流选项
        self.nxajpm = nxajpm           # XAJ参数数量
        self.nwtspm = nwtspm           # 权重参数数量
        self.nroutpm = nroutpm         # 汇流参数数量
        self.pcorr = pcorr             # 降水校正参数

    def forward(self, x, z, doDropMC=False):
        """
        深度学习参数反演+静态XAJ模型前向传播函数
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            气象强迫数据，形状为[time, basin, var]
            var包括：P(mm/d)降雨, PET(mm/d)潜在蒸散发
        z : torch.Tensor
            静态属性数据，形状为[time, basin, ninv]
            用于RNN参数反演的输入特征
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            XAJ模型输出，包含径流模拟结果
        """
        # =============================================================================
        # RNN参数反演 (RNN Parameter Inversion)
        # =============================================================================
        # 使用RNN从静态属性反演XAJ模型参数
        # 使用RNN从静态属性反演XAJ模型参数
        Gen = self.lstminv(z)  # RNN输出，形状为[time, basin, ntp]
        
        # 提取最后一个时间步的参数作为XAJ模型参数
        Params0 = Gen[-1, :, :]  # 形状为[basin, ntp]
        ngage = Params0.shape[0]  # 流域数量
        
        # =============================================================================
        # 参数解析和变换 (Parameter Parsing and Transformation)
        # =============================================================================
        # XAJ参数：从RNN输出中提取XAJ相关参数
        xajpara0 = Params0[:, 0:self.nxajpm]  # XAJ参数部分
        # 使用sigmoid激活函数将参数限制在[0,1]范围内
        xajpara = torch.sigmoid(xajpara0).view(ngage, self.nfea, self.nmul)
        
        # 汇流参数：从RNN输出中提取汇流相关参数
        routpara0 = Params0[:, self.nxajpm:self.nxajpm+self.nroutpm]
        if self.comprout is False:
            # 简单汇流：每个流域2个参数
            routpara = torch.sigmoid(routpara0)
        else:
            # 组件汇流：每个组件2个参数
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)
        
        # 多组件权重：从RNN输出中提取权重参数
        if self.nwtspm == 0:
            # 不使用权重：简单平均
            wts = None
        else:
            # 使用权重：学习各组件的重要性
            wtspara = Params0[:, self.nxajpm+self.nroutpm:self.nxajpm+self.nroutpm+self.nwtspm]
            wts = F.softmax(wtspara, dim=-1)  # 使用softmax确保权重和为1
        
        # 降水校正参数：从RNN输出中提取降水校正参数
        if self.pcorr is None:
            corrpara = None
        else:
            corrpara0 = Params0[:, self.nxajpm+self.nroutpm+self.nwtspm:self.nxajpm+self.nroutpm+self.nwtspm+1]
            corrpara = torch.sigmoid(corrpara0)
        
        # =============================================================================
        # XAJ水文模拟 (XAJ Hydrological Simulation)
        # =============================================================================
        # 使用反演得到的参数运行XAJ模型
        out = self.XAJ(
            x, 
            parameters=xajpara, 
            mu=self.nmul, 
            muwts=wts, 
            rtwts=routpara, 
            bufftime=self.inittime,
            routOpt=self.routOpt, 
            comprout=self.comprout, 
            corrwts=corrpara, 
            pcorr=self.pcorr
        )
        return out

