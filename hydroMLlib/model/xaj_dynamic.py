import torch  
import torch.nn as nn
from .hydroRouting import UH_conv, UH_gamma
from .xaj_static import XAJMul
from .dLmodels import (
    CudnnLstmModel, CudnnGruModel, CudnnBiLstmModel,
    CudnnBiGruModel, CudnnRnnModel, CudnnCnnLstmModel, CudnnCnnBiLstmModel,
)

import torch
import torch.nn as nn

class XAJMulTD(nn.Module):
    """
    动态参数版多组件新安江模型（XAJMulTD）

    核心思想：
    1. 沿用静态新安江的物理链条不变：
       三层蒸散 → 蓄满产流 → 三层张力水更新 → 自由水库调蓄 → 三水源分配 → 线性退水 → (可选)汇流
    2. 但允许“部分参数”随时间变化（比如 KE、WM、SM、EX 等），实现对气候/下垫面/人类活动的时变响应
    3. 用 tdlst 指定哪些参数要随时间走，其余的用 staind 那一帧当作“静态底”
    4. 用 dydrop 做一个“动态参数的退火/抖动”：有一定概率回落到静态值，防止每一时刻都过拟合
    5. 数值稳定性完全沿用静态版 + HBVMul 风格：状态初始化用 (zeros + eps)，分母/幂底全 clamp
    """

    def __init__(self):
        super(XAJMulTD, self).__init__()

    def forward(self, x,
                parameters,  # [T, B, 12, mu] 每一步的动态参数（归一化）
                staind,      # int，静态基准参数的时间索引
                tdlst,       # list[int] 要动态化的参数索引（1-based）
                mu,
                muwts=None,
                rtwts=None,
                bufftime=0,
                outstate=False,
                routOpt=False,
                comprout=False,
                corrwts=None,
                pcorr=None,
                dydrop=0.0):
        """
        x: [time, basin, var]  这里 var 至少有 P, PET；你前面写成了 x[...,2] 是 PET
        parameters: [time, basin, 12, mu] 动态参数序列(0~1)
        staind: 用哪一帧的参数当作“静态底”
        tdlst: 哪几个参数是动态的（1-based，对应下面 parascaLst 的顺序）
        mu: 组件数
        dydrop: 动态参数 dropout 概率，0=都用动态，1=都回退静态
        """

        # ------------------------------------------------------------
        # 0. 数值保护常数：所有除法、幂次底数都用它兜底
        # ------------------------------------------------------------
        PRECS = 1e-6

        # ============================================================
        # 1. 预热 / 初始水文状态
        #    目的：让 WU/WL/WD/S/QI/QG/FR 不从 0 开始而是从一个“合理的湿度”开始
        #    做法有两种：
        #    (1) 给 bufftime>0：就真的跑一遍静态 XAJ，把末态接过来；
        #    (2) 否则：像 HBVMul 那样给一堆 very small > 0 的初值
        # ============================================================
        if bufftime > 0:
            # 1.1 用静态 XAJ 跑预热段
            with torch.no_grad():
                x_init = x[:bufftime, :, :]                 # [buff, B, var]
                # 预热阶段取当时最后一帧的动态参数当“静态参数”
                par_init = parameters[bufftime - 1, :, :, :]  # [B, 12, mu]
                init_model = XAJMul()                       # 你前面静态版的类
                Qs_init, WU, WL, WD, S, FR, QI, QG = init_model(
                    x_init,
                    par_init,
                    mu,
                    muwts=muwts,
                    rtwts=rtwts,
                    bufftime=0,
                    outstate=True,
                    routOpt=False,
                    comprout=False,
                    corrwts=corrwts,
                    pcorr=pcorr,
                )
        else:
            # 1.2 无预热：完全仿 HBVMul 的初始化方式一
            Ngrid = x.shape[1]
            device = x.device if hasattr(x, "device") else torch.device("cpu")
            # 各个水库都给一个 1e-3 的小正数，既表示“几乎没水”，又能避免后面除以 0
            WU = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 上层张力水
            WL = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 中层张力水
            WD = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 深层张力水
            S  = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 自由水库水深
            QI = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 壤中线性库
            QG = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 地下线性库
            FR = (torch.zeros([Ngrid, mu], dtype=torch.float32, device=device) + 0.001)  # 上一时段产流面积

        # ============================================================
        # 2. 切掉预热段的强迫，拆成 P / PET
        # ============================================================
        P   = x[bufftime:, :, 0]   # [T', B]
        PET = x[bufftime:, :, 2]   # [T', B]
        Nstep, Ngrid = P.shape
        device = P.device

        # ============================================================
        # 3. 降水订正（可选）：P ← P * α(basin)
        #    典型用例：模型外降水存在系统偏差，需要乘个系数拉回正确量级
        # ============================================================
        if pcorr is not None:
            # parPCORR: [B]，每个格点有一个修正系数
            parPCORR = pcorr[0] + corrwts[:, 0] * (pcorr[1] - pcorr[0])
            # 对全时段降水缩放
            P = parPCORR.unsqueeze(0).repeat(Nstep, 1) * P

        # 把 P/PET 扩展到 mu 维，这样后面所有物理过程都能“对 mu 组参数一次性并行”
        Pm   = P.unsqueeze(-1).repeat(1, 1, mu)    # [T', B, mu]
        PETm = PET.unsqueeze(-1).repeat(1, 1, mu)  # [T', B, mu]

        # ============================================================
        # 4. 动态参数的“静态底 + 动态覆盖 + dropout”机制
        # ============================================================
        # 4.1 这 12 个参数的物理范围要和静态版 XAJ 一致
        parascaLst = [
            [0.3, 2.0],     # 0 ke   蒸散折算系数
            [0.0, 5.0],     # 1 b    张力水容量曲线指数
            [0.01, 100.0],  # 2 wum  上层张力水容量
            [0.01, 100.0],  # 3 wlm  中层张力水容量
            [0.01, 200.0],  # 4 wm   总张力水容量
            [0.09, 1.0],    # 5 c    中/深层蒸散折减系数
            [0.01, 100.0],  # 6 sm   自由水库最大水深
            [0.0, 10.0],    # 7 ex   自由水库非线性指数
            [0.01, 0.7],    # 8 ki   自由水库→壤中比例
            [0.01, 0.7],    # 9 kg   自由水库→地下比例
            [0.0, 0.998],   # 10 ci  壤中线性退水系数
            [0.0, 0.998],   # 11 cg  地下线性退水系数
        ]

        # 4.2 把“正式期”的参数拿出来（已经去掉 bufftime）
        #     parAll: [T', B, 12, mu]
        parAll = parameters[bufftime:, :, :, :]

        # 4.3 先把所有动态参数从 0~1 的无量纲映射回物理范围
        parAllTrans = torch.zeros_like(parAll)
        for ip, (pmin, pmax) in enumerate(parascaLst):
            parAllTrans[:, :, ip, :] = pmin + parAll[:, :, ip, :] * (pmax - pmin)

        # 4.4 再做一份“静态版”，就是把 staind 那一帧的参数复制到所有时间
        #     parStaFull: [T', B, 12, mu]
        total_steps = parAllTrans.shape[0]
        # ALL/PUB/PUR 模式的动态参数长度不同，因此统一在物理模型内裁剪 staind，
        # 防止外部传入的索引超过当前时间序列范围（例如 TestBuff 对应训练期长度）。
        if total_steps == 0:
            raise ValueError("parAllTrans 时间长度为 0，无法获取静态参数基准帧")
        staind = 0 if staind is None else staind
        staind = max(0, min(staind, total_steps - 1))
        parStaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat(Nstep, 1, 1, 1)

        # 4.5 最终要用的参数先拷贝一份静态的，后面只对 tdlst 里面的参数做动态替换
        parUseFull = parStaFull.clone()

        # 4.6 动态参数 dropout 掩码：
        #     - dydrop = 0 → 全用动态
        #     - dydrop = 0.3 → 有 30% 的时空点强制回退静态
        if dydrop > 0.0:
            # 这里做成 [T', B, 1, 1]，表示同一时刻同一格点所有参数组件共用一个 mask
            drop_mask = torch.bernoulli(
                torch.full((Nstep, Ngrid, 1, 1), fill_value=dydrop, device=device)
            )
        else:
            drop_mask = torch.zeros((Nstep, Ngrid, 1, 1), device=device)

        # 4.7 对 tdlst 指定的参数做“静态+动态”的混合
        #     注意 tdlst 是 1-based，我们要转成 0-based
        for idx1 in tdlst:
            i = idx1 - 1
            dyn_val = parAllTrans[:, :, i, :]               # 动态值
            sta_val = parStaFull[:, :, i, :]                # 静态底
            # drop_mask=1 → 用静态；=0 → 用动态
            mixed = dyn_val * (1 - drop_mask.squeeze(-1)) + sta_val * drop_mask.squeeze(-1)
            parUseFull[:, :, i, :] = mixed

        # ============================================================
        # 5. 通量时间序列缓存（和 HBVMul/XAJMul 一样）
        #    都是 [T', B, mu]，这样聚合时直接 mean(-1) 或乘 muwts 即可
        # ============================================================
        QS_inflow = torch.zeros([Nstep, Ngrid, mu], dtype=torch.float32, device=device)  # 坡面/快流
        QI_inflow = torch.zeros_like(QS_inflow)                                          # 壤中流
        QG_inflow = torch.zeros_like(QS_inflow)                                          # 地下流
        ET_series = torch.zeros_like(QS_inflow)                                          # 实际蒸散
        FR0 = FR.clone()                                                                 # 上一时段产流面积

        # ============================================================
        # 6. 主时间循环：核心水文过程
        #    和静态版完全一样，只是每一步的参数换成了 parUseFull[t]
        # ============================================================
        for t in range(Nstep):
            # 6.1 这一时刻的 12 个物理参数取出来
            par_t = parUseFull[t]  # [B, 12, mu]
            (parKE, parB, parWUM, parWLM, parWM,
             parC, parSM, parEX, parKI_, parKG_, parCI, parCG) = [
                par_t[:, i, :] for i in range(par_t.shape[1])
            ]

            # 因为 WM、WUM、WLM 可能是动态的，所以深层容量也要每时刻重算
            wdm = torch.clamp(parWM - parWUM - parWLM, min=0.0)

            # 动态参数下也必须保证 KI + KG < 1，否则自由水会当期被抽空
            sum_k = parKI_ + parKG_
            parKI = torch.where(sum_k < 1.0,
                                parKI_,
                                (1 - PRECS) * parKI_ / (sum_k + PRECS))
            parKG = torch.where(sum_k < 1.0,
                                parKG_,
                                (1 - PRECS) * parKG_ / (sum_k + PRECS))

            # 6.2 本时刻强迫
            P_t   = Pm[t, :, :]                    # [B, mu]
            PET_t = PETm[t, :, :] * parKE          # 折算后 PET

            # --------------------------------------------------------
            # 6.3 三层蒸散（能量→上层→中层→深层）
            # --------------------------------------------------------
            # 上层：EU = min(PET, WU + P)
            EU = torch.min(PET_t, WU + P_t)

            # 剩余需水
            D = torch.clamp(PET_t - EU, min=0.0)

            # 中层：用“湿润度差”(WL - c*WLM) 做一个平滑的切换
            wet_excess = torch.clamp(WL - parC * parWLM, min=0.0)
            wet_ratio  = wet_excess / (wet_excess + 1.0)  # 落在 (0,1)

            # 湿润公式：按含水占比供水
            EL_wet = D * WL / (parWLM + PRECS)
            # 干旱公式：按 c 限制，不能超过中层现有水
            EL_dry = torch.min(parC * D, WL)

            # 平滑混合后的中层蒸散
            EL = wet_ratio * EL_wet + (1 - wet_ratio) * EL_dry
            EL = torch.clamp(EL, min=0.0)
            EL = torch.min(EL, WL)

            # 深层蒸散：上面没满足的再按 c 扣一点
            D_res = torch.clamp(D - EL, min=0.0)
            ED = torch.min(parC * D_res, WD)

            # 当日实际蒸散
            E = EU + EL + ED
            ET_series[t, :, :] = E

            # --------------------------------------------------------
            # 6.4 蓄满产流（新安江的核心容量曲线）
            # --------------------------------------------------------
            PE = torch.clamp(P_t - E, min=0.0)  # 扣掉蒸散后的有效水
            W  = WU + WL + WD                   # 当前土壤湿度

            # 容量曲线最大等效水深
            WMM = parWM * (1.0 + parB)
            # 幂底；1 - W/WM 代表剩余可蓄比例，必须是正的
            base_W = torch.clamp(1.0 - W / (parWM + PRECS), min=PRECS)
            # A 是“湿度已经贡献出来的等效水深”
            A = WMM * (1.0 - base_W ** (1.0 / (1.0 + parB)))

            # 产流分段判据
            ratio = torch.clamp((PE + A) / (WMM + PRECS), max=1.0)
            R_per = torch.where(
                PE + A < WMM,
                # 非线性段：土壤还没完全蓄满
                PE - (parWM - W) + parWM * (1.0 - ratio) ** (1.0 + parB),
                # 线性段：超量产流
                PE - (parWM - W)
            )
            R_per = torch.clamp(R_per, min=0.0)

            # --------------------------------------------------------
            # 6.5 三层张力水更新（有水就填、没水就不动，全部用 clamp）
            # --------------------------------------------------------
            dW = P_t - E - R_per  # 本时段透水区的净入库量，可正可负

            # 增水回填：按上→中→深
            dpos = torch.clamp(dW, min=0.0)
            fill_u = torch.min(dpos, torch.clamp(parWUM - WU, min=0.0))
            WU = WU + fill_u
            dpos = dpos - fill_u

            fill_l = torch.min(dpos, torch.clamp(parWLM - WL, min=0.0))
            WL = WL + fill_l
            dpos = dpos - fill_l

            fill_d = torch.min(dpos, torch.clamp(wdm - WD, min=0.0))
            WD = WD + fill_d

            # 亏水回吐：也按上→中→深，防止任何一层出现负水量
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
            # 6.6 自由水库调蓄 + 两段式快速径流
            # --------------------------------------------------------
            # 6.6.1 产流面积 FR：有雨就按 R/PE，没有雨就沿用上一时段
            FR = torch.where(PE > 0.0, R_per / (PE + PRECS), FR0)
            FR = torch.clamp(FR, 0.0, 1.0)

            # 6.6.2 面积换算：把上一时段 FR0 下的水深 S 换算到当前 FR
            S_eq = S * FR0 / (FR + PRECS)
            S_eq = torch.clamp(S_eq, max=parSM)

            # 6.6.3 自由水库出流能力 AU（XAJ 的非线性自由水库公式）
            SMM = parSM * (1.0 + parEX)
            ratio_s = torch.clamp(1.0 - S_eq / (parSM + PRECS), min=PRECS)
            AU = SMM * (1.0 - ratio_s ** (1.0 / (1.0 + parEX)))

            # 6.6.4 坡面快流：仍然两段式，且限制 RS ≤ R_per
            ratio_pa = torch.clamp((PE + AU) / (SMM + PRECS), max=1.0)
            RS_part1 = (PE + S_eq - parSM + parSM * (1.0 - ratio_pa) ** (1.0 + parEX)) * FR
            RS_part2 = (PE + S_eq - parSM) * FR
            RS = torch.where(PE + AU < SMM, RS_part1, RS_part2)
            RS = torch.clamp(RS, min=0.0)
            RS = torch.min(RS, R_per)

            # 6.6.5 自由水库回蓄：没出成快速径流的那部分回到自由水库
            S = S_eq + (R_per - RS) / (FR + PRECS)
            S = torch.clamp(S, max=parSM)

            # 6.6.6 更新当前这一步的产流面积，供下一步面积换算
            FR0 = FR

            # --------------------------------------------------------
            # 6.7 三水源分配 + 线性退水
            # --------------------------------------------------------
            # 自由水库中剩下的水，按 KI/KG 分成壤中补给和地下补给
            RI = parKI * S * FR
            RG = parKG * S * FR
            S = S * (1.0 - parKI - parKG)  # 自由水库真正留下的

            # 线性退水：经典的一阶递推
            QI = parCI * QI + (1.0 - parCI) * RI
            QG = parCG * QG + (1.0 - parCG) * RG

            # 记录各分量
            QS_inflow[t, :, :] = RS
            QI_inflow[t, :, :] = QI
            QG_inflow[t, :, :] = QG

        # ============================================================
        # 7. 多组件合成
        #    先把每个分量在 mu 维上做平均/加权，
        #    再看需不需要汇流
        # ============================================================
        QSave = QS_inflow.mean(-1, keepdim=True)
        QIave = QI_inflow.mean(-1, keepdim=True)
        QGave = QG_inflow.mean(-1, keepdim=True)
        ETave = ET_series.mean(-1, keepdim=True)

        river_inflow = QS_inflow + QI_inflow + QG_inflow  # [T', B, mu]

        if muwts is None:
            inflow_mean = river_inflow.mean(-1)  # [T', B]
        else:
            inflow_mean = (river_inflow * muwts).sum(-1)

        # ============================================================
        # 8. 汇流（可选）：和静态版一样走 Gamma 单位线
        # ============================================================
        if routOpt:
            # 看是先合成再汇流，还是每个组件单独汇流
            if comprout:
                Qsim = river_inflow.view(Nstep, Ngrid * mu)
            else:
                Qsim = inflow_mean

            # 路由参数范围固定：按照静态版来的
            tempa = 0.0 + rtwts[:, 0] * (2.9 - 0.0)
            tempb = 0.0 + rtwts[:, 1] * (6.5 - 0.0)

            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)

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
            # 不汇流就直接输出面平均入河量
            Qs = inflow_mean.unsqueeze(-1)

        # ============================================================
        # 9. 输出控制
        # ============================================================
        if outstate:
            # 要末态就全吐出去，方便你分段跑或做 warm start
            return Qs, WU, WL, WD, S, FR, QI, QG
        else:
            # 和静态版一样：出口流量 + 三分量 + 实际蒸散
            Qall = torch.cat((Qs, QSave, QIave, QGave, ETave), dim=-1)
            return Qall


class MultiInv_XAJTDModel(nn.Module):
    """
    深度学习参数反演 + 动态参数 XAJ 模型

    参考 MultiInv_HBVTDModel 的结构：
    - RNN 反演得到整段时间的 XAJ 参数序列：[Time, Basin, (nfea*nmul)]
    - 可选学习组件权重、可选路由参数（按 comprout 决定形状）
    - 可选降水订正参数（1 个），范围由 pcorr 在物理模型中缩放
    - 调用动态物理模型 XAJMulTD 做水文模拟
    """

    def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5,
                 inittime=0, routOpt=False, comprout=False,
                 compwts=False, staind=-1, tdlst=None, dydrop=0.0,
                 pcorr=None, rnn_type='lstm'):
        super(MultiInv_XAJTDModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea            # xaj物理参数为 12
        self.nmul = nmul
        self.hiddeninv = hiddeninv
        self.rnn_type = rnn_type.lower()

        # 参数总数：动态 XAJ 参数时间序列 + 路由 + 组件权重 + 可选降水订正
        nxajpm = nfea * nmul
        if comprout is False:
            nroutpm = 2
        else:
            nroutpm = nmul * 2
        nwtspm = (nmul if compwts else 0)
        ntp = nxajpm + nroutpm + nwtspm + (0 if pcorr is None else 1)

        # RNN 选择
        if self.rnn_type == 'lstm':
            self.lstminv = CudnnLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'gru':
            self.lstminv = CudnnGruModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bilstm':
            self.lstminv = CudnnBiLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'bigru':
            self.lstminv = CudnnBiGruModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'rnn':
            self.lstminv = CudnnRnnModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnlstm':
            self.lstminv = CudnnCnnLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        elif self.rnn_type == 'cnnbilstm':
            self.lstminv = CudnnCnnBiLstmModel(nx=ninv, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
        else:
            raise ValueError("rnn_type must be one of: 'lstm', 'gru', 'bilstm', 'bigru', 'rnn', 'cnnlstm', 'cnnbilstm'")

        # 物理模型
        self.XAJ = XAJMulTD()

        # 运行属性
        self.inittime = inittime
        self.routOpt = routOpt
        self.comprout = comprout
        self.nxajpm = nxajpm
        self.nwtspm = nwtspm
        self.nroutpm = nroutpm
        self.staind = staind if staind is not None else -1
        self.tdlst = tdlst or []
        self.dydrop = dydrop
        self.pcorr = pcorr

    def forward(self, x, z, doDropMC=False):
        # 反演参数时间序列
        Params0 = self.lstminv(z)              # [Time, Basin, ntp]
        ntstep = Params0.shape[0]
        ngage = Params0.shape[1]

        # 动态 XAJ 参数（整段时间）
        xajpara0 = Params0[:, :, 0:self.nxajpm]
        xajpara = torch.sigmoid(xajpara0).view(ntstep, ngage, self.nfea, self.nmul)

        # 路由参数（取最后一时刻）
        routpara0 = Params0[-1, :, self.nxajpm:self.nxajpm + self.nroutpm]
        if self.comprout is False:
            routpara = torch.sigmoid(routpara0)                 # [B, 2]
        else:
            routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2)  # [B*mu, 2]

        # 组件权重（可选，取最后一时刻）
        if self.nwtspm == 0:
            wts = None
        else:
            wtspara = Params0[-1, :, self.nxajpm + self.nroutpm:self.nxajpm + self.nroutpm + self.nwtspm]
            wts = torch.softmax(wtspara, dim=-1)

        # 降水订正（可选，取最后一时刻）
        if self.pcorr is None:
            corrpara = None
        else:
            start = self.nxajpm + self.nroutpm + self.nwtspm
            corrpara0 = Params0[-1, :, start:start + 1]
            corrpara = torch.sigmoid(corrpara0)

        # 物理模拟
        out = self.XAJ(
            x,
            parameters=xajpara,
            staind=self.staind,
            tdlst=self.tdlst,
            mu=self.nmul,
            muwts=wts,
            rtwts=routpara,
            bufftime=self.inittime,
            outstate=False,
            routOpt=self.routOpt,
            comprout=self.comprout,
            corrwts=corrpara,
            pcorr=self.pcorr,
            dydrop=self.dydrop,
        )
        return out
