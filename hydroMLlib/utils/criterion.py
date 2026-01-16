"""
损失函数工具模块

本模块提供水文模型训练中使用的各种损失函数，包括组合损失函数和
基于Nash-Sutcliffe效率(NSE)的损失函数。主要用于dPLHBV模型的训练优化。

主要功能：
- 组合损失函数：结合RMSE和对数平方根RMSE的混合损失
- NSE损失函数：基于Nash-Sutcliffe效率的批量损失计算
- 平方根NSE损失：使用RMSE和标准差的NSE变体
- 支持多变量输出和缺失值处理
"""

import torch
import numpy as np


class RmseLossComb(torch.nn.Module):
    """
    组合RMSE损失函数类
    
    结合标准RMSE和对数平方根RMSE的混合损失函数，用于水文模型训练。
    该损失函数能够同时优化高流量和低流量的预测精度，特别适用于径流预测。
    
    数学公式:
        Loss = (1-α) * RMSE + α * Log-Sqrt-RMSE
        其中:
        - RMSE = sqrt(mean((pred - true)²))
        - Log-Sqrt-RMSE = sqrt(mean((log10(sqrt(pred+β)+0.1) - log10(sqrt(true+β)+0.1))²))
    
    参数:
        alpha (float): 对数平方根RMSE的权重系数
                      - 范围: [0, 1]
                      - 0: 仅使用标准RMSE
                      - 1: 仅使用对数平方根RMSE
                      - 0.5: 两种损失等权重
        beta (float): 数值稳定性参数，防止log(0)错误
                      - 默认值: 1e-6
                      - 用于处理接近零的预测值
    
    输入维度:
        output: [batch_size, seq_len, n_variables] - 模型预测值
        target: [batch_size, seq_len, n_variables] - 真实值
    
    特性:
        - 支持多变量输出
        - 自动处理NaN值（缺失值）
        - 数值稳定性优化
        - 适用于径流预测任务
    """
    
    def __init__(self, alpha, beta=1e-6):
        """
        初始化组合RMSE损失函数
        
        参数:
            alpha (float): 对数平方根RMSE的权重系数
            beta (float): 数值稳定性参数，默认1e-6
        """
        super(RmseLossComb, self).__init__()
        self.alpha = alpha  # 对数平方根RMSE的权重
        self.beta = beta    # 数值稳定性参数

    def forward(self, output, target):
        """
        前向传播计算组合损失
        
        参数:
            output (torch.Tensor): 模型预测值 [batch_size, seq_len, n_variables]
            target (torch.Tensor): 真实值 [batch_size, seq_len, n_variables]
        
        返回:
            torch.Tensor: 标量损失值
        
        计算过程:
            1. 对每个变量分别计算损失
            2. 计算标准RMSE损失
            3. 计算对数平方根RMSE损失
            4. 按权重组合两种损失
            5. 对所有变量损失求和
        """
        ny = target.shape[2]  # 获取变量数量
        loss = 0
        
        # 对每个输出变量分别计算损失
        for k in range(ny):
            # 获取第k个变量的预测值和真实值
            p0 = output[:, :, k]  # 预测值
            t0 = target[:, :, k]  # 真实值
            
            # 计算对数平方根变换
            # 添加beta和0.1确保数值稳定性
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            
            # 创建有效值掩码（排除NaN值）
            mask = t0 == t0  # NaN != NaN 为True，有效值 == 有效值 为True
            p = p0[mask]     # 提取有效预测值
            t = t0[mask]     # 提取有效真实值
            
            # 计算标准RMSE损失
            loss1 = torch.sqrt(((p - t)**2).mean())
            
            # 为对数平方根损失创建掩码
            mask1 = t1 == t1
            pa = p1[mask1]   # 提取有效对数平方根预测值
            ta = t1[mask1]   # 提取有效对数平方根真实值
            
            # 计算对数平方根RMSE损失
            loss2 = torch.sqrt(((pa - ta)**2).mean())
            
            # 按权重组合两种损失
            temp = (1.0 - self.alpha) * loss1 + self.alpha * loss2
            
            # 累加到总损失
            loss = loss + temp
            
        return loss


class NSELossBatch(torch.nn.Module):
    """
    批量NSE损失函数类
    
    基于Nash-Sutcliffe效率(NSE)的批量损失函数，参考Fredrick 2019年的实现。
    该损失函数使用流域径流的标准差进行归一化，适用于多流域批量训练。
    
    数学公式:
        Loss = mean((pred - true)² / (std + ε)²)
        其中:
        - std: 各流域径流的标准差
        - ε: 数值稳定性参数
    
    参数:
        stdarray (numpy.ndarray): 所有流域径流的标准差数组
                                 - 形状: [n_basins]
                                 - 用于归一化不同流域的损失
        eps (float): 数值稳定性参数，防止除零错误
                    - 默认值: 0.1
                    - 添加到标准差中避免除零
    
    输入维度:
        output: [batch_size, n_basins, 1] - 模型预测值
        target: [batch_size, n_basins, 1] - 真实值
        igrid: [batch_size] - 流域索引数组
    
    特性:
        - 支持批量多流域训练
        - 自动归一化不同流域的损失
        - 处理缺失值
        - 适用于径流预测任务
    """
    
    def __init__(self, stdarray, eps=0.1):
        """
        初始化批量NSE损失函数
        
        参数:
            stdarray (numpy.ndarray): 流域径流标准差数组
            eps (float): 数值稳定性参数，默认0.1
        """
        super(NSELossBatch, self).__init__()
        self.std = stdarray  # 流域径流标准差
        self.eps = eps       # 数值稳定性参数

    def forward(self, output, target, igrid):
        """
        前向传播计算批量NSE损失
        
        参数:
            output (torch.Tensor): 模型预测值 [batch_size, n_basins, 1]
            target (torch.Tensor): 真实值 [batch_size, n_basins, 1]
            igrid (torch.Tensor): 流域索引 [batch_size]
        
        返回:
            torch.Tensor: 标量损失值
        
        计算过程:
            1. 根据流域索引获取对应的标准差
            2. 计算预测误差的平方
            3. 用标准差归一化误差
            4. 计算平均损失
        """
        nt = target.shape[0]  # 获取时间步数
        
        # 根据流域索引复制标准差到对应时间步
        stdse = np.tile(self.std[igrid], (nt, 1))
        
        # 将标准差转换为PyTorch张量并移到正确设备
        device = output.device
        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(device)
        
        # 提取预测值和真实值（假设只有一个输出变量）
        p0 = output[:, :, 0]  # 预测值 [时间, 流域]
        t0 = target[:, :, 0] # 真实值 [时间, 流域]
        
        # 创建有效值掩码（排除NaN值）
        mask = t0 == t0
        p = p0[mask]         # 有效预测值
        t = t0[mask]         # 有效真实值
        stdw = stdbatch[mask] # 对应的标准差
        
        # 计算平方误差
        sqRes = (p - t)**2
        
        # 用标准差归一化误差（添加eps防止除零）
        normRes = sqRes / (stdw + self.eps)**2
        
        # 计算平均损失
        loss = torch.mean(normRes)
        
        return loss


class NSESqrtLossBatch(torch.nn.Module):
    """
    平方根NSE损失函数类
    
    基于Nash-Sutcliffe效率的变体，使用RMSE和标准差进行归一化。
    参考Fredrick 2019年的实现，适用于多流域批量训练。
    
    数学公式:
        Loss = mean(sqrt((pred - true)²) / (std + ε))
        其中:
        - sqrt((pred - true)²): 绝对误差
        - std: 各流域径流的标准差
        - ε: 数值稳定性参数
    
    参数:
        stdarray (numpy.ndarray): 所有流域径流的标准差数组
                                 - 形状: [n_basins]
                                 - 用于归一化不同流域的损失
        eps (float): 数值稳定性参数，防止除零错误
                    - 默认值: 0.1
                    - 添加到标准差中避免除零
    
    输入维度:
        output: [batch_size, n_basins, 1] - 模型预测值
        target: [batch_size, n_basins, 1] - 真实值
        igrid: [batch_size] - 流域索引数组
    
    特性:
        - 使用绝对误差而非平方误差
        - 支持批量多流域训练
        - 自动归一化不同流域的损失
        - 处理缺失值
        - 对异常值更鲁棒
    """
    
    def __init__(self, stdarray, eps=0.1):
        """
        初始化平方根NSE损失函数
        
        参数:
            stdarray (numpy.ndarray): 流域径流标准差数组
            eps (float): 数值稳定性参数，默认0.1
        """
        super(NSESqrtLossBatch, self).__init__()
        self.std = stdarray  # 流域径流标准差
        self.eps = eps       # 数值稳定性参数

    def forward(self, output, target, igrid):
        """
        前向传播计算平方根NSE损失
        
        参数:
            output (torch.Tensor): 模型预测值 [batch_size, n_basins, 1]
            target (torch.Tensor): 真实值 [batch_size, n_basins, 1]
            igrid (torch.Tensor): 流域索引 [batch_size]
        
        返回:
            torch.Tensor: 标量损失值
        
        计算过程:
            1. 根据流域索引获取对应的标准差
            2. 计算预测误差的绝对值
            3. 用标准差归一化绝对误差
            4. 计算平均损失
        """
        nt = target.shape[0]  # 获取时间步数
        
        # 根据流域索引复制标准差到对应时间步
        stdse = np.tile(self.std[igrid], (nt, 1))
        
        # 将标准差转换为PyTorch张量并移到正确设备
        device = output.device
        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(device)
        
        # 提取预测值和真实值（假设只有一个输出变量）
        p0 = output[:, :, 0]  # 预测值 [时间, 流域]
        t0 = target[:, :, 0] # 真实值 [时间, 流域]
        
        # 创建有效值掩码（排除NaN值）
        mask = t0 == t0
        p = p0[mask]         # 有效预测值
        t = t0[mask]         # 有效真实值
        stdw = stdbatch[mask] # 对应的标准差
        
        # 计算绝对误差（平方根）
        sqRes = torch.sqrt((p - t)**2)
        
        # 用标准差归一化绝对误差（添加eps防止除零）
        normRes = sqRes / (stdw + self.eps)
        
        # 计算平均损失
        loss = torch.mean(normRes)
        
        return loss
