# -*- coding: utf-8 -*-
"""
RNN模型和水文模型实现模块 (RNN Models and Hydrological Model Implementation Module)

本模块包含用于水文深度学习的各种RNN模型和HBV水文模型的PyTorch实现。
主要功能包括：

1. RNN模型类：
   - CudnnLstmModel: LSTM模型
   - CudnnGruModel: GRU模型  
   - CudnnBiLstmModel: 双向LSTM模型
   - CudnnBiGruModel: 双向GRU模型
   - CudnnRnnModel: 基础RNN模型
   - CudnnCnnLstmModel: CNN-LSTM混合模型
   - CudnnCnnBiLstmModel: CNN-BiLSTM混合模型

2. HBV水文模型类：
   - HBVMul: 多组件静态参数HBV模型
   - HBVMulET: 带ET形状参数的多组件HBV模型
   - HBVMulTD: 多组件动态参数HBV模型
   - HBVMulTDET: 带ET形状参数的多组件动态HBV模型

3. 深度学习+水文模型组合类：
   - MultiInv_HBVModel: 深度学习参数反演+静态HBV模型
   - MultiInv_HBVTDModel: 深度学习参数反演+动态HBV模型

4. 辅助函数：
   - UH_conv: 单位线卷积函数
   - UH_gamma: Gamma分布单位线生成函数

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# 抑制所有警告信息，保持输出清洁
warnings.filterwarnings("ignore")



class CudnnLstmModel(torch.nn.Module):
    """
    LSTM模型类 (LSTM Model Class)
    
    基于PyTorch实现的LSTM模型，用于时间序列预测和参数反演。
    该模型使用LSTM作为核心网络结构，适用于水文时间序列建模。
    
    参数 (Parameters):
    ----------
    nx : int
        输入特征维度，即输入数据的特征数量
    ny : int  
        输出特征维度，即输出数据的特征数量
    hiddenSize : int
        LSTM隐藏层大小，控制模型的复杂度和表达能力
    dr : float, default=0.5
        Dropout比率，用于防止过拟合，范围[0,1]
        
    网络结构 (Network Architecture):
    ----------
    1. 线性输入层: nx -> hiddenSize
    2. ReLU激活函数
    3. LSTM层: hiddenSize -> hiddenSize  
    4. 线性输出层: hiddenSize -> ny
    
    用途 (Usage):
    ----------
    主要用于HBV模型参数的深度学习反演，将静态属性映射为水文模型参数。
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx                    # 输入特征维度
        self.ny = ny                    # 输出特征维度
        self.hiddenSize = hiddenSize    # LSTM隐藏层大小
        self.ct = 0                     # 计数器（保留用于兼容性）
        self.nLayer = 1                # LSTM层数（当前固定为1层）
        
        # 线性输入层：将输入特征映射到隐藏层维度
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # LSTM层：使用PyTorch内置的LSTM实现
        # batch_first=False: 输入格式为(seq_len, batch, input_size)
        # dropout仅在多层LSTM时生效
        self.lstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # 线性输出层：将隐藏层输出映射到目标维度
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU标志（保留用于兼容性）

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        前向传播函数 (Forward Pass Function)
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            输入张量，形状为(seq_len, batch_size, nx)
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
        dropoutFalse : bool, default=False  
            是否禁用Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            输出张量，形状为(seq_len, batch_size, ny)
        """
        # 输入层：线性变换 + ReLU激活
        x0 = F.relu(self.linearIn(x))
        
        # LSTM层：处理时间序列信息
        # outLSTM: LSTM输出，形状为(seq_len, batch_size, hidden_size)
        # hn, cn: 最终隐藏状态和细胞状态
        outLSTM, (hn, cn) = self.lstm(x0)
        
        # 输出层：线性变换得到最终结果
        out = self.linearOut(outLSTM)
        return out
        
class CudnnGruModel(torch.nn.Module):
    """
    GRU模型类 (GRU Model Class)
    
    基于PyTorch实现的GRU模型，用于时间序列预测和参数反演。
    GRU是LSTM的简化版本，具有更少的参数但通常能达到相似的性能。
    
    参数 (Parameters):
    ----------
    nx : int
        输入特征维度，即输入数据的特征数量
    ny : int  
        输出特征维度，即输出数据的特征数量
    hiddenSize : int
        GRU隐藏层大小，控制模型的复杂度和表达能力
    dr : float, default=0.5
        Dropout比率，用于防止过拟合，范围[0,1]
        
    网络结构 (Network Architecture):
    ----------
    1. 线性输入层: nx -> hiddenSize
    2. ReLU激活函数
    3. GRU层: hiddenSize -> hiddenSize  
    4. 线性输出层: hiddenSize -> ny
    
    用途 (Usage):
    ----------
    主要用于HBV模型参数的深度学习反演，相比LSTM具有更快的训练速度。
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnGruModel, self).__init__()
        self.nx = nx                    # 输入特征维度
        self.ny = ny                    # 输出特征维度
        self.hiddenSize = hiddenSize    # GRU隐藏层大小
        self.ct = 0                     # 计数器（保留用于兼容性）
        self.nLayer = 1                # GRU层数（当前固定为1层）
        
        # 线性输入层：将输入特征映射到隐藏层维度
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # GRU层：使用PyTorch内置的GRU实现
        # batch_first=False: 输入格式为(seq_len, batch, input_size)
        # dropout仅在多层GRU时生效
        self.gru = nn.GRU(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # 线性输出层：将隐藏层输出映射到目标维度
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU标志（保留用于兼容性）
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        前向传播函数 (Forward Pass Function)
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            输入张量，形状为(seq_len, batch_size, nx)
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
        dropoutFalse : bool, default=False  
            是否禁用Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            输出张量，形状为(seq_len, batch_size, ny)
        """
        # 输入层：线性变换 + ReLU激活
        x0 = F.relu(self.linearIn(x))
        
        # GRU层：处理时间序列信息
        # outGRU: GRU输出，形状为(seq_len, batch_size, hidden_size)
        # hn: 最终隐藏状态（GRU没有细胞状态）
        outGRU, hn = self.gru(x0)
        
        # 输出层：线性变换得到最终结果
        out = self.linearOut(outGRU)
        return out
        
class CudnnBiLstmModel(torch.nn.Module):
    """
    双向LSTM模型类 (Bidirectional LSTM Model Class)
    
    基于PyTorch实现的双向LSTM模型，能够同时利用过去和未来的信息。
    双向LSTM通过前向和后向两个LSTM层处理序列，能够捕获更丰富的时序特征。
    
    参数 (Parameters):
    ----------
    nx : int
        输入特征维度，即输入数据的特征数量
    ny : int  
        输出特征维度，即输出数据的特征数量
    hiddenSize : int
        单向LSTM隐藏层大小，双向LSTM总输出维度为hiddenSize*2
    dr : float, default=0.5
        Dropout比率，用于防止过拟合，范围[0,1]
        
    网络结构 (Network Architecture):
    ----------
    1. 线性输入层: nx -> hiddenSize
    2. ReLU激活函数
    3. 双向LSTM层: hiddenSize -> hiddenSize*2  
    4. 线性输出层: hiddenSize*2 -> ny
    
    用途 (Usage):
    ----------
    主要用于HBV模型参数的深度学习反演，特别适合需要利用完整时序信息的场景。
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnBiLstmModel, self).__init__()
        self.nx = nx                    # 输入特征维度
        self.ny = ny                    # 输出特征维度
        self.hiddenSize = hiddenSize    # 单向LSTM隐藏层大小
        self.ct = 0                     # 计数器（保留用于兼容性）
        self.nLayer = 1                # LSTM层数（当前固定为1层）
        
        # 线性输入层：将输入特征映射到隐藏层维度
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # 双向LSTM层：使用PyTorch内置的双向LSTM实现
        # bidirectional=True: 启用双向处理
        # 输出维度为hiddenSize*2（前向+后向）
        self.bilstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            bidirectional=True,
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # 线性输出层：处理双向LSTM的输出（维度为hiddenSize*2）
        self.linearOut = torch.nn.Linear(hiddenSize * 2, ny)
        self.gpu = 1                    # GPU标志（保留用于兼容性）
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        前向传播函数 (Forward Pass Function)
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            输入张量，形状为(seq_len, batch_size, nx)
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
        dropoutFalse : bool, default=False  
            是否禁用Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            输出张量，形状为(seq_len, batch_size, ny)
        """
        # 输入层：线性变换 + ReLU激活
        x0 = F.relu(self.linearIn(x))
        
        # 双向LSTM层：处理时间序列信息
        # outBiLSTM: 双向LSTM输出，形状为(seq_len, batch_size, hidden_size*2)
        # hn, cn: 最终隐藏状态和细胞状态（包含前向和后向）
        outBiLSTM, (hn, cn) = self.bilstm(x0)
        
        # 输出层：线性变换得到最终结果
        out = self.linearOut(outBiLSTM)
        return out
        
class CudnnBiGruModel(torch.nn.Module):
    """
    双向GRU模型类 (Bidirectional GRU Model Class)
    
    基于PyTorch实现的双向GRU模型，能够同时利用过去和未来的信息。
    双向GRU通过前向和后向两个GRU层处理序列，能够捕获更丰富的时序特征。
    GRU是LSTM的简化版本，具有更少的参数但通常能达到相似的性能。
    
    参数 (Parameters):
    ----------
    nx : int
        输入特征维度，即输入数据的特征数量
    ny : int  
        输出特征维度，即输出数据的特征数量
    hiddenSize : int
        单向GRU隐藏层大小，双向GRU总输出维度为hiddenSize*2
    dr : float, default=0.5
        Dropout比率，用于防止过拟合，范围[0,1]
        
    网络结构 (Network Architecture):
    ----------
    1. 线性输入层: nx -> hiddenSize
    2. ReLU激活函数
    3. 双向GRU层: hiddenSize -> hiddenSize*2  
    4. 线性输出层: hiddenSize*2 -> ny
    
    用途 (Usage):
    ----------
    主要用于HBV模型参数的深度学习反演，特别适合需要利用完整时序信息的场景。
    相比双向LSTM具有更快的训练速度和更少的参数。
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnBiGruModel, self).__init__()
        self.nx = nx                    # 输入特征维度
        self.ny = ny                    # 输出特征维度
        self.hiddenSize = hiddenSize    # 单向GRU隐藏层大小
        self.ct = 0                     # 计数器（保留用于兼容性）
        self.nLayer = 1                # GRU层数（当前固定为1层）
        
        # 线性输入层：将输入特征映射到隐藏层维度
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # 双向GRU层：使用PyTorch内置的双向GRU实现
        # bidirectional=True: 启用双向处理
        # 输出维度为hiddenSize*2（前向+后向）
        self.bigru = nn.GRU(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            bidirectional=True,
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # 线性输出层：处理双向GRU的输出（维度为hiddenSize*2）
        self.linearOut = torch.nn.Linear(hiddenSize * 2, ny)
        self.gpu = 1                    # GPU标志（保留用于兼容性）
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        前向传播函数 (Forward Pass Function)
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            输入张量，形状为(seq_len, batch_size, nx)
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
        dropoutFalse : bool, default=False  
            是否禁用Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            输出张量，形状为(seq_len, batch_size, ny)
        """
        # 输入层：线性变换 + ReLU激活
        x0 = F.relu(self.linearIn(x))
        
        # 双向GRU层：处理时间序列信息
        # outBiGRU: 双向GRU输出，形状为(seq_len, batch_size, hidden_size*2)
        # hn: 最终隐藏状态（包含前向和后向）
        outBiGRU, hn = self.bigru(x0)
        
        # 输出层：线性变换得到最终结果
        out = self.linearOut(outBiGRU)
        return out
        
class CudnnRnnModel(torch.nn.Module):
    """
    基础RNN模型类 (Basic RNN Model Class)
    
    基于PyTorch实现的基础RNN模型，是最简单的时间序列神经网络。
    RNN通过隐藏状态在时间步之间传递信息，能够处理序列数据。
    相比LSTM和GRU，RNN结构更简单但容易出现梯度消失问题。
    
    参数 (Parameters):
    ----------
    nx : int
        输入特征维度，即输入数据的特征数量
    ny : int  
        输出特征维度，即输出数据的特征数量
    hiddenSize : int
        RNN隐藏层大小，控制模型的复杂度和表达能力
    dr : float, default=0.5
        Dropout比率，用于防止过拟合，范围[0,1]
        
    网络结构 (Network Architecture):
    ----------
    1. 线性输入层: nx -> hiddenSize
    2. ReLU激活函数
    3. RNN层: hiddenSize -> hiddenSize  
    4. 线性输出层: hiddenSize -> ny
    
    用途 (Usage):
    ----------
    主要用于HBV模型参数的深度学习反演，适用于简单的时序建模任务。
    相比LSTM和GRU具有最少的参数，但表达能力相对较弱。
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnRnnModel, self).__init__()
        self.nx = nx                    # 输入特征维度
        self.ny = ny                    # 输出特征维度
        self.hiddenSize = hiddenSize    # RNN隐藏层大小
        self.ct = 0                     # 计数器（保留用于兼容性）
        self.nLayer = 1                # RNN层数（当前固定为1层）
        
        # 线性输入层：将输入特征映射到隐藏层维度
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # RNN层：使用PyTorch内置的RNN实现
        # batch_first=False: 输入格式为(seq_len, batch, input_size)
        # nonlinearity='tanh': 使用tanh激活函数（可选'tanh'或'relu'）
        # dropout仅在多层RNN时生效
        self.rnn = nn.RNN(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0,
            nonlinearity='tanh'  # 'tanh' or 'relu'
        )
        
        # 线性输出层：将隐藏层输出映射到目标维度
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU标志（保留用于兼容性）
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        前向传播函数 (Forward Pass Function)
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            输入张量，形状为(seq_len, batch_size, nx)
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
        dropoutFalse : bool, default=False  
            是否禁用Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            输出张量，形状为(seq_len, batch_size, ny)
        """
        # 输入层：线性变换 + ReLU激活
        x0 = F.relu(self.linearIn(x))
        
        # RNN层：处理时间序列信息
        # outRNN: RNN输出，形状为(seq_len, batch_size, hidden_size)
        # hn: 最终隐藏状态
        outRNN, hn = self.rnn(x0)
        
        # 输出层：线性变换得到最终结果
        out = self.linearOut(outRNN)
        return out


class CudnnCnnLstmModel(torch.nn.Module):
    """
    CNN-LSTM模型类 (CNN-LSTM Model Class)
    
    基于PyTorch实现的CNN-LSTM混合模型，结合了卷积神经网络和LSTM的优势。
    该模型先用CNN提取空间/特征维度的局部模式，然后用LSTM处理时间序列信息。
    CNN-LSTM模型能够同时捕获特征间的局部相关性和时间序列的长期依赖关系。
    
    参数 (Parameters):
    ----------
    nx : int
        输入特征维度，即输入数据的特征数量
    ny : int  
        输出特征维度，即输出数据的特征数量
    hiddenSize : int
        LSTM隐藏层大小，控制模型的复杂度和表达能力
    dr : float, default=0.5
        Dropout比率，用于防止过拟合，范围[0,1]
    cnn_out_channels : int, default=None
        CNN输出通道数，如果为None则使用hiddenSize
    kernel_size : int, default=3
        CNN卷积核大小，控制感受野范围
        
    网络结构 (Network Architecture):
    ----------
    1. CNN特征提取层：
       - 1D卷积层: 在特征维度上进行卷积，提取局部特征模式
       - 批归一化: 加速训练并提高稳定性
       - ReLU激活函数
       - Dropout层
    2. 线性投影层: CNN输出 -> hiddenSize
    3. LSTM层: hiddenSize -> hiddenSize  
    4. 线性输出层: hiddenSize -> ny
    
    用途 (Usage):
    ----------
    主要用于XAJ模型参数的深度学习反演，特别适合需要同时考虑特征间关系和时间依赖的场景。
    相比纯LSTM模型，CNN-LSTM能够更好地捕获静态属性之间的局部相关性。
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, cnn_out_channels=None, kernel_size=3):
        super(CudnnCnnLstmModel, self).__init__()
        self.nx = nx                    # 输入特征维度
        self.ny = ny                    # 输出特征维度
        self.hiddenSize = hiddenSize    # LSTM隐藏层大小
        self.ct = 0                     # 计数器（保留用于兼容性）
        self.nLayer = 1                # LSTM层数（当前固定为1层）
        
        # CNN参数设置
        self.cnn_out_channels = cnn_out_channels if cnn_out_channels is not None else hiddenSize
        self.kernel_size = kernel_size
        
        # CNN特征提取层：在特征维度上进行1D卷积
        # 输入形状: (batch, channels=1, features=nx)
        # 输出形状: (batch, channels=cnn_out_channels, features=nx)
        # 使用padding='same'保持特征维度不变
        self.conv1d = nn.Conv1d(
            in_channels=1,              # 输入通道数（将每个时间步的特征视为单通道）
            out_channels=self.cnn_out_channels,  # 输出通道数
            kernel_size=kernel_size,    # 卷积核大小
            padding=(kernel_size - 1) // 2,  # 保持特征维度不变
            padding_mode='zeros'
        )
        
        # 批归一化：加速训练并提高稳定性
        self.bn = nn.BatchNorm1d(self.cnn_out_channels)
        
        # Dropout层：防止过拟合
        self.dropout_cnn = nn.Dropout(dr)
        
        # 线性投影层：将CNN输出投影到LSTM输入维度
        # CNN输出维度为 cnn_out_channels * nx，需要投影到 hiddenSize
        self.linear_proj = torch.nn.Linear(self.cnn_out_channels, hiddenSize)
        
        # LSTM层：处理时间序列信息
        # batch_first=False: 输入格式为(seq_len, batch, input_size)
        # dropout仅在多层LSTM时生效
        self.lstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # 线性输出层：将隐藏层输出映射到目标维度
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU标志（保留用于兼容性）

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        前向传播函数 (Forward Pass Function)
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            输入张量，形状为(seq_len, batch_size, nx)
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
        dropoutFalse : bool, default=False  
            是否禁用Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            输出张量，形状为(seq_len, batch_size, ny)
        """
        seq_len, batch_size, nx = x.shape
        
        # 步骤1: CNN特征提取
        # 将输入重塑为 (batch_size * seq_len, 1, nx) 以便进行1D卷积
        x_reshaped = x.reshape(batch_size * seq_len, 1, nx)
        
        # 1D卷积：在特征维度上提取局部模式
        x_conv = self.conv1d(x_reshaped)  # (batch*seq, cnn_out_channels, nx)
        
        # 批归一化
        x_conv = self.bn(x_conv)
        
        # ReLU激活
        x_conv = F.relu(x_conv)
        
        # Dropout
        if not dropoutFalse:
            x_conv = self.dropout_cnn(x_conv)
        
        # 将CNN输出重塑回 (seq_len, batch_size, cnn_out_channels)
        # 对特征维度取平均，得到每个时间步的CNN特征表示
        x_conv = x_conv.reshape(batch_size * seq_len, self.cnn_out_channels, nx)
        x_conv = x_conv.mean(dim=2)  # 平均池化: (batch*seq, cnn_out_channels)
        x_conv = x_conv.reshape(seq_len, batch_size, self.cnn_out_channels)
        
        # 步骤2: 线性投影到LSTM输入维度
        x_proj = F.relu(self.linear_proj(x_conv))  # (seq_len, batch_size, hiddenSize)
        
        # 步骤3: LSTM层处理时间序列
        # outLSTM: LSTM输出，形状为(seq_len, batch_size, hidden_size)
        # hn, cn: 最终隐藏状态和细胞状态
        outLSTM, (hn, cn) = self.lstm(x_proj)
        
        # 步骤4: 输出层得到最终结果
        out = self.linearOut(outLSTM)
        return out


class CudnnCnnBiLstmModel(torch.nn.Module):
    """
    CNN-BiLSTM模型类 (CNN-BiLSTM Model Class)
    
    基于PyTorch实现的CNN-BiLSTM混合模型，结合了卷积神经网络和双向LSTM的优势。
    该模型先用CNN提取空间/特征维度的局部模式，然后用双向LSTM处理时间序列信息。
    CNN-BiLSTM模型能够同时捕获特征间的局部相关性、时间序列的长期依赖关系以及双向时序信息。
    
    参数 (Parameters):
    ----------
    nx : int
        输入特征维度，即输入数据的特征数量
    ny : int  
        输出特征维度，即输出数据的特征数量
    hiddenSize : int
        单向LSTM隐藏层大小，双向LSTM总输出维度为hiddenSize*2
    dr : float, default=0.5
        Dropout比率，用于防止过拟合，范围[0,1]
    cnn_out_channels : int, default=None
        CNN输出通道数，如果为None则使用hiddenSize
    kernel_size : int, default=3
        CNN卷积核大小，控制感受野范围
        
    网络结构 (Network Architecture):
    ----------
    1. CNN特征提取层：
       - 1D卷积层: 在特征维度上进行卷积，提取局部特征模式
       - 批归一化: 加速训练并提高稳定性
       - ReLU激活函数
       - Dropout层
    2. 线性投影层: CNN输出 -> hiddenSize
    3. 双向LSTM层: hiddenSize -> hiddenSize*2  
    4. 线性输出层: hiddenSize*2 -> ny
    
    用途 (Usage):
    ----------
    主要用于XAJ模型参数的深度学习反演，特别适合需要同时考虑特征间关系、时间依赖和双向时序信息的场景。
    相比CNN-LSTM模型，CNN-BiLSTM能够利用过去和未来的信息，提供更丰富的时序特征表示。
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, cnn_out_channels=None, kernel_size=3):
        super(CudnnCnnBiLstmModel, self).__init__()
        self.nx = nx                    # 输入特征维度
        self.ny = ny                    # 输出特征维度
        self.hiddenSize = hiddenSize    # 单向LSTM隐藏层大小
        self.ct = 0                     # 计数器（保留用于兼容性）
        self.nLayer = 1                # LSTM层数（当前固定为1层）
        
        # CNN参数设置
        self.cnn_out_channels = cnn_out_channels if cnn_out_channels is not None else hiddenSize
        self.kernel_size = kernel_size
        
        # CNN特征提取层：在特征维度上进行1D卷积
        # 输入形状: (batch, channels=1, features=nx)
        # 输出形状: (batch, channels=cnn_out_channels, features=nx)
        # 使用padding保持特征维度不变
        self.conv1d = nn.Conv1d(
            in_channels=1,              # 输入通道数（将每个时间步的特征视为单通道）
            out_channels=self.cnn_out_channels,  # 输出通道数
            kernel_size=kernel_size,    # 卷积核大小
            padding=(kernel_size - 1) // 2,  # 保持特征维度不变
            padding_mode='zeros'
        )
        
        # 批归一化：加速训练并提高稳定性
        self.bn = nn.BatchNorm1d(self.cnn_out_channels)
        
        # Dropout层：防止过拟合
        self.dropout_cnn = nn.Dropout(dr)
        
        # 线性投影层：将CNN输出投影到LSTM输入维度
        # CNN输出维度为 cnn_out_channels，需要投影到 hiddenSize
        self.linear_proj = torch.nn.Linear(self.cnn_out_channels, hiddenSize)
        
        # 双向LSTM层：处理时间序列信息（前向和后向）
        # batch_first=False: 输入格式为(seq_len, batch, input_size)
        # bidirectional=True: 启用双向处理
        # 输出维度为hiddenSize*2（前向+后向）
        # dropout仅在多层LSTM时生效
        self.bilstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            bidirectional=True,  # 启用双向LSTM
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # 线性输出层：处理双向LSTM的输出（维度为hiddenSize*2）
        self.linearOut = torch.nn.Linear(hiddenSize * 2, ny)
        self.gpu = 1                    # GPU标志（保留用于兼容性）

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        前向传播函数 (Forward Pass Function)
        
        参数 (Parameters):
        ----------
        x : torch.Tensor
            输入张量，形状为(seq_len, batch_size, nx)
        doDropMC : bool, default=False
            是否在推理时使用Monte Carlo Dropout（当前未实现）
        dropoutFalse : bool, default=False  
            是否禁用Dropout（当前未实现）
            
        返回 (Returns):
        ----------
        torch.Tensor
            输出张量，形状为(seq_len, batch_size, ny)
        """
        seq_len, batch_size, nx = x.shape
        
        # 步骤1: CNN特征提取
        # 将输入重塑为 (batch_size * seq_len, 1, nx) 以便进行1D卷积
        x_reshaped = x.reshape(batch_size * seq_len, 1, nx)
        
        # 1D卷积：在特征维度上提取局部模式
        x_conv = self.conv1d(x_reshaped)  # (batch*seq, cnn_out_channels, nx)
        
        # 批归一化
        x_conv = self.bn(x_conv)
        
        # ReLU激活
        x_conv = F.relu(x_conv)
        
        # Dropout
        if not dropoutFalse:
            x_conv = self.dropout_cnn(x_conv)
        
        # 将CNN输出重塑回 (seq_len, batch_size, cnn_out_channels)
        # 对特征维度取平均，得到每个时间步的CNN特征表示
        x_conv = x_conv.reshape(batch_size * seq_len, self.cnn_out_channels, nx)
        x_conv = x_conv.mean(dim=2)  # 平均池化: (batch*seq, cnn_out_channels)
        x_conv = x_conv.reshape(seq_len, batch_size, self.cnn_out_channels)
        
        # 步骤2: 线性投影到LSTM输入维度
        x_proj = F.relu(self.linear_proj(x_conv))  # (seq_len, batch_size, hiddenSize)
        
        # 步骤3: 双向LSTM层处理时间序列
        # outBiLSTM: 双向LSTM输出，形状为(seq_len, batch_size, hidden_size*2)
        # hn, cn: 最终隐藏状态和细胞状态（包含前向和后向）
        outBiLSTM, (hn, cn) = self.bilstm(x_proj)
        
        # 步骤4: 输出层得到最终结果
        out = self.linearOut(outBiLSTM)
        return out

