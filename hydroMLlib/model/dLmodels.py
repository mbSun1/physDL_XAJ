# -*- coding: utf-8 -*-
"""
RNN Models and Hydrological Model Implementations

This module provides PyTorch implementations of RNN backbones and HBV-based
hydrological model components used in hydrology-oriented deep learning.

Contents:
- RNN model classes:
  - CudnnLstmModel, CudnnGruModel
  - CudnnBiLstmModel, CudnnBiGruModel
  - CudnnRnnModel
  - CudnnCnnLstmModel, CudnnCnnBiLstmModel
- HBV model classes:
  - HBVMul, HBVMulET, HBVMulTD, HBVMulTDET
- Hybrid (DL + hydrological model) classes:
  - MultiInv_HBVModel, MultiInv_HBVTDModel
- Helper functions:
  - UH_conv, UH_gamma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")



class CudnnLstmModel(torch.nn.Module):
    """
    LSTM model.

    A PyTorch LSTM backbone used for time-series prediction and parameter
    inversion in hydrology-oriented deep learning.

    Args:
        nx: Input feature dimension.
        ny: Output feature dimension.
        hiddenSize: Hidden size of the LSTM.
        dr: Dropout rate in [0, 1].
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx                    # input feature dimension
        self.ny = ny                    # output feature dimension
        self.hiddenSize = hiddenSize    # LSTM hidden size
        self.ct = 0                     # counter (kept for compatibility)
        self.nLayer = 1                 # number of layers (fixed to 1)
        
        # Linear input layer: nx -> hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # LSTM layer (seq_len, batch, input_size); dropout effective only for nLayer>1
        self.lstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # Linear output layer: hiddenSize -> ny
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU flag (kept for compatibility)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, nx).
            doDropMC: Whether to use MC Dropout at inference (not implemented).
            dropoutFalse: Whether to disable dropout (not implemented).

        Returns:
            Output tensor of shape (seq_len, batch_size, ny).
        """
        # Input projection + ReLU
        x0 = F.relu(self.linearIn(x))
        
        # LSTM over sequence
        outLSTM, (hn, cn) = self.lstm(x0)
        
        # Output projection
        out = self.linearOut(outLSTM)
        return out
        
class CudnnGruModel(torch.nn.Module):
    """
    GRU model.

    A PyTorch GRU backbone used for time-series prediction and parameter
    inversion. GRU is a simplified alternative to LSTM with fewer parameters.

    Args:
        nx: Input feature dimension.
        ny: Output feature dimension.
        hiddenSize: Hidden size of the GRU.
        dr: Dropout rate in [0, 1].
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnGruModel, self).__init__()
        self.nx = nx                    # input feature dimension
        self.ny = ny                    # output feature dimension
        self.hiddenSize = hiddenSize    # GRU hidden size
        self.ct = 0                     # counter (kept for compatibility)
        self.nLayer = 1                 # number of layers (fixed to 1)
        
        # Linear input layer: nx -> hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # GRU layer (seq_len, batch, input_size); dropout effective only for nLayer>1
        self.gru = nn.GRU(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # Linear output layer: hiddenSize -> ny
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU flag (kept for compatibility)
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, nx).
            doDropMC: Whether to use MC Dropout at inference (not implemented).
            dropoutFalse: Whether to disable dropout (not implemented).

        Returns:
            Output tensor of shape (seq_len, batch_size, ny).
        """
        # Input projection + ReLU
        x0 = F.relu(self.linearIn(x))
        
        # GRU over sequence (GRU has no cell state)
        outGRU, hn = self.gru(x0)
        
        # Output projection
        out = self.linearOut(outGRU)
        return out
        
class CudnnBiLstmModel(torch.nn.Module):
    """
    Bidirectional LSTM model.

    Uses a bidirectional LSTM to leverage both past and future context in a
    sequence, which can provide richer temporal representations.

    Args:
        nx: Input feature dimension.
        ny: Output feature dimension.
        hiddenSize: Hidden size of a single direction (output is 2*hiddenSize).
        dr: Dropout rate in [0, 1].
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnBiLstmModel, self).__init__()
        self.nx = nx                    # input feature dimension
        self.ny = ny                    # output feature dimension
        self.hiddenSize = hiddenSize    # hidden size per direction
        self.ct = 0                     # counter (kept for compatibility)
        self.nLayer = 1                 # number of layers (fixed to 1)
        
        # Linear input layer: nx -> hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # Bidirectional LSTM (output dim is 2*hiddenSize)
        self.bilstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            bidirectional=True,
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # Linear output layer: 2*hiddenSize -> ny
        self.linearOut = torch.nn.Linear(hiddenSize * 2, ny)
        self.gpu = 1                    # GPU flag (kept for compatibility)
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, nx).
            doDropMC: Whether to use MC Dropout at inference (not implemented).
            dropoutFalse: Whether to disable dropout (not implemented).

        Returns:
            Output tensor of shape (seq_len, batch_size, ny).
        """
        # Input projection + ReLU
        x0 = F.relu(self.linearIn(x))
        
        # Bidirectional LSTM over sequence
        outBiLSTM, (hn, cn) = self.bilstm(x0)
        
        # Output projection
        out = self.linearOut(outBiLSTM)
        return out
        
class CudnnBiGruModel(torch.nn.Module):
    """
    Bidirectional GRU model.

    Uses a bidirectional GRU to leverage both past and future context in a
    sequence. Compared to a bidirectional LSTM, it typically trains faster and
    uses fewer parameters.

    Args:
        nx: Input feature dimension.
        ny: Output feature dimension.
        hiddenSize: Hidden size per direction (output is 2*hiddenSize).
        dr: Dropout rate in [0, 1].
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnBiGruModel, self).__init__()
        self.nx = nx                    # input feature dimension
        self.ny = ny                    # output feature dimension
        self.hiddenSize = hiddenSize    # hidden size per direction
        self.ct = 0                     # counter (kept for compatibility)
        self.nLayer = 1                 # number of layers (fixed to 1)
        
        # Linear input layer: nx -> hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # Bidirectional GRU (output dim is 2*hiddenSize)
        self.bigru = nn.GRU(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            bidirectional=True,
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # Linear output layer: 2*hiddenSize -> ny
        self.linearOut = torch.nn.Linear(hiddenSize * 2, ny)
        self.gpu = 1                    # GPU flag (kept for compatibility)
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, nx).
            doDropMC: Whether to use MC Dropout at inference (not implemented).
            dropoutFalse: Whether to disable dropout (not implemented).

        Returns:
            Output tensor of shape (seq_len, batch_size, ny).
        """
        # Input projection + ReLU
        x0 = F.relu(self.linearIn(x))
        
        # Bidirectional GRU over sequence
        outBiGRU, hn = self.bigru(x0)
        
        # Output projection
        out = self.linearOut(outBiGRU)
        return out
        
class CudnnRnnModel(torch.nn.Module):
    """
    Basic RNN model.

    A vanilla RNN backbone implemented in PyTorch. It is simpler than LSTM/GRU
    but can suffer from vanishing gradients on long sequences.

    Args:
        nx: Input feature dimension.
        ny: Output feature dimension.
        hiddenSize: Hidden size of the RNN.
        dr: Dropout rate in [0, 1].
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnRnnModel, self).__init__()
        self.nx = nx                    # input feature dimension
        self.ny = ny                    # output feature dimension
        self.hiddenSize = hiddenSize    # RNN hidden size
        self.ct = 0                     # counter (kept for compatibility)
        self.nLayer = 1                 # number of layers (fixed to 1)
        
        # Linear input layer: nx -> hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        
        # RNN layer (seq_len, batch, input_size)
        # nonlinearity: 'tanh' or 'relu'
        self.rnn = nn.RNN(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0,
            nonlinearity='tanh'  # 'tanh' or 'relu'
        )
        
        # Linear output layer: hiddenSize -> ny
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU flag (kept for compatibility)
        
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, nx).
            doDropMC: Whether to use MC Dropout at inference (not implemented).
            dropoutFalse: Whether to disable dropout (not implemented).

        Returns:
            Output tensor of shape (seq_len, batch_size, ny).
        """
        # Input projection + ReLU
        x0 = F.relu(self.linearIn(x))
        
        # RNN over sequence
        outRNN, hn = self.rnn(x0)
        
        # Output projection
        out = self.linearOut(outRNN)
        return out


class CudnnCnnLstmModel(torch.nn.Module):
    """
    CNN-LSTM model.

    A hybrid backbone that applies 1D convolutions along the feature dimension
    to extract local patterns, then uses an LSTM to model temporal dynamics.

    Args:
        nx: Input feature dimension.
        ny: Output feature dimension.
        hiddenSize: LSTM hidden size.
        dr: Dropout rate in [0, 1].
        cnn_out_channels: Number of CNN output channels (defaults to hiddenSize).
        kernel_size: 1D convolution kernel size.
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, cnn_out_channels=None, kernel_size=3):
        super(CudnnCnnLstmModel, self).__init__()
        self.nx = nx                    # input feature dimension
        self.ny = ny                    # output feature dimension
        self.hiddenSize = hiddenSize    # LSTM hidden size
        self.ct = 0                     # counter (kept for compatibility)
        self.nLayer = 1                 # number of layers (fixed to 1)
        
        # CNN parameters
        self.cnn_out_channels = cnn_out_channels if cnn_out_channels is not None else hiddenSize
        self.kernel_size = kernel_size
        
        # CNN feature extractor (1D conv over feature dimension)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=self.cnn_out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            padding_mode='zeros'
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(self.cnn_out_channels)
        
        # Dropout
        self.dropout_cnn = nn.Dropout(dr)
        
        # Project CNN features to LSTM input dimension
        self.linear_proj = torch.nn.Linear(self.cnn_out_channels, hiddenSize)
        
        # LSTM over time
        self.lstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # Output projection
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1                    # GPU flag (kept for compatibility)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, nx).
            doDropMC: Whether to use MC Dropout at inference (not implemented).
            dropoutFalse: Whether to disable dropout (not implemented).

        Returns:
            Output tensor of shape (seq_len, batch_size, ny).
        """
        seq_len, batch_size, nx = x.shape
        
        # Step 1: CNN feature extraction
        x_reshaped = x.reshape(batch_size * seq_len, 1, nx)
        
        # 1D convolution over features
        x_conv = self.conv1d(x_reshaped)  # (batch*seq, cnn_out_channels, nx)
        
        # Batch norm
        x_conv = self.bn(x_conv)
        
        # ReLU
        x_conv = F.relu(x_conv)
        
        # Dropout
        if not dropoutFalse:
            x_conv = self.dropout_cnn(x_conv)
        
        # Reshape back and average-pool over feature dimension
        x_conv = x_conv.reshape(batch_size * seq_len, self.cnn_out_channels, nx)
        x_conv = x_conv.mean(dim=2)  # (batch*seq, cnn_out_channels)
        x_conv = x_conv.reshape(seq_len, batch_size, self.cnn_out_channels)
        
        # Step 2: project to LSTM input dim
        x_proj = F.relu(self.linear_proj(x_conv))  # (seq_len, batch_size, hiddenSize)
        
        # Step 3: LSTM over time
        outLSTM, (hn, cn) = self.lstm(x_proj)
        
        # Step 4: output projection
        out = self.linearOut(outLSTM)
        return out


class CudnnCnnBiLstmModel(torch.nn.Module):
    """
    CNN-BiLSTM model.

    A hybrid backbone that applies 1D convolutions along the feature dimension
    to extract local patterns, then uses a bidirectional LSTM to model temporal
    dynamics with both past and future context.

    Args:
        nx: Input feature dimension.
        ny: Output feature dimension.
        hiddenSize: Hidden size per direction (output is 2*hiddenSize).
        dr: Dropout rate in [0, 1].
        cnn_out_channels: Number of CNN output channels (defaults to hiddenSize).
        kernel_size: 1D convolution kernel size.
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, cnn_out_channels=None, kernel_size=3):
        super(CudnnCnnBiLstmModel, self).__init__()
        self.nx = nx                    # input feature dimension
        self.ny = ny                    # output feature dimension
        self.hiddenSize = hiddenSize    # hidden size per direction
        self.ct = 0                     # counter (kept for compatibility)
        self.nLayer = 1                 # number of layers (fixed to 1)
        
        # CNN parameters
        self.cnn_out_channels = cnn_out_channels if cnn_out_channels is not None else hiddenSize
        self.kernel_size = kernel_size
        
        # CNN feature extractor (1D conv over feature dimension)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=self.cnn_out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            padding_mode='zeros'
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(self.cnn_out_channels)
        
        # Dropout
        self.dropout_cnn = nn.Dropout(dr)
        
        # Project CNN features to LSTM input dimension
        self.linear_proj = torch.nn.Linear(self.cnn_out_channels, hiddenSize)
        
        # Bidirectional LSTM over time (output dim is 2*hiddenSize)
        self.bilstm = nn.LSTM(
            input_size=hiddenSize, 
            hidden_size=hiddenSize, 
            num_layers=1, 
            batch_first=False, 
            bidirectional=True,
            dropout=dr if self.nLayer > 1 else 0
        )
        
        # Linear output layer: 2*hiddenSize -> ny
        self.linearOut = torch.nn.Linear(hiddenSize * 2, ny)
        self.gpu = 1                    # GPU flag (kept for compatibility)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (seq_len, batch_size, nx).
            doDropMC: Whether to use MC Dropout at inference (not implemented).
            dropoutFalse: Whether to disable dropout (not implemented).

        Returns:
            Output tensor of shape (seq_len, batch_size, ny).
        """
        seq_len, batch_size, nx = x.shape
        
        # Step 1: CNN feature extraction
        x_reshaped = x.reshape(batch_size * seq_len, 1, nx)
        
        # 1D convolution over features
        x_conv = self.conv1d(x_reshaped)  # (batch*seq, cnn_out_channels, nx)
        
        # Batch norm
        x_conv = self.bn(x_conv)
        
        # ReLU
        x_conv = F.relu(x_conv)
        
        # Dropout
        if not dropoutFalse:
            x_conv = self.dropout_cnn(x_conv)
        
        # Reshape back and average-pool over feature dimension
        x_conv = x_conv.reshape(batch_size * seq_len, self.cnn_out_channels, nx)
        x_conv = x_conv.mean(dim=2)  # (batch*seq, cnn_out_channels)
        x_conv = x_conv.reshape(seq_len, batch_size, self.cnn_out_channels)
        
        # Step 2: project to LSTM input dim
        x_proj = F.relu(self.linear_proj(x_conv))  # (seq_len, batch_size, hiddenSize)
        
        # Step 3: bidirectional LSTM over time
        outBiLSTM, (hn, cn) = self.bilstm(x_proj)
        
        # Step 4: output projection
        out = self.linearOut(outBiLSTM)
        return out

