# -*- coding: utf-8 -*-
"""
RNN模型兼容模块 (RNN Model Compatibility Module)

本模块用于向后兼容，将 dLmodels.py 中的 RNN 模型类重新导出。
旧版本的模型文件可能引用了 hydroMLlib.model.rnn，本模块确保这些模型可以正常加载。
"""

# 从 dLmodels 导入所有 RNN 相关类
from .dLmodels import (
    CudnnLstmModel,
    CudnnGruModel,
    CudnnBiLstmModel,
    CudnnBiGruModel,
    CudnnRnnModel,
)

# 导出所有类，确保可以被 pickle 正确加载
__all__ = [
    'CudnnLstmModel',
    'CudnnGruModel',
    'CudnnBiLstmModel',
    'CudnnBiGruModel',
    'CudnnRnnModel',
]

