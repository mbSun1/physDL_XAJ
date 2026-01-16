# -*- coding: utf-8 -*-
"""
RNN Model Compatibility Module

This module exists for backward compatibility and re-exports the RNN model
classes defined in `dLmodels.py`.

Older serialized models may reference `hydroMLlib.model.rnn`; keeping this
module ensures they can still be loaded.
"""

# Import and re-export RNN-related classes from dLmodels
from .dLmodels import (
    CudnnLstmModel,
    CudnnGruModel,
    CudnnBiLstmModel,
    CudnnBiGruModel,
    CudnnRnnModel,
)

# Export all classes so pickling works reliably
__all__ = [
    'CudnnLstmModel',
    'CudnnGruModel',
    'CudnnBiLstmModel',
    'CudnnBiGruModel',
    'CudnnRnnModel',
]

