"""
Backward-compatibility bridge for MS-VMUNet model architecture.
Core implementation resides in `models/ms_vmunet.py`.
"""
from models.ms_vmunet import (
    MultiScaleConv,
    FlexibleScanMambaBlock,
    ModularSkipConnection,
    HighFrequencyDifferenceSkip,
    MSVMUNetNextGen,
    MSVMUNet,
    get_nextgen_model
)

__all__ = [
    "MultiScaleConv",
    "FlexibleScanMambaBlock",
    "ModularSkipConnection",
    "HighFrequencyDifferenceSkip",
    "MSVMUNetNextGen",
    "MSVMUNet",
    "get_nextgen_model"
]
