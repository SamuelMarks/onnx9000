"""Neural Network Module framework."""

from .module import Module

__all__ = ["Module"]
from .linear import Linear

__all__.append("Linear")
from .conv import Conv1d, Conv2d, Conv3d

__all__.extend(["Conv1d", "Conv2d", "Conv3d"])
from .pool import AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d

__all__.extend(["MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d"])
from .conv import ConvTranspose1d, ConvTranspose2d

__all__.extend(["ConvTranspose1d", "ConvTranspose2d"])
from .pool import AdaptiveAvgPool2d

__all__.append("AdaptiveAvgPool2d")
from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)

__all__.extend(
    [
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "LayerNorm",
        "GroupNorm",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
    ]
)
from .dropout import Dropout, Dropout2d, Dropout3d

__all__.extend(["Dropout", "Dropout2d", "Dropout3d"])
from .embedding import Embedding

__all__.append("Embedding")
from .rnn import GRU, LSTM, RNN

__all__.extend(["RNN", "LSTM", "GRU"])
from .containers import ModuleDict, ModuleList, ParameterList, Sequential

__all__.extend(["Sequential", "ModuleList", "ModuleDict", "ParameterList"])
from .identity import Identity

__all__.append("Identity")
from .flatten import Flatten, Unflatten

__all__.extend(["Flatten", "Unflatten"])
