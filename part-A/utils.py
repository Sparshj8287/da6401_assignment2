from enum import Enum

from torch import nn


class FilterOrg(int, Enum):
    doubling = 2
    halving = 0.5
    equal = 1


class Activation(Enum):
    relu = nn.ReLU
    gelu = nn.GELU
    silu = nn.SiLU
    mish = nn.Mish
    leaky_relu = nn.LeakyReLU 


class Pooling(Enum):
    maxpooling = nn.MaxPool2d
    avgpooling = nn.AvgPool2d
