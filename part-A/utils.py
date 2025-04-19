from enum import Enum
from torch import nn


class FilterOrg(int, Enum):
    doubling = 2
    halving = 0.5
    equal = 1


class Activation(Enum):
    relu = nn.ReLU         # Rectified Linear Unit
    gelu = nn.GELU         # Gaussian Error Linear Unit
    silu = nn.SiLU         # Sigmoid Linear Unit (also known as Swish)
    mish = nn.Mish         # Mish activation function
    leaky_relu = nn.LeakyReLU  # Leaky ReLU


class Pooling(Enum):
    maxpooling = nn.MaxPool2d  # Takes the maximum value in each window
    avgpooling = nn.AvgPool2d  # Takes the average value in each window
