from enum import Enum

from torch import nn


class LastLayerStrategy(Enum):
    replace = nn.Linear(in_features=2048, out_features=10)
    append = nn.Sequential(nn.ReLU(), nn.Linear(in_features=2048, out_features=10))
