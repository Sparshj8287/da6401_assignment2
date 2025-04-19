from enum import Enum
from torch import nn


class LastLayerStrategy(Enum):
    """
    Defines strategies for modifying the final layer of a model.

    Attributes:
        replace: Replaces the final layer with a single Linear layer.
        append: Appends an activation (ReLU) followed by a Linear layer to the existing final layer.
    """
    replace = nn.Linear(in_features=2048, out_features=10)
    append = nn.Sequential(nn.ReLU(), nn.Linear(in_features=2048, out_features=10))
