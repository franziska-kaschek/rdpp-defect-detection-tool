"""
Backbone registry for encoder–decoder model pairs.

Maps backbone names (used in configs) to their corresponding
encoder and decoder constructor functions.
"""

from src.model.resnet import (
    resnet18,
    resnet34,
    resnet50,
    wide_resnet50_2,
)
from src.model.de_resnet import (
    de_resnet18,
    de_resnet34,
    de_resnet50,
    de_wide_resnet50_2,
)

BACKBONE_FNS = {
    "resnet18": (resnet18, de_resnet18),
    "resnet34": (resnet34, de_resnet34),
    "resnet50": (resnet50, de_resnet50),
    "wide_resnet50_2": (wide_resnet50_2, de_wide_resnet50_2),
}
