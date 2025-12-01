"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .common import (
    get_activation, 
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .presnet import PResNet, ResNet
from .compressed_presnet import (
    CompressedResNetBlockStem,
    CompressedResNetLumaFusion,
    CompressedResNetLumaFusionPruned,
    CompressedResNetReconstruction,
    build_compressed_backbone,
)
from .test_resnet import MResNet

try:  # Optional dependency for timm backbones
    from .timm_model import TimmModel  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    TimmModel = None

try:
    from .torchvision_model import TorchVisionModel
except ModuleNotFoundError:  # pragma: no cover
    TorchVisionModel = None

try:  # Optional CSPResNet entry
    from .presnet import PResNet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    PResNet = None

try:
    from .csp_darknet import CSPDarkNet, CSPPAN
except ModuleNotFoundError:  # pragma: no cover
    CSPDarkNet = CSPPAN = None

try:
    from .efficientvit_backbone import EfficientViTBackbone
except ModuleNotFoundError:  # pragma: no cover
    EfficientViTBackbone = None