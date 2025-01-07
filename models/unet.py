import segmentation_models_pytorch as smp
from typing import Any


class Unet(smp.Unet):
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(
            in_channels=int(config['num_channels']),
            classes=int(config['num_classes']),
            encoder_name='resnet50',
            encoder_weights=None
        )