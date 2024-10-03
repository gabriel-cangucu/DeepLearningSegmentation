import segmentation_models_pytorch as smp
from typing import Any


class Unet(smp.Unet):
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(encoder_weights=None)

        self.in_channels = int(config['num_channels'])
        self.classes = int(config['num_classes'])