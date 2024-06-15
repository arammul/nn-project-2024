from typing import Dict, List, Optional, Union
from segmentation_models_pytorch import Unet

import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationHead,
    SegmentationModel
)

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder 
from segmentation_models_pytorch.base import initialization as init

class MultiHeadUnet(Unet):

    def __init__(
        self,
        dataset_configs: List[Dict[str, str]],
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        activation: Optional[Union[str, callable]] = None,
    ):
        SegmentationModel.__init__(self)

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_heads = nn.ModuleDict()
        for config in dataset_configs:
            self.segmentation_heads[config["name"]] = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=config["num_classes"],
                activation=activation,
                kernel_size=3,
            )

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        for head in self.segmentation_heads.values():
            init.initialize_head(head)


    def forward(self, x, dataset_name):
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_heads[dataset_name](decoder_output)

        return masks