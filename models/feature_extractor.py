import torch
import torch.nn as nn
import torchvision
import logging
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor

from models.basemodel import BaseModel

logger = logging.getLogger(__name__)

class DecoderBlock(nn.Module):
    def __init__(
        self, previous, out, ksize=3, num_convs=1, norm=nn.BatchNorm2d, padding="zeros"
    ):
        super().__init__()
        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous if i == 0 else out,
                out,
                kernel_size=ksize,
                padding=ksize // 2,
                bias=norm is None,
                padding_mode=padding,
            )
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        _, _, hp, wp = previous.shape
        _, _, hs, ws = skip.shape
        # Calculate scale
        scale = 2 ** np.round(np.log2(np.array([hs / hp, ws / wp])))
        # Upsampling feature maps
        upsampled = nn.functional.interpolate(
            previous, scale_factor=scale.tolist(), mode="bilinear", align_corners=False
        )
        
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        if (hu <= hs) and (wu <= ws):
            skip = skip[:, :, :hu, :wu]
        elif (hu >= hs) and (wu >= ws):
            skip = nn.functional.pad(skip, [0, wu - ws, 0, hu - hs])
        else:
            raise ValueError(
                f"Inconsistent skip vs upsampled shapes: {(hs, ws)}, {(hu, wu)}"
            )

        return self.layers(skip) + upsampled


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, **kw):
        super().__init__()
        self.first = nn.Conv2d(
            in_channels_list[-1], out_channels, 1, padding=0, bias=True
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(c, out_channels, ksize=1, **kw)
                for c in in_channels_list[::-1][1:]
            ]
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, layers):
        feats = None
        for idx, x in enumerate(reversed(layers.values())):
            if feats is None:
                feats = self.first(x)
            else:
                feats = self.blocks[idx - 1](feats, x)
        out = self.out(feats)
        return out

class FeatureExtractor(BaseModel):
    default_conf = {
        "pretrained": True,
        "input_dim": 3,
        "output_dim": 128,  # of channels in output feature maps
        "encoder": "???",  # torchvision net as string
        "decoder_norm": "nn.BatchNorm2d",  # normalization ind decoder blocks
    }
    # two images 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def build_encoder(self, conf):
        assert isinstance(conf.encoder, str)
        # mark
        if conf.pretrained:
            assert conf.input_dim == 3
        Encoder = getattr(torchvision.models, conf.encoder)

        kw = {}
        if conf.encoder.startswith("resnet"):
            layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
            kw["replace_stride_with_dilation"] = [False, False, False]
        else:
            raise NotImplementedError(conf.encoder)

        encoder = Encoder(weights="DEFAULT" if conf.pretrained else None, **kw)
        encoder = create_feature_extractor(encoder, return_nodes=layers)

        return encoder, layers
    
    def _init(self, conf):
        # Preprocessing
        self.register_buffer("mean_", torch.tensor(self.mean), persistent=False)
        self.register_buffer("std_", torch.tensor(self.std), persistent=False)

        # Encoder
        self.encoder, self.layers = self.build_encoder(conf)
        s = 128
        inp = torch.zeros(1, 3, s, s)
        features = list(self.encoder(inp).values())  # 存储编码器的输出特征图
        self.skip_dims = [x.shape[1] for x in features] # 存储每个特征图的通道数
        self.layer_strides = [s / f.shape[-1] for f in features] # 计算每个特征图相对于输入图像的下采样比例
        self.scales = [self.layer_strides[0]] # 记录第一个特征图的下采样比例

        # Decoder
        norm = eval(conf.decoder_norm) if conf.decoder_norm else None
        self.decoder = FPN(self.skip_dims, out_channels=conf.output_dim, norm=norm)
        
        logger.debug(
            "Built feature extractor with layers {name:dim:stride}:\n"
            f"{list(zip(self.layers, self.skip_dims, self.layer_strides))}\n"
            f"and output scales {self.scales}."
        )
    
    def _forward(self, image):
        image = (image - self.mean_[:, None, None]) / self.std_[:, None, None]
        skip_features = self.encoder(image)
        output = self.decoder(skip_features)
        
        pred = {"feature_maps": [output], "skip_features": skip_features}

        return pred
