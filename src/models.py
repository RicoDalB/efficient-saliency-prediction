from __future__ import annotations

# Neural network architectures for visual sailency prediction.

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import (MobileNet_V2_Weights, ResNet18_Weights, mobilenet_v2, resnet18)

ModelName = Literal["light_single", "light_multi", "heavy_multi"]
FeatureDict = dict[str, Tensor]

# Return largest useful Groupnorm group count 
def _group_count(channels: int, preferred_group: int = 8) -> int:
    for groups in range(min(preferred_group, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1

# ImageNet-pretrained mobileNetV2 returning four feature scales
class MobileNetV2Encoder(nn.Module):
    out_channels = {
        "s4": 24,
        "s8": 32,
        "s16": 96,
        "s32": 320,
    }
    _tap_indices = {
        3: "s4",
        6: "s8",
        13: "s16",
        17: "s32",
    }

    # Create MobileNetV2 encoder
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        
        backbone = mobilenet_v2(weights=weights)

        self.features = backbone.features[:18]

    def forward(self, x: Tensor) -> FeatureDict:
        outputs: FeatureDict = {}

        for index, layer in enumerate(self.features):
            x = layer(x)
            key = self._tap_indices.get(index)
            if key is not None:
                outputs[key] = x

        if set(outputs) != set(self.out_channels):
            raise RuntimeError("MobileNetV2 feature are incomplete")
        
        return outputs
    

# Heavier feature extractor, larger than mobile, return the same four-scale interface
class ResNet18Encoder(nn.Module):
    out_channels = {
        "s4": 64,
        "s8": 128,
        "s16": 256,
        "s32": 512,
    }
    # Create ResNet-18 encoder
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def farward(self, x: Tensor) -> FeatureDict:
        x = self.stem(x)
        s4 = self.layer1(x)
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)

        return {
            "s4": s4,
            "s8": s8,
            "s16": s16,
            "s32": s32,
        }
    
# Depthwise-separable refinement used after each upsampling step
class RefinementBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_group_count(channels),
                num_channels=channels,
            ),
            # ReLU 
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
    

# Decode only the deepest feature map for the light_single model
class SingleScaleDecoder(nn.Module):
    def __init__(self, deepest_channels: int, decoder_channels: int = 32) -> None:
        super().__init__()

        self.deep_projection = nn.Conv2d(
            in_channels=deepest_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            bias=False,
        )

        self.refine16 = RefinementBlock(decoder_channels)
        self.refine8 = RefinementBlock(decoder_channels)
        self.refine4 = RefinementBlock(decoder_channels)

        self.prediction_head = nn.Conv2d(
            in_channels=decoder_channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, features: FeatureDict, outpu_size: tuple[int, int],) -> Tensor:
        x = self.deep_projection(features["s32"])

        x = F.interpolate(
            x,
            size=features["s16"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = self.refine16(x)

        x = F.interpolate(
            x,
            size=features["s8"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.refine8(x)

        x = F.interpolate(
            x,
            size=features["s4"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.refine4(x)

        logits = self.prediction_head(x)

        # Resize
        return F.interpolate(logits, size=outpu_size, mode="bilinear", align_corners=False,)
    

# Top-down decoder shared by light_multi and heavy_multi
class MultiScaleDecoder(nn.Module):
    def __init__(self, encoder_channels: dict[str, int], decoder_channels: int = 32,) -> None:
        super().__init__()
        required_keys = {"s4", "s8", "s16", "s32"}

        if set(encoder_channels) != required_keys:
            raise ValueError(
                "encoder_channels must contain exactly "
                f"{sorted(required_keys)}, received {sorted(encoder_channels)}"
            )

        # Every encoder stage has a different number of channels.
        # MobileNetV2: 24, 32, 96, 320
        # ResNet-18:   64, 128, 256, 512

        self.projections = nn.ModuleDict(
            {
                key: nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=decoder_channels,
                    kernel_size=1,
                    bias=False,
                )
                for key, in_channels in encoder_channels.items()
            }
        )

        self.refine16 = RefinementBlock(decoder_channels)
        self.refine8 = RefinementBlock(decoder_channels)
        self.refine4 = RefinementBlock(decoder_channels)

        self.prediction_head = nn.Conv2d(
            in_channels=decoder_channels,
            out_channels=1,
            kernel_size=1,
        )

    # Unsample a deep feature and add a same-scale skip feature. 
    def _upsample_add(self, x: Tensor, lateral: Tensor) -> Tensor:
        x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)

        return x + lateral
    
    def forward(self, features: FeatureDict, output_size: tuple[int, int],)-> Tensor:
        projected = {
            key: projection(features[key])
            for key, projection in self.projections.items()
        }

        x = projected["s32"]

        x = self._upsample_add(x, projected["s16"])
        x = self.refine16(x)

        x = self._upsample_add(x, projected["s8"])
        x = self.refine8(x)

        x = self._upsample_add(x, projected["s4"])
        x = self.refine4(x)

        logits = self.prediction_head(x)

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class SaliencyModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "images must have shape [B, 3, H, W], "
                f"received {tuple(images.shape)}"
            )
        output_size = tuple(images.shape[-2:])
        features = self.encoder(images)

        return self.decoder(features, output_size)



# Contruct one of the three models, light single, light multi or heavy multi
def build_model(name: ModelName, *, pretrained: bool = True, decoder_channels: int = 32,) -> SaliencyModel:

    if decoder_channels <= 0:
        raise ValueError("decoder_channels must be greater than zero")

    if name == "light_single":
        encoder = MobileNetV2Encoder(pretrained=pretrained)
        decoder = SingleScaleDecoder(
            deepest_channels=encoder.out_channels["s32"],
            decoder_channels=decoder_channels,
        )
        return SaliencyModel(encoder, decoder)

    if name == "light_multi":
        encoder = MobileNetV2Encoder(pretrained=pretrained)
        decoder = MultiScaleDecoder(
            encoder_channels=encoder.out_channels,
            decoder_channels=decoder_channels,
        )
        return SaliencyModel(encoder, decoder)

    if name == "heavy_multi":
        encoder = ResNet18Encoder(pretrained=pretrained)
        decoder = MultiScaleDecoder(
            encoder_channels=encoder.out_channels,
            decoder_channels=decoder_channels,
        )
        return SaliencyModel(encoder, decoder)

    raise ValueError(
        f"Unknown model name {name!r}. "
        "Expected 'light_single', 'light_multi', or 'heavy_multi'."
    )


# Count nn parameters
def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


# Return encoder decoder tot and trainable parameter count
def parameter_summary(model: SaliencyModel) -> dict[str, int]:
    return {
        "encoder": count_parameters(model.encoder),
        "decoder": count_parameters(model.decoder),
        "total": count_parameters(model),
        "trainable": count_parameters(model, trainable_only=True),
    }
