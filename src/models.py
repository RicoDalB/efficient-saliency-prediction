from __future__ import annotations

""" Neural network architectures for visual sailency prediction.
All models follow the same tensor contract:

- input:  RGB tensor with shape ``[B, 3, H, W]``;
- output: raw saliency logits with shape ``[B, 1, H, W]``.

The logits are intentionally not passed through sigmoid or softmax here.
``src.losses_metrics`` converts them into a spatial probability distribution."""

from typing import Literal

import torch 
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import (MobileNet_V2_Weights, ResNet18_Weights, mobilenet_v2, resnet18)

ModelName = Literal["light_single", "light_multi", "heavy_multi"]
FeatureDict = dict[str, Tensor]

# Return largest useful Groupnorm group count dividing channels
def _group_count(channels: int, preferred_group: int = 8) -> int:
    for groups in range(min(preferred_group, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1

# ImageNet-pretrained mobileNetV2 returning four feature scales
class MobileNetV2Encoder(nn.Module):
    # MovileNetV2 is designed for image classification, out one class vector, for Sailency we do not need the classification head
    # so we colletc features map from four different depths of network
    out_channels = {
        "s4": 24,
        "s8": 32,
        "s16": 96,
        "s32": 320,
    }

    # During farward pass layes are visited in sequence, at these indices we save current tensor in the feature dictionary
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
        
        # Build TorchVision MobileNetV2 model
        backbone = mobilenet_v2(weights=weights)

        # We stop before final classification becouse with 320 channesl already conteins deepest info we need so keep model smaller and faster
        self.features = backbone.features[:18]

    # Extract four MobileNet feature maps, x = RGB imgae, return dict containing "s4"...
    def forward(self, x: Tensor) -> FeatureDict:
        outputs: FeatureDict = {}

        # We apply one block at time so we can save intermediate outputs at selected depths
        for index, layer in enumerate(self.features):
            x = layer(x)
            # Check layer index is one of selected in _tap_indices
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

        # stem is first part of ResNet-18
        # image -> 7x7 convolution -> BatchNorm -> ReLU -> Max pooling
        # after stem spatial res is approx 1/4 of the original image
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        # Store four residual stages
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def farward(self, x: Tensor) -> FeatureDict:
        x = self.stem(x)
        # Layer 1 keep approx stride 4
        s4 = self.layer1(x)
        # Layer 2 reduce res to approx stride 8
        s8 = self.layer2(s4)
        # Layer 3 reduce res to appeox stride 16
        s16 = self.layer3(s8)
        # Layer 4 produces deepest stride-32 representation
        s32 = self.layer4(s16)

        return {
            "s4": s4,
            "s8": s8,
            "s16": s16,
            "s32": s32,
        }
    

"""Lightweight processing block used after each upsampling step.
The block contains:
    depthwise 3x3 convolution
        Processes spatial information separately inside each channel.
    pointwise 1x1 convolution
        Mixes information across channels.
    GroupNorm
        Stabilizes activations and works well with small batch sizes.
    ReLU
        Adds non-linearity so the block can learn complex functions.
A depthwise-separable block is cheaper than a normal full 3x3 convolution.""" 
class RefinementBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # Depthwise convolution: (groups=channels) each input is convoled indipendently with its onw 3x3 filter
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            # Pointwise convolution: 1x1 convolution combines info from different channels while keeping spatial 
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False,
            ),
            # Normalize groups of channels, 
            nn.GroupNorm(
                num_groups=_group_count(channels),
                num_channels=channels,
            ),
            # ReLU 
            nn.ReLU(inplace=True),
        )

    def farward(self, x: Tensor) -> Tensor:
        return self.block(x)
    

"""Decoder used by the `light_single` model.
This decoder uses only the deepest feature values, s32.
It still reads the spatial sizes of s16, s8 and s4 so that it
knows the correct dimensions for interpolation but it does NOT add
    or concatenate their feature values.
Data flow:
    s32 values
        -> 1x1 projection to 32 channels
        -> upsample to s16 size -> refine
        -> upsample to s8 size  -> refine
        -> upsample to s4 size  -> refine
        -> 1-channel prediction
        -> resize to original image size
"""
class SingleScaleDecoder(nn.Module):
    def __init__(self, deepest_channels: int, decoder_channels: int = 32) -> None:
        super().__init__()

        # Reduce the number of deepest encoder channels to a small shared
        # decoder width. For MobileNetV2 this converts 320 channels to 32.
        self.deep_projection = nn.Conv2d(
            in_channels=deepest_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            bias=False,
        )

        # One lightweight refinement block after every upsampling stage.
        self.refine16 = RefinementBlock(decoder_channels)
        self.refine8 = RefinementBlock(decoder_channels)
        self.refine4 = RefinementBlock(decoder_channels)

        # Convert the final 32-channel feature map into one saliency-logit map.
        self.prediction_head = nn.Conv2d(
            in_channels=decoder_channels,
            out_channels=1,
            kernel_size=1,
        )

    # Decode deapest feature map into full-resolution logits
    def forward(self, features: FeatureDict, outpu_size: tuple[int, int],) -> Tensor:
        x = self.deep_projection(features["s32"])

        # Increase res so match stride-16 -> Not used
        x = F.interpolate(
            x,
            size=features["s16"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Learn how to improve the upsampled representation.
        x = self.refine16(x)

        # Repeat the same process for the stride-8 spatial size.
        x = F.interpolate(
            x,
            size=features["s8"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.refine8(x)

        # Repeat again for the stride-4 spatial size.
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
    

"""Top-down decoder used by `light_multi` and `heavy_multi`.
Unlike the single-scale decoder, this decoder uses feature VALUES from all
four encoder depths.
Data flow:
    projected s32
        -> upsample and add projected s16 -> refine
        -> upsample and add projected s8  -> refine
        -> upsample and add projected s4  -> refine
        -> one-channel prediction
        -> resize to original image size
The deeper feature gives semantic understanding, while shallower features
restore spatial detail and localization.
"""  
class MultiScaleDecoder(nn.Module):
    def __init__(self, encoder_channels: dict[str, int], decoder_channels: int = 32,) -> None:
        super().__init__()
        # Both encoders must provide exactly these four feature levels.
        required_keys = {"s4", "s8", "s16", "s32"}

        # Fail early if an encoder does not satisfy the expected interface.
        if set(encoder_channels) != required_keys:
            raise ValueError(
                "encoder_channels must contain exactly "
                f"{sorted(required_keys)}, received {sorted(encoder_channels)}"
            )

        # Every encoder stage has a different number of channels.
        #
        # MobileNetV2: 24, 32, 96, 320
        # ResNet-18:   64, 128, 256, 512
        #
        # Before adding features, all of them must have the same channel count.
        # Therefore, each one passes through a separate 1x1 projection that
        # converts it to ``decoder_channels`` channels, normally 32.
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

        # Refinement blocks used after each skip fusion.
        self.refine16 = RefinementBlock(decoder_channels)
        self.refine8 = RefinementBlock(decoder_channels)
        self.refine4 = RefinementBlock(decoder_channels)

        # Final 1x1 convolution that converts 32 decoder channels to one map.
        self.prediction_head = nn.Conv2d(
            in_channels=decoder_channels,
            out_channels=1,
            kernel_size=1,
        )

    # Unsample a deep feature and add a same-scale skip feature. Lateral is projected encoder feature from shallower stage, more spatial details
    def _unsample_add(self, x: Tensor, lateral: Tensor) -> Tensor:
        x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)

        # Add two tensor elem by elem
        return x + lateral
    
    # Fuse all feature scale and return full-res logits
    def forward(self, features: FeatureDict, output_size: tuple[int, int],)-> Tensor:
        # Apply the correct 1x1 projection to each encoder feature.
        projected = {
            key: projection(features[key])
            for key, projection in self.projections.items()
        }

        # Start from the deepest semantic feature.
        x = projected["s32"]

        # Upsample s32 to s16 size, add s16 detail, then refine the result.
        x = self._upsample_add(x, projected["s16"])
        x = self.refine16(x)

        # Upsample the result to s8 size, add s8 detail, then refine.
        x = self._upsample_add(x, projected["s8"])
        x = self.refine8(x)

        # Upsample the result to s4 size, add the finest selected detail,
        # then refine once more.
        x = self._upsample_add(x, projected["s4"])
        x = self.refine4(x)

        # Convert the final decoder representation into one saliency-logit map.
        logits = self.prediction_head(x)

        # Resize from stride-4 resolution to the exact original image size.
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


"""Simple wrapper joining an encoder and a decoder.
The wrapper gives all three project models the same public interface:
    logits = model(images)
Training code therefore does not need to know which encoder or decoder is being used."""
class SaliencyModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        # Register encoder and decoder as PyTorch submodules. Their parameters
        # will automatically appear in model.parameters(), checkpoints, etc.
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: Tensor) -> Tensor:
        """Predict saliency logits for a batch of RGB images."""

        # Validate the expected image tensor structure.
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "images must have shape [B, 3, H, W], "
                f"received {tuple(images.shape)}"
            )

        # Save the original height and width so that the decoder can return an
        # output aligned with the target saliency map.
        output_size = tuple(images.shape[-2:])

        # Extract hierarchical CNN features.
        features = self.encoder(images)

        # Convert those features into a one-channel full-resolution map.
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
