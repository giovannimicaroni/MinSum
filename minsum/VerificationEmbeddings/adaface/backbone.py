"""AdaFace IR backbones compatible with the official pretrained checkpoints.

The network layout follows mk-minchul/AdaFace's MIT-licensed ``net.py``:
https://github.com/mk-minchul/AdaFace/blob/master/net.py
"""

from __future__ import annotations

from collections import namedtuple

import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.view(inputs.size(0), -1)


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(inputs)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return inputs * scale


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.res_layer(inputs) + self.shortcut_layer(inputs)


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


Bottleneck = namedtuple("Bottleneck", ["in_channel", "depth", "stride"])


def _get_block(
    in_channel: int,
    depth: int,
    num_units: int,
    stride: int = 2,
) -> list[Bottleneck]:
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def _get_blocks(num_layers: int) -> list[list[Bottleneck]]:
    units_by_depth = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 14, 3),
        100: (3, 13, 30, 3),
    }
    try:
        units = units_by_depth[num_layers]
    except KeyError as exc:
        raise ValueError(f"Unsupported AdaFace backbone depth: {num_layers}") from exc
    channels = ((64, 64), (64, 128), (128, 256), (256, 512))
    return [
        _get_block(in_channel, depth, num_units)
        for (in_channel, depth), num_units in zip(channels, units)
    ]


class Backbone(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        num_layers: int,
        mode: str = "ir",
    ) -> None:
        super().__init__()
        if input_size not in ((112, 112), (224, 224)):
            raise ValueError("AdaFace input_size must be 112x112 or 224x224")
        if mode not in ("ir", "ir_se"):
            raise ValueError("AdaFace mode must be 'ir' or 'ir_se'")

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        unit_module = BasicBlockIR if mode == "ir" else BasicBlockIRSE
        modules = []
        for block in _get_blocks(num_layers):
            for unit in block:
                modules.append(unit_module(unit.in_channel, unit.depth, unit.stride))
        self.body = nn.Sequential(*modules)

        spatial_size = 7 if input_size == (112, 112) else 14
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * spatial_size * spatial_size, 512),
            nn.BatchNorm1d(512, affine=False),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.input_layer(inputs)
        features = self.body(features)
        features = self.output_layer(features)
        norms = torch.norm(features, 2, 1, keepdim=True)
        return features / norms.clamp_min(1e-12), norms


def build_model(model_name: str = "ir_50") -> Backbone:
    architectures = {
        "ir_18": (18, "ir"),
        "ir_34": (34, "ir"),
        "ir_50": (50, "ir"),
        "ir_101": (100, "ir"),
        "ir_se_50": (50, "ir_se"),
        "ir_se_101": (100, "ir_se"),
    }
    try:
        num_layers, mode = architectures[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(architectures))
        raise ValueError(
            f"Unsupported AdaFace architecture {model_name!r}. Choose one of: {supported}"
        ) from exc
    return Backbone(input_size=(112, 112), num_layers=num_layers, mode=mode)
