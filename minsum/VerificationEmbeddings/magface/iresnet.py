"""MagFace iResNet backbones.

This follows the official MagFace inference backbone:
https://github.com/IrvingMeng/MagFace/blob/master/models/iresnet.py
"""

from __future__ import annotations

import torch
from torch import nn


def _conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("IBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in IBasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-05, momentum=0.9)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        output = self.bn1(inputs)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.prelu(output)
        output = self.conv2(output)
        output = self.bn3(output)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return output + identity


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(
        self,
        layers: tuple[int, int, int, int],
        embedding_size: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: tuple[bool, bool, bool] | None = None,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = (False, False, False)
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must have 3 values")

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.bn2 = nn.BatchNorm2d(512 * IBasicBlock.expansion, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * IBasicBlock.expansion * self.fc_scale, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=2e-05, momentum=0.9)
        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * IBasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * IBasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * IBasicBlock.expansion, eps=2e-05, momentum=0.9),
            )

        layers = [
            IBasicBlock(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        ]
        self.inplanes = planes * IBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                IBasicBlock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, IBasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.prelu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.bn2(output)
        output = self.dropout(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return self.features(output)


def build_iresnet(architecture: str, embedding_size: int = 512) -> IResNet:
    layers_by_architecture = {
        "iresnet18": (2, 2, 2, 2),
        "iresnet34": (3, 4, 6, 3),
        "iresnet50": (3, 4, 14, 3),
        "iresnet100": (3, 13, 30, 3),
    }
    try:
        layers = layers_by_architecture[architecture]
    except KeyError as exc:
        supported = ", ".join(sorted(layers_by_architecture))
        raise ValueError(
            f"Unsupported MagFace architecture {architecture!r}. Choose one of: {supported}"
        ) from exc
    return IResNet(layers=layers, embedding_size=embedding_size)
