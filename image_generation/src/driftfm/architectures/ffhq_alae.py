from __future__ import annotations

# Copyright 2019-2020 Stanislav Pidhorskyi
# Modified for standalone checkpoint import and inference in Drift Flow Matching.
# Portions of this file are adapted from the official ALAE implementation:
# https://github.com/podgorskiy/ALAE
# Licensed under the Apache License, Version 2.0.

from contextlib import contextmanager
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from ..training.checkpoints import load_checkpoint, save_checkpoint
from ..utils import normalize_path, resolve_latest_alias

_OFFICIAL_MODEL_DEFAULTS: dict[str, Any] = {
    "START_CHANNEL_COUNT": 64,
    "MAX_CHANNEL_COUNT": 512,
    "LAYER_COUNT": 6,
    "LATENT_SPACE_SIZE": 256,
    "DLATENT_AVG_BETA": 0.995,
    "TRUNCATIOM_PSI": 0.7,
    "TRUNCATIOM_CUTOFF": 8,
    "STYLE_MIXING_PROB": 0.9,
    "MAPPING_LAYERS": 5,
    "CHANNELS": 3,
    "GENERATOR": "GeneratorDefault",
    "ENCODER": "EncoderDefault",
}
_PROJECT_CKPT_FORMAT = "driftfm_ffhq_alae_v1"


@contextmanager
def _temporary_module_root(path: str | Path | None):
    if path in (None, ""):
        yield
        return
    module_root = str(resolve_latest_alias(normalize_path(path, resolve_latest=False)))
    inserted = False
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(module_root)
            except ValueError:
                pass


@dataclass(slots=True)
class FFHQALAEConfig:
    start_channel_count: int = 16
    max_channel_count: int = 512
    layer_count: int = 9
    latent_size: int = 512
    dlatent_avg_beta: float | None = 0.995
    truncation_psi: float | None = None
    truncation_cutoff: int | None = None
    style_mixing_prob: float | None = None
    mapping_layers: int = 8
    channels: int = 3
    generator: str = "GeneratorDefault"
    encoder: str = "EncoderDefault"

    @property
    def image_size(self) -> int:
        return 2 ** (self.layer_count + 1)

    @property
    def num_style_layers(self) -> int:
        return 2 * self.layer_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_channel_count": self.start_channel_count,
            "max_channel_count": self.max_channel_count,
            "layer_count": self.layer_count,
            "latent_size": self.latent_size,
            "dlatent_avg_beta": self.dlatent_avg_beta,
            "truncation_psi": self.truncation_psi,
            "truncation_cutoff": self.truncation_cutoff,
            "style_mixing_prob": self.style_mixing_prob,
            "mapping_layers": self.mapping_layers,
            "channels": self.channels,
            "generator": self.generator,
            "encoder": self.encoder,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FFHQALAEConfig:
        return cls(**dict(payload))

    @classmethod
    def from_official_yaml_payload(cls, payload: dict[str, Any]) -> FFHQALAEConfig:
        model_payload = dict(_OFFICIAL_MODEL_DEFAULTS)
        model_payload.update(dict(payload.get("MODEL", {})))
        return cls(
            start_channel_count=int(model_payload["START_CHANNEL_COUNT"]),
            max_channel_count=int(model_payload["MAX_CHANNEL_COUNT"]),
            layer_count=int(model_payload["LAYER_COUNT"]),
            latent_size=int(model_payload["LATENT_SPACE_SIZE"]),
            dlatent_avg_beta=(
                None
                if model_payload.get("DLATENT_AVG_BETA") is None
                else float(model_payload["DLATENT_AVG_BETA"])
            ),
            truncation_psi=(
                None if model_payload.get("TRUNCATIOM_PSI") is None else float(model_payload["TRUNCATIOM_PSI"])
            ),
            truncation_cutoff=(
                None
                if model_payload.get("TRUNCATIOM_CUTOFF") is None
                else int(model_payload["TRUNCATIOM_CUTOFF"])
            ),
            style_mixing_prob=(
                None
                if model_payload.get("STYLE_MIXING_PROB") is None
                else float(model_payload["STYLE_MIXING_PROB"])
            ),
            mapping_layers=int(model_payload["MAPPING_LAYERS"]),
            channels=int(model_payload["CHANNELS"]),
            generator=str(model_payload["GENERATOR"]),
            encoder=str(model_payload["ENCODER"]),
        )


class _EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        gain: float = float(np.sqrt(2.0)),
        lrmul: float = 1.0,
        implicit_lreq: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.std = 0.0
        self.gain = gain
        self.lrmul = lrmul
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.std = self.gain / np.sqrt(self.in_features) * self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0.0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0.0, std=self.std / self.lrmul)
            setattr(self.weight, "lr_equalization_coef", self.std)
            if self.bias is not None:
                setattr(self.bias, "lr_equalization_coef", self.lrmul)
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.implicit_lreq:
            bias = None if self.bias is None else self.bias * self.lrmul
            return F.linear(inputs, self.weight * self.std, bias)
        return F.linear(inputs, self.weight, self.bias)


def _as_tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class _EqualizedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        gain: float = float(np.sqrt(2.0)),
        transpose: bool = False,
        transform_kernel: bool = False,
        lrmul: float = 1.0,
        implicit_lreq: bool = True,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _as_tuple(kernel_size)
        self.stride = _as_tuple(stride)
        self.padding = _as_tuple(padding)
        self.output_padding = _as_tuple(output_padding)
        self.dilation = _as_tuple(dilation)
        self.groups = groups
        self.gain = gain
        self.lrmul = lrmul
        self.transpose = transpose
        self.fan_in = int(np.prod(self.kernel_size)) * in_channels // groups
        self.transform_kernel = transform_kernel
        if transpose:
            self.weight = Parameter(torch.empty(in_channels, out_channels // groups, *self.kernel_size))
        else:
            self.weight = Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.std = 0.0
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.std = self.gain / np.sqrt(self.fan_in) * self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0.0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0.0, std=self.std / self.lrmul)
            setattr(self.weight, "lr_equalization_coef", self.std)
            if self.bias is not None:
                setattr(self.bias, "lr_equalization_coef", self.lrmul)
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.transform_kernel:
            weight = F.pad(weight, (1, 1, 1, 1), mode="constant")
            if self.transpose:
                weight = (
                    weight[:, :, 1:, 1:]
                    + weight[:, :, :-1, 1:]
                    + weight[:, :, 1:, :-1]
                    + weight[:, :, :-1, :-1]
                )
            else:
                weight = (
                    weight[:, :, 1:, 1:]
                    + weight[:, :, :-1, 1:]
                    + weight[:, :, 1:, :-1]
                    + weight[:, :, :-1, :-1]
                ) * 0.25
        bias = self.bias
        if not self.implicit_lreq:
            bias = None if bias is None else bias * self.lrmul
            weight = weight * self.std
        if self.transpose:
            return F.conv_transpose2d(
                inputs,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return F.conv2d(
            inputs,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class _EqualizedConvTranspose2d(_EqualizedConv2d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["transpose"] = True
        super().__init__(*args, **kwargs)


def _pixel_norm(inputs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return inputs * torch.rsqrt(torch.mean(inputs.pow(2.0), dim=1, keepdim=True) + epsilon)


def _style_mod(x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1)


def _upscale2d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x = torch.reshape(x, [batch, channels, height, 1, width, 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, [batch, channels, height * factor, width * factor])


def _downscale2d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    return F.avg_pool2d(x, factor, factor)


class _Blur(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = np.array([1, 2, 1], dtype=np.float32)
        kernel = kernel[:, np.newaxis] * kernel[np.newaxis, :]
        kernel /= np.sum(kernel)
        self.register_buffer("weight", torch.tensor(kernel).view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class _EncodeBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, latent_size: int, *, fused_scale: bool = True) -> None:
        super().__init__()
        self.conv_1 = _EqualizedConv2d(inputs, inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.empty(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False)
        self.blur = _Blur(inputs)
        if fused_scale:
            self.conv_2 = _EqualizedConv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            self.fused_scale = True
        else:
            self.conv_2 = _EqualizedConv2d(inputs, outputs, 3, 1, 1, bias=False)
            self.fused_scale = False
        self.bias_2 = nn.Parameter(torch.empty(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)
        self.style_1 = _EqualizedLinear(2 * inputs, latent_size)
        self.style_2 = _EqualizedLinear(2 * outputs, latent_size)
        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - mean) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((mean, std), dim=1)
        x = self.instance_norm_1(x)
        x = self.conv_2(self.blur(x))
        if not self.fused_scale:
            x = _downscale2d(x)
        x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - mean) ** 2, dim=[2, 3], keepdim=True))
        style_2 = torch.cat((mean, std), dim=1)
        x = self.instance_norm_2(x)
        w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
        w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1]))
        return x, w1, w2


class _DecodeBlock(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        latent_size: int,
        *,
        has_first_conv: bool = True,
        fused_scale: bool = True,
        layer: int = 0,
    ) -> None:
        super().__init__()
        self.has_first_conv = has_first_conv
        self.fused_scale = fused_scale
        if has_first_conv:
            if fused_scale:
                self.conv_1 = _EqualizedConvTranspose2d(
                    inputs,
                    outputs,
                    3,
                    2,
                    1,
                    bias=False,
                    transform_kernel=True,
                )
            else:
                self.conv_1 = _EqualizedConv2d(inputs, outputs, 3, 1, 1, bias=False)
        self.blur = _Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.zeros(1, outputs, 1, 1))
        self.bias_1 = nn.Parameter(torch.zeros(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_1 = _EqualizedLinear(latent_size, 2 * outputs, gain=1.0)
        self.conv_2 = _EqualizedConv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.noise_weight_2 = nn.Parameter(torch.zeros(1, outputs, 1, 1))
        self.bias_2 = nn.Parameter(torch.zeros(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_2 = _EqualizedLinear(latent_size, 2 * outputs, gain=1.0)
        self.layer = layer

    def _inject_noise(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        noise: bool | str,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if noise:
            if noise == "batch_constant":
                rand = torch.randn(
                    1,
                    1,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                    generator=generator,
                )
            else:
                rand = torch.randn(
                    x.shape[0],
                    1,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                    generator=generator,
                )
            return torch.addcmul(x, value=1.0, tensor1=weight, tensor2=rand)
        scale = math.pow(self.layer + 1, 0.5)
        return x + scale * torch.exp(-x * x / (2.0 * scale * scale)) / math.sqrt(2 * math.pi) * 0.8

    def forward(
        self,
        x: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        noise: bool | str,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.has_first_conv:
            if not self.fused_scale:
                x = _upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)
        x = self._inject_noise(x, self.noise_weight_1, noise, generator=generator)
        x = x + self.bias_1
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_1(x)
        x = _style_mod(x, self.style_1(s1))
        x = self.conv_2(x)
        x = self._inject_noise(x, self.noise_weight_2, noise, generator=generator)
        x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)
        return _style_mod(x, self.style_2(s2))


class _FromRGB(nn.Module):
    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__()
        self.from_rgb = _EqualizedConv2d(channels, outputs, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.from_rgb(x), 0.2)


class _ToRGB(nn.Module):
    def __init__(self, inputs: int, channels: int) -> None:
        super().__init__()
        self.to_rgb = _EqualizedConv2d(inputs, channels, 1, 1, 0, gain=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_rgb(x)


class _EncoderDefault(nn.Module):
    def __init__(self, startf: int, maxf: int, layer_count: int, latent_size: int, channels: int = 3) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.latent_size = latent_size
        self.from_rgb: nn.ModuleList[_FromRGB] = nn.ModuleList()
        self.encode_block: nn.ModuleList[_EncodeBlock] = nn.ModuleList()
        mul = 2
        inputs = startf
        resolution = 2 ** (layer_count + 1)
        for _ in range(layer_count):
            outputs = min(maxf, startf * mul)
            self.from_rgb.append(_FromRGB(channels, inputs))
            fused_scale = resolution >= 128
            self.encode_block.append(
                _EncodeBlock(inputs, outputs, latent_size, fused_scale=fused_scale)
            )
            resolution //= 2
            inputs = outputs
            mul *= 2

    def encode(self, x: torch.Tensor, lod: int) -> torch.Tensor:
        styles = torch.zeros(x.shape[0], 1, self.latent_size, device=x.device, dtype=x.dtype)
        x = self.from_rgb[self.layer_count - lod - 1](x)
        for index in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[index](x)
            styles[:, 0] += s1 + s2
        return styles

    def encode2(self, x: torch.Tensor, lod: int, blend: float) -> torch.Tensor:
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size, device=x.device, dtype=x.dtype)
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend
        x_prev = F.avg_pool2d(x_orig, 2, 2)
        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x = torch.lerp(x_prev, x, blend)
        for index in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[index](x)
            styles[:, 0] += s1 + s2
        return styles

    def forward(self, x: torch.Tensor, lod: int, blend: float) -> torch.Tensor:
        if blend == 1:
            return self.encode(x, lod)
        return self.encode2(x, lod, blend)


class _GeneratorDefault(nn.Module):
    def __init__(self, startf: int, maxf: int, layer_count: int, latent_size: int, channels: int = 3) -> None:
        super().__init__()
        self.layer_count = layer_count
        mul = 2 ** (layer_count - 1)
        inputs = min(maxf, startf * mul)
        self.const = Parameter(torch.empty(1, inputs, 4, 4))
        init.ones_(self.const)
        self.layer_to_resolution = [0 for _ in range(layer_count)]
        self.decode_block: nn.ModuleList[_DecodeBlock] = nn.ModuleList()
        self.to_rgb: nn.ModuleList[_ToRGB] = nn.ModuleList()
        resolution = 2
        for index in range(layer_count):
            outputs = min(maxf, startf * mul)
            has_first_conv = index != 0
            fused_scale = resolution * 2 >= 128
            self.decode_block.append(
                _DecodeBlock(
                    inputs,
                    outputs,
                    latent_size,
                    has_first_conv=has_first_conv,
                    fused_scale=fused_scale,
                    layer=index,
                )
            )
            resolution *= 2
            self.layer_to_resolution[index] = resolution
            self.to_rgb.append(_ToRGB(outputs, channels))
            inputs = outputs
            mul //= 2

    def decode(
        self,
        styles: torch.Tensor,
        lod: int,
        noise: bool | str,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = self.const
        for index in range(lod + 1):
            x = self.decode_block[index](
                x,
                styles[:, 2 * index + 0],
                styles[:, 2 * index + 1],
                noise,
                generator=generator,
            )
        return self.to_rgb[lod](x)

    def decode2(
        self,
        styles: torch.Tensor,
        lod: int,
        blend: float,
        noise: bool | str,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = self.const
        for index in range(lod):
            x = self.decode_block[index](
                x,
                styles[:, 2 * index + 0],
                styles[:, 2 * index + 1],
                noise,
                generator=generator,
            )
        x_prev = self.to_rgb[lod - 1](x)
        x = self.decode_block[lod](
            x,
            styles[:, 2 * lod + 0],
            styles[:, 2 * lod + 1],
            noise,
            generator=generator,
        )
        x = self.to_rgb[lod](x)
        x_prev = F.interpolate(x_prev, size=self.layer_to_resolution[lod])
        return torch.lerp(x_prev, x, blend)

    def forward(
        self,
        styles: torch.Tensor,
        lod: int,
        blend: float,
        noise: bool | str,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if blend == 1:
            return self.decode(styles, lod, noise, generator=generator)
        return self.decode2(styles, lod, blend, noise, generator=generator)


class _MappingBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, *, lrmul: float) -> None:
        super().__init__()
        self.fc = _EqualizedLinear(inputs, outputs, lrmul=lrmul)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.fc(x), 0.2)


class _MappingD(nn.Module):
    def __init__(self, *, mapping_layers: int, latent_size: int, dlatent_size: int, mapping_fmaps: int) -> None:
        super().__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[_EqualizedLinear] = nn.ModuleList()
        for index in range(mapping_layers):
            outputs = 2 * dlatent_size if index == mapping_layers - 1 else mapping_fmaps
            self.map_blocks.append(_EqualizedLinear(inputs, outputs, lrmul=0.1))
            inputs = outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.map_blocks:
            x = block(x)
        return x[:, 0, x.shape[2] // 2]


class _MappingF(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        mapping_layers: int,
        latent_size: int,
        dlatent_size: int,
        mapping_fmaps: int,
    ) -> None:
        super().__init__()
        inputs = dlatent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        self.map_blocks: nn.ModuleList[_MappingBlock] = nn.ModuleList()
        for index in range(mapping_layers):
            outputs = latent_size if index == mapping_layers - 1 else mapping_fmaps
            self.map_blocks.append(_MappingBlock(inputs, outputs, lrmul=0.1))
            inputs = outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pixel_norm(x)
        for block in self.map_blocks:
            x = block(x)
        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)


class _DLatent(nn.Module):
    def __init__(self, dlatent_size: int, layer_count: int) -> None:
        super().__init__()
        self.register_buffer("buff", torch.zeros(layer_count, dlatent_size, dtype=torch.float32))


class FFHQALAE(nn.Module):
    def __init__(self, config: FFHQALAEConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config if config is not None else FFHQALAEConfig(**kwargs)
        cfg = self.config
        if cfg.generator != "GeneratorDefault" or cfg.encoder != "EncoderDefault":
            raise ValueError("Only GeneratorDefault/EncoderDefault are supported by the internal FFHQALAE runtime")
        self.mapping_d = _MappingD(
            mapping_layers=3,
            latent_size=cfg.latent_size,
            dlatent_size=cfg.latent_size,
            mapping_fmaps=cfg.latent_size,
        )
        self.mapping_f = _MappingF(
            num_layers=cfg.num_style_layers,
            mapping_layers=cfg.mapping_layers,
            latent_size=cfg.latent_size,
            dlatent_size=cfg.latent_size,
            mapping_fmaps=cfg.latent_size,
        )
        self.decoder = _GeneratorDefault(
            startf=cfg.start_channel_count,
            layer_count=cfg.layer_count,
            maxf=cfg.max_channel_count,
            latent_size=cfg.latent_size,
            channels=cfg.channels,
        )
        self.encoder = _EncoderDefault(
            startf=cfg.start_channel_count,
            layer_count=cfg.layer_count,
            maxf=cfg.max_channel_count,
            latent_size=cfg.latent_size,
            channels=cfg.channels,
        )
        self.dlatent_avg = _DLatent(cfg.latent_size, self.mapping_f.num_layers)
        self.layer_count = cfg.layer_count
        self.latent_size = cfg.latent_size

    @property
    def image_size(self) -> int:
        return self.config.image_size

    def _styles_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 2:
            return latents[:, None, :].repeat(1, self.mapping_f.num_layers, 1)
        if latents.ndim == 3:
            if latents.shape[1] != self.mapping_f.num_layers:
                raise ValueError(
                    f"Expected {self.mapping_f.num_layers} style layers, received {latents.shape[1]}"
                )
            return latents
        raise ValueError(f"Expected latents with rank 2 or 3, received shape={tuple(latents.shape)}")

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        styles = self.encoder(images, self.layer_count - 1, 1.0)
        return styles[:, 0]

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        *,
        noise: bool | str = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.decoder(
            self._styles_from_latents(latents),
            self.layer_count - 1,
            1.0,
            noise,
            generator=generator,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))

    def project_state_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "mapping_d": self.mapping_d.state_dict(),
            "mapping_f": self.mapping_f.state_dict(),
            "dlatent_avg": self.dlatent_avg.state_dict(),
        }

    def load_project_state_dict(self, state: dict[str, Any]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.mapping_d.load_state_dict(state["mapping_d"])
        self.mapping_f.load_state_dict(state["mapping_f"])
        self.dlatent_avg.load_state_dict(state["dlatent_avg"])

    def to_project_checkpoint(self, *, source: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format": _PROJECT_CKPT_FORMAT,
            "config": self.config.to_dict(),
            "state": self.project_state_dict(),
            "source": {} if source is None else dict(source),
        }

    def save_project_checkpoint(self, path: str | Path, *, source: dict[str, Any] | None = None) -> Path:
        checkpoint_path = normalize_path(path, resolve_latest=False)
        save_checkpoint(checkpoint_path, self.to_project_checkpoint(source=source))
        return checkpoint_path

    @classmethod
    def from_official_checkpoint_payload(
        cls,
        payload: dict[str, Any],
        *,
        config: FFHQALAEConfig,
    ) -> FFHQALAE:
        model = cls(config=config)
        models = dict(payload.get("models", payload))
        model.decoder.load_state_dict(models["generator_s"])
        model.encoder.load_state_dict(models["discriminator_s"])
        model.mapping_d.load_state_dict(models["mapping_tl_s"])
        model.mapping_f.load_state_dict(models["mapping_fl_s"])
        model.dlatent_avg.load_state_dict(models["dlatent_avg"])
        return model

    @classmethod
    def load_project_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> FFHQALAE:
        checkpoint_path = resolve_latest_alias(normalize_path(path, resolve_latest=False))
        checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
        if checkpoint.get("format") != _PROJECT_CKPT_FORMAT:
            raise ValueError(f"Unsupported FFHQ ALAE checkpoint format: {checkpoint.get('format')!r}")
        model = cls(config=FFHQALAEConfig.from_dict(dict(checkpoint["config"])))
        model.load_project_state_dict(dict(checkpoint["state"]))
        return model


def load_official_alae_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    module_root: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = resolve_latest_alias(normalize_path(path, resolve_latest=False))
    with _temporary_module_root(module_root):
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


__all__ = [
    "FFHQALAE",
    "FFHQALAEConfig",
    "load_official_alae_checkpoint",
]
