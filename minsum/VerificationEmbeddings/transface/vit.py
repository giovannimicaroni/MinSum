"""Minimal TransFace Vision Transformer backbone.

This mirrors the official TransFace inference backbone closely enough for
checkpoint loading while avoiding an extra runtime dependency on timm.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU6,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class VITBatchNorm(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_token,
            3,
            self.num_heads,
            embed_dim // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_patches: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.ReLU6,
        norm_layer: str = "ln",
    ) -> None:
        super().__init__()
        if norm_layer == "bn":
            self.norm1 = VITBatchNorm(num_features=num_patches)
            self.norm2 = VITBatchNorm(num_features=num_patches)
        elif norm_layer == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}")
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 9,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height != self.img_size[0] or width != self.img_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) does not match "
                f"model ({self.img_size[0]}*{self.img_size[1]})."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 9,
        in_channels: int = 3,
        num_classes: int = 512,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: str = "ln",
        mask_ratio: float = 0.0,
        using_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_patches=num_patches,
                )
                for i in range(depth)
            ]
        )
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == "bn":
            self.norm = VITBatchNorm(self.num_patches)
        else:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}")
        self.feature = nn.Sequential(
            nn.Linear(embed_dim * num_patches, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, eps=2e-5),
            nn.Linear(embed_dim, num_classes, bias=False),
            nn.BatchNorm1d(num_classes, eps=2e-5),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.senet = nn.Sequential(
            nn.Linear(embed_dim * num_patches, num_patches, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_patches, num_patches, bias=False),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length, dim = x.size()
        len_keep = int(length * (1 - mask_ratio))
        noise = torch.rand(batch_size, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        mask = torch.ones([batch_size, length], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        if self.training and self.mask_ratio > 0:
            x, _, ids_restore = self.random_masking(x, self.mask_ratio)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.float())
        if self.training and self.mask_ratio > 0:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(
                x_,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
            )
        original = x
        flat = torch.reshape(x, (batch_size, self.num_patches * self.embed_dim))
        patch_weights = self.senet(flat)
        patch_weights = torch.reshape(patch_weights, (batch_size, self.num_patches, 1))
        weighted = patch_weights * original
        return torch.reshape(weighted, (batch_size, self.num_patches * self.embed_dim)), patch_weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, patch_weights = self.forward_features(x)
        out_x = torch.reshape(x, (x.shape[0], self.num_patches, self.embed_dim))
        patch_entropy = torch.std(out_x, dim=2)
        features = self.feature(x)
        return features, patch_weights, patch_entropy


def build_transface(name: str, num_features: int = 512) -> VisionTransformer:
    configs = {
        "vit_s_dp005_mask_0": dict(embed_dim=512, depth=12, drop_path_rate=0.05, mask_ratio=0.0),
        "vit_b_dp005_mask_005": dict(embed_dim=512, depth=24, drop_path_rate=0.05, mask_ratio=0.05),
        "vit_l_dp005_mask_005": dict(embed_dim=768, depth=24, drop_path_rate=0.05, mask_ratio=0.05),
        "vit_s": dict(embed_dim=512, depth=12, drop_path_rate=0.1, mask_ratio=0.1),
        "vit_b": dict(embed_dim=512, depth=24, drop_path_rate=0.1, mask_ratio=0.1),
    }
    if name not in configs:
        raise ValueError(f"Unsupported TransFace network: {name}")
    return VisionTransformer(
        img_size=112,
        patch_size=9,
        num_classes=num_features,
        num_heads=8,
        norm_layer="ln",
        **configs[name],
    )
