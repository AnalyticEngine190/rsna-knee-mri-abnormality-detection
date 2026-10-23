"""Extracted tri-planar CNN model components from the author's ConvNeXt training code.

The architecture is provided for research/portfolio inspection; it requires
PyTorch and TIMM and does not reproduce the complete training environment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm
from torch.utils.checkpoint import checkpoint

BACKBONE = "convnext_small"
EXPECTED_SHAPE = (16, 288, 288)
SLICE_BATCH_SIZE = 64
FREEZE_BN_STATS = True
PRETRAINED = False  # Offline-safe default; supply weights separately.
PLANES = ("axial", "coronal", "sagittal")
TARGET_COLS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float().clamp(min=self.eps)
        p_fp32 = self.p.float().clamp(min=1.0, max=6.0)
        z = x_fp32.pow(p_fp32).mean(dim=(-2, -1)).clamp(min=self.eps)
        return z.pow(1.0 / p_fp32).to(x.dtype)


class SliceTransformer(nn.Module):
    def __init__(self, dim: int, num_slices: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Transformer dim={dim} must be divisible by num_heads={num_heads}")
        self.num_slices = num_slices
        self.pos_embed = nn.Parameter(torch.randn(1, num_slices, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, slice_valid: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed.to(dtype=x.dtype)
        safe_valid = slice_valid.clone()
        empty_rows = ~safe_valid.any(dim=1)
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True
        encoded = self.encoder(x, src_key_padding_mask=~safe_valid)
        encoded = self.norm(encoded)
        weights = slice_valid.unsqueeze(-1).to(dtype=encoded.dtype)
        numerator = (encoded * weights).sum(dim=1)
        denominator = weights.sum(dim=1).clamp(min=1.0)
        return numerator / denominator


# Run this cell to update PlaneEncoder & TriPlanarModel in memory for ConvNeXt
# Re-run this cell so TriPlanarModel defaults to spatial mean pooling for ConvNeXt
class PlaneEncoder(nn.Module):
    def __init__(
        self,
        backbone_name=BACKBONE,
        emb_dim=256,
        num_slices=EXPECTED_SHAPE[0],
        slice_batch_size=SLICE_BATCH_SIZE,
        gradient_checkpointing=True,
        freeze_bn_stats=FREEZE_BN_STATS,
        drop_path_rate=0.1,
        pretrained=PRETRAINED,
    ):
        super().__init__()
        self.slice_batch_size = int(slice_batch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=1,
            drop_path_rate=drop_path_rate,
        )
        self.feat_dim = int(self.backbone.num_features)

        # ConvNeXt uses GELU/LayerNorm (signed features), so Phase 2 trained with mean pooling.
        # ResNet50d uses ReLU (>=0 features), so it used GeM pooling.
        self.use_gem = not str(backbone_name).lower().startswith("convnext")
        self.gem = GeM() if self.use_gem else None

        self.slice_encoder = SliceTransformer(self.feat_dim, num_slices=num_slices)
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def _backbone_forward(self, image_chunk: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled() and self.gradient_checkpointing:
            feature_map = checkpoint(self.backbone, image_chunk, use_reentrant=False)
        else:
            feature_map = self.backbone(image_chunk)

        if self.gem is not None:
            return self.gem(feature_map)
        return feature_map.float().mean(dim=(-2, -1)).to(feature_map.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, slices, height, width = x.shape
        slice_valid = x.abs().flatten(2).sum(dim=2) > 0
        valid_idx = torch.nonzero(slice_valid.reshape(-1), as_tuple=True)[0]
        flat_images = x.reshape(batch * slices, 1, height, width)
        valid_images = flat_images.index_select(0, valid_idx)
        min_chunk = 2 if (self.training and not self.freeze_bn_stats) else 1
        pooled_chunks = [
            self._backbone_forward(valid_images[start:end])
            for start, end in iter_chunk_bounds(int(valid_images.shape[0]), self.slice_batch_size, min_chunk)
        ]
        pooled_valid = torch.cat(pooled_chunks, dim=0) if pooled_chunks else x.new_zeros((0, self.feat_dim))
        flat_features = pooled_valid.new_zeros((batch * slices, self.feat_dim))
        if pooled_valid.numel() > 0:
            flat_features = flat_features.index_copy(0, valid_idx, pooled_valid)
        features = self.slice_encoder(
            flat_features.reshape(batch, slices, self.feat_dim),
            slice_valid,
        )
        return self.proj(features)

# Notebook status print removed from reusable module.


class TriPlanarModel(nn.Module):
    def __init__(
        self,
        backbone_name=BACKBONE,
        emb_dim=256,
        dropout=0.3,
        slice_batch_size=SLICE_BATCH_SIZE,
        pretrained=PRETRAINED,
        freeze_bn_stats=FREEZE_BN_STATS,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoders = nn.ModuleDict({
            plane: PlaneEncoder(
                backbone_name=backbone_name,
                emb_dim=emb_dim,
                slice_batch_size=slice_batch_size,
                pretrained=pretrained,
                freeze_bn_stats=freeze_bn_stats,
            )
            for plane in PLANES
        })
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 3 + len(PLANES), emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, len(TARGET_COLS)),
        )

    def forward(self, ax, cor, sag, presence):
        emb_list = []
        for idx, (plane, vol) in enumerate(zip(PLANES, (ax, cor, sag))):
            valid_idx = torch.nonzero(presence[:, idx] > 0.5, as_tuple=True)[0]
            if valid_idx.numel() == 0:
                emb_list.append(vol.new_zeros((vol.shape[0], self.emb_dim)))
                continue
            valid_emb = self.encoders[plane](vol.index_select(0, valid_idx))
            emb = valid_emb.new_zeros((vol.shape[0], self.emb_dim))
            emb = emb.index_copy(0, valid_idx, valid_emb)
            emb_list.append(emb)
        fused = torch.cat([*emb_list, presence.to(dtype=emb_list[0].dtype)], dim=1)
        return self.head(fused)


def iter_chunk_bounds(n: int, chunk_size: int, min_chunk: int):
    """Yield [start, end) chunks. A leftover smaller than min_chunk is absorbed."""
    if n <= 0:
        return
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        remaining = n - end
        if 0 < remaining < min_chunk:
            end = n
        yield start, end
        start = end

