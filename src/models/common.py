"""Shared components adapted from the project's tri-planar MRI training notebooks.

These modules illustrate the model architecture. They do not embed pretrained weights
or imply compatibility with every historical checkpoint revision.
"""
from __future__ import annotations

import torch
from torch import nn

TARGETS = (
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture',
)
PLANES = ('axial', 'coronal', 'sagittal')
SLICES = 16


class GeM(nn.Module):
    """Learnable generalized mean over a CNN feature map, as used by NFNet."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([p], dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float().clamp(min=self.eps)
        p = self.p.float().clamp(min=1., max=6.)
        pooled = xf.pow(p).mean(dim=(-2, -1)).clamp(min=self.eps)
        return pooled.pow(1. / p).to(dtype=x.dtype)


class SliceTransformer(nn.Module):
    """2-layer pre-norm slice Transformer with optional gated focal pooling.

    All-missing slices are handled without NaN or artificial gradient signal.
    """
    def __init__(self, dim: int, num_slices: int = SLICES, heads: int = 4,
                 dropout: float = 0.1, focal: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError('feature size must be divisible by attention heads')
        self.pos_embed = nn.Parameter(torch.randn(1, num_slices, dim) * .02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout,
                                            activation='gelu', batch_first=True,
                                            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.focal = bool(focal)
        if focal:
            self.slice_score = nn.Sequential(nn.Linear(dim, max(1, dim // 4)),
                                             nn.Tanh(), nn.Linear(max(1, dim // 4), 1))
            self.focal_gate = nn.Parameter(torch.full((dim,), .10))

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or valid.shape != features.shape[:2]:
            raise ValueError('expected features [batch,slices,channels], mask [batch,slices]')
        if features.shape[1] != self.pos_embed.shape[1]:
            raise ValueError('incorrect number of slices')
        valid = valid.bool()
        safe = valid.clone()
        empty = ~safe.any(dim=1)
        safe[empty, 0] = True
        x = self.norm(self.encoder(features + self.pos_embed.to(dtype=features.dtype),
                                   src_key_padding_mask=~safe))
        w = valid.unsqueeze(-1).to(dtype=x.dtype)
        average = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)
        if not self.focal:
            return average
        scores = self.slice_score(x).squeeze(-1).masked_fill(~safe, -1e4)
        attn = scores.softmax(dim=1).unsqueeze(-1) * w
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        focused = (x * attn).sum(dim=1)
        return average + self.focal_gate.to(dtype=x.dtype) * focused


def pooled_valid_slices(volume: torch.Tensor, encode, channels: int,
                        chunk_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Only forward nonzero slices; invalid slices get zero features."""
    if volume.ndim != 4 or volume.shape[1] != SLICES:
        raise ValueError('each plane must be [batch,16,height,width]')
    batch, slices, h, w = volume.shape
    valid = volume.abs().flatten(2).sum(-1) > 0
    idx = torch.nonzero(valid.flatten(), as_tuple=True)[0]
    flat = volume.reshape(batch * slices, 1, h, w)
    if len(idx):
        chosen = flat.index_select(0, idx)
        blocks = [encode(chosen[i:i + chunk_size])
                  for i in range(0, len(idx), chunk_size)]
        out = torch.cat(blocks, dim=0)
        if out.shape != (len(idx), channels):
            raise ValueError('slice backbone returned invalid feature dimensions')
        dense = out.new_zeros((batch * slices, channels)).index_copy(0, idx, out)
    else:
        dense = flat.new_zeros((batch * slices, channels))
    return dense.reshape(batch, slices, channels), valid


class ThreePlaneFusion(nn.Module):
    """Concatenate embeddings and observed-plane mask for the 12-label head."""
    def __init__(self, emb_dim: int = 256, dropout: float = .3):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(3 * emb_dim + 3, emb_dim),
                                  nn.LayerNorm(emb_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(emb_dim, len(TARGETS)))

    def forward(self, embeddings: list[torch.Tensor], presence: torch.Tensor) -> torch.Tensor:
        if len(embeddings) != 3 or presence.ndim != 2 or presence.shape[1] != 3:
            raise ValueError('need 3 plane embeddings and [batch,3] presence')
        fused = torch.cat([*embeddings, presence.to(dtype=embeddings[0].dtype)], dim=1)
        return self.head(fused)
