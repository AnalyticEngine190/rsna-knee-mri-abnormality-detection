"""One-channel, GeM-pooled tri-planar NFNet model based on the RSNA notebooks.

No pretrained weights are redistributed. TIMM is required for a real NFNet run;
pass backbone_factory in synthetic tests to avoid downloading model weights.
"""
from __future__ import annotations

from collections.abc import Callable
import torch
from torch import nn
from .common import GeM, SliceTransformer, ThreePlaneFusion, PLANES, pooled_valid_slices


def _timm_nfnet(pretrained: bool) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError('Install timm to create the NFNet backbone') from exc
    return timm.create_model('eca_nfnet_l0', pretrained=pretrained, in_chans=1,
                             num_classes=0, global_pool='')


class NFNetPlaneEncoder(nn.Module):
    def __init__(self, *, pretrained: bool = False,
                 backbone_factory: Callable[[], nn.Module] | None = None,
                 embedding_dim: int = 256, chunk_size: int = 4,
                 freeze_bn_stats: bool = True):
        super().__init__()
        self.backbone = backbone_factory() if backbone_factory else _timm_nfnet(pretrained)
        self.feat_dim = int(self.backbone.num_features)
        self.pool = GeM(p=3., eps=1e-6)
        self.slice_encoder = SliceTransformer(self.feat_dim, heads=4)
        self.proj = nn.Sequential(nn.Linear(self.feat_dim, embedding_dim),
                                  nn.LayerNorm(embedding_dim), nn.GELU())
        self.chunk_size = int(chunk_size)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be positive')

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for mod in self.backbone.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        return self

    def _extract(self, one_channel: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(one_channel)
        if feature_map.ndim != 4:
            raise ValueError('NFNet must return [N,C,H,W] features')
        return self.pool(feature_map)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        feats, valid = pooled_valid_slices(volume, self._extract, self.feat_dim,
                                           self.chunk_size)
        return self.proj(self.slice_encoder(feats, valid))


class TriPlanarNFNet(nn.Module):
    """Pass (axial, coronal, sagittal, presence) to get 12 raw logits."""
    def __init__(self, *, pretrained: bool = False,
                 backbone_factory: Callable[[], nn.Module] | None = None,
                 embedding_dim: int = 256, chunk_size: int = 4):
        super().__init__()
        self.encoders = nn.ModuleDict({
            plane: NFNetPlaneEncoder(pretrained=pretrained,
                                     backbone_factory=backbone_factory,
                                     embedding_dim=embedding_dim, chunk_size=chunk_size)
            for plane in PLANES
        })
        self.embedding_dim = embedding_dim
        self.fusion = ThreePlaneFusion(embedding_dim)

    def forward(self, axial: torch.Tensor, coronal: torch.Tensor,
                sagittal: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for index, (name, plane) in enumerate(zip(PLANES, (axial, coronal, sagittal))):
            present_idx = torch.nonzero(presence[:, index] > .5, as_tuple=True)[0]
            if len(present_idx):
                sub = self.encoders[name](plane.index_select(0, present_idx))
                zeros = sub.new_zeros((plane.shape[0], self.embedding_dim))
                embeddings.append(zeros.index_copy(0, present_idx, sub))
            else:
                embeddings.append(plane.new_zeros((plane.shape[0], self.embedding_dim)))
        return self.fusion(embeddings, presence)
