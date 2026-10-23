"""DINOv2 tri-planar architecture: CLS/mean/focal tokens + anatomical priors.

Clean implementation adapted from the author's high-resolution ViT notebook.
This API is not guaranteed to load checkpoints from earlier script revisions.
"""
from __future__ import annotations

from collections.abc import Callable
import math
import torch
from torch import nn
from .common import PLANES, TARGETS, SliceTransformer, ThreePlaneFusion, pooled_valid_slices


PLANE_PRIORS = {
    'ACL': (0, 2), 'MCL': (0, 1),
    'Medial Meniscus': (1, 2), 'Lateral Meniscus': (1, 2),
    'Medial OA': (1, 2), 'Lateral OA': (1, 2),
    'PF OA': (0, 2), 'Effusion': (0, 2), 'Synovitis': (0, 2),
    "Baker's": (0, 2), 'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2),
}


def _timm_dinov2(pretrained: bool) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError('Install timm to create the DINOv2 backbone') from exc
    return timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=pretrained,
                             num_classes=0, in_chans=3, img_size=518,
                             dynamic_img_size=True, dynamic_img_pad=True)


class DinoPlaneEncoder(nn.Module):
    def __init__(self, *, pretrained: bool = False,
                 backbone_factory: Callable[[], nn.Module] | None = None,
                 embedding_dim: int = 256, chunk_size: int = 2,
                 unfreeze_last: int = 6):
        super().__init__()
        self.backbone = backbone_factory() if backbone_factory else _timm_dinov2(pretrained)
        self.feat_dim = int(self.backbone.num_features)
        self.prefix_tokens = int(getattr(self.backbone, 'num_prefix_tokens', 1))
        self.token_proj = nn.Sequential(nn.LayerNorm(3 * self.feat_dim),
                                        nn.Dropout(.2), nn.Linear(3 * self.feat_dim, self.feat_dim),
                                        nn.GELU())
        heads = next((h for h in (6, 8, 4, 12, 3, 2, 1) if self.feat_dim % h == 0), 1)
        self.slice_encoder = SliceTransformer(self.feat_dim, heads=heads, focal=True)
        self.proj = nn.Sequential(nn.Linear(self.feat_dim, embedding_dim),
                                  nn.LayerNorm(embedding_dim), nn.GELU())
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        self._freeze_prefix(unfreeze_last)

    def _freeze_prefix(self, unfreeze_last: int) -> None:
        blocks = getattr(self.backbone, 'blocks', None)
        if blocks is None:
            # Mock/flexible backbones can omit blocks; real TIMM ViTs have them.
            return
        n = len(blocks)
        if not 0 < unfreeze_last <= n:
            raise ValueError(f'unfreeze_last must be in [1,{n}]')
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in list(blocks)[n - unfreeze_last:]:
            for param in block.parameters():
                param.requires_grad = True
        final_norm = getattr(self.backbone, 'norm', None)
        if final_norm is not None:
            for param in final_norm.parameters():
                param.requires_grad = True

    def _extract(self, mono: torch.Tensor) -> torch.Tensor:
        clipped = 3. * torch.tanh(mono / 3.)
        rgb = clipped.expand(-1, 3, -1, -1).contiguous()
        tokens = self.backbone.forward_features(rgb)
        if tokens.ndim != 3 or tokens.shape[-1] != self.feat_dim:
            raise ValueError('expected DINOv2 forward_features -> [N,tokens,channels]')
        if self.prefix_tokens < 1 or tokens.shape[1] <= self.prefix_tokens:
            raise ValueError('DINOv2 model needs at least one prefix and patch token')
        cls = tokens[:, 0]
        patches = tokens[:, self.prefix_tokens:]
        avg = patches.mean(dim=1)
        topk = max(1, patches.shape[1] // 8)
        focal = patches.topk(topk, dim=1).values.mean(dim=1)
        return self.token_proj(torch.cat((cls, avg, focal), dim=-1))

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        feats, valid = pooled_valid_slices(volume, self._extract, self.feat_dim,
                                           self.chunk_size)
        return self.proj(self.slice_encoder(feats, valid))


class PlanePriorHead(nn.Module):
    """Per-label plane queries with the notebook's axial/coronal/sagittal priors."""
    def __init__(self, emb_dim: int = 256, dropout: float = .3,
                 prior_strength: float = .55):
        super().__init__()
        self.plane_emb = nn.Parameter(torch.randn(3, emb_dim) * .02)
        self.query = nn.Parameter(torch.randn(len(TARGETS), emb_dim) * .02)
        self.drop = nn.Dropout(dropout)
        self.out_weight = nn.Parameter(torch.randn(len(TARGETS), emb_dim) * (emb_dim ** -.5))
        self.out_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        m = torch.zeros(len(TARGETS), 3)
        for k, t in enumerate(TARGETS):
            for i in PLANE_PRIORS[t]:
                m[k, i] = float(prior_strength)
        self.register_buffer('plane_prior', m)
        self.emb_dim = emb_dim

    def forward(self, emb: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        h = emb + self.plane_emb.unsqueeze(0)
        att = torch.einsum('bph,oh->bop', h, self.query) / math.sqrt(self.emb_dim)
        att = att + self.plane_prior.unsqueeze(0)
        absent = (presence[:, None, :] < .5)
        empty = absent.all(dim=-1, keepdim=True)
        masked = absent & ~empty
        att = att.masked_fill(masked, -1e4).softmax(dim=-1)
        ctx = self.drop(torch.einsum('bop,bph->boh', att, h))
        return (ctx * self.out_weight.unsqueeze(0)).sum(-1) + self.out_bias


class TriPlanarDINOv2(nn.Module):
    def __init__(self, *, pretrained: bool = False,
                 backbone_factory: Callable[[], nn.Module] | None = None,
                 embedding_dim: int = 256, chunk_size: int = 2,
                 unfreeze_last: int = 6):
        super().__init__()
        self.encoders = nn.ModuleDict({plane: DinoPlaneEncoder(
            pretrained=pretrained, backbone_factory=backbone_factory,
            embedding_dim=embedding_dim, chunk_size=chunk_size,
            unfreeze_last=unfreeze_last) for plane in PLANES})
        self.embedding_dim = embedding_dim
        self.fusion = ThreePlaneFusion(embedding_dim)
        self.prior_head = PlanePriorHead(embedding_dim)
        self.prior_gate = nn.Parameter(torch.full((len(TARGETS),), .10))

    def forward(self, axial: torch.Tensor, coronal: torch.Tensor,
                sagittal: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for index, (name, plane) in enumerate(zip(PLANES, (axial, coronal, sagittal))):
            chosen = torch.nonzero(presence[:, index] > .5, as_tuple=True)[0]
            if len(chosen):
                sub = self.encoders[name](plane.index_select(0, chosen))
                zeros = sub.new_zeros((plane.shape[0], self.embedding_dim))
                embeddings.append(zeros.index_copy(0, chosen, sub))
            else:
                embeddings.append(plane.new_zeros((plane.shape[0], self.embedding_dim)))
        base = self.fusion(embeddings, presence)
        priors = self.prior_head(torch.stack(embeddings, dim=1), presence)
        return base + self.prior_gate * priors
