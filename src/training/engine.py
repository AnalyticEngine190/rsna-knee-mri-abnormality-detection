"""Minimal, data-agnostic training/evaluation helpers for 12-label knee MRI models."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from src.models.common import TARGETS


def masked_weighted_bce(logits, labels, weights, pos_weight=None):
    """BCE for missing / soft labels, sample weights, and optional positive weights."""
    if logits.shape != labels.shape or logits.ndim != 2:
        raise ValueError('logits and targets should both be [batch,targets]')
    if weights.ndim != 1 or weights.shape[0] != logits.shape[0]:
        raise ValueError('sample weights must be [batch]')
    valid = torch.isfinite(labels)
    if not valid.any():
        raise ValueError('no finite labels')
    values = torch.where(valid, labels, torch.zeros_like(labels))
    if ((values < 0) | (values > 1)).any():
        raise ValueError('soft targets must be in [0,1]')
    w = weights.to(device=logits.device, dtype=torch.float32)
    if not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError('sample weights must be positive finite')
    pw = None if pos_weight is None else pos_weight.to(device=logits.device, dtype=torch.float32)
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        loss = F.binary_cross_entropy_with_logits(logits.float(), values.float(),
                                                  pos_weight=pw, reduction='none')
    indicator = valid.float() * w[:, None]
    return (loss * indicator).sum() / indicator.sum().clamp(min=1)


def macro_roc_auc(scores: np.ndarray, labels: np.ndarray):
    """Return target-wise AUCs and macro AUC; skips one-class or missing targets."""
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.shape != labels.shape or scores.ndim != 2 or scores.shape[1] != len(TARGETS):
        raise ValueError('scores and labels must have shape [studies,12]')
    measured = {}
    for j, t in enumerate(TARGETS):
        good = np.isfinite(scores[:, j]) & np.isfinite(labels[:, j])
        gold = labels[good, j]
        if gold.size >= 2 and set(np.unique(gold)) == {0, 1}:
            measured[t] = float(roc_auc_score(gold, scores[good, j]))
    return (float(np.mean(list(measured.values()))) if measured else float('nan'), measured)


def train_epoch(model, loader, optimizer, *, device='cpu', pos_weight=None):
    model.train()
    total, count = 0., 0
    for batch in loader:
        ax, cor, sag, presence, target, weight = [x.to(device) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        logits = model(ax, cor, sag, presence)
        loss = masked_weighted_bce(logits, target, weight, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        total += float(loss.detach().item()) * len(ax)
        count += len(ax)
    if count == 0:
        raise ValueError('training dataset is empty')
    return total / count


@torch.no_grad()
def validate(model, loader, *, device='cpu'):
    model.eval()
    pred, true = [], []
    for batch in loader:
        ax, cor, sag, presence, target, weight = [x.to(device) for x in batch]
        out = model(ax, cor, sag, presence)
        pred.append(out.float().cpu().numpy())
        true.append(target.float().cpu().numpy())
    if not pred:
        raise ValueError('validation dataset is empty')
    return macro_roc_auc(np.concatenate(pred), np.concatenate(true))
