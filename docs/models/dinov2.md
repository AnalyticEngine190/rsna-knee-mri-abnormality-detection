# DINOv2 ViT-S/14 triplanar branch

**Origin:** Curated reimplementation of the user's `dinov2_training(1).py` and high-resolution fold-recovery experiments. This is not a pretrained DINOv2 release, a full reproduction of the 0.957 ensemble, or an exact checkpoint-compatible archive.

- **Input:** each plane is a tensor of 16 sampled grayscale MRI slices. Soft-clip foreground z-scores with `3*tanh(x/3)`; replicate grayscale to 3 channels for ViT-S/14.
- **Feature extraction:** use `timm`'s dynamic-image-size DINOv2 model, with explicit ViT patch tokens. Pool the CLS token, average patch tokens, and per-channel top-k focal patch tokens (`max(1, N//8)`). A learnable projection maps `3*hidden_dim -> hidden_dim`.
- **Volume fusion:** a two-layer slice Transformer with masked mean pooling and learnable focal-attention pooling. Each plane produces a 256-wide embedding.
- **Prediction:** concatenate the three plane embeddings plus plane-presence mask to predict 12 logits. A learned per-target plane-prior head is added with learnable gates.
- **Transfer learning:** the experiment froze an early ViT block prefix and fine-tuned the last six transformer blocks. The cleaned module defaults to **no automatic downloading** of weights.
- **Resolution:** Phase 2 experiments used 336px; Phase 3 aimed for 392px but one checkpoint recorded a fold-specific size. Always honor checkpoint metadata during reproduction.

## Programmatic use

```python
from src.models.triplanar_dinov2 import TriPlanarDINOv2
model = TriPlanarDINOv2(pretrained=False)  # requires TIMM; random backbone unless weights are separately supplied
logits = model(axial, coronal, sagittal, presence)  # [batch,12]
```

**Validation:** source compiled and a small mock-backbone test covers output shapes, gradients, missing-plane masking, and finite logits. Full GPU training and checkpoint parity are not validated in this repository.

## Known limitations

No model checkpoint, DICOM images, patient labels, or private test data are redistributed. Training quality depends on externally supplied legal data, fold-consistent labels, and correct preprocessing. Focal top-k pooling is a per-feature heuristic, not clinical localization. Anatomy priors are explicit architectural assumptions and not a substitute for evaluation.
