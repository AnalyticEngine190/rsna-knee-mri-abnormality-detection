# ECA-NFNet-L0 tri-planar branch

**Origin:** Curated reimplementation from the user's `eca_nfnet_l0 (2)(1).py` plus supporting Phase-3 scripts. No original checkpoint weights included; this is not a byte-identical checkpoint-compatible copy.

- **Input:** native 1-channel grayscale MRI slices, sampled 16 slices per anatomical plane. Missing planes use a binary 3-element presence mask.
- **CNN backbone:** `eca_nfnet_l0` via TIMM, with `num_classes=0` and `global_pool=''` to keep spatial feature maps.
- **Pooling:** learnable generalized-mean pooling (GeM; initial power `p=3.0`, eps `1e-6`, clamp to [1,6]). For ConvNeXt, the project instead preferred mean pooling because signed LayerNorm features are not handled well by positive-only GeM.
- **Slice aggregation:** two-layer pre-norm Transformer followed by valid-slice mean pooling.
- **Fusion:** 256-dimensional embedding for each plane plus a plane-mask vector -> 12-label classification head.
- **Training:** original experiments used five grouped folds, Phase 2 up to 320px and later 384px; BatchNorm statistics were held fixed when configured. Gold-standard validation was very small and AUC could be undefined in individual folds.

## Programmatic use

```python
from src.models.triplanar_nfnet import TriPlanarNFNet
model = TriPlanarNFNet(pretrained=False)  # needs TIMM; load approved weights separately
logits = model(axial, coronal, sagittal, presence)
```

**Validation:** a synthetic toy-backbone smoke test checks output shape, backpropagation, and missing-plane handling. This is not evidence of clinical performance, full training reproducibility, or compatibility with the original saved `.pth` files.

## Limitations

Training requires precomputed permitted MRI cache, a verified patient-level fold map, report/soft targets and expert labels with appropriate authorization. No external assets are distributed.
