# Compressed ResNet – Luma-Fusion Pruned Variant

Luma-fusion pruned shrinks the ResNet body to match aggressive coefficient pruning. It keeps the luma/chroma fusion stem from the standard luma-fusion adapter but scales channel counts and block depths according to the chosen `coeff-window`.

## Pipeline
- Regenerate ResNet residual stages with narrower channel widths that scale with the frequency budget (minimum width 8).
- Feed luminance coefficients through a stride-2 Conv→BN→SiLU block sized to the pruned backbone.
- Project upsampled Cb/Cr coefficients with a companion 3×3 Conv→BN→SiLU branch.
- Fuse the branches via a 1×1 convolution and upsample by 4× before entering the slimmed residual stack.
- Rebuild the classification head so its input width matches the new backbone output channels.

## When to use
- You target extremely low bandwidth (e.g. coeff-window 1 or 2) and want the network capacity to reflect the available information.
- You need a lighter inference footprint for edge devices while keeping the compressed training pipeline.

## CLI quickstart
```bash
python -m src.nn.backbone.train_compressed_backbones \
  --variant luma-fusion-pruned \
  --train-dirs dataset/classification/imagenet1k0 dataset/classification/imagenet1k1 \
                dataset/classification/imagenet1k2 dataset/classification/imagenet1k3 \
  --val-dir dataset/classification/imagenet1kvalid \
  --coeff-window 2 --range-mode studio \
  --epochs 90 --batch-size 256 --amp
```
Expect to revisit learning-rate or weight-decay settings when you deviate strongly from the default window size; the reduced parameter count can make optimisation more sensitive.

## Implementation reference
- Core logic: `CompressedResNetLumaFusionPruned` in `src/nn/backbone/compressed_presnet.py`.
- Training entry point: `src/nn/backbone/train_compressed_backbones.py`.
