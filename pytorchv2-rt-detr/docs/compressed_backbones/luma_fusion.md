# Compressed ResNet – Luma-Fusion Variant

Luma-fusion keeps the model in coefficient space but downsamples the luminance stream before merging with chroma features. This mirrors early fusion strategies used in video codecs while maintaining a modest compute footprint.

## Pipeline
- Insert a 3×3 stride-2 Conv→BN→SiLU block on the luminance coefficients, producing a half-resolution feature map with widened channel count.
- Upsample Cb/Cr coefficients to the luminance grid and project them with a dedicated 3×3 Conv→BN→SiLU branch.
- Concatenate the luma and chroma embeddings, fuse with a 1×1 convolution, then upsample by 4× to restore the standard ResNet spatial scale before the residual stack.

## When to use
- You want to reduce compute in the stem by working at lower spatial resolution while preserving chroma cues.
- You are experimenting with coefficient budgets (`coeff-window` < 8) and need extra capacity to compensate for missing frequencies.

## CLI quickstart
```bash
python -m src.nn.backbone.train_compressed_backbones \
  --train-dirs dataset/classification/imagenet1k0 dataset/classification/imagenet1k1 \
                dataset/classification/imagenet1k2 dataset/classification/imagenet1k3 \
  --val-dir dataset/classification/imagenet1kvalid \
  --variant luma-fusion \
  --coeff-window 4 --range-mode studio \
  --epochs 90 --batch-size 256 --amp
```
Adjust `--coeff-window` depending on your bandwidth target; smaller windows pair well with higher hidden widths when chroma detail is scarce.

## Implementation reference
- Core logic: `CompressedResNetLumaFusion` in `src/nn/backbone/compressed_presnet.py`.
- Training entry point: `src/nn/backbone/train_compressed_backbones.py`.
