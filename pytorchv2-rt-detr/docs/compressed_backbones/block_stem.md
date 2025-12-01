# Compressed ResNet – Block-Stem Variant

The block-stem adapter consumes DCT coefficients directly and learns separate projections for luminance and chrominance before entering the residual stack. It skips RGB reconstruction entirely, trading interpretability for speed.

## Pipeline
- Treat the 64-coefficient luminance blocks as a dense feature map with stride 8.
- Project luminance blocks with a 3×3 Conv→BN→SiLU stack to match the ResNet stem width.
- Concatenate upsampled Cb/Cr coefficient maps (nearest-neighbour) and run a chroma projection path.
- Fuse the two branches with a 1×1 convolution and upsample 2× to align with the residual block grid.
- Continue through the unmodified residual stages.

## When to use
- You want to avoid the inverse colour transform while keeping a stem at full spatial resolution.
- Latency matters and you can lean on the network to learn colour mixing from coefficient space.

## CLI quickstart
```bash
python -m src.nn.backbone.train_compressed_backbones \
  --train-dirs dataset/classification/imagenet1k0 dataset/classification/imagenet1k1 \
                dataset/classification/imagenet1k2 dataset/classification/imagenet1k3 \
  --val-dir dataset/classification/imagenet1kvalid \
  --variant block-stem \
  --coeff-window 4 --range-mode studio \
  --epochs 90 --batch-size 256 --amp
```
For stable training, keep BatchNorm statistics in sync with your coefficient window; extreme pruning (window=1) may require lower learning rates.

## Implementation reference
- Core logic: `CompressedResNetBlockStem` in `src/nn/backbone/compressed_presnet.py`.
- Training entry point: `src/nn/backbone/train_compressed_backbones.py`.
