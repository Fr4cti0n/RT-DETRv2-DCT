# Compressed ResNet – Reconstruction Variant

This variant decodes the low-frequency DCT payload back into RGB before feeding the backbone. It targets parity with a standard ResNet stem while preserving the storage/bandwidth benefits of the compressed input format.

## Pipeline
- Decode 8×8 Y, Cb, Cr blocks with a fixed inverse-DCT basis.
- Upsample chroma planes back to the luminance resolution (4:2:0 → 4:4:4).
- Convert YCbCr to RGB in the requested range (`studio` by default).
- Run a light 3×3 refinement block (Conv→BN→SiLU→Conv) to compensate for quantisation artefacts.
- Apply ImageNet mean/std normalisation and funnel the tensor through the vanilla ResNet stem.

## When to use
- You need a drop-in replacement for RGB training pipelines, including pretrained weight transfer.
- You plan to compare reconstruction fidelity against RGB baselines.

## CLI quickstart
```bash
python -m src.nn.backbone.train_compressed_backbones \
  --train-dirs dataset/classification/imagenet1k0 dataset/classification/imagenet1k1 \
                dataset/classification/imagenet1k2 dataset/classification/imagenet1k3 \
  --val-dir dataset/classification/imagenet1kvalid \
  --variant reconstruction \
  --coeff-window 4 --range-mode studio \
  --epochs 90 --batch-size 256 --amp
```
Set `--coeff-window` to the desired low-frequency window (1, 2, 4 or 8). Smaller windows shrink the chroma content but reduce bandwidth.

## Implementation reference
- Core logic: `CompressedResNetReconstruction` in `src/nn/backbone/compressed_presnet.py`.
- Training entry point: `src/nn/backbone/train_compressed_backbones.py`.
