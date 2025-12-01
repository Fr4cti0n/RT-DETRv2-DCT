# Training Guide for Backbone Experiments

This guide summarises the training entry points and options for every backbone configuration in the repo, covering both RGB and DCT-compressed pipelines.

## 1. Dataset prep
- Expect ImageNet-style folders under `dataset/classification/imagenet1k*` as shipped with the repo structure.
- Optional caps for quick smoke tests: use `--max-train-images` / `--max-val-images` (standard trainer) or the same flags on the compressed trainer.

## 2. Standard RGB backbones (`train_backbones.py`)
```bash
python -m src.nn.backbone.train_backbones \
  --model resnet34 \  # or cspdarknet53 / efficientvit_m4
  --train-dirs dataset/classification/imagenet1k0 dataset/classification/imagenet1k1 \
                dataset/classification/imagenet1k2 dataset/classification/imagenet1k3 \
  --val-dir dataset/classification/imagenet1kvalid \
  --output-dir output/imagenet_resnet34_backbone \
  --epochs 100 --batch-size 256 --amp
```
Key flags:
- `--model`: `resnet34`, `cspdarknet53`, `efficientvit_m4`.
- `--input-format`: `rgb` (default) or `compressed` when pairing with the compressed adapter.
- Compression toggles (only valid for `resnet34`): `--compression-coeff-window`, `--compression-range-mode`, `--compressed-backbone`.

## 3. Compressed ResNet trainer (`train_compressed_backbones.py`)
Designed for rapid iteration on compressed variants. Example:
```bash
python -m src.nn.backbone.train_compressed_backbones \
  --variant reconstruction \  # or block-stem / luma-fusion / luma-fusion-pruned
  --train-dirs dataset/classification/imagenet1k0 dataset/classification/imagenet1k1 \
                dataset/classification/imagenet1k2 dataset/classification/imagenet1k3 \
  --val-dir dataset/classification/imagenet1kvalid \
  --coeff-window 4 --range-mode studio \
  --epochs 100 --batch-size 256 --amp
```
Useful options:
- `--max-train-images`, `--max-val-images`: subsample datasets for debugging.
- `--channels-last`: enable NHWC layout if you benchmark throughput.
- `--save-every`, `--save-best`: control checkpoint cadence.

## 4. Choosing a variant
- **Reconstruction**: full RGB rebuild before the ResNet stem; easiest drop-in baseline.
- **Block-Stem**: keeps coefficients in the stem; good for latency-sensitive inference.
- **Luma-Fusion**: mixes downsampled luma with chroma in feature space; balances cost vs detail.
- **Luma-Fusion-Pruned**: scaled-down residual stack tuned for aggressive coefficient pruning.

Each variant has a dedicated explainer:
- `docs/compressed_backbones/reconstruction.md`
- `docs/compressed_backbones/block_stem.md`
- `docs/compressed_backbones/luma_fusion.md`
- `docs/compressed_backbones/luma_fusion_pruned.md`

## 5. Troubleshooting checklist
- Verify the DCT pipeline with `src/nn/backbone/debug_compressed_reconstruction.py` before long runs.
- For small coefficient windows, consider lowering `--lr` or increasing weight decay.
- When using `--input-format compressed` in `train_backbones.py`, ensure the variant matches `--compressed-backbone` and the dataset transform emits `(y_blocks, cbcr_blocks)` tuples.

Happy training!
