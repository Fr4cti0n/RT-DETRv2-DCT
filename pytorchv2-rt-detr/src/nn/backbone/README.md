# Compressed backbone training

![Accuracy vs retained information fraction (top-1)](../../plots/Resnet34_accuracy.png)

This folder hosts standalone trainers for ImageNet-style classification backbones, including DCT-compressed variants used by RT-DETRv2.

## Setup
- Install deps from the repo root: `pip install -r requirements.txt`.
- (Optional) activate the provided env: `source detr-compressed/bin/activate`.
- Datasets: place ImageNet-style splits under `dataset/classification/` (train/val folders matching `ImageNetDataset`).

## Train RGB backbones
Runs standard classification training defined in [src/nn/backbone/train_backbones.py](src/nn/backbone/train_backbones.py).
```bash
# ResNet-34 RGB
python -m src.nn.backbone.train_backbones \
  --model resnet34 \
  --train-dirs dataset/classification/imagenet_train \
  --val-dir dataset/classification/imagenet_val \
  --output-dir output/imagenet_resnet34_backbone \
  --batch-size 256 --epochs 100
```
- Switch `--model` to `cspdarknet53` or `efficientvit_m4` for other presets.

## Train DCT-compressed backbones
Uses the DCT adapters that match the detector configs in [configs/rtdetrv2/lumafusion_coeffs/](configs/rtdetrv2/lumafusion_coeffs/). Script: [src/nn/backbone/train_compressed_backbones.py](src/nn/backbone/train_compressed_backbones.py).
```bash
python -m src.nn.backbone.train_compressed_backbones \
  --train-dirs dataset/classification/imagenet_train \
  --val-dir dataset/classification/imagenet_val \
  --output-dir output/compressed_resnet34 \
  --variant luma-fusion \
  --coeff-count-luma 64 --coeff-count-chroma 64 \
  --batch-size 256 --epochs 90
```
- Adjust coefficient windows/counts with `--coeff-window-*` or `--coeff-count-*` to mirror the detector YAML.
- `--trim-coefficients` (default) reduces payload depth; use `--no-trim-coefficients` to keep all 64.
- To resume the latest matching run, pass `--auto-resume`.

## Notes
- Training logs and checkpoints are written under the chosen `--output-dir`.
- wandb logging stays disabled unless you add the `--wandb` flag; the scripts run offline by default.
- Inference benchmarking for compressed backbones lives in [src/nn/backbone/inference_benchmark.py](src/nn/backbone/inference_benchmark.py).
