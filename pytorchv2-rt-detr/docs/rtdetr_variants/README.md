# RT-DETRv2 Compressed-Input Recipes

This directory documents the launch commands for every RT-DETRv2 experiment that consumes DCT-compressed inputs. Each variant pairs a `CompressedPResNet` backbone with the detector stack defined in `configs/rtdetrv2/`.

## Usage checklist
- Place COCO 2017 data under `dataset/detection/coco/` (matching the default YAML).
- Activate the project environment (e.g. `source detr-compressed/bin/activate`).
- Run experiments from the repo root so relative paths inside the configs resolve correctly.
- If you trained a backbone checkpoint, provide it via the `CompressedPResNet.compressed_pretrained` key (defaults to `null`).

## Variant index
- Reconstruction: `reconstruction.md`
- Block-Stem windows: `blockstem_coeff1.md`, `blockstem_coeff2.md`, `blockstem_coeff4.md`, `blockstem_coeff8.md`
- Luma-Fusion windows: `lumafusion_coeff1.md`, `lumafusion_coeff2.md`, `lumafusion_coeff4.md`, `lumafusion_coeff8.md`
- Luma-Fusion-Pruned windows: `lumafusion_pruned_coeff1.md`, `lumafusion_pruned_coeff2.md`, `lumafusion_pruned_coeff4.md`, `lumafusion_pruned_coeff8.md`

Each page describes the adapter, expected coefficient window, and a ready-to-run `tools/train.py` command.
