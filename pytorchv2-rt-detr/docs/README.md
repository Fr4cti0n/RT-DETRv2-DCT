# RT-DETRV2 for intra coding VOP mpeg4 part2 — docs hub

This directory collects the notes for running RT-DETRv2 on MPEG-4 part2 intra-coded VOPs using DCT-domain inputs.

## What you will find
- Compressed backbones overview: [compressed_backbones/README.md](compressed_backbones/README.md)
- Detector variant recipes and launch commands: [rtdetr_variants/README.md](rtdetr_variants/README.md)
- Reconstruction details: [compressed_backbones/reconstruction.md](compressed_backbones/reconstruction.md) and [rtdetr_variants/reconstruction.md](rtdetr_variants/reconstruction.md)

## Quick start reminders
- Activate the compressed environment: `source detr-compressed/bin/activate`.
- Keep COCO 2017 data at `dataset/detection/coco/` so configs resolve as-is.
- Typical launch (swap the YAML for other windows/backbones):
  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --master_port=9909 --nproc_per_node=4 tools/train.py \
    -c configs/rtdetrv2/lumafusion_coeffs/rtdetrv2_r34vd_lumafusion_coeffY64_Cb64_Cr64_120e_coco.yml \
    --use-amp --seed 0
  ```

## Plots
- COCO AP vs retained information: [../plots/RTDETRv2_AP.png](../plots/RTDETRv2_AP.png)

## Back to the project
- Main overview and setup: [../README.md](../README.md)
