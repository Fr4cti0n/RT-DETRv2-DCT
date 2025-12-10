# RT-DETRv2 – Luma-Fusion Pruned (coeff=2)

This run keeps a 2×2 coefficient window while shrinking the residual stages to match the reduced frequency content, giving a tight balance between accuracy and throughput.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff2_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion-pruned`, `coeff_window: 2`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff2_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Watch validation accuracy versus the non-pruned luma-fusion runs to quantify the benefit of channel trimming.
