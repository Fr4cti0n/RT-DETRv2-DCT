# RT-DETRv2 – Luma-Fusion Pruned (coeff=4)

Here the pruned adapter keeps a 4×4 coefficient window while maintaining a reduced channel footprint compared to the standard luma-fusion variant, offering solid accuracy with lower compute.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff4_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion-pruned`, `coeff_window: 4`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff4_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Use this setup when you want near-baseline quality with a trimmed backbone ready for deployment.
