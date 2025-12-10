# RT-DETRv2 – Luma-Fusion Pruned (coeff=1)

The pruned luma-fusion adapter scales down residual widths in proportion to the coefficient window. With coeff=1 it matches the extreme compression regime and offers the lightest detector backbone.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff1_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion-pruned`, `coeff_window: 1`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff1_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Plan for longer convergence; the reduced channel budget benefits from conservative learning rates and EMA tracking.
