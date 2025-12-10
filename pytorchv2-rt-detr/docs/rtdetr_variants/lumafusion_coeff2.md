# RT-DETRv2 – Luma-Fusion (coeff=2)

This run keeps a 2×2 low-frequency window per luminance block within the luma-fusion adapter, delivering a strong quality/cost compromise.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff2_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion`, `coeff_window: 2`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff2_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Use this profile as the default starting point when weighing luma-fusion against block-stem and reconstruction baselines.
