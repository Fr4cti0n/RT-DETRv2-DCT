# RT-DETRv2 – Luma-Fusion (coeff=1)

The luma-fusion adapter downsamples the luminance coefficients and fuses chroma projections before re-expanding to match the residual stages. With coeff=1 it operates on the DC term only, maximising compression.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff1_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion`, `coeff_window: 1`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff1_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Monitor gradients closely; the aggressive window may benefit from longer warmup or EMA to stabilise training.
