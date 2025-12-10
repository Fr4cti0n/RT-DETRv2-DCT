# RT-DETRv2 – Luma-Fusion (coeff=4)

Here the adapter consumes a 4×4 low-frequency window while mixing chroma projections in feature space, bringing accuracy close to full-resolution inputs with modest extra cost.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff4_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion`, `coeff_window: 4`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff4_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Expect accuracy similar to reconstruction while shaving stem latency; use this as the high-quality luma-fusion baseline.
