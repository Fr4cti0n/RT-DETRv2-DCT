# RT-DETRv2 – Block-Stem (coeff=8)

The coeff=8 flavour forwards all DCT coefficients through the block-stem adapter, effectively matching RGB capacity while benefiting from the learned compressed stem.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff8_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: block-stem`, `coeff_window: 8`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff8_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Use this run as an upper-bound when benchmarking block-stem compression versus reconstruction-style baselines.
