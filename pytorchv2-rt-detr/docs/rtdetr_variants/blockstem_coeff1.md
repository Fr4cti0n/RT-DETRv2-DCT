# RT-DETRv2 – Block-Stem (coeff=1)

This profile keeps only the DC term from each luminance block while routing chroma coefficients through the block-stem adapter. It is the most compact block-stem configuration and prioritises bandwidth over high-frequency detail.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff1_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: block-stem`, `coeff_window: 1`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff1_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

With such aggressive pruning, monitor loss early in training; if the run diverges, extend warmup or drop the base learning rate by ~20%.
