# RT-DETRv2 – Block-Stem (coeff=2)

The coeff=2 block-stem run retains a 2×2 low-frequency window for each luma block, offering a balance between compression and reconstruction quality while avoiding RGB decoding.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff2_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: block-stem`, `coeff_window: 2`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff2_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

This setting often converges cleanly; use it as a baseline when exploring throughput/accuracy trade-offs for block-stem adapters.
