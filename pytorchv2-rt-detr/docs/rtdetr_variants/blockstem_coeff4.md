# RT-DETRv2 – Block-Stem (coeff=4)

This configuration keeps a 4×4 band of luma coefficients per block, delivering higher fidelity while still skipping the RGB reconstruction stage.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff4_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: block-stem`, `coeff_window: 4`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_blockstem_coeff4_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Expect accuracy close to the 8×8 window while retaining most of the latency gains from the compressed stem.
