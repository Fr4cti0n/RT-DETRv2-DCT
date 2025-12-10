# RT-DETRv2 – Luma-Fusion Pruned (coeff=8)

The coeff=8 pruned configuration forwards all DCT coefficients through the slimmed luma-fusion adapter, targeting the best accuracy-to-compute ratio within the pruned family.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff8_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion-pruned`, `coeff_window: 8`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_pruned_coeff8_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Run this variant to benchmark pruned adapters against block-stem and full luma-fusion baselines at comparable fidelity.
