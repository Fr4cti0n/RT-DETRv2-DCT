# RT-DETRv2 – Luma-Fusion (coeff=8)

This configuration forwards the complete set of DCT coefficients through the luma-fusion adapter, maximising fidelity while retaining the adapter's fused stem design.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff8_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: luma-fusion`, `coeff_window: 8`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff8_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Leverage this run as the upper bound when quantifying luma-fusion compression trade-offs. The input pipeline mirrors the backbone pretraining setup: raw RGB tensors go directly into the DCT compressor (no ImageNet mean/std normalisation beforehand) and the resulting coefficients are normalised via `NormalizeDCTCoefficientsFromFile` using the ImageNet stats at `configs/dct_stats/imagenet_coeff8_studio.pt`.

## Using a pretrained backbone for evaluation
To reuse the ImageNet-trained backbone stored at `output/compressed_resnet34/luma-fusion_coeff8/model_best.pth`, point the config at that checkpoint and launch validation only:

```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_lumafusion_coeff8_120e_coco.yml \
  --test-only \
  -u CompressedPResNet.compressed_pretrained=output/compressed_resnet34/luma-fusion_coeff8/model_best.pth \
     CompressedPResNet.strict_load=False
```

This command loads the backbone weights, skips training, and runs a validation pass. Replace `--test-only` with training flags if you want to fine-tune the detector starting from the same pretrained backbone.
