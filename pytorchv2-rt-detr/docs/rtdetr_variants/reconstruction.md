# RT-DETRv2 – Reconstruction Adapter

The reconstruction pipeline decodes 8×8 DCT blocks back to RGB space before handing images to the ResNet stages. Use this variant as the baseline when comparing against compressed-input adapters.

## Config reference
- YAML: `configs/rtdetrv2/rtdetrv2_r34vd_reconstruction_120e_coco.yml`
- Backbone: `CompressedPResNet` with `compression_variant: reconstruction`

## Launch command
```bash
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r34vd_reconstruction_120e_coco.yml \
  --devices 0,1,2,3 --amp
```

Tweak `--devices` (or `CUDA_VISIBLE_DEVICES`) to match your hardware. If you have a pretrained RGB reconstruction checkpoint, set `CompressedPResNet.compressed_pretrained` in the YAML before launching.
