#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_rt_detr_compressed_A100_gpu.sl"

coeff_values=(64 48 32 16 0)
luma_coeff=64

for coeff in "${coeff_values[@]}"; do
    job_name="lfc4_cb${coeff}"
    echo "Submitting job for Cb/Cr=${coeff}"
    sbatch --job-name="${job_name}" \
        --export=ALL,COEFF_COUNT_LUMA=${luma_coeff},COEFF_COUNT_CB=${coeff},COEFF_COUNT_CR=${coeff} \
        "${SLURM_SCRIPT}"
done
