#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_rt_detr_compressed_A100_gpu.sl"

y_values=(48 32 16 0)
cbcr_values=(64 48 32 16 0)

for y_coeff in "${y_values[@]}"; do
    for chroma_coeff in "${cbcr_values[@]}"; do
        job_name="lfc4_y${y_coeff}_cb${chroma_coeff}"
        echo "Submitting job for Y=${y_coeff}, Cb/Cr=${chroma_coeff}"
        sbatch --job-name="${job_name}" \
            --export=ALL,COEFF_COUNT_LUMA=${y_coeff},COEFF_COUNT_CB=${chroma_coeff},COEFF_COUNT_CR=${chroma_coeff} \
            "${SLURM_SCRIPT}"
    done
done
