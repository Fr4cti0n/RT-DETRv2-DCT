#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_rt_detrv2_train_h200_gpu.sl"
CONFIG_DIR="configs/rtdetrv2/lumafusion_coeffs"

# Sweep Cb/Cr coefficient counts for a fixed luma budget.
coeff_values=(64 48 32 16 0)
luma_coeff=64

for coeff in "${coeff_values[@]}"; do
    config_file="${CONFIG_DIR}/rtdetrv2_r34vd_lumafusion_coeffY${luma_coeff}_Cb${coeff}_Cr${coeff}_120e_coco.yml"

    if [[ ! -f "${SCRIPT_DIR}/${config_file}" ]]; then
        echo "Skipping missing config: ${config_file}" 1>&2
        continue
    fi

    job_name="lfc4_cb${coeff}"
    echo "Submitting job for ${config_file}" | tee /dev/stderr

    CONFIG_FILE="${config_file}" sbatch --job-name="${job_name}" "${SLURM_SCRIPT}"
done
