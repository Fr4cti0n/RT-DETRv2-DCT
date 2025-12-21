#!/bin/bash

# Slurm submission script, 
# yolov11 torch 

# Job name
#SBATCH -J "yolov11_zip_reco_adam"

# Batch output file
#SBATCH --output run_yolov11_rgb.o%J

# Batch error file
#SBATCH --error run_yolov11_rgb.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition gpu

# GPUs per compute node
#   8 (maximum) for gpu
#   8 (maximum) for hpda
#SBATCH --gres=gpu:a100:1

# ----------------------------
# processes / tasks
#SBATCH -n 1

# ----------------------------
# CPUs per task
# Set the number of cpu in proportion to the number of GPU's devices :
#   gpu: until 8 cores / device
#   hpda: until 8 cores / device
#SBATCH --cpus-per-task 8

# ----------------------------
# Maximum memory per compute node (MB)
#SBATCH --mem 120000 
# ----------------------------

# ------------------------
# Job time (hh:mm:ss)
#SBATCH --time 24:00:00
# ------------------------

##SBATCH --mail-type ALL
# User e-mail address
##SBATCH --mail-user aduche@insa-rouen.fr

# environments
# ---------------------------------
PROJECTNAME=YOLOv11_sequence
module purge
module load aidl/pytorch/2.5.1-cuda12.4
export PYTHONUSERBASE=~/packages/$PROJECTNAME
export PATH=$PATH:~/packages/$PROJECTNAME/
# ---------------------------------

# Copy script input data and go to working directory 
cp -r $(pwd)/YOLOv11_COMPRESSED_CRIANN/ $LOCAL_WORK_DIR
LOCAL_PATH_YOLO="$(pwd)"
DATASET="Datasets/DIRIF"
cd $LOCAL_WORK_DIR/ 
echo Working directory : $PWD
cd YOLOv11_COMPRESSED_CRIANN/
mkdir -p results/"$DATASET"
echo "$DATASET"
srun python main.py --train --data-dir "$LOCAL_PATH_YOLO" --fm-coeff 0.00005 --dataset "$DATASET" --optimizer "sgd"
#srun python nets/nn.py
# Move output data to target directory
mkdir -p $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/"results_compressed_fm_adam_roerder_dirif"
mv $LOCAL_WORK_DIR/YOLOv11_COMPRESSED_CRIANN/results/* $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/"results_compressed_fm_adam_roerder_dirif"


