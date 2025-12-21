#!/bin/bash

# Slurm submission script, 
# yolov11 torch 

# Job name
#SBATCH -J "lfc4"

# Batch output file
#SBATCH --output run_rt_rec.o%J

# Batch error file
#SBATCH --error run_rt_rec.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition gpu_h200

# GPUs per compute node
#   8 (maximum) for gpu
#   8 (maximum) for hpda
#SBATCH --gres=gpu:h200:4

# ----------------------------
# processes / tasks
#SBATCH -n 1
#SBATCH --nodes 1

# ----------------------------
# CPUs per task
# Set the number of cpu in proportion to the number of GPU's devices :
#   gpu: until 8 cores / device
#   hpda: until 8 cores / device
#SBATCH --cpus-per-task 32

# ----------------------------
# Maximum memory per compute node (MB)
#SBATCH --mem 80000 
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
cp -r $(pwd)/RT-DETRv2-DCT/pytorchv2-rt-detr/ $LOCAL_WORK_DIR
LOCAL_PATH_YOLO="$(pwd)"
DATASET="/home/2017018/aduche02/Datasets/classification_imagenet"
cd $LOCAL_WORK_DIR/ 
echo Working directory : $PWD
cd pytorchv2-rt-detr//
mkdir -p results/"$DATASET"
echo "$DATASET"
export WANDB_API_KEY="ff82e925973a4616bfb09d403b18fe712beeef97"
srun torchrun --standalone --nproc_per_node=4 -m src.nn.backbone.train_compressed_backbones \
	--distributed \
	--variant luma-fusion \
	--train-dirs "$DATASET"/imagenet1k0 "$DATASET"/imagenet1k1 "$DATASET"/imagenet1k2 "$DATASET"/imagenet1k3 \
	--val-dir "$DATASET"/imagenet1kvalid \
	--coeff-count-luma 64 \
	--coeff-count-cb 64 \
	--coeff-count-cr 64 \
	--trim-coefficients \
	--dct-stats configs/dct_stats/imagenet_coeff8_studio.pt \
	--range-mode studio \
	--epochs 100 \
	--batch-size 1024 \
	--amp \
	--wandb \
	--save-every 10 \
	--save-best \
	--warmup-epochs 1 \
	--time-limit-hours 1.5 \
	--image-size 256
#srun python nets/nn.py
# Move output data to target directory
mkdir -p $SLURM_SUBMIT_DIR/"RT-DETR"/"compresser/"$SLURM_JOB_ID
mv $LOCAL_WORK_DIR/pytorchv2-rt-detr/output/compressed_resnet34/* $SLURM_SUBMIT_DIR/RT-DETRv2-DCT/pytorchv2-rt-detr/output/compressed_resnet34/


