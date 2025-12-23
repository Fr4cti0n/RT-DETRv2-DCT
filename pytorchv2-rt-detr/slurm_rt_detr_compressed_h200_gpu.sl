#!/bin/bash

# Slurm submission script, 
# yolov11 torch 

# Job name
#SBATCH -J "lfc4"

# Batch output file
#SBATCH --output run_rt_rec.o%J

# Batch error file
#SBATCH --error run_rt_rec.e%J

# Partition / GPU selection
#SBATCH --partition gpu_h200
#SBATCH --gres=gpu:h200:1

# processes / tasks
#SBATCH -n 1
#SBATCH --cpus-per-task 8

# Maximum memory per compute node (MB)
#SBATCH --mem 80000

# Job time (hh:mm:ss)
#SBATCH --time 24:00:00

##SBATCH --mail-type ALL
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
# Allow overrides via sbatch --export while keeping sensible defaults.
COEFF_COUNT_LUMA=${COEFF_COUNT_LUMA:-64}
COEFF_COUNT_CB=${COEFF_COUNT_CB:-64}
COEFF_COUNT_CR=${COEFF_COUNT_CR:-64}
srun python -m src.nn.backbone.train_compressed_backbones \
	--variant luma-fusion \
	--train-dirs "$DATASET"/imagenet1k0 "$DATASET"/imagenet1k1 "$DATASET"/imagenet1k2 "$DATASET"/imagenet1k3 \
	--val-dir "$DATASET"/imagenet1kvalid \
	--coeff-count-luma "$COEFF_COUNT_LUMA" \
	--coeff-count-cb "$COEFF_COUNT_CB" \
	--coeff-count-cr "$COEFF_COUNT_CR" \
	--trim-coefficients \
	--dct-stats configs/dct_stats/imagenet_coeff8_studio.pt \
	--range-mode studio \
	--epochs 100 \
	--batch-size 256 \
	--amp \
	--wandb \
	--save-every 10 \
	--save-best \
	--warmup-epochs 10 \
	--time-limit-hours 22.5 \
	--image-size 256
#srun python nets/nn.py
# Move output data to target directory
mkdir -p $SLURM_SUBMIT_DIR/"RT-DETR"/"compresser/"$SLURM_JOB_ID
mv $LOCAL_WORK_DIR/pytorchv2-rt-detr/output/compressed_resnet34/* $SLURM_SUBMIT_DIR/RT-DETRv2-DCT/pytorchv2-rt-detr/output/compressed_resnet34/


