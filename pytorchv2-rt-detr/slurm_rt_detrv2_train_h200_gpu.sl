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
cd $LOCAL_WORK_DIR/ 
echo Working directory : $PWD
cd pytorchv2-rt-detr//
CONFIG_FILE=${CONFIG_FILE:-configs/rtdetrv2/lumafusion_coeffs/rtdetrv2_r34vd_lumafusion_coeffY32_Cb16_Cr16_120e_coco.yml}
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yml)
RUN_OUTPUT_DIR="output/${CONFIG_BASENAME}"
RESULTS_TAG=${RESULTS_TAG:-$CONFIG_BASENAME}
mkdir -p "results/$RESULTS_TAG"
echo "Results tag: $RESULTS_TAG"

DATASET_ROOT=${DATASET_ROOT:-}
TRAIN_IMG_DIR=${TRAIN_IMG_DIR:-}
TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-}
VAL_IMG_DIR=${VAL_IMG_DIR:-}
VAL_ANN_FILE=${VAL_ANN_FILE:-}

if [ -n "$DATASET_ROOT" ]; then
	TRAIN_IMG_DIR=${TRAIN_IMG_DIR:-${DATASET_ROOT%/}/train2017}
	TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-${DATASET_ROOT%/}/annotations_trainval2017/annotations/instances_train2017.json}
	VAL_IMG_DIR=${VAL_IMG_DIR:-${DATASET_ROOT%/}/val2017}
	VAL_ANN_FILE=${VAL_ANN_FILE:-${DATASET_ROOT%/}/annotations_trainval2017/annotations/instances_val2017.json}
fi

UPDATE_ARGS=(
	train_dataloader.total_batch_size=16
	val_dataloader.total_batch_size=8
)

if [ -n "$TRAIN_IMG_DIR" ]; then
	UPDATE_ARGS+=(train_dataloader.dataset.img_folder=$TRAIN_IMG_DIR)
fi
if [ -n "$TRAIN_ANN_FILE" ]; then
	UPDATE_ARGS+=(train_dataloader.dataset.ann_file=$TRAIN_ANN_FILE)
fi
if [ -n "$VAL_IMG_DIR" ]; then
	UPDATE_ARGS+=(val_dataloader.dataset.img_folder=$VAL_IMG_DIR)
fi
if [ -n "$VAL_ANN_FILE" ]; then
	UPDATE_ARGS+=(val_dataloader.dataset.ann_file=$VAL_ANN_FILE)
fi
export WANDB_API_KEY="ff82e925973a4616bfb09d403b18fe712beeef97"
srun python tools/train.py \
	--config "$CONFIG_FILE" \
	--backbone-root output/compressed_resnet34 \
	--update "${UPDATE_ARGS[@]}" \
	--device cuda \
	--use-amp \
	--wandb \
	--time-limit-hours 22.5
    --auto-resume \
	--checkpoint-root output/detr_compressed34
#srun python nets/nn.py
# Move output data to target directory
if [ -d "$LOCAL_WORK_DIR/pytorchv2-rt-detr/${RUN_OUTPUT_DIR}" ]; then
	mkdir -p "$SLURM_SUBMIT_DIR/RT-DETRv2-DCT/pytorchv2-rt-detr/${RUN_OUTPUT_DIR}"
	mv "$LOCAL_WORK_DIR/pytorchv2-rt-detr/${RUN_OUTPUT_DIR}"/* "$SLURM_SUBMIT_DIR/RT-DETRv2-DCT/pytorchv2-rt-detr/${RUN_OUTPUT_DIR}/"
fi
