#!/bin/bash
#SBATCH --nodes=8                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Important for distributed usage (1 task per node)
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8                  # 8 GPUs per node
##SBATCH --exclusive
#SBATCH --qos=normal

set -x -e

# (Optional) Set up Hugging Face cache directory
export HF_HOME=/path/to/user/cache

# Activate your conda environment
source /path/to/user/miniconda3/etc/profile.d/conda.sh
conda activate smolvlm

# Debug prints
echo "Python path: $(which python)"
python -c "import sys; print('Sys path:', sys.path)"
python -c "import torch; print('PyTorch version:', torch.__version__, '\nPyTorch location:', torch.__file__)"
python -c "import transformers; print('Transformers location:', transformers.__file__)"
which torchrun

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6001
export WORLD_SIZE=$((SLURM_NNODES * 8))
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=0

# User-defined variables
MODEL_NAME="HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_PATH="scripts/mixtures/onevision_no_mammoth_more_image_balanced.yaml"
DATA_FOLDER="/path/to/user/apollo-dataset/"
RUN_NAME="final-vision-balanced-500-visionunfrozen-newmix"

# Debug prints for environment
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"

cd /path/to/user/smolvlmvideo

export PYTHONPATH="/path/to/user/smolvlmvideo:$PYTHONPATH"
srun torchrun \
    --nproc_per_node=8 \
    --nnodes="${SLURM_NNODES}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    smolvlm/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --model_max_length 8192 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 7 \
    --vision_tower_lr 5e-6 \
    --tune_vision_tower True \
    --connector_lr 1e-4 \
    --language_model_lr 2e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --peft_enable False \
    --logging_steps 1 \
    --data_mixture $DATA_PATH \
    --data_folder $DATA_FOLDER \
    --dataloader_drop_last True \
    --dataloader_num_workers 2 \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1 \
    --gradient_checkpointing True \
    --packed False \
    --mask_user_tokens True \
    --add_media_intro_outro True \
    --video_target_size 512 \
    --image_target_size 2048 \
    --max_frames 64 \
    --fps 1 \
    --use_liger_kernel False \
    --report_to wandb \
    --run_name $RUN_NAME



#    --optim paged_adamw_8bit \