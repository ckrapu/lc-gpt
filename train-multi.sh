#!/bin/bash

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected on this device"
    exit 1
fi

echo "Detected $NUM_GPUS GPU(s) on this device"

# Create CUDA_VISIBLE_DEVICES string (0,1,2,... for all available GPUs)
GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))

# NCCL Settings for RTX A5000 GPUs
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes (from default 10 minutes)
export NCCL_P2P_DISABLE=1  # Disable P2P for consumer GPUs like RTX A5000
export NCCL_IB_DISABLE=1   # Disable InfiniBand if not available
export CUDA_VISIBLE_DEVICES=$GPU_LIST  # Set GPUs dynamically

echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# PyTorch distributed settings
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Optional: Set NCCL socket interface if needed (uncomment and adjust if necessary)
# export NCCL_SOCKET_IFNAME=eth0

# Run training with accelerate
accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision fp16 \
    --dynamo_backend no \
    train.py \
    --config configs/randar_nlcd_128.yaml \
    "$@"