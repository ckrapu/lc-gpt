#!/bin/bash

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected on this device"
    exit 1
fi

echo "Detected $NUM_GPUS GPU(s) on this device"

# Check if any GPU is Hopper or Blackwell
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eiq 'Hopper|H100|H200|Blackwell|B100|B200|GB200'; then
    DYNAMO_FLAG="--dynamo_backend inductor"
    echo "Hopper or Blackwell GPU detected: enabling Dynamo backend."
else
    DYNAMO_FLAG="--dynamo_backend no"
    echo "No Hopper or Blackwell GPU detected: Dynamo backend disabled."
fi

# Create CUDA_VISIBLE_DEVICES string (0,1,2,... for all available GPUs)
GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))

# NCCL Settings for RTX A5000 GPUs
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes (from default 10 minutes)
export NCCL_P2P_DISABLE=0  # Disable P2P for consumer GPUs like RTX A5000
export NCCL_IB_DISABLE=0   # Disable InfiniBand if not available
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
    $DYNAMO_FLAG \
    train.py \
    --config configs/randar_nlcd_128_large.yaml \
    "$@"
