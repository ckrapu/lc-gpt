#!/bin/bash

# NCCL Settings for RTX A5000 GPUs
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes (from default 10 minutes)
export NCCL_P2P_DISABLE=1  # Disable P2P for consumer GPUs like RTX A5000
export NCCL_IB_DISABLE=1   # Disable InfiniBand if not available
export CUDA_VISIBLE_DEVICES=0,1  # Explicitly set GPUs

# PyTorch distributed settings
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Optional: Set NCCL socket interface if needed (uncomment and adjust if necessary)
# export NCCL_SOCKET_IFNAME=eth0

# Run training with accelerate
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --dynamo_backend no \
    train.py \
    --config configs/randar_nlcd_32_tokenized.yaml \
    "$@"