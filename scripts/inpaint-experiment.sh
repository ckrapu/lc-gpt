#!/bin/bash

# This script iterates over mask_ratio fractions and saves the data to logs/

# Create logs directory if it doesn't exist
mkdir -p logs

# Define mask ratios
mask_ratios=(0.2 0.4 0.6 0.7 0.8 0.9 0.925 0.95 0.975 0.99 0.995 0.998)
mask_types=(interior random)
n_images=100
# Run experiments for each mask ratio
for ratio in "${mask_ratios[@]}"; do
    for mask_type in "${mask_types[@]}"; do
        echo "Running experiment with mask_ratio=$ratio and mask_type=$mask_type"

        # Format ratio for filename (replace . with _)
        ratio_str=$(echo $ratio | sed 's/\./_/')
        
        # Run the evaluation
        python tools/eval-inpaint-single.py \
            --config configs/randar_nlcd_32.yaml \
            --gpt-ckpt results/randar_nlcd_32/checkpoints/final \
            --n-images $n_images \
            --verbose \
            --save-results "logs/inpaint_eval_results_interior_${ratio_str}.json" \
            --mask-type $mask_type \
            --mask-ratio $ratio

        echo "Completed mask_ratio=$ratio, mask_type=$mask_type"
        echo "Results saved to logs/inpaint_eval_results_${mask_type}_${ratio_str}.json"
    done
done

echo "All experiments completed!"