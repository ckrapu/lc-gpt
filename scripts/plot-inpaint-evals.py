#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Find all inpaint evaluation result files
log_dir = Path("logs")
eval_files = sorted(log_dir.glob("inpaint_eval_results*.json"))

# Extract data
mask_ratios = []
accuracies = []

for file in eval_files:
    # Extract mask ratio from filename (e.g., _0_2.json -> 0.2, _0_998.json -> 0.998)
    name_parts = file.stem.split('_')
    if len(name_parts) >= 3:
        ratio_str = '_'.join(name_parts[-2:])  # Get last two parts (e.g., "0_2")
        mask_ratio = float(ratio_str.replace('_', '.'))
        
        # Read JSON and extract accuracy
        with open(file) as f:
            data = json.load(f)
            accuracy = data['avg_accuracy']
            
        mask_ratios.append(mask_ratio)
        accuracies.append(accuracy)

# Sort by mask ratio
sorted_pairs = sorted(zip(mask_ratios, accuracies))
mask_ratios, accuracies = zip(*sorted_pairs)

# Calculate log10(1 - mask_ratio) for x-axis
x_values = [np.log10(1 - r) if r < 1 else np.nan for r in mask_ratios]

# Create plot
plt.figure(figsize=(5, 3))
plt.plot(x_values, accuracies, 'o-', linewidth=2, markersize=8, color='k')
plt.xlabel('Inpaint mask fraction ($\log_{10}$ scale)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.2)

# Set x-axis to show actual mask ratios
ax = plt.gca()
ax.set_xticks(x_values)
ax.set_xticklabels([f'{r:.1%}' for r in mask_ratios])
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()

plt.savefig('results/visualizations/inpaint_accuracy_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Processed {len(eval_files)} evaluation files")
print(f"Mask ratios: {mask_ratios}")
print(f"Accuracies: {accuracies}")