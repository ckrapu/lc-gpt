#!/usr/bin/env python3
"""
Generate a visualization showing the probability of development from multiple
land use / land cover samples produced by a generative machine learning model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import rasterio
import glob

# NLCD class mapping and colors
nlcd_to_rgb = {
    11: (0.278, 0.420, 0.627),  # Open Water
    12: (0.820, 0.867, 0.976),  # Perennial Ice/Snow
    21: (0.867, 0.788, 0.788),  # Developed, Open Space
    22: (0.847, 0.576, 0.510),  # Developed, Low Intensity
    23: (0.929, 0.0, 0.0),      # Developed, Medium Intensity
    24: (0.667, 0.0, 0.0),      # Developed, High Intensity
    31: (0.698, 0.678, 0.639),  # Barren Land (Rock/Sand/Clay)
    41: (0.408, 0.667, 0.388),  # Deciduous Forest
    42: (0.110, 0.388, 0.188),  # Evergreen Forest
    43: (0.710, 0.788, 0.557),  # Mixed Forest
    51: (0.647, 0.549, 0.188),  # Dwarf Scrub
    52: (0.800, 0.729, 0.486),  # Shrub/Scrub
    71: (0.886, 0.886, 0.757),  # Grassland/Herbaceous
    72: (0.788, 0.788, 0.467),  # Sedge/Herbaceous
    73: (0.600, 0.757, 0.278),  # Lichens
    74: (0.467, 0.678, 0.576),  # Moss
    81: (0.859, 0.847, 0.239),  # Pasture/Hay
    82: (0.667, 0.439, 0.157),  # Cultivated Crops
    90: (0.729, 0.847, 0.918),  # Woody Wetlands
    95: (0.439, 0.639, 0.729),  # Emergent Herbaceous Wetlands
}

nlcd_to_name = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/Clay)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    51: "Dwarf Scrub",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous",
    73: "Lichens",
    74: "Moss",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}

# Developed land cover classes (21-24)
DEVELOPED_CLASSES = [21, 22, 23, 24]

def load_geotiff(filepath):
    """Load a GeoTiff file and return the data array."""
    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read first band
    return data

def create_nlcd_colormap_image(data):
    """Convert NLCD data to RGB image using the NLCD colormap."""
    # Create RGB image
    height, width = data.shape
    rgb_image = np.zeros((height, width, 3))
    
    # Map each NLCD class to its RGB color
    for nlcd_class, rgb_color in nlcd_to_rgb.items():
        mask = data == nlcd_class
        rgb_image[mask] = rgb_color
    
    return rgb_image

def calculate_development_probability(samples):
    """
    Calculate the probability of development for each pixel across samples.
    
    Args:
        samples: List of 2D arrays containing NLCD class values
    
    Returns:
        2D array with probability values (0-1) for each pixel
    """
    if not samples:
        raise ValueError("No samples provided")
    
    # Get shape from first sample
    height, width = samples[0].shape
    n_samples = len(samples)
    
    # Initialize development count array
    development_count = np.zeros((height, width))
    
    # Count developed pixels across samples
    for sample in samples:
        # Create mask for developed classes
        is_developed = np.isin(sample, DEVELOPED_CLASSES)
        development_count += is_developed
    
    # Calculate probability
    prob_development = development_count / n_samples
    
    return prob_development

def find_changing_bbox(samples, padding=30):
    """
    Find the bounding box of regions that change across samples.
    
    Args:
        samples: List of 2D arrays containing NLCD class values
        padding: Number of pixels to pad the bounding box
    
    Returns:
        Tuple of (min_row, max_row, min_col, max_col)
    """
    if len(samples) < 2:
        # If only one sample, return full extent
        h, w = samples[0].shape
        return 0, h, 0, w
    
    # Find pixels that differ between any pair of samples
    h, w = samples[0].shape
    changes_mask = np.zeros((h, w), dtype=bool)
    
    # Compare each sample with the first one
    reference = samples[0]
    for sample in samples[1:]:
        changes_mask |= (sample != reference)
    
    # Also compare pairs of samples to catch all changes
    for i in range(len(samples)-1):
        for j in range(i+1, len(samples)):
            changes_mask |= (samples[i] != samples[j])
    
    # Find bounding box of changes
    rows, cols = np.where(changes_mask)
    
    if len(rows) == 0:
        # No changes found, return center region
        return h//4, 3*h//4, w//4, 3*w//4
    
    min_row = max(0, rows.min() - padding)
    max_row = min(h, rows.max() + padding + 1)
    min_col = max(0, cols.min() - padding)
    max_col = min(w, cols.max() + padding + 1)
    
    return min_row, max_row, min_col, max_col

def main():
    """Main function to create the visualization."""
    # Setup paths
    data_dir = Path("data/geotiff_exports/resampled")
    output_dir = Path("results/visualizations/case_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find Fort Hood samples
    fort_hood_pattern = str(data_dir / "ft_hood_inpainted_100m_sample*.tif")
    fort_hood_files = sorted(glob.glob(fort_hood_pattern))
    
    if not fort_hood_files:
        raise FileNotFoundError(f"No Fort Hood samples found matching pattern: {fort_hood_pattern}")
    
    print(f"Found {len(fort_hood_files)} Fort Hood samples")
    
    # Load all samples
    samples = []
    for filepath in fort_hood_files:
        data = load_geotiff(filepath)
        samples.append(data)
        print(f"Loaded {Path(filepath).name}: shape {data.shape}")
    
    # Use first sample as the base Fort Hood domain
    fort_hood_base = samples[0]
    
    # Calculate probability of development
    prob_dev = calculate_development_probability(samples)
    
    # Find bounding box of changing regions
    min_row, max_row, min_col, max_col = find_changing_bbox(samples, padding=30)
    print(f"\nBounding box of changes: rows [{min_row}:{max_row}], cols [{min_col}:{max_col}]")
    print(f"Cropped region size: {max_row-min_row} x {max_col-min_col}")
    
    # Crop all data to bounding box
    fort_hood_cropped = fort_hood_base[min_row:max_row, min_col:max_col]
    prob_dev_cropped = prob_dev[min_row:max_row, min_col:max_col]
    samples_cropped = [s[min_row:max_row, min_col:max_col] for s in samples]
    
    # Create figure with gridspec - add row for colorbar
    fig = plt.figure(figsize=(20, 15))
    # Create a gridspec with 6 rows: 1 for colorbar, 4 for main plots, 1 for samples
    gs = gridspec.GridSpec(6, 6, figure=fig, hspace=0.05, wspace=0.02, 
                          height_ratios=[0.05, 1, 1, 1, 1, 1])
    
    # Row 0: Horizontal colorbar (spans all columns)
    ax_cbar = fig.add_subplot(gs[0, :])
    
    # Rows 1-4, columns 0-2: Fort Hood domain with NLCD colors
    ax_domain = fig.add_subplot(gs[1:5, :3])
    domain_rgb = create_nlcd_colormap_image(fort_hood_cropped)
    ax_domain.imshow(domain_rgb)
    ax_domain.set_xticks([])
    ax_domain.set_yticks([])
    # Add (a) label in top left corner
    ax_domain.text(0.02, 0.98, 'a', transform=ax_domain.transAxes,
                   fontsize=28, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
    
    # Rows 1-4, columns 3-5: Probability of development heatmap
    ax_prob = fig.add_subplot(gs[1:5, 3:])
    im = ax_prob.imshow(prob_dev_cropped, cmap='hot', vmin=0, vmax=1)
    ax_prob.set_xticks([])
    ax_prob.set_yticks([])
    # Add (b) label in top left corner
    ax_prob.text(0.02, 0.98, 'b', transform=ax_prob.transAxes,
                 fontsize=28, fontweight='bold', va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
    
    # Add horizontal colorbar in the dedicated row
    # Create a mappable object for the colorbar with explicit normalization
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap='hot')
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Probability of development', labelpad=10, fontsize=24)
    
    # Explicitly set the ticks and labels
    tick_locations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Move colorbar to top
    ax_cbar.xaxis.set_ticks_position('top')
    ax_cbar.xaxis.set_label_position('top')
    ax_cbar.tick_params(labelsize=20)
    
    # Row 5: Six sample visualizations
    for i in range(6):
        ax_sample = fig.add_subplot(gs[5, i])
        if i < len(samples_cropped):
            sample_rgb = create_nlcd_colormap_image(samples_cropped[i])
            ax_sample.imshow(sample_rgb)
        ax_sample.set_xticks([])
        ax_sample.set_yticks([])

        ax_sample.text(0.02, 0.98, f'{i+1}', transform=ax_sample.transAxes,
                 fontsize=20, fontweight='bold', va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5))
    
    # Use tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "fort_hood_development_probability.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total samples analyzed: {len(samples)}")
    print(f"  Full image dimensions: {fort_hood_base.shape}")
    print(f"  Cropped dimensions: {fort_hood_cropped.shape}")
    print(f"  Mean P(development) in cropped region: {prob_dev_cropped.mean():.3f}")
    print(f"  Max P(development): {prob_dev_cropped.max():.3f}")
    print(f"  Min P(development): {prob_dev_cropped.min():.3f}")
    print(f"  Pixels always developed in crop: {(prob_dev_cropped == 1.0).sum()}")
    print(f"  Pixels never developed in crop: {(prob_dev_cropped == 0.0).sum()}")

if __name__ == "__main__":
    main()