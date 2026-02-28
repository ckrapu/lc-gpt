#!/usr/bin/env python
"""Simple script to generate unconditional land cover images with varying temperatures."""

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from safetensors.torch import load_file
import sys
sys.path.append('..')

from RandAR.utils import instantiate_from_config
from RandAR.dataset.nlcd_dataset import NLCDDataset, detokenize


@dataclass
class Config:
    scripts_path: Path = Path(__file__).parent
    top_path: Path = scripts_path.parent
    config_path: Path = top_path / "configs/randar_nlcd_128_large.yaml"
    checkpoint_path: Path = top_path / "results/models/randar_nlcd_128_large/checkpoints/iter_180000"
    data_path: Path = top_path / "data/data_128_final.npz"
    save_path: Path = top_path / "results/visualizations/unconditional_generation.png"
    temperatures: tuple = (0.8, 0.9, 1.0, 1.1, 1.2)
    n_rows: int = 4
    top_k: int = 0
    top_p: float = 1.0
    num_inference_steps: int = 32
    tokenized_width: int = 64
    tokenized_height: int = 64
    d: int = 2
    raw_width: int = 128
    raw_height: int = 128
    figsize: tuple = (15, 12)
    

def main():
    cfg = Config()
    
    # Load model config and initialize
    with open(cfg.config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model = instantiate_from_config(config_dict['ar_model'])
    
    # Load checkpoint
    model_file = cfg.checkpoint_path / "model.safetensors"
    if model_file.exists():
        model.load_state_dict(load_file(model_file))
    else:
        print(f"Model file not found at {model_file}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    
    # Load decode table
    data = np.load(cfg.data_path)
    decode_table = data['decode_table']
    
    # Generate images for each temperature
    all_rgb_images = []
    
    for temp in cfg.temperatures:
        print(f"Generating with temperature {temp}")
        
        # Generate batch for this temperature
        cond = torch.zeros(cfg.n_rows, dtype=torch.long).to(device)
        aux = torch.zeros(cfg.n_rows, 2).to(device)
        
        print(f"Running generation with cond of shape {cond.shape} and aux of shape {aux.shape}")
        with torch.no_grad():
            gen_indices = model.generate(
                cond=cond,
                token_order=None,
                aux=aux,
                cfg_scales=[1.0, 1.0],
                num_inference_steps=cfg.num_inference_steps,
                temperature=temp,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )
        
        # Decode tokens to land cover
        gen_images = gen_indices.cpu().numpy()
        temp_rgb_images = []
        
        for i in range(cfg.n_rows):
            tokens = gen_images[i].reshape(cfg.tokenized_height, cfg.tokenized_width).flatten()
            decoded = detokenize(tokens, decode_table)
            
            # Convert to RGB
            h, w = decoded.shape
            rgb_img = np.zeros((h, w, 3), dtype=np.float32)
            for class_id, rgb in NLCDDataset.NLCD_TO_RGB.items():
                mask = decoded == class_id
                rgb_img[mask] = rgb
            temp_rgb_images.append(rgb_img)
        
        all_rgb_images.append(temp_rgb_images)
    
    # Plot 4x5 grid (4 rows, 5 columns for temperatures)
    fig, axes = plt.subplots(cfg.n_rows, len(cfg.temperatures), figsize=cfg.figsize)
    
    for col, temp in enumerate(cfg.temperatures):
        for row in range(cfg.n_rows):
            axes[row, col].imshow(all_rgb_images[col][row])
            axes[row, col].axis('off')
            
            # Add temperature label at the top of each column
            if row == 0:
                axes[row, col].text(64, -10, f'T = {temp}', 
                                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(cfg.save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Generated {cfg.n_rows * len(cfg.temperatures)} images")


if __name__ == "__main__":
    main()