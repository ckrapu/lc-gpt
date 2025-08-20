#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
sys.path.append("./")

from accelerate import Accelerator
from omegaconf import OmegaConf
from RandAR.utils import instantiate_from_config
from RandAR.utils.inpainting import generate_inpainting

plt.style.use('dark_background')

def nlcd_to_rgb(data):
    """Convert NLCD data values to RGB image using the colormap."""
    from RandAR.model.nlcd_tokenizer import NLCDTokenizer
    return NLCDTokenizer().nlcd_to_rgb(data)

def main():
    parser = argparse.ArgumentParser(description="Create Markov chain sampling animation using RandAR model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--t", type=int, default=100, help="Number of Gibbs sampling iterations")
    parser.add_argument("--output-file", type=str, default="results/visualizations/sample_chain.gif", help="Output animation filename (optional)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--size", type=int, default=32, help="Size of each NLCD image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--fps", type=int, default=30, help="Animation frame rate")
    parser.add_argument("--n-chains", type=int, default=1, help="Number of parallel Markov chains to run")
    parser.add_argument("--save-array", type=str, default="results/visualizations/samples.npz", help="Save samples as numpy array to this file (.npz)")
    parser.add_argument("--frame-interval", type=int, default=10, help="Save animation frames every N samples (default: 10)")
    
    args = parser.parse_args()
    
    # No validation needed since both have defaults now
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load config
    config = OmegaConf.load(args.config)
    device = torch.device(args.device)
    
    print(f"Loading config from: {args.config}")
    print(f"Using checkpoint: {args.gpt_ckpt}")
    print(f"Device: {device}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"Number of parallel chains: {args.n_chains}")
    print(f"Gibbs sampling iterations: {args.t}")
    if args.output_file:
        print(f"Animation output file: {args.output_file}")
        print(f"Frame interval: {args.frame_interval} (save every {args.frame_interval}th step)")
    if args.save_array:
        print(f"Array output file: {args.save_array}")
    
    # Create dataset to get vocab info
    dataset = instantiate_from_config(config.dataset)
    vocab_size = dataset.vocab_size
    
    # Update config with actual vocab size
    config.ar_model.params.vocab_size = vocab_size
    
    # Create model
    model = instantiate_from_config(config.ar_model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Load checkpoint
    if args.gpt_ckpt:
        print(f"Loading checkpoint from: {args.gpt_ckpt}")
        
        # Handle both single checkpoint file and accelerate checkpoint directory
        if os.path.isdir(args.gpt_ckpt):
            # Accelerate checkpoint directory
            mixed_precision = "no" if args.device == "cpu" else "bf16"
            accelerator = Accelerator(mixed_precision=mixed_precision)
            model = accelerator.prepare(model)
            accelerator.load_state(args.gpt_ckpt)
            model = accelerator.unwrap_model(model)
            model = model.to(device)
        else:
            # Single checkpoint file
            ckpt = torch.load(args.gpt_ckpt, map_location=device)
            if 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict)
        
        print("Checkpoint loaded successfully!")
    
    # Ensure model is fully on the target device
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # Initialize with random images from the dataset for multiple chains
    n_chains = args.n_chains
    current_images = torch.zeros(n_chains, args.size, args.size, dtype=torch.long, device=device)
    
    print(f"Initializing {n_chains} parallel Markov chains")
    for chain_idx in range(n_chains):
        random_idx = random.randint(0, len(dataset) - 1)
        real_tokens, _, _ = dataset[random_idx]
        current_images[chain_idx] = real_tokens.reshape(args.size, args.size).to(device)
        print(f"Chain {chain_idx}: Starting from dataset index {random_idx}")
    
    # Initialize storage for samples: shape (T+1, N, H, W)
    all_samples = torch.zeros(args.t + 1, n_chains, args.size, args.size, dtype=torch.long)
    all_samples[0] = current_images.cpu()
    
    # Create frames list for animation (all chains)
    all_frames = []  # List of lists: all_frames[chain_idx][time_step]
    for chain_idx in range(n_chains):
        frames = []
        frames.append(nlcd_to_rgb((current_images[chain_idx].cpu().numpy() + 1)))
        all_frames.append(frames)
    
    print("Starting Gibbs sampling...")
    
    # Main Gibbs sampling loop
    cond = torch.tensor([0], dtype=torch.long, device=device)  # Dummy condition
    
    for step in tqdm(range(args.t), desc="Gibbs sampling"):
        with torch.no_grad():
            # Update each chain independently
            for chain_idx in range(n_chains):
                # Pick random position for this chain
                pos = random.randint(0, args.size * args.size - 1)
                row, col = pos // args.size, pos % args.size
                
                # Prepare conditioning: all positions except target
                known_positions = torch.tensor([p for p in range(args.size * args.size) if p != pos], device=device)
                known_tokens = current_images[chain_idx].flatten()[known_positions]
                unknown_positions = torch.tensor([pos], device=device)
                
                gen_indices = generate_inpainting(
                    model=model,
                    cond=cond,
                    known_tokens=known_tokens,
                    known_positions=known_positions,
                    unknown_positions=unknown_positions,
                    cfg_scales=[1.0, 1.0],
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                
                # Extract the sampled token for the target position
                new_token = gen_indices[0, pos].item()
                
                # Update image for this chain
                current_images[chain_idx, row, col] = new_token
            
            # Store samples for all chains at this timestep
            all_samples[step + 1] = current_images.cpu()
            
            # Add frames for animation (all chains) - only every frame_interval steps
            if step % args.frame_interval == 0 or step == args.t - 1:  # Always save last frame too
                for chain_idx in range(n_chains):
                    all_frames[chain_idx].append(nlcd_to_rgb((current_images[chain_idx].cpu().numpy())))
    
    print(f"Gibbs sampling complete. Generated {len(all_frames[0])} frames per chain (every {args.frame_interval} steps).")
    print(f"Sample array shape: {all_samples.shape} (T, N, H, W)")
    
    # Save samples as numpy array if requested
    if args.save_array:
        print(f"Saving samples to {args.save_array}...")
        np.savez_compressed(args.save_array, 
                          samples=all_samples.numpy(),
                          metadata={
                              'n_chains': n_chains,
                              'n_iterations': args.t,
                              'image_size': args.size,
                              'temperature': args.temperature,
                              'top_k': args.top_k,
                              'top_p': args.top_p,
                              'seed': args.seed,
                              'frame_interval': args.frame_interval
                          })
        print(f"Samples saved to {args.save_array}")
    
    # Create animation only if output file is specified
    if args.output_file:
        print("Creating multi-chain animation...")
        
        # Calculate grid layout for chains
        import math
        n_cols = min(n_chains, 4)  # Max 4 columns
        n_rows = math.ceil(n_chains / n_cols)
        
        # Calculate figure size
        fig_width = n_cols * 4  # 4 inches per column
        fig_height = n_rows * 4  # 4 inches per row
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('black')
        
        # Handle different subplot configurations
        if n_chains == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        # Initialize images for each subplot
        ims = []
        for chain_idx in range(n_chains):
            ax = axes[chain_idx]
            ax.axis('off')
            ax.set_facecolor('black')
            
            im = ax.imshow(all_frames[chain_idx][0], interpolation='nearest')
            ims.append(im)
        
        # Hide unused subplots
        for i in range(n_chains, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        def animate(frame_idx):
            for chain_idx in range(n_chains):
                ims[chain_idx].set_data(all_frames[chain_idx][frame_idx])
            
            return ims
        
        # Create animation
        total_frames = len(all_frames[0])
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames, interval=1000//args.fps, repeat=True, blit=False
        )
        
        # Save animation
        print(f"Saving animation to {args.output_file}...")
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(args.output_file, writer=writer)
        
        print(f"Animation saved successfully to {args.output_file}!")
        print(f"Animation contains {total_frames} frames at {args.fps} fps")
        print(f"Total duration: {total_frames / args.fps:.1f} seconds")

if __name__ == "__main__":
    main() 