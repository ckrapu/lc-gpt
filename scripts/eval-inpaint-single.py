import numpy as np
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("./")

from accelerate import Accelerator
from omegaconf import OmegaConf
from RandAR.utils import instantiate_from_config
from RandAR.utils.inpainting import generate_inpainting
from RandAR.model.nlcd_tokenizer import NLCDTokenizer


def nlcd_to_rgb(data):
    """Convert NLCD data values to RGB image using the colormap."""
    # Determine vocab size from data
    vocab_size = int(data.max()) + 1
    return NLCDTokenizer(vocab_size=vocab_size).nlcd_to_rgb(data)

def generate_mask(size, mask_ratio, mask_type="interior", seed=None):
    """Generate a mask for inpainting based on the specified type.
    
    Args:
        size: Image size (assumes square images)
        mask_ratio: Ratio of pixels to mask
        mask_type: Type of mask - 'interior' or 'random'
        seed: Random seed for reproducibility (only used for 'random' type)
    
    Returns:
        np.ndarray: Boolean mask where True indicates pixels to inpaint
    """
    mask = np.zeros((size, size), dtype=bool)
    
    if mask_type == "interior":

        # For example if we have mask_ratio = 0.40
        # and the image size is 32x32, we would expect the number
        # of masked pixels to be 0.40 * 32 * 32 = 409.6
        # so we figure out the size of that region as a square
        # then mask out a square chunk in the middle with that size.
        masked_region_size = np.sqrt(mask_ratio * (size*size))
        dist_from_edge = int(masked_region_size / 2)

        # Create mask for interior rectangular region
        start_h = dist_from_edge    
        end_h = size - dist_from_edge
        start_w = dist_from_edge
        end_w = size - dist_from_edge
        print(f"Horizontal observed width: {start_w} , vertical observed height: {start_h}")
        mask[start_h:end_h, start_w:end_w] = True
    
    elif mask_type == "random":
        # Create mask with randomly selected pixels
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        total_pixels = size * size
        num_masked = int(total_pixels * mask_ratio)
        
        # Generate random indices to mask
        flat_indices = rng.choice(total_pixels, num_masked, replace=False)
        
        # Convert flat indices to 2D coordinates
        for idx in flat_indices:
            row = idx // size
            col = idx % size
            mask[row, col] = True
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    return mask

def visualize_inpainting_samples(model, dataset, device, args):
    """
    Visualize inpainting samples for a few random images.
    Each row corresponds to a single image from the dataset; the first column is the ground truth image,
    and the subsequent columns are the inpainted samples.

    If the number of visualized images is 4 and the number of samples is 3
    then the figure should have 4 rows and 2+1 columns.

    Args:
        model: The RandAR model
        dataset: Dataset to sample from
        device: Device to run on
        args: Command line arguments
    """
    # Sample n_plots random images
    dataset_indices = random.sample(range(len(dataset)), args.n_plots)
    
    # Create figure
    fig, axes = plt.subplots(args.n_plots, args.n_samples_viz + 1, figsize=(2*(args.n_samples_viz + 1), 2*args.n_plots))
    fig.patch.set_facecolor('black')
    
    if args.n_plots == 1:
        axes = axes[None, :]
    
    for img_idx, dataset_idx in enumerate(dataset_indices):
        # Load image
        real_tokens, label, _ = dataset[dataset_idx]
        
        # Generate mask based on mask type
        # Use a deterministic seed for each image to ensure consistency across samples
        mask_seed = args.seed + dataset_idx if args.mask_type == "random" else None
        mask = generate_mask(args.size, args.mask_ratio, args.mask_type, seed=mask_seed)
        
        # Get positions and tokens
        known_positions = []
        known_tokens = []
        unknown_positions = []
        
        for i in range(args.size):
            for j in range(args.size):
                pos = i * args.size + j
                if mask[i, j]:  # Masked region (to be inpainted)
                    unknown_positions.append(pos)
                else:  # Unmasked region (visible context)
                    known_positions.append(pos)
                    known_tokens.append(real_tokens[pos].item())
        
        # Convert to tensors
        bs = 1
        cond = torch.tensor([0], dtype=torch.long, device=device)  # Dummy class
        known_positions = torch.tensor(known_positions, dtype=torch.long)
        known_tokens = torch.tensor(known_tokens, dtype=torch.long)
        unknown_positions = torch.tensor(unknown_positions, dtype=torch.long)
        
        # Show original image
        real_img = real_tokens.reshape(args.size, args.size).numpy()
        axes[img_idx, 0].imshow(nlcd_to_rgb(real_img), interpolation='nearest')
        axes[img_idx, 0].axis('off')
        
        # Generate n_samples_viz inpainted versions
        for j in range(args.n_samples_viz):
            with torch.no_grad():
                gen_indices = generate_inpainting(
                    model=model,
                    cond=cond,
                    known_tokens=known_tokens,
                    known_positions=known_positions,
                    unknown_positions=unknown_positions,
                    cfg_scales=(args.cfg_scale, args.cfg_scale),
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                
                # Convert to 2D image
                gen_indices_np = gen_indices[0].cpu().numpy()
                gen_img = np.zeros((args.size, args.size))
                
                # Fill in known pixels
                for pos, token in zip(known_positions, known_tokens):
                    row, col = pos // args.size, pos % args.size
                    gen_img[row, col] = token.item()
                
                # Fill in generated pixels
                for pos in unknown_positions:
                    row, col = pos // args.size, pos % args.size
                    gen_img[row, col] = gen_indices_np[pos]
                
                # Show generated image
                axes[img_idx, j+1].imshow(nlcd_to_rgb(gen_img), interpolation='nearest')
                axes[img_idx, j+1].axis('off')

    
    plt.tight_layout()
    plt.savefig(args.save_plot, bbox_inches='tight', facecolor='black')
    print(f"Plot saved to: {args.save_plot}")
    
    plt.show()

def evaluate_inpainting_single_image(model, 
                                   real_tokens: torch.Tensor,
                                   mask: np.ndarray,
                                   device: torch.device,
                                   size: int = 16,
                                   cfg_scales: tuple = (1.0, 1.0),
                                   temperature: float = 1.0,
                                   top_k: int = 0,
                                   top_p: float = 1.0):
    """
    Evaluate inpainting on a single image and return metrics.
    
    Args:
        model: The RandAR model
        real_tokens: Ground truth tokens [size*size]
        mask: Boolean mask indicating pixels to inpaint [size, size]
        device: Device to run on
        size: Image size
        cfg_scales: CFG scale range
        temperature: Sampling temperature  
        top_k: Top-k sampling
        top_p: Top-p sampling
    
    Returns:
        dict: Contains 'accuracy', 'log_loss', 'num_masked_pixels'
    """
    model.eval()
    
    # Get positions and tokens
    known_positions = []
    known_tokens = []
    unknown_positions = []
    unknown_true_tokens = []
    
    for i in range(size):
        for j in range(size):
            pos = i * size + j
            if mask[i, j]:  # Masked region (to be inpainted)
                unknown_positions.append(pos)
                unknown_true_tokens.append(real_tokens[pos].item())
            else:  # Unmasked region (visible context)
                known_positions.append(pos)
                known_tokens.append(real_tokens[pos].item())
    
    # Convert to tensors
    bs = 1
    cond = torch.tensor([0], dtype=torch.long, device=device)  # Dummy class
    known_positions = torch.tensor(known_positions, dtype=torch.long)
    known_tokens = torch.tensor(known_tokens, dtype=torch.long)
    unknown_positions = torch.tensor(unknown_positions, dtype=torch.long)
    unknown_true_tokens = torch.tensor(unknown_true_tokens, dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Get inpainting results
        gen_indices = generate_inpainting(
            model=model,
            cond=cond,
            known_tokens=known_tokens,
            known_positions=known_positions,
            unknown_positions=unknown_positions,
            cfg_scales=cfg_scales,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Extract predicted tokens for masked positions
        gen_indices_np = gen_indices[0].cpu().numpy()
        predicted_tokens = torch.tensor([gen_indices_np[pos] for pos in unknown_positions], 
                                      dtype=torch.long, device=device)
        
        # Calculate accuracy
        accuracy = (predicted_tokens == unknown_true_tokens).float().mean().item()
        
        # For log loss, we need to get the logits for the true tokens
        # We'll run a forward pass with the ground truth sequence to get logits
        log_loss = calculate_log_loss_for_positions(
            model, cond, real_tokens, unknown_positions, device, size
        )
        
        return {
            'accuracy': accuracy,
            'log_loss': log_loss,
            'num_masked_pixels': len(unknown_positions),
            'predicted_tokens': predicted_tokens.cpu().numpy(),
            'true_tokens': unknown_true_tokens.cpu().numpy()
        }


def calculate_log_loss_for_positions(model, cond, real_tokens, positions, device, size):
    """
    Calculate log loss for specific positions using teacher forcing.
    
    Args:
        model: The RandAR model
        cond: Condition tensor
        real_tokens: Ground truth tokens for the full image
        positions: Positions to calculate loss for
        device: Device to run on
        size: Image size
    
    Returns:
        float: Average log loss for the specified positions
    """
    # Ensure real_tokens is on the correct device
    real_tokens = real_tokens.to(device)
    
    # Create a random token order that includes our positions
    # We'll put our target positions at the end so we can get their logits
    all_positions = torch.arange(size * size, device=device)
    
    # Handle positions input (could be list or tensor)
    if isinstance(positions, torch.Tensor):
        target_positions = positions.to(device, dtype=torch.long)
    else:
        target_positions = torch.tensor(positions, device=device, dtype=torch.long)
    
    # Create token order: non-target positions first, then target positions
    non_target_mask = torch.ones(size * size, dtype=torch.bool, device=device)
    non_target_mask[target_positions] = False
    non_target_positions = all_positions[non_target_mask]
    
    # Shuffle non-target positions and put target positions at the end
    non_target_shuffled = non_target_positions[torch.randperm(len(non_target_positions))]
    token_order = torch.cat([non_target_shuffled, target_positions]).unsqueeze(0)
    
    # Get the tokens in this order
    ordered_tokens = real_tokens[token_order[0]]
    
    # Run forward pass to get logits
    logits, _, _ = model(
        idx=ordered_tokens.unsqueeze(0), 
        cond_idx=cond,
        token_order=token_order,
        targets=ordered_tokens.unsqueeze(0)
    )
    
    # Extract logits for target positions (they're at the end)
    num_targets = len(target_positions)
    target_logits = logits[0, -num_targets:]  # [num_targets, vocab_size]
    target_true_tokens = ordered_tokens[-num_targets:]  # [num_targets]
    
    # Calculate cross-entropy loss
    log_loss = F.cross_entropy(target_logits, target_true_tokens, reduction='mean').item()
    
    return log_loss


def evaluate_inpainting_dataset(model, dataset, device, args):
    """
    Evaluate inpainting on n random images from the dataset.
    
    Args:
        model: The RandAR model  
        dataset: Dataset to sample from
        device: Device to run on
        args: Command line arguments
    
    Returns:
        dict: Aggregated results
    """
    # Sample n random images
    dataset_indices = random.sample(range(len(dataset)), min(args.n_images, len(dataset)))
    
    results = {
        'accuracies': [],
        'log_losses': [],
        'num_masked_pixels': [],
        'image_results': []
    }
    
    print(f"Evaluating inpainting on {len(dataset_indices)} random images...")
    
    for i, dataset_idx in enumerate(tqdm(dataset_indices, desc="Evaluating images")):
        # Load image
        real_tokens, label, _ = dataset[dataset_idx]
        
        # Generate mask based on mask type
        # Use a deterministic seed for each image to ensure reproducibility
        mask_seed = args.seed + dataset_idx if args.mask_type == "random" else None
        mask = generate_mask(args.size, args.mask_ratio, args.mask_type, seed=mask_seed)
        
        # Evaluate this image
        image_result = evaluate_inpainting_single_image(
            model=model,
            real_tokens=real_tokens,
            mask=mask,
            device=device,
            size=args.size,
            cfg_scales=(args.cfg_scale, args.cfg_scale),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Store results
        results['accuracies'].append(image_result['accuracy'])
        results['log_losses'].append(image_result['log_loss'])
        results['num_masked_pixels'].append(image_result['num_masked_pixels'])
        results['image_results'].append({
            'dataset_idx': dataset_idx,
            'accuracy': image_result['accuracy'],
            'log_loss': image_result['log_loss'],
            'num_masked': image_result['num_masked_pixels']
        })
        
        if args.verbose and i < 5:  # Print details for first 5 images
            print(f"  Image {dataset_idx}: Accuracy={image_result['accuracy']:.4f}, "
                  f"Log Loss={image_result['log_loss']:.4f}, "
                  f"Masked Pixels={image_result['num_masked_pixels']}")
    
    # Calculate aggregated metrics
    avg_accuracy = np.mean(results['accuracies'])
    std_accuracy = np.std(results['accuracies'])
    avg_log_loss = np.mean(results['log_losses'])
    std_log_loss = np.std(results['log_losses'])
    total_masked_pixels = np.sum(results['num_masked_pixels'])
    
    return {
        'mask_type': args.mask_type,
        'mask_ratio': args.mask_ratio,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_log_loss': avg_log_loss,
        'std_log_loss': std_log_loss,
        'total_masked_pixels': int(total_masked_pixels),
        'n_images': len(dataset_indices),
        'individual_results': results['image_results']
    }


def main(args):
    # Set random seeds for reproducibility
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
    print(f"Mask type: {args.mask_type}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Number of test images: {args.n_images}")
    print(f"Random seed: {args.seed}")
    
    # Create dataset
    dataset = instantiate_from_config(config.dataset)
    vocab_size = dataset.vocab_size
    
    # Update config with actual vocab size
    config.ar_model.params.vocab_size = vocab_size
    
    # Create model
    model = instantiate_from_config(config.ar_model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    else:
        print("No checkpoint provided, using randomly initialized model.")
    
    # Ensure model is fully on the target device
    model = model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    
    if args.visualize:
        visualize_inpainting_samples(model, dataset, device, args)
    else:
        # Run evaluation
        results = evaluate_inpainting_dataset(model, dataset, device, args)
        
        # Print results
        print("\n" + "="*60)
        print("NLCD INPAINTING EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset size: {len(dataset)}")
        print(f"Evaluated images: {results['n_images']}")
        print(f"Total masked pixels: {results['total_masked_pixels']}")
        print(f"Average masked pixels per image: {results['total_masked_pixels'] / results['n_images']:.1f}")
        print()
        print(f"ACCURACY (Token-level exact match):")
        print(f"  Mean: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"  Range: [{min([r['accuracy'] for r in results['individual_results']]):.4f}, "
              f"{max([r['accuracy'] for r in results['individual_results']]):.4f}]")
        print()
        print(f"LOG LOSS (Cross-entropy on true tokens):")
        print(f"  Mean: {results['avg_log_loss']:.4f} ± {results['std_log_loss']:.4f}")
        print(f"  Range: [{min([r['log_loss'] for r in results['individual_results']]):.4f}, "
              f"{max([r['log_loss'] for r in results['individual_results']]):.4f}]")
        print("="*60)
        
        # Save detailed results if requested
        if args.save_results:
            import json
            output_file = args.save_results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NLCD inpainting model accuracy and log loss")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n-images", type=int, default=100, help="Number of random images to evaluate")
    parser.add_argument("--size", type=int, default=32, help="Size of each NLCD image (default: 32)")
    parser.add_argument("--mask-ratio", type=float, default=0.5, 
                       help="Ratio of interior region to mask for inpainting (default: 0.5)")
    parser.add_argument("--mask-type", type=str, default="interior", choices=["interior", "random"],
                       help="Type of mask to use: 'interior' for center rectangle, 'random' for random pixels (default: interior)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                       help="Device to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale for sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results for first few images")
    parser.add_argument("--save-results", type=str, default=None, 
                       help="Path to save detailed results as JSON")
    parser.add_argument("--visualize", action="store_true", help="Visualize inpainting samples")
    parser.add_argument("--n-plots", type=int, default=3, help="Number of random images to visualize")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of inpainting samples per image")
    parser.add_argument("--save-plot", type=str, default='results/visualizations/inpaint.png', help="Path to save visualization plot")
    
    args = parser.parse_args()
    main(args) 