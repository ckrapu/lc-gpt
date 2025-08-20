import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append("./")

from omegaconf import OmegaConf
from RandAR.utils import instantiate_from_config


def calculate_perplexity(model, tokenizer, data_loader, device, num_samples=None):
    """Calculate perplexity of the model on the given dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(tqdm(data_loader, desc="Calculating perplexity")):
            if num_samples is not None and samples_processed >= num_samples:
                break
                
            x, y = x.to(device), y.to(device)
            image_tokens = x  # Already flattened in NLCD dataset
            cond = y.reshape(-1)
            
            # Forward pass
            logits, loss, _ = model(image_tokens, cond, targets=image_tokens)
            
            # Accumulate loss and token count
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
            samples_processed += batch_size
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss, samples_processed


def main(args):
    # Load config
    config = OmegaConf.load(args.config)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    
    print(f"Loading config from: {args.config}")
    print(f"Using checkpoint: {args.gpt_ckpt}")
    print(f"Device: {device}")
    print(f"Evaluating on {args.n if args.n else 'full'} dataset")
    
    # Create dataset
    dataset = instantiate_from_config(config.dataset)
    
    # Limit dataset size if specified
    if args.n is not None:
        # Create a subset of the dataset
        subset_indices = list(range(min(args.n, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
        drop_last=False,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocabulary size: {dataset.dataset.vocab_size if hasattr(dataset, 'dataset') else dataset.vocab_size}")
    
    # Create model
    model = instantiate_from_config(config.ar_model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint
    if args.gpt_ckpt:
        print(f"Loading checkpoint from: {args.gpt_ckpt}")
        
        # Handle both single checkpoint file and accelerate checkpoint directory
        if os.path.isdir(args.gpt_ckpt):
            # Accelerate checkpoint directory
            from accelerate import Accelerator
            # Use CPU for accelerator if device is CPU
            mixed_precision = "no" if args.device == "cpu" else "bf16"
            accelerator = Accelerator(mixed_precision=mixed_precision)
            model = accelerator.prepare(model)
            accelerator.load_state(args.gpt_ckpt)
            model = accelerator.unwrap_model(model)
            # Ensure model is on the correct device after unwrapping
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
    
    # Create tokenizer (not used for loss calculation but needed for compatibility)
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    
    # Calculate perplexity
    print("Starting perplexity calculation...")
    perplexity, avg_loss, samples_processed = calculate_perplexity(
        model, tokenizer, data_loader, device, args.n
    )
    
    # Print results
    print("\n" + "="*50)
    print("PERPLEXITY EVALUATION RESULTS")
    print("="*50)
    print(f"Samples processed: {samples_processed}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("="*50)
    
    # Save results to file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Samples processed: {samples_processed}\n")
            f.write(f"Average loss: {avg_loss:.4f}\n")
            f.write(f"Perplexity: {perplexity:.4f}\n")
        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate perplexity of RandAR model on NLCD data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n", type=int, default=None, help="Number of samples to evaluate (default: full dataset)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for evaluation")
    parser.add_argument("--output-file", type=str, default=None, help="Output file to save results")
    
    args = parser.parse_args()
    main(args)
