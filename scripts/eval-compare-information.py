#!/usr/bin/env python3
"""
Compare information content between language and land cover data.
"""

import numpy as np
import torch
from collections import Counter
import argparse
import os
import requests
from tqdm import tqdm
import tiktoken
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from RandAR.utils import instantiate_from_config
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Compare information content between language and land cover data")
    parser.add_argument("--text-path", type=str, default="data/tiny_shakespeare.txt")
    parser.add_argument("--config", type=str, default="configs/randar_nlcd_32.yaml")
    parser.add_argument("--max-language-tokens", type=int, default=None, help="Maximum language tokens to analyze")
    parser.add_argument("--max-land-cover-samples", type=int, default=None, help="Maximum land cover samples to analyze")
    parser.add_argument("--tokenizer", type=str, default="tiktoken", 
                        choices=["gpt2", "gpt-3.5-turbo", "tiktoken"], 
                        help="Tokenizer to use for language data")
    args = parser.parse_args()
    
    # Download text data if needed
    if not os.path.exists(args.text_path):
        print("Downloading text data...")
        os.makedirs(os.path.dirname(args.text_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        with open(args.text_path, 'w') as f:
            f.write(text)
    
    # Analyze language data
    print("\n=== Language Data ===")
    with open(args.text_path, 'r') as f:
        text = f.read()
    
    # Get tokenizer
    print("Using tiktoken cl100k_base tokenizer (GPT-4)")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    vocab_size = tokenizer.n_vocab
    
    if args.max_language_tokens:
        tokens = tokens[:args.max_language_tokens]
    
    print(f"Text length: {len(text)} characters")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Compression ratio: {len(text) / len(tokens):.2f} characters per token")
    
    # Compute bigram statistics efficiently
    print("Computing language bigram statistics...")
    bigram_counts = Counter()
    unigram_counts = Counter(tokens)
    
    # Process in chunks for memory efficiency
    chunk_size = 100000
    for i in tqdm(range(0, len(tokens) - 1, chunk_size)):
        end = min(i + chunk_size + 1, len(tokens))
        chunk = tokens[i:end]
        for j in range(len(chunk) - 1):
            bigram_counts[(chunk[j], chunk[j+1])] += 1
    
    print(f"Unique tokens used: {len(unigram_counts)}")
    print(f"Unique bigrams: {len(bigram_counts)}")
    
    # Compute entropies
    total_u = sum(unigram_counts.values())
    total_b = sum(bigram_counts.values())
    
    h_marginal = -sum((c/total_u) * np.log2(c/total_u) for c in unigram_counts.values() if c > 0)
    h_joint = -sum((c/total_b) * np.log2(c/total_b) for c in bigram_counts.values() if c > 0)
    
    # Fast conditional entropy computation
    h_conditional = h_joint - h_marginal
    
    print(f"\nEntropy Statistics:")
    print(f"  Marginal entropy H(X): {h_marginal:.4f} bits")
    print(f"  Joint entropy H(X1,X2): {h_joint:.4f} bits")
    print(f"  Conditional entropy H(X2|X1): {h_conditional:.4f} bits")
    print(f"  Information per token: {h_conditional:.4f} bits")
    
    lang_info = h_conditional
    
    # Analyze land cover data
    print("\n=== Land Cover Data ===")

    
    config = OmegaConf.load(args.config)
    dataset = instantiate_from_config(config.dataset)
    
    if args.max_land_cover_samples:
        indices = list(range(min(args.max_land_cover_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"Dataset size: {len(dataset)} samples")
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    print(f"Vocabulary size: {base_dataset.vocab_size}")
    
    # Get image dimensions
    first_sample, _, _ = dataset[0] if hasattr(dataset, '__getitem__') else dataset.dataset[0]
    image_size = int(np.sqrt(len(first_sample)))
    print(f"Image size: {image_size}x{image_size} = {len(first_sample)} tokens per image")
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    bigram_counts = Counter()
    unigram_counts = Counter()
    
    print("Computing land cover bigram statistics...")
    for batch, _, _ in tqdm(loader):
        for sample in batch:
            tokens = sample.numpy().flatten()
            unigram_counts.update(tokens)
            
            # Reshape to 2D
            sample_2d = tokens.reshape(image_size, image_size)
            
            # Horizontal bigrams
            for i in range(image_size):
                for j in range(image_size - 1):
                    bigram_counts[(sample_2d[i, j], sample_2d[i, j+1])] += 1
            
            # Vertical bigrams
            for i in range(image_size - 1):
                for j in range(image_size):
                    bigram_counts[(sample_2d[i, j], sample_2d[i+1, j])] += 1
    
    print(f"\nTotal tokens: {sum(unigram_counts.values())}")
    print(f"Unique bigrams: {len(bigram_counts)}")
    print(f"Total bigrams: {sum(bigram_counts.values())}")
    
    # Compute entropies
    total_u = sum(unigram_counts.values())
    total_b = sum(bigram_counts.values())
    
    h_marginal = -sum((c/total_u) * np.log2(c/total_u) for c in unigram_counts.values() if c > 0)
    h_joint = -sum((c/total_b) * np.log2(c/total_b) for c in bigram_counts.values() if c > 0)
    h_conditional = h_joint - h_marginal
    
    print(f"\nEntropy Statistics:")
    print(f"  Marginal entropy H(X): {h_marginal:.4f} bits")
    print(f"  Joint entropy H(X1,X2): {h_joint:.4f} bits")
    print(f"  Conditional entropy H(X2|X1): {h_conditional:.4f} bits")
    print(f"  Information per token: {h_conditional:.4f} bits")
    
    land_info = h_conditional
    
    # Summary
    print("\n=== Comparison Summary ===")
    print(f"\nLanguage Data (using {args.tokenizer} tokenizer):")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Unique tokens in sample: {len(unigram_counts)}")
    print(f"  Information per token: {lang_info:.4f} bits")
    print(f"  Perplexity (2^H): {2**lang_info:.2f}")
    
    print(f"\nLand Cover Data:")
    print(f"  Vocabulary size: {base_dataset.vocab_size}")
    print(f"  Information per token: {land_info:.4f} bits")
    print(f"  Perplexity (2^H): {2**land_info:.2f}")


if __name__ == "__main__":
    main()