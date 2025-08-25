#!/usr/bin/env python3
import numpy as np
import argparse
from pathlib import Path

def extract_patches(images, D):
    """Extract all DxD patches from images."""
    N, H, W = images.shape
    patches = images.reshape(N, H//D, D, W//D, D).transpose(0,1,3,2,4).reshape(-1, D, D)
    return patches

def tokenize_images(images, decode_table, D):
    """Tokenize images using the decode table."""
    N, H, W = images.shape
    patches = extract_patches(images, D)
    
    # Flatten patches and decode table for comparison
    patches_flat = patches.reshape(len(patches), -1)
    decode_flat = decode_table.reshape(len(decode_table), -1)
    
    # Find indices of patches in decode table
    tokens = np.zeros(len(patches), dtype=np.int32)
    for i, patch in enumerate(patches_flat):
        tokens[i] = np.where((decode_flat == patch).all(axis=1))[0][0]
    
    # Reshape to tokenized images
    return tokens.reshape(N, H//D, W//D)

def combine_datasets(image_size, ratios, output_file, tokenizer_downsample_ratio=2):
    """Combine multiple resolution datasets with unified tokenization."""
    all_train_images = []
    all_test_images = []
    all_train_data = []
    all_test_data = []
    all_train_coords = []
    all_test_coords = []
    train_labels = []
    test_labels = []
    
    # First pass: collect all land cover images
    for ratio in ratios:
        input_file = f"../data/data_size{image_size}_ratio{ratio}.npz"
        print(f"Loading {input_file}...")
        
        try:
            data = np.load(input_file)
            
            # Extract land cover channel from train_data and test_data
            train_lc = data['train_data'][:, 0, :, :]  # First channel is land cover
            test_lc = data['test_data'][:, 0, :, :]
            
            all_train_images.append(train_lc)
            all_test_images.append(test_lc)
            
            # Store full data for later
            all_train_data.append(data['train_data'])
            all_test_data.append(data['test_data'])
            all_train_coords.append(data['train_coords'])
            all_test_coords.append(data['test_coords'])
            
            # Create labels
            train_labels.extend([ratio] * len(train_lc))
            test_labels.extend([ratio] * len(test_lc))
            
            print(f"  Added {len(train_lc)} train and {len(test_lc)} test samples with ratio {ratio}")
            
        except FileNotFoundError:
            print(f"  Warning: {input_file} not found, skipping...")
            continue
    
    if not all_train_images:
        print("No data files found!")
        return
    
    # Concatenate all land cover images
    train_images = np.concatenate(all_train_images, axis=0)
    test_images = np.concatenate(all_test_images, axis=0)
    
    print(f"\nCreating unified decode table from {len(train_images)} train and {len(test_images)} test images...")
    
    # Create unified decode table from all images
    D = tokenizer_downsample_ratio
    all_patches = np.vstack([extract_patches(train_images, D), extract_patches(test_images, D)])
    decode_table, inverse = np.unique(all_patches.reshape(len(all_patches), -1), 
                                      axis=0, return_inverse=True)
    decode_table = decode_table.reshape(-1, D, D)
    
    print(f"Created decode table with {len(decode_table)} unique tokens")
    
    # Tokenize images using unified decode table
    n_train_patches = (train_images.shape[0] * train_images.shape[1] * train_images.shape[2]) // (D * D)
    train_tokens = inverse[:n_train_patches]
    test_tokens = inverse[n_train_patches:]
    
    train_images_tokenized = train_tokens.reshape(train_images.shape[0], train_images.shape[1]//D, train_images.shape[2]//D)
    test_images_tokenized = test_tokens.reshape(test_images.shape[0], test_images.shape[1]//D, test_images.shape[2]//D)
    
    # Concatenate all other data
    train_data = np.concatenate(all_train_data, axis=0)
    test_data = np.concatenate(all_test_data, axis=0)
    train_coords = np.concatenate(all_train_coords, axis=0)
    test_coords = np.concatenate(all_test_coords, axis=0)
    
    # Save combined dataset
    print(f"\nSaving combined dataset to {output_file}")
    print(f"Total train samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    
    np.savez_compressed(
        output_file,
        train_data=train_data,
        test_data=test_data,
        train_coords=train_coords,
        test_coords=test_coords,
        train_data_tokenized=train_images_tokenized,
        test_data_tokenized=test_images_tokenized,
        decode_table=decode_table,
        train_label=np.array(train_labels),
        test_label=np.array(test_labels)
    )
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple resolution datasets")
    parser.add_argument("--image-size", type=int, default=64, help="Image size")
    parser.add_argument("--ratios", type=int, nargs="+", default=[2, 4, 8], 
                        help="Downsample ratios to combine")
    parser.add_argument("--output", type=str, default="../data/combined_multi_res.npz",
                        help="Output file path")
    parser.add_argument("--tokenizer-downsample-ratio", type=int, default=2,
                        help="Tokenizer downsample ratio (default: 2)")
    
    args = parser.parse_args()
    
    combine_datasets(args.image_size, args.ratios, args.output, args.tokenizer_downsample_ratio)