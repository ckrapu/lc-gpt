import torch
from torch.utils.data import Dataset
import numpy as np
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


class NLCDDataset(Dataset):
    """Dataset that loads NLCD data from npz file.
    Assumes that we have a .npz file with the following keys mapping to arrays with provided dtypes:
    - train_data: (N, 2, H, W) (2nd band is ignored)
    - train_coords: (N, 2)
    - test_data: (N, 2, H, W) (2nd band is ignored)
    - test_coords: (N, 2)
    """
    
    def __init__(self, data_path, split='train', max_samples=None):
        self.data_path = data_path
        self.split = split
        
        
        # Load the data
        data = np.load(data_path)
        
        if split == 'train':
            self.data = data['train_data'][:, 0, :, :]  # First channel only
            self.coords = data['train_coords'] if 'train_coords' in data else None
        else:
            self.data = data['test_data'][:, 0, :, :]  # First channel only
            self.coords = data['test_coords'] if 'test_coords' in data else None
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data[:max_samples]
            if self.coords is not None:
                self.coords = self.coords[:max_samples]
        
        # Get unique values and create mapping
        self.unique_values = np.unique(self.data)
        self.value_to_idx = {val: idx for idx, val in enumerate(self.unique_values)}
        self.idx_to_value = {idx: val for val, idx in self.value_to_idx.items()}
        self.vocab_size = len(self.unique_values)
        
        print(f"Loaded {split} split with {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Unique values: {self.unique_values}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the HxW image
        img = self.data[idx]
        
        # Convert to token indices
        tokens = torch.zeros_like(torch.from_numpy(img), dtype=torch.long)
        for val, token_idx in self.value_to_idx.items():
            tokens[img == val] = token_idx
        
        # Flatten 
        tokens = tokens.flatten()
        
        # Dummy class label (we could use coords or other metadata later)
        label = torch.tensor(0, dtype=torch.long)
        
        return tokens, label, idx 