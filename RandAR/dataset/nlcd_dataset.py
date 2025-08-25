import torch
from torch.utils.data import Dataset
import numpy as np



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
        
        data = np.load(data_path)

        self.data_train = data['train_data_tokenized']
        self.coords_train = data['train_coords'] if 'train_coords' in data else None

        self.data_test = data['test_data_tokenized']
        self.coords_test = data['test_coords'] if 'test_coords' in data else None
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.data_train):
            self.data_train = self.data_train[:max_samples]
            if self.coords_train is not None:
                self.coords_train = self.coords_train[:max_samples]
        if max_samples is not None and max_samples < len(self.data_test):
            self.data_test = self.data_test[:max_samples]
            if self.coords_test is not None:
                self.coords_test = self.coords_test[:max_samples]
                self.coords_train = self.coords_train[:max_samples]

        
        self.vocab_size = len(data['decode_table'])
        self.decode_table = data.get("decode_table")
        self.unique_values = np.arange(len(self.decode_table)) if self.decode_table is not None else np.unique(self.data_train)

        # Get unique values and create mapping
        self.value_to_idx = {val: idx for idx, val in enumerate(self.unique_values)}
        self.idx_to_value = {idx: val for val, idx in self.value_to_idx.items()}

        self.image_shape = self.data_train.shape[-2], self.data_train.shape[-1]

        print(f"Loaded {split} split with {len(self.data_train)} training samples")
        print(f"Training data shape: {self.data_train.shape}")
        print(f"Unique values: {self.unique_values}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def __len__(self):

        if self.split == 'train':
            self.data = self.data_train
        elif self.split == 'test':
            self.data = self.data_test
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            tokens = torch.from_numpy(self.data_train[idx]).flatten()  
        elif self.split == 'test':
            tokens = torch.from_numpy(self.data_test[idx]).flatten()
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Dummy class label (we could use coords or other metadata later)
        label = torch.tensor(0, dtype=torch.long)
        
        return tokens, label, idx 