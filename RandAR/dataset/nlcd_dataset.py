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

    Optionally, there may also be arrays `train_labels` and `test_labels` which should be of shape (N,)
    or (N, 1). These are discrete class labels which should be applied to the whole image, and are currently
    used to encode the scale/resolution fo the original data.

    """
    
    def __init__(self, data_path, split='train', max_samples=None):
        self.data_path = data_path
        self.split = split

        data = np.load(data_path)

        self.data_train   = data['train_data_tokenized']
        self.coords_train = data.get('train_coords', None)
        self.labels_train = data.get('train_labels', None)

        self.data_test   = data['test_data_tokenized']
        self.coords_test = data.get('test_coords', None)
        self.labels_test = data.get('test_labels', None)

        # Check for auxiliary data in the npz file
        self.aux_train = data.get('train_aux', None)
        self.aux_test = data.get('test_aux', None)
        
        # Log whether auxiliary data is present
        if self.aux_train is not None:
            print(f"Found auxiliary training data with shape: {self.aux_train.shape}")
            self.aux_dim = self.aux_train.shape[1]  # Use actual dimension from data
        else:
            print(f"No auxiliary training data found in {data_path}, will use zero vectors of dimension {self.aux_dim}")

        if self.aux_test is not None:
            print(f"Found auxiliary test data with shape: {self.aux_test.shape}")
        else:
            print(f"No auxiliary test data found in {data_path}, will use zero vectors of dimension {aux_dim}")
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.data_train):
            self.data_train = self.data_train[:max_samples]
            if self.coords_train is not None:
                self.coords_train = self.coords_train[:max_samples]
            if self.aux_train is not None:
                self.aux_train = self.aux_train[:max_samples]
        if max_samples is not None and max_samples < len(self.data_test):
            self.data_test = self.data_test[:max_samples]
            if self.coords_test is not None:
                self.coords_test = self.coords_test[:max_samples]
                self.coords_train = self.coords_train[:max_samples]
            if self.aux_test is not None:
                self.aux_test = self.aux_test[:max_samples]

        
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
            # Get auxiliary data if available, otherwise use zeros
            if self.aux_train is not None:
                aux = torch.from_numpy(self.aux_train[idx]).float()
            else:
                aux = torch.zeros(self.aux_dim, dtype=torch.float32)

            if self.labels_train is not None:
                label = torch.from_numpy(self.labels_train[idx]).long()
                print(f"Found unique labels from the `train_labels` data: {torch.unique(label)}")
            else:
                label = torch.tensor(0, dtype=torch.long)
                
        elif self.split == 'test':
            tokens = torch.from_numpy(self.data_test[idx]).flatten()
            # Get auxiliary data if available, otherwise use zeros
            if self.aux_test is not None:
                aux = torch.from_numpy(self.aux_test[idx]).float()
            else:
                aux = torch.zeros(self.aux_dim, dtype=torch.float32)

            if self.labels_test is not None:
                label = torch.from_numpy(self.labels_test[idx]).long()
            else:
                label = torch.tensor(0, dtype=torch.long)
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return tokens, label, aux