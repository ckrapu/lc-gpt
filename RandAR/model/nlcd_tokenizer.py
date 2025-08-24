import torch
import numpy as np
import logging

from typing import Union

logger = logging.getLogger(__name__)


class NLCDTokenizer:
    """Tokenizer for NLCD dataset - handles mapping between indices and NLCD values."""
    
    def __init__(self, vocab_size=None, value_mapping=None,):
        self.vocab_size = vocab_size
        self.codebook_embed_dim = 1  # Dummy value for compatibility
        
        # Store the mapping if provided
        if value_mapping:
            self.idx_to_value = value_mapping
            self.value_to_idx = {v: k for k, v in value_mapping.items()}
        else:
            # Default identity mapping
            self.idx_to_value = {i: i for i in range(vocab_size)}
            self.value_to_idx = {i: i for i in range(vocab_size)}
        
        # NLCD colormap
        self.lut = {
            11: (0.278, 0.420, 0.627),
            12: (0.820, 0.867, 0.976),
            21: (0.867, 0.788, 0.788),
            22: (0.847, 0.576, 0.510),
            23: (0.929, 0.0, 0.0),
            24: (0.667, 0.0, 0.0),
            31: (0.698, 0.678, 0.639),
            41: (0.408, 0.667, 0.388),
            42: (0.110, 0.388, 0.188),
            43: (0.710, 0.788, 0.557),
            51: (0.647, 0.549, 0.188),
            52: (0.800, 0.729, 0.486),
            71: (0.886, 0.886, 0.757),
            72: (0.788, 0.788, 0.467),
            73: (0.600, 0.757, 0.278),
            74: (0.467, 0.678, 0.576),
            81: (0.859, 0.847, 0.239),
            82: (0.667, 0.439, 0.157),
            90: (0.729, 0.847, 0.918),
            95: (0.439, 0.639, 0.729),  
        }

        self.lut_arr = np.zeros((100, 3))
        for code, rgb in self.lut.items():
            self.lut_arr[code] = rgb

        # Create mapping from data values to LUT keys
        # Data values 1-16 map to NLCD classes (for visualization)
        lut_keys = list(self.lut.keys())
        self.data_to_lut = {}
        for i in range(len(lut_keys)):
            self.data_to_lut[i] = lut_keys[i]
        
    def __call__(self, x):
        # Pass-through for compatibility
        return x, 0.0
    
    def nlcd_to_rgb(self, data):
        """Convert NLCD data values to RGB image using the colormap.
        
        Args:
            data: 2D numpy array with NLCD data values (1-16, 0 for ungenerated)
            
        Returns:
            RGB image as numpy array of shape (height, width, 3) with values in [0, 1]
        """
        height, width = data.shape
        rgb_img = np.zeros((height, width, 3))
        
        for i in range(height):
            for j in range(width):
                data_value = data[i, j]
                lut_key = self.data_to_lut[data_value]
                rgb = self.lut[lut_key]
                rgb_img[i, j] = rgb
        return rgb_img

    
    
    def encode_indices(self, x: np.ndarray):
        
        # For NLCD dataset, tokens are already indices
        return x.flatten()

    def decode_codes_to_img(self, codes:Union[np.ndarray, torch.Tensor], decode_table: np.ndarray, scale: int = 10) -> np.ndarray:
        """
        Convert token codes back to images for visualization using NLCD colormap. Assumes that `codes`
        is passed as an array of size (batch_size, seq_len) and that the values may need to be remapped
        from tokens back into pixel space, with the mapped patches of pixels larger than the original tokens in terms
        of shape, i.e. a token of shape (1, ) corresponds to a patch of shape (2, 2). 

        Assumes that the "lookup_table" is an array with row indexes corresponding to tokens
        and other dims corresponding to the raw values. The final image shape is required so we
        know how much space each token is supposed to take in pixel space. Also, assumes square patches. Typically
        of shape (65000, 2, 2) for a moderately sized codebook.   
        """

        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().numpy()

        batch_size = codes.shape[0]
        patch_shape = decode_table.shape[1:]


        shape_before_map = int(np.sqrt(codes.shape[1])), int(np.sqrt(codes.shape[1]))
        shape_after_map = (shape_before_map[0]*patch_shape[0], shape_before_map[1]*patch_shape[1])

        imgs_rgb = np.zeros((batch_size, *shape_after_map, 3), dtype=np.uint8)

        for b in range(batch_size):
            decoded_values = decode_table[codes[b]]  # (seq_len, patch_h, patch_w)
            
            for patch_idx in range(codes.shape[1]):
                patch_row = patch_idx // shape_before_map[1]
                patch_col = patch_idx % shape_before_map[1]
                
                start_row = patch_row * patch_shape[0]
                start_col = patch_col * patch_shape[1]
                
                patch = decoded_values[patch_idx]
                for i in range(patch_shape[0]):
                    for j in range(patch_shape[1]):
                        nlcd_value = int(patch[i, j])
                        rgb = self.lut_arr[nlcd_value]
                        imgs_rgb[b, start_row + i, start_col + j] = (rgb * 255).astype(np.uint8)
        
        # Upscale using nearest neighbor
        upscaled = np.repeat(np.repeat(imgs_rgb, scale, axis=1), scale, axis=2)
        
        return upscaled 