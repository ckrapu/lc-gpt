import torch
import torch.nn as nn
import numpy as np


class NLCDTokenizer(nn.Module):
    """Tokenizer for NLCD dataset - handles mapping between indices and NLCD values."""
    
    def __init__(self, vocab_size=None, value_mapping=None):
        super().__init__()
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
        
        # Create mapping from data values to LUT keys
        # Data values 1-16 map to NLCD classes (for visualization)
        lut_keys = list(self.lut.keys())
        self.data_to_lut = {}
        for i in range(len(lut_keys)):
            self.data_to_lut[i] = lut_keys[i]
        
    def forward(self, x):
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

    
    
    def encode_indices(self, x):
        # For NLCD dataset, tokens are already indices
        return x.flatten()
    
    def decode_codes_to_img(self, codes, image_size):
        """Convert token codes back to images for visualization using NLCD colormap."""
        batch_size = codes.shape[0]
        side_len = int(np.sqrt(codes.shape[1]))
        
        # Reshape to 2D
        imgs = codes.reshape(batch_size, side_len, side_len)
        
        # Map indices back to NLCD values
        imgs_values = torch.zeros_like(imgs)
        for idx, value in self.idx_to_value.items():
            imgs_values[imgs == idx] = value
        
        # Create RGB images using the NLCD colormap
        imgs_rgb = np.zeros((batch_size, side_len, side_len, 3), dtype=np.uint8)
        
        for b in range(batch_size):
            for i in range(side_len):
                for j in range(side_len):
                    data_value = imgs_values[b, i, j].item()
                    # Convert indices (0-15) to NLCD values (1-16) for colormap lookup
                    nlcd_value = data_value + 1 if data_value > 0 else 0
                    if nlcd_value in self.data_to_lut:
                        lut_key = self.data_to_lut[nlcd_value]
                        rgb = self.lut[lut_key]
                        imgs_rgb[b, i, j] = [int(r * 255) for r in rgb]
                    else:
                        # Default to black for unknown values
                        imgs_rgb[b, i, j] = [0, 0, 0]
        
        # Resize if needed
        if side_len != image_size:
            from PIL import Image
            resized_imgs = []
            for img in imgs_rgb:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((image_size, image_size), Image.NEAREST)
                resized_imgs.append(np.array(pil_img))
            imgs_rgb = np.stack(resized_imgs)
        
        return imgs_rgb 