import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
import rasterio.windows
from pyproj import CRS, Transformer
import cv2
from pathlib import Path
from typing import Tuple, Sequence, Optional


import numpy as np

def detokenize(tokens: np.ndarray, decode_table: np.ndarray) -> np.ndarray:
    """
    Replace each integer in a 2D token grid with its DxD patch from decode_table.

    Args:
        tokens: 2D array of shape (H, W) with integer labels in [0, N-1].
        decode_table: 3D array of shape (N, D, D) mapping label -> DxD patch.

    Returns:
        2D array of shape (H*D, W*D) with patches tiled in raster order.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must be 2D (H, W)")
    if decode_table.ndim != 3:
        raise ValueError("decode_table must be 3D (N, D, D)")
    N, D, D2 = decode_table.shape
    if D != D2:
        raise ValueError("decode_table must have square patches (D, D)")
    if tokens.size > 0:
        tmin, tmax = int(tokens.min()), int(tokens.max())
        if tmin < 0 or tmax >= N:
            raise ValueError(f"tokens out of range [0, {N-1}] (min={tmin}, max={tmax})")

    # (H, W, D, D)
    patches = decode_table[tokens]
    # Reorder to (H, D, W, D) then collapse to (H*D, W*D)
    return patches.transpose(0, 2, 1, 3).reshape(tokens.shape[0] * D, tokens.shape[1] * D)
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
    
    # Class-level parameters for data paths (relative to data directory)
    DATA_DIR = Path(__file__).parent.parent.parent / 'data'
    DEM_DIR = DATA_DIR / 'dem'
    NLCD_PATH = DATA_DIR / 'nlcd_2021_land_cover_l48_20230630_cog.tif'
    MERGED_DEM_PATH = DEM_DIR / 'merged_conus_dem.tif'
    
    # NLCD to RGB mapping for visualization
    NLCD_TO_RGB = {
        11: (0.278, 0.420, 0.627),  # Open Water
        12: (0.820, 0.867, 0.976),  # Perennial Ice/Snow
        21: (0.867, 0.788, 0.788),  # Developed, Open Space
        22: (0.847, 0.576, 0.510),  # Developed, Low Intensity
        23: (0.929, 0.0, 0.0),      # Developed, Medium Intensity
        24: (0.667, 0.0, 0.0),      # Developed, High Intensity
        31: (0.698, 0.678, 0.639),  # Barren Land
        41: (0.408, 0.667, 0.388),  # Deciduous Forest
        42: (0.110, 0.388, 0.188),  # Evergreen Forest
        43: (0.710, 0.788, 0.557),  # Mixed Forest
        51: (0.647, 0.549, 0.188),  # Dwarf Scrub
        52: (0.800, 0.729, 0.486),  # Shrub/Scrub
        71: (0.886, 0.886, 0.757),  # Grassland/Herbaceous
        72: (0.788, 0.788, 0.467),  # Sedge/Herbaceous
        73: (0.600, 0.757, 0.278),  # Lichens
        74: (0.467, 0.678, 0.576),  # Moss
        81: (0.859, 0.847, 0.239),  # Pasture/Hay
        82: (0.667, 0.439, 0.157),  # Cultivated Crops
        90: (0.729, 0.847, 0.918),  # Woody Wetlands
        95: (0.439, 0.639, 0.729),  # Emergent Herbaceous Wetlands
    }
    
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
            self.aux_dim = 2  # Default dimension for slope and aspect
            print(f"No auxiliary training data found in {data_path}, will use zero vectors of dimension {self.aux_dim}")

        if self.aux_test is not None:
            print(f"Found auxiliary test data with shape: {self.aux_test.shape}")
        else:
            print(f"No auxiliary test data found in {data_path}, will use zero vectors of dimension {self.aux_dim}")
        
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
    
    @classmethod
    def extract(cls, 
                                         bbox: Sequence[float], 
                                         use_elevation: bool = True,
                                         downsample_resolution: int = 1,
                                         as_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract land cover data along with underlying elevation data (slope and aspect).
        
        Args:
            bbox: Sequence of 4 floats (west, south, east, north) in lat-long (WGS84)
            use_elevation: Whether to include elevation data (slope/aspect)
            downsample_resolution: Downsampling ratio for the land cover data
            as_rgb: If True, return land cover as RGB (HxWx3), else as integer labels
            
        Returns:
            land_cover: Array of land cover data, either:
                - Integer labels of shape (H, W) if as_rgb=False
                - RGB values of shape (H, W, 3) if as_rgb=True
            terrain_features: Array of shape (2, H, W) with slope and aspect
        """
        from scipy.stats import mode
        
        # Set up CRS transformers
        working_crs = CRS.from_string('EPSG:4326')  # WGS84
        
        # Extract bounding box coordinates
        west, south, east, north = bbox
        
        # Open NLCD dataset
        with rasterio.open(cls.NLCD_PATH) as nlcd_src:
            data_crs = nlcd_src.crs
            from_working_crs = Transformer.from_crs(working_crs, data_crs, always_xy=True)
            
            # Convert bbox to data CRS
            bbox_left, bbox_bottom = from_working_crs.transform(west, south)
            bbox_right, bbox_top = from_working_crs.transform(east, north)
            
            # Get pixel bounds
            row_start, col_start = nlcd_src.index(bbox_left, bbox_top)
            row_end, col_end = nlcd_src.index(bbox_right, bbox_bottom)
            
            # Ensure correct order
            row_start, row_end = min(row_start, row_end), max(row_start, row_end)
            col_start, col_end = min(col_start, col_end), max(col_start, col_end)
            
            # Read land cover data
            window = rasterio.windows.Window(
                col_start, row_start,
                col_end - col_start,
                row_end - row_start
            )
            land_cover = nlcd_src.read(1, window=window)
        
        # Downsample if requested
        if downsample_resolution > 1:
            h, w = land_cover.shape
            new_h, new_w = h // downsample_resolution, w // downsample_resolution
            
            # Crop to make divisible
            land_cover = land_cover[:new_h * downsample_resolution, 
                                  :new_w * downsample_resolution]
            
            # Reshape and take mode
            reshaped = land_cover.reshape(new_h, downsample_resolution, 
                                         new_w, downsample_resolution)
            reshaped = reshaped.swapaxes(1, 2).reshape(new_h * new_w, 
                                                       downsample_resolution * downsample_resolution)
            block_modes, _ = mode(reshaped, axis=1)
            land_cover = block_modes.reshape(new_h, new_w).astype(np.uint8)
        
        # Convert to RGB if requested
        if as_rgb:
            h, w = land_cover.shape
            rgb_image = np.zeros((h, w, 3), dtype=np.float32)
            for class_id, rgb in cls.NLCD_TO_RGB.items():
                mask = land_cover == class_id
                rgb_image[mask] = rgb
            land_cover = rgb_image
        
        # Extract elevation data if requested
        if use_elevation:
            terrain_features = cls._extract_terrain_features(bbox, land_cover.shape[:2], 
                                                            downsample_resolution)
        else:
            # Return zeros for slope and aspect
            h, w = land_cover.shape[:2] if as_rgb else land_cover.shape
            terrain_features = np.zeros((2, h, w), dtype=np.float32)
        
        return land_cover, terrain_features
    
    @classmethod
    def _extract_terrain_features(cls, bbox: Sequence[float], 
                                 target_shape: Tuple[int, int],
                                 downsample_resolution: int) -> np.ndarray:
        """
        Extract slope and aspect from DEM data.
        
        Args:
            bbox: Bounding box (west, south, east, north) in WGS84
            target_shape: Target (H, W) shape for output
            downsample_resolution: Downsampling ratio used for land cover
            
        Returns:
            Array of shape (2, H, W) with slope (degrees) and aspect (degrees)
        """
        west, south, east, north = bbox
        h_target, w_target = target_shape
        
        # Pixel size in meters (30m NLCD resolution * downsample ratio)
        pixel_size = 30.0 * downsample_resolution
        
        # Initialize output
        terrain_features = np.zeros((2, h_target, w_target), dtype=np.float32)
        
        try:
            # Open DEM file
            with rasterio.open(cls.MERGED_DEM_PATH) as dem_src:
                # Create window from bounds
                window = rasterio.windows.from_bounds(
                    west, south, east, north, 
                    transform=dem_src.transform
                )
                
                # Read DEM data
                dem_data = dem_src.read(1, window=window)
                
                # Handle nodata values
                nodata_mask = (dem_data == dem_src.nodata) | (dem_data > 4430.0)
                
                if np.sum(nodata_mask) / dem_data.size > 0.25:
                    # Too much missing data, return zeros
                    return terrain_features
                
                # Interpolate missing values if any
                if np.any(nodata_mask):
                    dem_data = cv2.inpaint(
                        dem_data.astype(np.float32),
                        nodata_mask.astype(np.uint8),
                        inpaintRadius=3,
                        flags=cv2.INPAINT_TELEA
                    )
                
                # Resize to target shape
                dem_data = cv2.resize(dem_data, (w_target, h_target), 
                                     interpolation=cv2.INTER_LINEAR)
                
                # Calculate slope and aspect
                dy, dx = np.gradient(dem_data, pixel_size)
                
                # Slope in degrees
                slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                slope = np.degrees(slope_rad)
                
                # Aspect in degrees (0-360, where 0=North)
                aspect_rad = np.arctan2(-dx, dy)
                aspect = np.degrees(aspect_rad) % 360
                
                terrain_features[0] = slope
                terrain_features[1] = aspect
                
        except Exception as e:
            print(f"Warning: Could not extract terrain features: {e}")
            # Return zeros on error
            pass
        
        return terrain_features
    
    
    


if __name__ == "__main__":
    # Test the extract_land_cover_with_elevation method
    import matplotlib.pyplot as plt
    
    # Test with a small bounding box (Denver area)
    test_bbox = (-105.1, 39.6, -104.9, 39.8)  # (west, south, east, north)
    
    print("Testing NLCDDataset.extract_land_cover_with_elevation()")
    print(f"Bounding box: {test_bbox}")
    print("-" * 50)
    
    # Test 1: Extract with integer labels and elevation
    print("\nTest 1: Integer labels with elevation data")
    land_cover_int, terrain = NLCDDataset.extract_land_cover_with_elevation(
        bbox=test_bbox,
        use_elevation=True,
        downsample_resolution=2,
        as_rgb=False
    )
    print(f"Land cover shape: {land_cover_int.shape}, dtype: {land_cover_int.dtype}")
    print(f"Terrain features shape: {terrain.shape}, dtype: {terrain.dtype}")
    print(f"Unique land cover classes: {np.unique(land_cover_int)}")
    print(f"Slope range: {terrain[0].min():.2f}° - {terrain[0].max():.2f}°")
    print(f"Aspect range: {terrain[1].min():.2f}° - {terrain[1].max():.2f}°")
    
    # Test 2: Extract as RGB without elevation
    print("\nTest 2: RGB format without elevation data")
    land_cover_rgb, terrain_empty = NLCDDataset.extract_land_cover_with_elevation(
        bbox=test_bbox,
        use_elevation=False,
        downsample_resolution=4,
        as_rgb=True
    )
    print(f"Land cover RGB shape: {land_cover_rgb.shape}, dtype: {land_cover_rgb.dtype}")
    print(f"Terrain features shape: {terrain_empty.shape}, dtype: {terrain_empty.dtype}")
    print(f"RGB value range: [{land_cover_rgb.min():.3f}, {land_cover_rgb.max():.3f}]")
    print(f"Terrain all zeros: {np.allclose(terrain_empty, 0)}")
    
    # Visualize results
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Plot integer land cover
    im1 = axes[0, 0].imshow(land_cover_int, cmap='tab20')
    axes[0, 0].set_title('Land Cover (Integer Labels)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # Plot RGB land cover
    axes[0, 1].imshow(land_cover_rgb)
    axes[0, 1].set_title('Land Cover (RGB)')
    axes[0, 1].axis('off')
    
    # Plot slope
    im2 = axes[0, 2].imshow(terrain[0], cmap='YlOrRd', vmin=0)
    axes[0, 2].set_title('Slope (degrees)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Plot aspect with cyclical colormap
    im3 = axes[1, 0].imshow(terrain[1], cmap='twilight', vmin=0, vmax=360)
    axes[1, 0].set_title('Aspect (degrees)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Plot slope histogram
    axes[1, 1].hist(terrain[0].flatten(), bins=30, edgecolor='black')
    axes[1, 1].set_title('Slope Distribution')
    axes[1, 1].set_xlabel('Slope (degrees)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot aspect histogram (polar)
    ax_polar = plt.subplot(2, 3, 6, projection='polar')
    aspect_rad = np.radians(terrain[1].flatten())
    counts, bins = np.histogram(aspect_rad, bins=36, range=(-np.pi, np.pi))
    bins_center = (bins[:-1] + bins[1:]) / 2
    ax_polar.bar(bins_center, counts, width=bins[1]-bins[0], edgecolor='black')
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title('Aspect Distribution', pad=20)
    
    
    plt.suptitle(f'Land Cover and Terrain Analysis\nBBox: {test_bbox}', fontsize=14)
    plt.tight_layout()
    plt.savefig('nlcd_test_output.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'nlcd_test_output.png'")
    plt.show()
    
    print("\nTest completed successfully!")