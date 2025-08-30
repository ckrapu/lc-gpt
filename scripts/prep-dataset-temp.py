# %% [markdown]
# # Overview
# This notebook contains a data cleaning pipeline for preparing a dataset for land cover data overlaid on digital elevation data. 
# 
# The expected outputs are:
# - Train and test files for image data, with shape (N, 2, H, W) with the first channel for LULC and second channel for DEM
# - Train and test geopackage files containing the bounding boxes for each image
# 
# The main steps in this notebook are:
# - Splitting up the domain into adjacent grid cells and extracting land cover data
# - Assigning portions of the domain into train and test splits using Sobol sequences
# - Downloading and merging DEM files for the same domain
# - Extracting DEM data for each land cover image
# - Saving these data to disk
# - Creating an animation of the final dataset

# %% [markdown]
# # 1. Imports

# %%
import cv2
import elevation
import geopandas as gpd
import logging
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import multiprocessing as mp
import numba as nb
import numpy as np
import os
import pandas as pd
import psutil
import rasterio
import time

from dataclasses import dataclass
from functools import partial
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from pyproj import CRS, Transformer
from scipy.stats import mode
from shapely.geometry import box
from tqdm.notebook import tqdm
from typing import Tuple
from dotenv import load_dotenv
import logging

BUCKET_NAME = "lc-inpaint"
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)

load_dotenv()

plt.style.use('dark_background')

%load_ext watermark
%watermark -iv

# %% [markdown]
# # 2. Setting up the config
# 

# %%
# These can be manipulated by Papermill when running
# from the command line
image_size = 128
downsample_ratio = 2
n_samples_max = 10_000_000
n_grid_unit = 30 # implies n_grid_unit**2 total grid cells


# %%

current_path = Path.cwd()
parent_path = current_path.parent

@dataclass
class DataPrepConfig:
    logging_level = logging.INFO

    # Geographic bounds (WGS84 coordinates)
    bbox_west: float = -119.0
    bbox_east: float = -64.0
    bbox_south: float = 22.0
    bbox_north: float = 49.0

    # Sampling parameters
    image_size: int = image_size  # Size of output images
    downsample_ratio: int = downsample_ratio
    max_fraction_reject_class: float = 0.9  # Maximum fraction of pixels allowed in a reject-eligible class
    area_fraction_test: float = 0.05  # Fraction of area to reserve for testing
    n_samples_max: int = n_samples_max
    n_grid_unit: int = n_grid_unit  # discrete units in each dimension for gridding the domain into discrete units
    n_samples_max_per_cell: int = n_samples_max // (n_grid_unit ** 2)

    # Data paths
    data_dir: Path = parent_path / 'data'
    dem_dir = Path(data_dir) / 'dem'

    nlcd_path: Path = data_dir / 'nlcd_2021_land_cover_l48_20230630.img' # Make sure you have this file before you start
    nlcd_path_tiff: Path = data_dir / 'nlcd_2021_land_cover_l48_20230630_cog.tif'
    output_path: Path  = data_dir / f'data_size{image_size}_ratio{downsample_ratio}.npz'
    output_path_test_gpkg: Path = data_dir / f"test_{image_size}_ratio{downsample_ratio}.gpkg"
    output_path_train_gpkg: Path = data_dir / f"train_{image_size}_ratio{downsample_ratio}.gpkg"
    split_save_path: Path = data_dir / f"split_{image_size}_ratio{downsample_ratio}.gpkg"
    merged_dem_path: Path = dem_dir / f"merged_conus_dem.tif"
    output_path_animation: Path = data_dir / f"animation_{image_size}_ratio{downsample_ratio}.gif"
    
    # Processing parameters
    random_seed: int = 827  # Random seed for reproducibility
    recompute_counts: bool = False  # Whether to recompute class counts
    show_plots: bool = True  # Whether to display plots
    
    # CRS parameters
    working_crs: str = 'EPSG:4326'  # CRS for geographic operations (WGS84)         
    
    nlcd_original_classes_for_reject = {11}
    nlcd_original_unknown_class = 0

    # Parameters for DEM processing
    dem_nodata_threshold: float = 0.25
    dem_product: str = 'SRTM1' # Choices are 'SRTM1' or 'SRTM3', lower resolution
    download_dem: bool = False 
    merge_dem: bool = False
    dem_elev_max: float = 4430.0 # Threshold for nodata values in DEM. Highest point in CONUS is 4421 m.

    # Create a tokenizer which allows us to represent images in patches instead of single pixels
    tokenizer_downsample_ratio: int = 2
    tokenizer_path = data_dir / f'tokenizer.pkl'

    upload_to_s3: bool = True

    # Mapping from raw NLCD classes to RGB colors for visualization
    nlcd_to_rgb  = {
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
    nlcd_to_name = {
        11: "Open Water",
        12: "Perennial Ice/Snow",
        21: "Developed, Open Space",
        22: "Developed, Low Intensity",
        23: "Developed, Medium Intensity",
        24: "Developed, High Intensity",
        31: "Barren Land (Rock/Sand/Clay)",
        41: "Deciduous Forest",
        42: "Evergreen Forest",
        43: "Mixed Forest",
        51: "Dwarf Scrub",
        52: "Shrub/Scrub",
        71: "Grassland/Herbaceous",
        72: "Sedge/Herbaceous",
        73: "Lichens",
        74: "Moss",
        81: "Pasture/Hay",
        82: "Cultivated Crops",
        90: "Woody Wetlands",
        95: "Emergent Herbaceous Wetlands"
    }
    

    def __post_init__(self):
        # Validate bbox coordinates
        if not (self.bbox_west < self.bbox_east):
            raise ValueError(f"Invalid bbox coordinates: bbox_west ({self.bbox_west}) should be less than bbox_east ({self.bbox_east})")
        if not (self.bbox_south < self.bbox_north):
            raise ValueError(f"Invalid bbox coordinates: bbox_south ({self.bbox_south}) should be less than bbox_north ({self.bbox_north})")
        
        # Validate file paths
        if not self.nlcd_path.is_file():
            raise FileNotFoundError(f"NLCD file not found at {self.nlcd_path}")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found at {self.data_dir}")
            

config = DataPrepConfig()

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=config.logging_level,
)

np.random.seed(827)

logging.info(f"Setting project data directory to {config.data_dir}")

def downsample_patch(patch: np.ndarray, ratio: int, force_divisible=False) -> np.ndarray:
    """Downsample a patch by taking the mode of each ratio x ratio window."""
    if ratio == 1:
        return patch
    
    if force_divisible:
        h, w = patch.shape
        new_h = (h // ratio) * ratio
        new_w = (w // ratio) * ratio
        patch = patch[:new_h, :new_w]
    
    # Reshape into blocks of size ratio x ratio
    h, w = patch.shape
    new_h, new_w = h // ratio, w // ratio
    
    reshaped = patch.reshape(new_h, ratio, new_w, ratio)
    
    # Move the two ratio axes adjacent so each block becomes one dimension
    # resulting shape: (new_h * new_w, ratio * ratio)
    reshaped = reshaped.swapaxes(1, 2).reshape(new_h * new_w, ratio * ratio)
    
    # mode(..., axis=1) finds the most frequent value in each row
    block_modes, _ = mode(reshaped, axis=1)
    
    # Reshape back to (new_h, new_w)
    downsampled = block_modes.reshape(new_h, new_w)
    
    return downsampled

def check_overlap(point_coords: Tuple[float, float], image_size_meters: float, 
                 grid_gdf: gpd.GeoDataFrame, split: str) -> bool:
    """Check if an image centered at point_coords overlaps with the specified split area."""
    x, y = point_coords
    half_size = image_size_meters / 2
    
    # Create a box representing the image extent in working CRS
    image_box = box(x - half_size, y - half_size,
                   x + half_size, y + half_size)
    
    # Check intersection with grid cells of the opposite split
    opposite_split = 'test' if split == 'train' else 'train'
    opposite_cells = grid_gdf[grid_gdf['split'] == opposite_split]
    
    return not any(image_box.intersects(cell) for cell in opposite_cells.geometry)

def is_within_bbox(x: float, y: float) -> bool:
    """Check if a point is within the specified bbox."""
    return (config.bbox_west <= x <= config.bbox_east and
            config.bbox_south <= y <= config.bbox_north)



# %% [markdown]
# ## Convert IMG file to cloud-optimized GeoTIFF

# %%
if os.path.exists(config.nlcd_path_tiff):
    logging.info(f"COG file already exists at {config.nlcd_path_tiff}, skipping conversion.")
else:
    logging.info(f"Converting {config.nlcd_path} to cloud-optimized GeoTIFF at {config.nlcd_path_tiff}...")
    import subprocess
    cmd =f"""gdal_translate \
        -of COG \
        -co COMPRESS=LZW \
        -co BIGTIFF=YES \
        -co NUM_THREADS=ALL_CPUS \
        -co BLOCKSIZE=512 \
        -co OVERVIEW_RESAMPLING=NEAREST \
        {config.nlcd_path} \
        {config.nlcd_path_tiff}
    """

    subprocess.run(cmd, shell=True, check=True)



# %% [markdown]
# # 3. Dataset Information and CRS Setup
# 

# %% [markdown]
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %%
# Open the NLCD dataset and print basic information
with rasterio.open(config.nlcd_path_tiff) as src:
    logging.info(f"Dataset CRS: {src.crs}")
    logging.info(f"Dataset bounds: {src.bounds}")
    logging.info(f"Dataset shape: {src.shape}")
    logging.info(f"Dataset resolution: {src.res}")
    logging.info(f"Dataset transform: {src.transform}")
    
    # Set up CRS transformers
    data_crs = src.crs
    working_crs = CRS.from_string(config.working_crs)
    
    # Create transformers for converting between CRS
    to_working_crs = Transformer.from_crs(data_crs, working_crs, always_xy=True)
    from_working_crs = Transformer.from_crs(working_crs, data_crs, always_xy=True)
    
    # Convert dataset bounds to working CRS for validation
    bounds = src.bounds
    ds_left, ds_bottom = to_working_crs.transform(bounds.left, bounds.bottom)
    ds_right, ds_top = to_working_crs.transform(bounds.right, bounds.top)
    
    
    logging.info("\nBounding box validation:")
    logging.info(f"Dataset bounds (lon/lat): {ds_left:.4f}, {ds_bottom:.4f}, {ds_right:.4f}, {ds_top:.4f}")
    logging.info(f"Selected bbox (lon/lat): {config.bbox_west}, {config.bbox_south}, {config.bbox_east}, {config.bbox_north}")    
    samples_x = src.shape[1] / config.downsample_ratio / config.image_size
    samples_y = src.shape[0] / config.downsample_ratio / config.image_size

    logging.info(f"Maximum number of sampled images from full dataset: {samples_x * samples_y:.0f}")

# %% [markdown]
# # 4. Class Counting and Mapping

# %%
# Function to compute class counts in a block
def compute_block_counts(data):
    unique, counts = np.unique(data, return_counts=True)
    return dict(zip(unique, counts))

# Calculate available memory
available_memory = psutil.virtual_memory().available
dtype_size = np.dtype('uint8').itemsize
max_elements = available_memory // (2 * dtype_size)  # Use half of available memory

if not config.recompute_counts:
    logging.info(f"Skipping class counts computation from raster; loading from file")
else:
    with rasterio.open(config.nlcd_path_tiff) as src:
        # Convert bbox to pixel coordinates
        bbox_left, bbox_bottom = from_working_crs.transform(config.bbox_west, config.bbox_south)
        bbox_right, bbox_top = from_working_crs.transform(config.bbox_east, config.bbox_north)
        
        # Get pixel bounds
        row_start, col_start = src.index(bbox_left, bbox_top)
        row_end, col_end = src.index(bbox_right, bbox_bottom)
        
        # Ensure correct order
        row_start, row_end = min(row_start, row_end), max(row_start, row_end)
        col_start, col_end = min(col_start, col_end), max(col_start, col_end)
        
        # Calculate block size for the bbox region
        bbox_height = row_end - row_start
        bbox_width = col_end - col_start
        total_pixels = bbox_height * bbox_width
        n_blocks = max(1, total_pixels // max_elements)
        block_height = bbox_height // n_blocks
        
        # Initialize counts dictionary
        total_counts = {}
        
        # Process data in blocks within the bbox
        for i in tqdm(range(row_start, row_end, block_height), desc='Computing class counts'):
            # Read a block of data
            window = rasterio.windows.Window(
                col_start, i, 
                col_end - col_start,
                min(block_height, row_end - i)
            )
            data = src.read(1, window=window)
            
            # Update counts
            block_counts = compute_block_counts(data)
            for k, v in block_counts.items():
                total_counts[k] = total_counts.get(k, 0) + v

    logging.info(f"Unique values present in the bbox: {len(total_counts)}: {total_counts.keys()}")

    # Drop the counts which are in class 0 (Unknown)
    _ = total_counts.pop(config.nlcd_original_unknown_class, None)

# %%
if config.recompute_counts:

    # Convert to DataFrame for better visualization
    classes_df = pd.DataFrame([
        {'class_value': k, 'count': v, 'name': config.nlcd_to_name.get(k, 'Unknown')} 
        for k, v in total_counts.items()
    ])
    classes_df['percentage'] = classes_df['count'] / classes_df['count'].sum() * 100
    classes_df = classes_df.sort_values('count', ascending=False)

    # Rename the index (currently unnamed) to "class"
    classes_df.index.name = 'class'
    classes_df = classes_df.sort_index()

    # Load the mapping from original class codes to RGB for plotting and add to the dataframe
    # We will use these for plotting later
    classes_df['RGB'] = classes_df['class_value'].map(config.nlcd_to_rgb)
    classes_df.to_parquet(Path(config.data_dir) / 'class_distribution.parquet')
    present_classes = sorted(total_counts.keys())


else:
    logging.info(f"Skipping class counts computation from raster; loading from file")
    classes_df = pd.read_parquet(Path(config.data_dir) / 'class_distribution.parquet')
    total_counts = classes_df.set_index('class_value')['count'].to_dict()
    classes_df['RGB'] = classes_df['class_value'].map(config.nlcd_to_rgb)


palette_series = classes_df['RGB']
lut = np.zeros((100, 3))
for code, color in config.nlcd_to_rgb.items():
    lut[code] = color

logging.info(f"Prepare lookup table for plotting with shape {lut.shape}")



# %% [markdown]
# #### Pixel counts by class
# 
# 

# %% [markdown]
# # 5. Grid Creation for Train/Test Split

# %%
from scipy.stats import qmc  # Built into scipy, no extra installation needed

# Set up grid for train/test split
with rasterio.open(config.nlcd_path_tiff) as src:
    # Create grid cells in working CRS using bbox
    x_edges = np.linspace(config.bbox_west, config.bbox_east, config.n_grid_unit + 1)
    y_edges = np.linspace(config.bbox_south, config.bbox_north, config.n_grid_unit + 1)
    
    # Create grid cell polygons
    grid_cells = []
    for i in range(len(x_edges)-1):
        for j in range(len(y_edges)-1):
            # Create polygon in working CRS
            polygon = {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [x_edges[i], y_edges[j]],
                        [x_edges[i+1], y_edges[j]],
                        [x_edges[i+1], y_edges[j+1]],
                        [x_edges[i], y_edges[j+1]],
                        [x_edges[i], y_edges[j]]
                    ]]
                },
                'properties': {'id': len(grid_cells)}
            }
            grid_cells.append(polygon)
    
    # Create GeoDataFrame in working CRS
    grid_gdf = gpd.GeoDataFrame.from_features(grid_cells, crs=working_crs)
    
    # Set up Sobol sequence generator
    n_cells = len(grid_gdf)
    n_test = int(n_cells * config.area_fraction_test)
    
    # Generate Sobol sequence and scale to unique grid indices
    sobol_points = qmc.Sobol(d=1, seed=config.random_seed).random(n=n_test)
    sobol_indices = (sobol_points.flatten() * (n_cells - 1)).astype(int)
    sobol_indices = np.unique(sobol_indices)
    
    # If we got fewer unique indices than needed, add random ones
    if len(sobol_indices) < n_test:
        additional_indices = np.random.choice(
            np.setdiff1d(np.arange(n_cells), sobol_indices),
            size=n_test - len(sobol_indices),
            replace=False
        )
        sobol_indices = np.concatenate([sobol_indices, additional_indices])
    
    # Assign splits
    grid_gdf['split'] = 'train'
    grid_gdf.loc[sobol_indices, 'split'] = 'test'
    logging.info(f"Generated {n_test} test cells out of {n_cells} total cells and saved to grid_gdf")

# Save the grid to a GeoPackage file
grid_gdf.to_file(config.split_save_path, driver='GPKG')
logging.info(f"Saved geodataframe for grid of train/test cells to {config.split_save_path}")

if config.show_plots:
    fig, ax = plt.subplots(figsize=(12, 8))
    grid_gdf[grid_gdf['split'] == 'train'].plot(ax=ax, color='c', alpha=0.3)
    grid_gdf[grid_gdf['split'] == 'test'].plot(ax=ax, color='m', alpha=0.3)

    for x in x_edges:
        ax.axvline(x, color='black', linestyle='--', alpha=0.5)
    for y in y_edges:
        ax.axhline(y, color='black', linestyle='--', alpha=0.5)

    # Manually create legend elements
    legend_elements = [
        mpatches.Patch(facecolor='c', alpha=0.3, label='Train unit'),
        mpatches.Patch(facecolor='m', alpha=0.3, label='Test unit')
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.set_title('Train/Test Grid Split')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()


# %% [markdown]
# # 6. Sampling Land Cover Images
# 

# %%
%load_ext line_profiler

# %%
@nb.jit(nopython=True, cache=True)
def downsample_patch_numba(patch, ratio):
    """Numba-optimized downsample by taking mode of each ratio x ratio window."""
    if ratio == 1:
        return patch.copy()

    h, w = patch.shape
    new_h, new_w = h // ratio, w // ratio
    result = np.zeros((new_h, new_w), dtype=patch.dtype)

    for i in range(new_h):
        for j in range(new_w):
            # Extract the block
            block = patch[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]

            # Find mode (most common value) in the block
            # For small blocks, simple counting is fast
            flat_block = block.flatten()
            unique_vals = np.unique(flat_block)
            max_count = 0
            mode_val = flat_block[0]

            for val in unique_vals:
                count = np.sum(flat_block == val)
                if count > max_count:
                    max_count = count
                    mode_val = val

            result[i, j] = mode_val

    return result


@nb.jit(nopython=True, cache=True)
def check_patch_validity(patch, unknown_class, reject_classes, max_reject_fraction):
    """Check if a patch is valid based on unknown pixels and reject class fraction."""
    # Check for unknown pixels
    if np.any(patch == unknown_class):
        return False

    # Check reject class fractions
    patch_size = patch.shape[0] * patch.shape[1]
    for reject_class in reject_classes:
        class_count = np.sum(patch == reject_class)
        if class_count / patch_size > max_reject_fraction:
            return False

    return True


@nb.jit(nopython=True, parallel=True, cache=True)
def process_patches_numba(cell_data, full_size_pixels, stride, unknown_class, 
                        reject_classes, max_reject_fraction, downsample_ratio,
                        row_start_global, col_start_global, max_samples_per_cell):
    """
    Numba-optimized patch extraction and processing.
    Returns: list of tuples (i_global, j_global, downsampled_patch)
    """
    h, w = cell_data.shape

    # Pre-allocate lists for results
    valid_i_positions = []
    valid_j_positions = []
    valid_patches = []

    sample_count = 0

    # Use parallel processing for row iteration
    for i in range(0, h - full_size_pixels + 1, stride):
        if max_samples_per_cell > 0 and sample_count >= max_samples_per_cell:
            break

        for j in range(0, w - full_size_pixels + 1, stride):
            if max_samples_per_cell > 0 and sample_count >= max_samples_per_cell:
                break

            # Extract patch
            patch = cell_data[i:i + full_size_pixels, j:j + full_size_pixels]

            # Check validity
            if check_patch_validity(patch, unknown_class, reject_classes, max_reject_fraction):
                # Downsample the patch
                downsampled = downsample_patch_numba(patch, downsample_ratio)

                # Store results (global pixel positions)
                valid_i_positions.append(row_start_global + i)
                valid_j_positions.append(col_start_global + j)
                valid_patches.append(downsampled)

                sample_count += 1

    return valid_i_positions, valid_j_positions, valid_patches


# Pre-compile the functions with dummy data to avoid first-run overhead
def precompile_numba_functions():
    """Pre-compile Numba functions to avoid JIT overhead during processing."""
    dummy_data = np.random.randint(0, 20, (512, 512), dtype=np.uint8)
    dummy_patch = np.random.randint(0, 20, (128, 128), dtype=np.uint8)
    reject_classes = np.array([11], dtype=np.uint8)

    # Trigger compilation
    _ = downsample_patch_numba(dummy_patch, 2)
    _ = check_patch_validity(dummy_patch, 0, reject_classes, 0.9)
    _ = process_patches_numba(dummy_data, 128, 64, 0, reject_classes, 0.9, 2, 0, 0, 10)

    logging.info("Numba functions pre-compiled successfully")

# Pre-compile on notebook load
precompile_numba_functions()

# %%
def process_cell_optimized(cell_data, args):
      """Process a single cell of data using pre-loaded NLCD array with Numba optimization.
      
      Args:
          cell_data: tuple of (cell_index, cell_geometry, cell_split, 
                             row_start_global, col_start_global, row_end_global, col_end_global)
          args: dict containing configuration and shared data parameters
      """
      cell_index, cell_geometry, cell_split, row_start_global, col_start_global, row_end_global, col_end_global = cell_data

      # Unpack configuration
      config = args['config']
      full_size_pixels = config.image_size * config.downsample_ratio
      stride = full_size_pixels // 2  # 50% overlap

      # Reconstruct the shared array
      if args['use_shared_memory']:
          shm = shared_memory.SharedMemory(name=args['shm_name'])
          nlcd_bbox_array = np.ndarray(
              args['array_shape'],
              dtype=args['array_dtype'],
              buffer=shm.buf
          )
      else:
          nlcd_bbox_array = args['nlcd_array']

      # Calculate local indices within the bbox array
      bbox_row_offset = args['bbox_row_start']
      bbox_col_offset = args['bbox_col_start']

      row_start_local = row_start_global - bbox_row_offset
      col_start_local = col_start_global - bbox_col_offset
      row_end_local = row_end_global - bbox_row_offset
      col_end_local = col_end_global - bbox_col_offset

      # Extract cell data from the pre-loaded array
      cell_data_array = nlcd_bbox_array[
          row_start_local:row_end_local,
          col_start_local:col_end_local
      ]

      # Convert reject classes set to numpy array for Numba
      reject_classes = np.array(list(config.nlcd_original_classes_for_reject), dtype=np.uint8)

      # Call Numba-optimized processing function
      valid_i_positions, valid_j_positions, valid_patches = process_patches_numba(
          cell_data_array.astype(np.uint8),  # Ensure uint8 dtype
          full_size_pixels,
          stride,
          config.nlcd_original_unknown_class,
          reject_classes,
          config.max_fraction_reject_class,
          config.downsample_ratio,
          row_start_global,
          col_start_global,
          config.n_samples_max_per_cell if config.n_samples_max_per_cell else -1
      )

      # Convert results back to the expected format
      results = []
      transform = args['transform']
      to_working_crs = args['to_working_crs']

      for i_global, j_global, downsampled in zip(valid_i_positions, valid_j_positions, valid_patches):
          # Convert pixel coordinates to geographic coordinates
          x_ul, y_ul = rasterio.transform.xy(transform, i_global, j_global)
          x_lr, y_lr = rasterio.transform.xy(
              transform,
              i_global + full_size_pixels,
              j_global + full_size_pixels
          )

          # Transform to working CRS
          x_ul_wgs, y_ul_wgs = to_working_crs.transform(x_ul, y_ul)
          x_lr_wgs, y_lr_wgs = to_working_crs.transform(x_lr, y_lr)

          bbox = box(x_ul_wgs, y_lr_wgs, x_lr_wgs, y_ul_wgs)
          results.append((downsampled, bbox, cell_split))

      # Clean up shared memory reference if used
      if args['use_shared_memory']:
          shm.close()

      return results


def sample_images_optimized(grid_gdf, config, use_multiprocessing=True, n_workers=None) -> tuple:
    """Optimized image sampling that pre-loads NLCD data once and uses Numba.
    
    Args:
        grid_gdf: GeoDataFrame containing grid cells
        config: Configuration object
        use_multiprocessing: If True, use multiprocessing; if False, use threading
        n_workers: Number of workers (defaults to CPU count - 1)
    
    Returns:
        tuple: (train_images, train_gdf, test_images, test_gdf)
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    logging.info("Starting optimized image sampling with Numba acceleration...")
    start_time = time.time()

    # Step 1: Pre-load the entire bbox region from NLCD
    logging.info("Pre-loading NLCD data for bbox region...")
    with rasterio.open(config.nlcd_path) as src:
        # Convert bbox to pixel coordinates
        bbox_left, bbox_bottom = from_working_crs.transform(config.bbox_west, config.bbox_south)
        bbox_right, bbox_top = from_working_crs.transform(config.bbox_east, config.bbox_north)

        # Get pixel bounds for the entire bbox
        bbox_row_start, bbox_col_start = src.index(bbox_left, bbox_top)
        bbox_row_end, bbox_col_end = src.index(bbox_right, bbox_bottom)

        # Ensure correct order
        bbox_row_start, bbox_row_end = min(bbox_row_start, bbox_row_end), max(bbox_row_start, bbox_row_end)
        bbox_col_start, bbox_col_end = min(bbox_col_start, bbox_col_end), max(bbox_col_start, bbox_col_end)

        # Read the entire bbox region once
        nlcd_bbox_array = src.read(
            1,
            window=rasterio.windows.Window(
                bbox_col_start, bbox_row_start,
                bbox_col_end - bbox_col_start,
                bbox_row_end - bbox_row_start
            )
        )

        # Store metadata we'll need
        src_transform = src.transform
        src_crs = src.crs

    array_size_mb = nlcd_bbox_array.nbytes / (1024 * 1024)
    logging.info(f"Loaded NLCD bbox array: shape={nlcd_bbox_array.shape}, size={array_size_mb:.2f} MB")

    # Step 2: Prepare cell data with pre-calculated indices
    cell_data_list = []
    for idx, cell in grid_gdf.iterrows():
        bounds = cell.geometry.bounds

        # Convert cell bounds to pixel coordinates
        cell_left, cell_bottom = from_working_crs.transform(bounds[0], bounds[1])
        cell_right, cell_top = from_working_crs.transform(bounds[2], bounds[3])

        with rasterio.open(config.nlcd_path) as src:
            row_start, col_start = src.index(cell_left, cell_top)
            row_end, col_end = src.index(cell_right, cell_bottom)

        # Ensure correct order
        row_start, row_end = min(row_start, row_end), max(row_start, row_end)
        col_start, col_end = min(col_start, col_end), max(col_start, col_end)

        cell_data_list.append((
            idx, cell.geometry, cell['split'],
            row_start, col_start, row_end, col_end
        ))

    # Step 3: Set up shared memory or direct array passing
    shm = None
    if use_multiprocessing:
        # Create shared memory for the array
        shm = shared_memory.SharedMemory(create=True, size=nlcd_bbox_array.nbytes)
        shared_array = np.ndarray(
            nlcd_bbox_array.shape,
            dtype=nlcd_bbox_array.dtype,
            buffer=shm.buf
        )
        shared_array[:] = nlcd_bbox_array[:]

        process_args = {
            'config': config,
            'use_shared_memory': True,
            'shm_name': shm.name,
            'array_shape': nlcd_bbox_array.shape,
            'array_dtype': nlcd_bbox_array.dtype,
            'bbox_row_start': bbox_row_start,
            'bbox_col_start': bbox_col_start,
            'transform': src_transform,
            'to_working_crs': to_working_crs
        }
    else:
        # For threading, we can pass the array directly
        process_args = {
            'config': config,
            'use_shared_memory': False,
            'nlcd_array': nlcd_bbox_array,
            'bbox_row_start': bbox_row_start,
            'bbox_col_start': bbox_col_start,
            'transform': src_transform,
            'to_working_crs': to_working_crs
        }

    # Step 4: Process cells in parallel
    logging.info(f"Processing cells using {n_workers} {'processes' if use_multiprocessing else 'threads'} with Numba...")

    if use_multiprocessing:
        with mp.Pool(n_workers) as pool:
            process_func = partial(process_cell_optimized, args=process_args)
            results = list(tqdm(
                pool.imap(process_func, cell_data_list),
                total=len(cell_data_list),
                desc="Processing grid cells (Numba-optimized)"
            ))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            process_func = partial(process_cell_optimized, args=process_args)
            futures = [executor.submit(process_func, cell_data) for cell_data in cell_data_list]
            results = [
                future.result()
                for future in tqdm(futures, desc="Processing grid cells (Numba-optimized)")
            ]

    # Step 5: Clean up shared memory if used
    if shm is not None:
        shm.close()
        shm.unlink()

    # Step 6: Flatten results and separate train/test
    train_images = []
    test_images = []
    train_bboxes = []
    test_bboxes = []

    for cell_results in results:
        for downsampled, bbox, split in cell_results:
            if split == 'train':
                train_images.append(downsampled)
                train_bboxes.append(bbox)
            else:
                test_images.append(downsampled)
                test_bboxes.append(bbox)

    # Create GeoDataFrames
    working_crs = grid_gdf.crs
    train_gdf = gpd.GeoDataFrame(geometry=train_bboxes, crs=working_crs)
    test_gdf = gpd.GeoDataFrame(geometry=test_bboxes, crs=working_crs)

    elapsed_time = time.time() - start_time
    logging.info(f"Numba-optimized sampling completed in {elapsed_time:.2f} seconds")

    return (
        np.array(train_images, dtype=np.uint8), train_gdf,
        np.array(test_images, dtype=np.uint8), test_gdf
    )


# Use the optimized version with Numba
train_images, train_gdf, test_images, test_gdf = sample_images_optimized(
    grid_gdf,
    config,
    use_multiprocessing=True,  # Set to False to use threading instead
    n_workers=None  # Will use CPU count - 1
)

print(f"Created {len(train_images)} train images and {len(test_images)} test images")

# %% [markdown]
# #### Data validation

# %%
# Make sure all images are in the valid integer range with no NaNs
logging.info(f"Extracted {len(train_images)} training images and {len(test_images)} testing images.")
assert np.all(np.isfinite(train_images))
assert np.all(np.isfinite(test_images))
logging.info("All images are free of null / NaN values.")

# Check that the images are the correct size
assert train_images.shape[1:] == (config.image_size, config.image_size), f"Train shape: {train_images.shape} should be {(config.image_size, config.image_size)}"
assert test_images.shape[1:] == (config.image_size, config.image_size), f"Test shape: {test_images.shape}, should be {(config.image_size, config.image_size)}"
logging.info("All images are the correct size.")


# %%
if config.show_plots:
    train_gdf.plot()

# %% [markdown]
# #### Plot sample images

# %%
n_images = 5

seen_classes = set()

if config.show_plots:
    fig, ax = plt.subplots(2, n_images, figsize=(n_images*2.5, 6))

    for i in range(n_images):
        for j, (images, geom, title) in enumerate(zip([train_images, test_images], [train_gdf.iloc[i].geometry, test_gdf.iloc[i].geometry], ["Train", "Test"])):
            ax[j, i].imshow(lut[images[i]])
            ax[j, i].set_title(f"{title} Image {i+1}")
            ax[j, i].axis('off')

            lat, lon = geom.centroid.xy
            ax[j, i].text(1.5, 5, f"{lat[0]:.3f}, {lon[0]:.3f}", color='black', fontsize=8,
                          bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))
            seen_classes.update(np.unique(images[i]))

    legend_handles = [mpatches.Patch(color=classes_df.loc[idx, "RGB"], label=classes_df.loc[idx, "name"]) for idx, _ in enumerate(seen_classes)]

    fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.show()
else:
    logging.info("Skipping display of sample NLCD images. Set `show_plots` to True to display.")


# %% [markdown]
# # 7. Class distribution across sampled images

# %%
def compute_class_distribution(images):
    unique, counts = np.unique(images, return_counts=True)
    total = counts.sum()
    return {cls: count/total for cls, count in zip(unique, counts)}

train_dist = compute_class_distribution(train_images)
test_dist = compute_class_distribution(test_images)

logging.info("\nFinal class distribution (original class ID: percentage):")
logging.info("\nTraining set:")
for cls_id, pct in train_dist.items():
    logging.info(f"{cls_id} ({config.nlcd_to_name[cls_id]}): {pct*100:.2f}%")



# %% [markdown]
# # 8. Downloading and matching with DEM data

# %% [markdown]
# #### Download data using `elevation`

# %%

if config.download_dem:
    n_dem_downloads, bounds = 625, (config.bbox_west, config.bbox_south, config.bbox_east, config.bbox_north)  # should be a square number
    # Define the bounding box for continental USA (approximate)
    # For testing, use a sample pair of values like below:
    # n_dem_downloads, bounds = 4, (-100.0, 28.0, -99.0, 29.0)  # should be a square number

    os.makedirs(config.dem_dir, exist_ok=True)

    # Calculate the number of splits in each dimension
    n_splits = int(n_dem_downloads ** 0.5)

    # Remove all files from the DEM directory
    for file in config.dem_dir.glob('*.tif'):
        file.unlink()

    # Split bounds into a grid and download DEM data
    with tqdm(total=n_dem_downloads, desc="Downloading DEM data") as pbar:
        for i in range(n_splits):
            for j in range(n_splits):
                west = bounds[0] + (bounds[2] - bounds[0]) * i / n_splits
                east = bounds[0] + (bounds[2] - bounds[0]) * (i + 1) / n_splits
                south = bounds[1] + (bounds[3] - bounds[1]) * j / n_splits
                north = bounds[1] + (bounds[3] - bounds[1]) * (j + 1) / n_splits

                assert west < east, f"West {west} should be less than east {east}"
                assert south < north, f"South {south} should be less than north {north}"
                
                dem_save_path = config.dem_dir / f'conus_dem_{i}_{j}.tif'
                elevation.clip(bounds=(west, south, east, north), output=dem_save_path, product=config.dem_product)

                # Check the statistics on the DEM
                with rasterio.open(dem_save_path) as src:
                    dem_data = src.read(1)
                    dem_nodata = src.nodata
                    dem_stats = {
                        'min': dem_data.min(),
                        'max': dem_data.max(),
                        'mean': dem_data.mean(),
                        'nodata': dem_nodata,
                        'nodata_fraction': np.mean(dem_data == dem_nodata)
                    }
                    logging.info(f"DEM statistics for {dem_save_path}: {dem_stats}")

                pbar.update(1)
else:
    logging.info("Skipping DEM download. Set `download_dem` to True to download.")

# %% [markdown]
# #### Merge into single contiguous DEM raster file

# %%
import glob
import os

from rasterio import merge

if config.merge_dem:
    # Create a list of all the GeoTIFF files
    search_pattern = os.path.join(config.dem_dir, "conus_dem_*.tif")
    dem_files = glob.glob(search_pattern)

    src_files_to_mosaic = []
    for file in dem_files:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge.merge(src_files_to_mosaic)

    # Copy the metadata from one of the input files
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(config.merged_dem_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        logging.info(f"Merged DEM saved to {config.merged_dem_path}")

    logging.info(f"Proportion of missing data in merged DEM: {np.mean(mosaic < config.dem_nodata_threshold):.2%}")

    for src in src_files_to_mosaic:
        src.close()

    # Delete variables to save on memory
    del mosaic



# %% [markdown]
# #### Check merged file metadata

# %%
# Print basic information about the merged GeoTIFF file
with rasterio.open(config.merged_dem_path) as merged_src:
    logging.info(f"Dataset CRS: {merged_src.crs}")
    logging.info(f"Dataset bounds: {merged_src.bounds}")
    logging.info(f"Dataset shape: {merged_src.shape}")
    logging.info(f"Dataset resolution: {merged_src.res}")
    logging.info(f"Dataset transform: {merged_src.transform}")
    logging.info(f"Missing data value: {merged_src.nodata}")
    logging.info(f"Data type: {merged_src.dtypes}")


# %% [markdown]
# #### Show merged file as elevation heatmap

# %%
# Load the image and run imshow
with rasterio.open(config.merged_dem_path) as src:
        downsample_stride = 100
        dem_data = src.read(1,
            out_shape=(
                src.count,
                int(src.height / downsample_stride),
                int(src.width / downsample_stride)
            ),
        resampling=rasterio.enums.Resampling.nearest
    )
if config.show_plots:
    
    # Calculate slope
    x, y = np.gradient(dem_data, src.res[0], src.res[1])
    slope = np.sqrt(x**2 + y**2)
    log_slope = np.log10(slope + 1)  # Adding 1 to avoid log(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Plot elevation
    im1 = axes[0].imshow(dem_data, cmap='terrain', extent=(bounds[0], bounds[2], bounds[1], bounds[3]), vmin=0)
    axes[0].set_title('Merged DEM Data')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', label='Elevation (meters)')
    
    # Plot log10 slope
    im2 = axes[1].imshow(log_slope, cmap='viridis', extent=(bounds[0], bounds[2], bounds[1], bounds[3]))
    axes[1].set_title('Log10 Slope')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', label='Log10 Slope')
    
    # Set the ticks to match the bounds
    for ax in axes:
        ax.set_xticks(np.linspace(bounds[0], bounds[2], num=3))
        ax.set_yticks(np.linspace(bounds[1], bounds[3], num=3))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax.grid(True, linestyle='--', alpha=0.8, color='k')
        
    plt.tight_layout()
    plt.show()
else:
    logging.info("Skipping display of merged DEM data. Set `show_plots` to True to display.")

# %%
dem_data.max()

# %% [markdown]
# # 9. Join elevation data with land cover data

# %%
def extract_dem_images(gdf: gpd.GeoDataFrame, dem_src: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract DEM images and calculate slope and aspect.
    
    Returns:
        dem_images: Array of shape (N, H, W) with normalized elevation (uint8)
        slopes: Array of shape (N, H, W) with slope in degrees
        aspects: Array of shape (N, H, W) with aspect in degrees (0-360)
        aux_features: Array of shape (N, 2) with mean slope and mean aspect
    """
    dem_images = []
    slopes = []
    aspects = []
    nodata_count = 0
    interpolate_count = 0
    
    # Pixel size in meters (30m NLCD resolution * downsample ratio)
    pixel_size = 30.0 * config.downsample_ratio
    
    print(f"Extracting DEM images for {len(gdf)} samples")
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extracting DEM images"):
        
        window = rasterio.windows.from_bounds(*row.geometry.bounds, transform=dem_src.transform) 
        
        dem_data = dem_src.read(1, window=window)

        # If the data is NaN or it's above the threshold, we set it to NaN
        is_nodata = np.logical_or(dem_data == dem_src.nodata, dem_data>config.dem_elev_max)
        nodata_fraction = np.mean(is_nodata)
        dem_data = dem_data.astype(np.float32)

        # If the read failed, the shape will be empty so we raise an alarm
        # If any failure cases occur, we want the resulting DEM array to be all NaNs
        # and have all dims with nonzero size.
        if len(dem_data.shape) == 0:
            dem_data  = np.empty((config.image_size, config.image_size)) * np.nan
            logging.debug(f"Failed to read window for row {row} with window {window}")
            nodata_count += 1
        elif any([dim == 0 for dim in dem_data.shape]):
            dem_data = np.empty((config.image_size, config.image_size)) * np.nan
            logging.debug(f"Window read for row {row} with bbox {row.geometry.bounds} has a zero dimension with shape {dem_data.shape}")
            nodata_count += 1
        elif nodata_fraction > config.dem_nodata_threshold:            
            dem_data *= np.nan
            nodata_count += 1
        elif np.any(is_nodata):
            # Interpolate NaN values using a spatially informed method
            dem_data = cv2.inpaint(dem_data, is_nodata.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            interpolate_count += 1

        assert np.all(np.isnan(dem_data)) or np.nanmax(dem_data) < config.dem_elev_max, f"DEM data contains values above nodata threshold: {dem_data.max()}"

        # Resize using cv2 to the desired image size
        if not np.any(np.isnan(dem_data)):
            dem_data = cv2.resize(dem_data, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR)
            
            # Calculate slope and aspect from raw DEM data (in meters)
            dy, dx = np.gradient(dem_data, pixel_size)
            
            # Calculate slope in degrees
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope = np.degrees(slope_rad)
            
            # Calculate aspect in degrees (0-360, where 0=North)
            aspect_rad = np.arctan2(-dx, dy)  # negative dx for proper orientation
            aspect = np.degrees(aspect_rad) % 360
        else:
            # For invalid DEM data, set slope and aspect to NaN
            slope = np.full((config.image_size, config.image_size), np.nan, dtype=np.float32)
            aspect = np.full((config.image_size, config.image_size), np.nan, dtype=np.float32)

        dem_images.append(dem_data)
        slopes.append(slope)
        aspects.append(aspect)

    logging.info(f"Number of images dropped due to nodata proportion exceeding threshold: {nodata_count} / {len(gdf)}")
    logging.info(f"Number of images with interpolation of missing values: {interpolate_count} / {len(gdf)}")
    
    # Convert to arrays
    dem_images = np.array(dem_images).astype(np.float32)
    slopes = np.array(slopes).astype(np.float32)
    aspects = np.array(aspects).astype(np.float32)
    
    # Create auxiliary features (mean slope and circular mean aspect)
    N = len(dem_images)
    aux_features = np.zeros((N, 2), dtype=np.float32)
    
    for i in range(N):
        if np.any(np.isnan(slopes[i])):
            # Set to zeros for invalid images
            aux_features[i] = [0.0, 0.0]
        else:
            # Mean slope
            aux_features[i, 0] = np.mean(slopes[i])
            
            # Circular mean for aspect
            aspect_rad = np.radians(aspects[i])
            mean_sin = np.mean(np.sin(aspect_rad))
            mean_cos = np.mean(np.cos(aspect_rad))
            mean_aspect = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
            aux_features[i, 1] = mean_aspect
    
    return dem_images, slopes, aspects, aux_features

with rasterio.open(config.merged_dem_path) as dem_src:
    train_dem_images, train_slopes, train_aspects, train_aux = extract_dem_images(train_gdf, dem_src)
    test_dem_images, test_slopes, test_aspects, test_aux = extract_dem_images(test_gdf, dem_src)

logging.info(f"Train auxiliary features shape: {train_aux.shape}")
logging.info(f"Test auxiliary features shape: {test_aux.shape}")
logging.info(f"Train aux stats - Slope: mean={train_aux[:, 0].mean():.2f}°, std={train_aux[:, 0].std():.2f}°")
logging.info(f"Train aux stats - Aspect: mean={train_aux[:, 1].mean():.2f}°, std={train_aux[:, 1].std():.2f}°")

# Normalize DEM images for visualization
# Offset all images to have a minimum of zero
train_dem_images -= train_dem_images.min(axis=(1, 2), keepdims=True)
test_dem_images -= test_dem_images.min(axis=(1, 2), keepdims=True)

train_dem_images = train_dem_images / train_dem_images.max(axis=(1, 2), keepdims=True)
test_dem_images  = test_dem_images / test_dem_images.max(axis=(1, 2), keepdims=True)

# Cast to uint8
train_dem_images = (train_dem_images * 255).astype(np.uint8)
test_dem_images = (test_dem_images * 255).astype(np.uint8)

# %% [markdown]
# ### Histogram of DEM values
# 

# %%
plt.hist(train_dem_images.flatten(), bins=100);


# %% [markdown]
# #### Show DEM images

# %%
# Plot several train and test images using elevation colormap
if config.show_plots:
    n_images = 3
    fig, ax = plt.subplots(2, n_images, figsize=(8, 5))

    # Pick random sample of train and test images to show
    sampled_train_indices = np.random.choice(len(train_dem_images), n_images, replace=False)
    sampled_test_indices = np.random.choice(len(test_dem_images), n_images, replace=False)

    for i in range(n_images):
        for j, (images, gdf, title, sampled_indices) in enumerate(zip(
                [train_dem_images, test_dem_images], 
                [train_gdf, test_gdf], 
                ["Train DEM", "Test DEM"],
                [sampled_train_indices, sampled_test_indices])):
            
            im = ax[j, i].imshow(images[sampled_indices[i]], cmap='terrain')
            ax[j, i].set_title(f"{title} Image {i+1}")
            ax[j, i].axis('off')
            centroid = gdf.iloc[sampled_indices[i]].geometry.centroid
            ax[j, i].text(1.5, 3, f"{centroid.y:.3f}, {centroid.x:.3f}",color='black', 
                          bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))
            # Add gridlines and lat/long overlay
            ax[j, i].grid(True, linestyle='--', alpha=0.8, color='k')
            ax[j, i].set_xticks(np.linspace(0, config.image_size, num=3))
            ax[j, i].set_yticks(np.linspace(0, config.image_size, num=3))
            ax[j, i].set_xticklabels(np.linspace(centroid.x - config.image_size // 2, centroid.x + config.image_size // 2, num=3).round(2))
            ax[j, i].set_yticklabels(np.linspace(centroid.y - config.image_size // 2, centroid.y + config.image_size // 2, num=3).round(2))
            
    # Add a common colorbar on the right-hand side
    cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Elevation (meters)')

    plt.tight_layout(rect=[0.1, 0, 1, 1])
    plt.show()
else:
    logging.info("Skipping DEM image visualization; set show_plots to True to display images.")


# %%
# Create visualization of NLCD and DEM data
n_images = 3
seen_classes = set()
show_legend = False
if config.show_plots:
    fig, axes = plt.subplots(2, n_images, figsize=(n_images*2.5, 6))

    # Pick random sample of train images to show
    
    sampled_indices = np.random.choice(len(train_images), n_images, replace=False)

    # Plot first n_images from training set
    for i, sample_idx in enumerate(sampled_indices):

        # Get DEM data for this image
        dem = train_dem_images[sample_idx]
        dem_min = dem.min()
        dem_relative = dem - dem_min
        
        # Calculate contours (relative to minimum elevation)
        levels = np.linspace(0, dem_relative.max(), 10)
        
        # Plot NLCD with contours
        axes[0, i].imshow(lut[train_images[sample_idx]])
        cs = axes[0, i].contour(dem_relative, levels=levels, colors='k', alpha=0.7, linewidths=0.5)
        axes[0, i].clabel(cs, inline=True, fontsize=8, fmt='%.0f')
        
        axes[0, i].axis('off')
        
        # Add lat/lon labels to image
        centroid = train_gdf.iloc[sample_idx].geometry.centroid
        axes[0, i].text(1.5, 5, f"{centroid.y:.3f}, {centroid.x:.3f}", color='black', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))
        axes[0, i].grid(True, linestyle='--', alpha=0.8, color='k')

        # Plot DEM
        im = axes[1, i].imshow(dem, cmap='terrain')
        axes[1, i].axis('off')
        seen_classes.update(np.unique(train_images[sample_idx]))
    
    axes[1, 0].set_ylabel(f'Elevation')
    axes[0, 0].set_ylabel(f'Land Cover')
    
    # Add colorbar for elevation below the subplots
    cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Elevation, relative to minimum (m)')

    # Add legend for NLCD classes above the subplots
    if show_legend:
        legend_handles = [mpatches.Patch(color=classes_df.loc[idx, "RGB"], 
                                        label=classes_df.loc[idx, "name"]) 
                        for idx in enumerate(seen_classes)]
        fig.legend(handles=legend_handles, loc='upper center', 
                bbox_to_anchor=(0.5, 0.99), ncol=4)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()
else:
    logging.info("Skipping visualization of NLCD and DEM data; set show_plots=True to enable.")


# %%
# Visualize slope and aspect for a few samples
if config.show_plots:
    n_images_shown = 3
    fig, axes = plt.subplots(3, n_images_shown, figsize=(n_images_shown*3, 9))

    sample_indices = np.random.choice(len(train_slopes), n_images_shown, replace=False)

    for i, idx in enumerate(sample_indices):
        # Original DEM
        axes[0, i].imshow(train_dem_images[idx], cmap='terrain')
        axes[0, i].set_title(f'DEM {i+1}')
        axes[0, i].axis('off')
        
        # Slope
        im_slope = axes[1, i].imshow(train_slopes[idx], cmap='YlOrRd', vmin=0, vmax=45)
        axes[1, i].set_title(f'Slope (mean: {train_aux[idx, 0]:.1f}°)')
        axes[1, i].axis('off')
        
        # Aspect
        im_aspect = axes[2, i].imshow(train_aspects[idx], cmap='hsv', vmin=0, vmax=360)
        axes[2, i].set_title(f'Aspect (mean: {train_aux[idx, 1]:.1f}°)')
        axes[2, i].axis('off')
    
    # Add colorbars
    cbar_slope = fig.add_axes([0.92, 0.38, 0.02, 0.2])
    fig.colorbar(im_slope, cax=cbar_slope, label='Slope (°)')
    
    cbar_aspect = fig.add_axes([0.92, 0.08, 0.02, 0.2])
    fig.colorbar(im_aspect, cax=cbar_aspect, label='Aspect (°)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
else:
    logging.info("Skipping visualization of slope and aspect. Set show_plots=True to enable.")

# %%

logging.info(f"Train auxiliary features shape: {train_aux.shape}")
logging.info(f"Test auxiliary features shape: {test_aux.shape}")
logging.info(f"Train aux stats - Slope: mean={train_aux[:, 0].mean():.2f}°, std={train_aux[:, 0].std():.2f}°")
logging.info(f"Train aux stats - Aspect: mean={train_aux[:, 1].mean():.2f}°, std={train_aux[:, 1].std():.2f}°")

# %% [markdown]
# # 10. Create tokenized arrays

# %%
'''
At this stage, we want to ge the unique DxD patches from both `train_images` and `test_images` which have shape (Ntrain, H, W) and (Ntest, H, W) respectively with integer datatype.
D is the token downsampling ratio here, so a ratio of D means that each patch is DxD and the shape of the tokenized image is (H // D, W // D). At the end, we
need the `train_images_tokenized`, `test_images_tokenized` and the `decode_table`(shape (K, D, D) where K is the number of unique tokens)
'''

D = config.tokenizer_downsample_ratio

# Extract all DxD patches
def extract_patches(images, D):
    N, H, W = images.shape
    patches = images.reshape(N, H//D, D, W//D, D).transpose(0,1,3,2,4).reshape(-1, D, D)
    return patches

# Get unique patches and create decode table
all_patches = np.vstack([extract_patches(train_images, D), extract_patches(test_images, D)])
decode_table, inverse = np.unique(all_patches.reshape(len(all_patches), -1), 
                                  axis=0, return_inverse=True)
decode_table = decode_table.reshape(-1, D, D)

# Tokenize images
n_train_patches = (train_images.shape[0] * train_images.shape[1] * train_images.shape[2]) // (D * D)
train_tokens = inverse[:n_train_patches]
test_tokens = inverse[n_train_patches:]

# Reshape to tokenized images
train_images_tokenized = train_tokens.reshape(train_images.shape[0], train_images.shape[1]//D, train_images.shape[2]//D)
test_images_tokenized = test_tokens.reshape(test_images.shape[0], test_images.shape[1]//D, test_images.shape[2]//D)

# Decode and verify first 3 images
def decode_images(tokenized, decode_table, D):
    N, th, tw = tokenized.shape
    decoded = decode_table[tokenized.flatten()].reshape(N, th, tw, D, D)
    return decoded.transpose(0,1,3,2,4).reshape(N, th*D, tw*D)

train_decoded = decode_images(train_images_tokenized[:3], decode_table, D)
test_decoded = decode_images(test_images_tokenized[:3], decode_table, D)

assert np.array_equal(train_decoded, train_images[:3]), "Train decoding mismatch!"
assert np.array_equal(test_decoded, test_images[:3]), "Test decoding mismatch!"
print(f"Decoding verified successfully! There are {decode_table.shape[0]} unique tokens.")

# %% [markdown]
# # 11. Concatenate data and save to disk + s3

# %%
save_data = True

# %%
if save_data:
    is_image_bad_train = np.any(np.isnan(train_dem_images), axis=(1, 2))
    is_image_kept_train = ~is_image_bad_train

    train_gdf_final = train_gdf[is_image_kept_train]
    logging.info(f"Removed {is_image_bad_train.sum()} images with missing DEM data from training set.")

    is_image_bad_test = np.any(np.isnan(test_dem_images), axis=(1, 2))
    is_image_kept_test = ~is_image_bad_test
    test_gdf_final = test_gdf[is_image_kept_test]
    logging.info(f"Removed {is_image_bad_test.sum()} images with missing DEM data from test set.")

    # Apply same mask to auxiliary features
    train_aux_final = train_aux[is_image_kept_train]
    test_aux_final = test_aux[is_image_kept_test]

    train_gdf_final.to_crs('EPSG:4326', inplace=True)
    test_gdf_final.to_crs('EPSG:4326', inplace=True)

    # Take arrays of shape (N, H, W) and stack them along the channel axis
    # which needs to be created for both data sets
    train_combined = np.stack([train_images[is_image_kept_train], train_dem_images[is_image_kept_train]], axis=1)
    test_combined = np.stack([test_images[is_image_kept_test],  test_dem_images[is_image_kept_test]], axis=1)

    train_gdf_path = config.output_path_train_gpkg
    train_gdf_final.to_file(train_gdf_path, driver='GPKG')
    train_gpkg_size = os.path.getsize(train_gdf_path)
    logging.info(f"Training sample location GeoDataFrame saved to {train_gdf_path} (Size: {train_gpkg_size / (1024 * 1024):.2f} MB)")

    test_gdf_path = config.output_path_test_gpkg
    test_gdf_final.to_file(test_gdf_path, driver='GPKG')
    test_gpkg_size = os.path.getsize(test_gdf_path)
    logging.info(f"Test sample location GeoDataFrame saved to {test_gdf_path} (Size: {test_gpkg_size / (1024 * 1024):.2f} MB)")

    # Save the lat-long coordinates of the training and test samples
    train_coords = np.stack(
        [train_gdf_final.centroid.x.values,
        train_gdf_final.centroid.y.values,],
    axis=1)

    test_coords = np.stack(
        [test_gdf_final.centroid.x.values,
        test_gdf_final.centroid.y.values],
    axis=1)

    # Save with auxiliary data included
    np.savez_compressed(
        config.output_path, 
        train_data=train_combined, 
        test_data=test_combined, 
        train_coords=train_coords,
        test_coords=test_coords,
        train_data_tokenized=train_images_tokenized,
        test_data_tokenized=test_images_tokenized,
        decode_table=decode_table,
        train_aux=train_aux_final,  # Add auxiliary features
        test_aux=test_aux_final      # Add auxiliary features
    )
    
    logging.info(f"Saved training and test data with auxiliary features to {config.output_path} (Size: {os.path.getsize(config.output_path) / (1024 * 1024):.2f} MB)")
    logging.info(f"Train aux shape: {train_aux_final.shape}, Test aux shape: {test_aux_final.shape}")
    
    if config.upload_to_s3:
        import boto3
        s3 = boto3.client('s3')
        bucket_name = BUCKET_NAME
        files_to_upload = [
            config.output_path,
            train_gdf_path,
            test_gdf_path, 
        ]

        for file in files_to_upload:
            logging.info(f"Uploading {file} to S3 bucket {bucket_name}...")
            try:
                s3.upload_file(str(file), bucket_name, file.name)
                logging.info(f"Uploaded {file} to S3 bucket {bucket_name} as {file.name}")

            except Exception as e:
                logging.error(f"Failed to upload {file} to S3 bucket {bucket_name}: {e}")

# %% [markdown]
# ### Show large, random sample of LULC images

# %%
plot_lulc_sample = False

if plot_lulc_sample:
    fig, axes = plt.subplots(10, 16, figsize=(48, 30))
    axes = axes.flatten()
    for i in range(16*10):
        ax = axes[i]
        random_index = np.random.randint(0, train_combined.shape[0])
        image = train_combined[random_index, 0].astype(np.uint8)
        ax.imshow(lut[image])
        ax.axis('off')
    plt.tight_layout()

# %% [markdown]
# # 12. Create animations

# %%
if not 'train_images_final' in locals():
    train_images_final = np.load(config.output_path)['train_data']

# %% [markdown]
# #### Multiple data samples

# %%
make_animation = False

# %%
if make_animation:
    class TerrainAnimator:
        def __init__(self, train_images, train_dem_images, lut, n_rows=4, n_cols=8):
            self.train_images = train_images
            self.train_dem_images = train_dem_images
            self.lut = lut
            self.n_rows = n_rows
            self.n_cols = n_cols
            self.exaggeration = 1
            
            # Pre-calculate mesh grid
            self.h, self.w = train_images[0].shape
            x = np.arange(self.w)
            y = np.arange(self.h)
            self.X, self.Y = np.meshgrid(x, y)
            
            # Initialize figure
            self.setup_figure()
            
        def setup_figure(self):
            plt.rcParams['figure.dpi'] = 300
            self.fig, self.axes = plt.subplots(
                self.n_rows, 
                self.n_cols, 
                figsize=(self.n_cols*1.4, self.n_rows*1.6),  # Reduced figure size
                subplot_kw={'projection': '3d'},
                constrained_layout=True  # Use constrained layout
            )
            self.fig.set_facecolor('black')
            self.fig.patch.set_alpha(1.0)
            # Reduce margins
            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
            
            # Select random indices once
            self.indices = np.random.choice(
                len(self.train_images), 
                self.n_rows * self.n_cols, 
                replace=False
            )
            
        def process_elevation(self, elevation):
            """Pre-process elevation data with Gaussian smoothing"""
            return cv2.GaussianBlur(elevation, (3, 3), 0)
            
        def create_surface(self, ax, idx):
            """Create a single surface plot"""
            land_cover = self.train_images[idx]
            elevation = self.process_elevation(self.train_dem_images[idx])
            
            surf = ax.plot_surface(
                self.X, self.Y,
                elevation * self.exaggeration,
                facecolors=self.lut[land_cover],
                shade=False,
                antialiased=False,
                rstride=1,
                cstride=1
            )

            ax.set_facecolor('black')
            
            # Configure view
            ax.view_init(elev=30, azim=45)
            ax.set_box_aspect([1, 1, 0.5])
            
            # Remove unnecessary elements
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.axis('off')

            ele_max = elevation.max()

            if ele_max < 100:
                zlim = 150
            elif ele_max < 200:
                zlim = 250
            else:
                zlim = max(300, ele_max * 3)

            ax.set_zlim(0, zlim)
            
            return surf
            
        def setup_plots(self):
            """Initialize all surface plots in parallel"""
            with ThreadPoolExecutor() as executor:
                self.surfaces = list(executor.map(
                    lambda args: self.create_surface(*args),
                    zip(self.axes.flatten(), self.indices)
                ))
            
            plt.subplots_adjust(hspace=-0.6, wspace=-0.2)  # Increased overlap between subplots
            
        def update(self, frame):
            """Animation update function"""
            for ax in self.axes.flatten():
                ax.view_init(elev=30, azim=frame)
            return self.surfaces
            
        def create_animation(self, frames=360, fps=30, out_path=config.output_path_animation):
            """Create and save the animation"""
            self.setup_plots()
            
            anim = FuncAnimation(
                self.fig,
                self.update,
                frames=frames,
                interval=1000/fps,
                blit=True
            )
            
            # Save with optimized settings
            anim.save(
                out_path,
                writer='pillow',
                fps=fps,
                savefig_kwargs={'facecolor': 'black'},
                progress_callback=lambda i, n: print(f'Saving frame {i}/{n}', end='\r')
            )
            plt.close()


    # Usage
    train_images_final[:,0]
    animator = TerrainAnimator(
        train_images_final[:,0], train_images_final[:,1], lut,
        n_rows=4, n_cols=4
    )
    animator.create_animation()

# %% [markdown]
# <img src="terrain_rotation.gif" width="1500" align="center">
# 

# %% [markdown]
# # Profiling

# %%
%whos


