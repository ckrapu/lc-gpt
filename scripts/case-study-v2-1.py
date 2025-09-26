'''
This script uses the LC-GPT model to inpaint missing regions in NLCD land cover data.
The missing regions are defined by geojson files and are inpainted at various resolutions.
This version improves on its predecessor by using a multiresolution inference scheme as follows:
- Perform inpainting at finest resolution that keeps the masked area <X% of the total context window
- Use the pixels at the coarse resolution inpaint to mask the logits for the finer resolution inpainting by only allowin
    tokens corresponding to raw pixel values which could be aggregated to the coarser inpainted value via majority vote. For example,
    if the coarse resolution pixel is `41`, then the fine resolution inpaint would be restricted to tokens corresponding to patches
    like `[[41, 41], [41, 42]]` since the majority is `41`. Ties are allowed here, so `[[41, 42], [42, 41]]` is also valid.
- For infilling large regions, proceed by randomly sampling context windows overlapping the inpainted region which have greater than Y% filled-in pixels
    where the infill is either ground truth (outside of the mask) or previously inpainted pixels (inside the mask).
'''

from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import sys
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
import rasterio.windows
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import rasterize as rio_rasterize
from omegaconf import OmegaConf

sys.path.append("./")
from RandAR.utils import instantiate_from_config
from RandAR.utils.inpainting import generate_inpainting
from RandAR.dataset.nlcd_dataset import detokenize
from RandAR.model.nlcd_tokenizer import NLCDTokenizer
import pickle


@dataclass
class Config:
    # IO
    config_yaml: str = "configs/randar_nlcd_128_large.yaml"
    ckpt_dir: str = "results/randar_nlcd_128_large/checkpoints/iter_180000"
    data_npz: str = "data/data_128_final.npz"  # for decode_table
    geojson_dir: str = "data/inpaint_regions"
    nlcd_img_path: str = "data/nlcd_2021_land_cover_l48_20230630.img"  # open .img (has .ige sidecar)
    output_dir: str = "results/case_study_v2"
    rasters_dir: str = output_dir + "/output_rasters"

    # Cases - list of bases to process (resolution will be calculated dynamically)
    bases: list = field(default_factory=lambda: [
        "ft_belvoir",
        "ft_custer_training_center",
        "ft_hood",
        "eglin_afb",
        "joint_base_lewis-mcchord",
        "vandenberg_afb",
    ])

    # Inference
    device: str = "cuda"
    window_tokens: int = 64  # 64x64 tokens → 128x128 raw pixels
    stride_tokens: int = 64  # no overlap by default; set 32 for overlap
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    cfg_scales: tuple[float, float] = (1.0, 1.0)
    seed: int = 42
    forbidden_nlcd: tuple[int, ...] = (11, 12, 90, 95)  # water/wetlands
    
    # Multiresolution parameters
    multiresolution: bool = True
    max_mask_ratio_coarse: float = 0.35  # Max mask coverage for coarsest resolution
    max_mask_ratio_fine: float = 0.70  # Max mask coverage for finer resolutions
    min_filled_ratio: float = 0.65  # Min ratio of filled pixels for window selection
    finest_resolution: int = 120  # Finest resolution to use (in meters)
    
    # Test mode
    test_mode: bool = False  # If True, use random tokens instead of model inference
    samples_per_base: int = 3 if test_mode else 10



def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S')


def load_decode_table(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    decode_table = data['decode_table']  # (T, D, D)
    logging.info(f"Loaded decode_table with shape {decode_table.shape} from {npz_path}")
    return decode_table


def compute_disallowed_token_indices(decode_table: np.ndarray,
                                     forbidden_nlcd: tuple[int, ...]) -> np.ndarray:
    flat = decode_table.reshape(decode_table.shape[0], -1)
    mask = np.isin(flat, np.array(forbidden_nlcd, dtype=flat.dtype)).any(axis=1)
    idx = np.where(mask)[0].astype(np.int64)
    logging.info(f"Identified {len(idx)} tokens that include NLCD classes {forbidden_nlcd}")
    return idx


def tokenize_image(raw: np.ndarray, decode_table: np.ndarray) -> np.ndarray:
    # D×D patch per token
    _, D, D2 = decode_table.shape
    assert D == D2, "decode_table must have square patches"
    H, W = raw.shape
    Hc = (H // D) * D
    Wc = (W // D) * D
    if (Hc, Wc) != (H, W):
        logging.info(f"Cropping raw from {(H, W)} to {(Hc, Wc)} to align with D={D}")
        raw = raw[:Hc, :Wc]

    Ht, Wt = H // D, W // D
    patches = raw.reshape(Ht, D, Wt, D).transpose(0, 2, 1, 3).reshape(Ht * Wt, D * D)
    decode_flat = decode_table.reshape(decode_table.shape[0], D * D)

    # Exact match per patch to row in decode_flat
    # Build hash for decode patches to indices
    # For small D (2) and 50k codes, brute-force compare via vectorization is acceptable in batches
    tokens = np.empty(Ht * Wt, dtype=np.int32)
    batch = 8192
    for start in range(0, patches.shape[0], batch):
        end = min(start + batch, patches.shape[0])
        p = patches[start:end]
        # Compare each p to decode_flat: (B, 1, P) == (1, T, P) → (B, T, P) → all axis=2
        eq = (p[:, None, :] == decode_flat[None, :, :]).all(axis=2)
        has_match = eq.any(axis=1)
        if np.all(has_match):
            idx = np.argmax(eq, axis=1)
            tokens[start:end] = idx.astype(np.int32)
        else:
            # Fallback: pick closest token by minimal Hamming distance
            # distance = number of unequal elements per code
            # Compute in chunks of codes to manage memory
            idx = np.empty(p.shape[0], dtype=np.int32)
            for i in range(p.shape[0]):
                if has_match[i]:
                    idx[i] = np.argmax(eq[i])
                else:
                    # Broadcast compare with XOR-like inequality
                    diff = (decode_flat != p[i][None, :])
                    dist = diff.sum(axis=1)
                    idx[i] = int(np.argmin(dist))
            tokens[start:end] = idx

    return tokens.reshape(Ht, Wt)


def coarsen_mask(mask_raw: np.ndarray, D: int) -> np.ndarray:
    H, W = mask_raw.shape
    Hc = (H // D) * D
    Wc = (W // D) * D
    if (Hc, Wc) != (H, W):
        mask_raw = mask_raw[:Hc, :Wc]
    Ht, Wt = Hc // D, Wc // D
    mask_tokens = mask_raw.reshape(Ht, D, Wt, D).any(axis=(1, 3))
    return mask_tokens


def coarsen_mask_by_ratio(mask_raw_30m: np.ndarray, ratio: int) -> np.ndarray:
    h, w = mask_raw_30m.shape
    new_h, new_w = h // ratio, w // ratio
    mask_raw_30m = mask_raw_30m[:new_h * ratio, :new_w * ratio]
    return mask_raw_30m.reshape(new_h, ratio, new_w, ratio).any(axis=(1, 3))


def calculate_optimal_resolution(base_name: str, nlcd_img: Path, geojson_dir: Path, 
                                max_mask_ratio: float = 0.35) -> tuple[int, float]:
    """
    Calculate the optimal resolution for a region such that a 128x128 pixel window
    contains fewer than max_mask_ratio (35%) masked pixels.
    Returns a tuple of (resolution in meters, mask ratio at that resolution).
    """
    resolutions = [30, 60, 120, 240, 480, 960]
    
    geojson_path = geojson_dir / f"base_{base_name}.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    
    with rasterio.open(nlcd_img) as nlcd:
        gdf_nlcd = gdf.to_crs(nlcd.crs)
        geom_nlcd = gdf_nlcd.geometry.union_all()
        bounds = geom_nlcd.bounds
        
        # For each resolution, check mask coverage in a 128x128 window
        for res_m in resolutions:
            ratio = int(round(res_m / 30.0))
            target_size_30m = 128 * ratio
            
            # Calculate window centered on mask
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            # Expand bounds to ensure we get a 128x128 window at this resolution
            expanded_bounds = (
                center_x - target_size_30m * nlcd.transform.a / 2,
                center_y - target_size_30m * abs(nlcd.transform.e) / 2,
                center_x + target_size_30m * nlcd.transform.a / 2,
                center_y + target_size_30m * abs(nlcd.transform.e) / 2
            )
            
            # Get window and rasterize mask
            window = rasterio.windows.from_bounds(*expanded_bounds, transform=nlcd.transform)
            window = window.round_offsets().round_lengths()
            
            # Calculate transform for this window
            x0 = nlcd.transform.c + window.col_off * nlcd.transform.a
            y0 = nlcd.transform.f + window.row_off * nlcd.transform.e
            
            # Rasterize mask at 30m resolution
            mask_30m = rio_rasterize(
                [(geom_nlcd, 1)],
                out_shape=(int(window.height), int(window.width)),
                transform=Affine(nlcd.transform.a, 0, x0, 0, nlcd.transform.e, y0),
                fill=0,
                all_touched=False,
                dtype='uint8'
            ).astype(bool)
            
            # Ensure we have exactly 128x128 at target resolution
            h, w = mask_30m.shape
            target_h = min(h, target_size_30m)
            target_w = min(w, target_size_30m)
            
            # Center crop if needed
            if h > target_size_30m:
                start_h = (h - target_size_30m) // 2
                mask_30m = mask_30m[start_h:start_h + target_size_30m, :]
            if w > target_size_30m:
                start_w = (w - target_size_30m) // 2
                mask_30m = mask_30m[:, start_w:start_w + target_size_30m]
            
            # Pad if needed
            if mask_30m.shape != (target_size_30m, target_size_30m):
                pad_h = target_size_30m - mask_30m.shape[0]
                pad_w = target_size_30m - mask_30m.shape[1]
                mask_30m = np.pad(mask_30m, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            # Calculate mask ratio at this resolution
            mask_coarse = coarsen_mask_by_ratio(mask_30m, ratio)
            mask_ratio = mask_coarse.sum() / mask_coarse.size
            
            logging.info(f"{base_name}: Testing resolution {res_m}m - mask coverage: {mask_ratio:.2%} ({mask_coarse.sum()}/{mask_coarse.size} pixels)")
            
            # If mask ratio is below threshold, this resolution is acceptable
            if mask_ratio < max_mask_ratio:
                logging.info(f"{base_name}: Selected resolution {res_m}m (mask coverage {mask_ratio:.2%} < {max_mask_ratio:.0%} threshold)")
                return res_m, mask_ratio
    
    # If no resolution meets the criteria, return the coarsest with its mask ratio
    # Need to recalculate the mask ratio for the coarsest resolution
    logging.warning(f"{base_name}: No resolution achieved < {max_mask_ratio:.0%} mask coverage, using maximum {resolutions[-1]}m")
    return resolutions[-1], mask_ratio  # Last calculated mask_ratio


def select_resolution_hierarchy(base_name: str, nlcd_img: Path, geojson_dir: Path, 
                               max_mask_ratio_coarse: float = 0.35,
                               max_mask_ratio_fine: float = 0.70,
                               finest_resolution: int = 30) -> list[tuple[int, float]]:
    """
    Select a hierarchy of resolutions for multiresolution inpainting.
    Returns list of (resolution, mask_ratio) tuples from coarse to fine.
    """
    resolutions = [960, 480, 240, 120, 60, 30]
    hierarchy = []
    
    geojson_path = geojson_dir / f"base_{base_name}.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    
    with rasterio.open(nlcd_img) as nlcd:
        gdf_nlcd = gdf.to_crs(nlcd.crs)
        geom_nlcd = gdf_nlcd.geometry.union_all()
        bounds = geom_nlcd.bounds
        
        # Find coarsest resolution with acceptable mask coverage
        coarsest_res = None
        for res_m in resolutions:
            if res_m < finest_resolution:
                continue
                
            ratio = int(round(res_m / 30.0))
            target_size_30m = 128 * ratio
            
            # Calculate window centered on mask
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            
            # Expand bounds to ensure we get a 128x128 window at this resolution
            expanded_bounds = (
                center_x - target_size_30m * nlcd.transform.a / 2,
                center_y - target_size_30m * abs(nlcd.transform.e) / 2,
                center_x + target_size_30m * nlcd.transform.a / 2,
                center_y + target_size_30m * abs(nlcd.transform.e) / 2
            )
            
            # Get window and rasterize mask
            window = rasterio.windows.from_bounds(*expanded_bounds, transform=nlcd.transform)
            window = window.round_offsets().round_lengths()
            
            # Calculate transform for this window
            x0 = nlcd.transform.c + window.col_off * nlcd.transform.a
            y0 = nlcd.transform.f + window.row_off * nlcd.transform.e
            
            # Rasterize mask at 30m resolution
            mask_30m = rio_rasterize(
                [(geom_nlcd, 1)],
                out_shape=(int(window.height), int(window.width)),
                transform=Affine(nlcd.transform.a, 0, x0, 0, nlcd.transform.e, y0),
                fill=0,
                all_touched=False,
                dtype='uint8'
            ).astype(bool)
            
            # Ensure we have exactly 128x128 at target resolution
            h, w = mask_30m.shape
            
            # Center crop if needed
            if h > target_size_30m:
                start_h = (h - target_size_30m) // 2
                mask_30m = mask_30m[start_h:start_h + target_size_30m, :]
            if w > target_size_30m:
                start_w = (w - target_size_30m) // 2
                mask_30m = mask_30m[:, start_w:start_w + target_size_30m]
            
            # Pad if needed
            if mask_30m.shape != (target_size_30m, target_size_30m):
                pad_h = target_size_30m - mask_30m.shape[0]
                pad_w = target_size_30m - mask_30m.shape[1]
                mask_30m = np.pad(mask_30m, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            # Calculate mask ratio at this resolution
            mask_coarse = coarsen_mask_by_ratio(mask_30m, ratio)
            mask_ratio = mask_coarse.sum() / mask_coarse.size
            
            # Check if this is our coarsest acceptable resolution
            if mask_ratio < max_mask_ratio_coarse:
                if coarsest_res is None:
                    coarsest_res = res_m
                    hierarchy.append((res_m, mask_ratio))
                    logging.info(f"{base_name}: Coarsest resolution {res_m}m (mask: {mask_ratio:.2%})")
                    break
        
        # If no coarse resolution found, use the coarsest available
        if not hierarchy:
            hierarchy.append((resolutions[0], mask_ratio))
            coarsest_res = resolutions[0]
            logging.warning(f"{base_name}: Using maximum resolution {resolutions[0]}m (mask: {mask_ratio:.2%})")
        
        # Add finer resolutions up to the threshold (from coarse to fine)
        finer_resolutions = [r for r in resolutions if r < coarsest_res and r >= finest_resolution]
        for res_m in finer_resolutions:
            if res_m >= coarsest_res or res_m < finest_resolution:
                continue
            
            ratio = int(round(res_m / 30.0))
            target_size_30m = 128 * ratio
            
            # Same calculation as above (could be refactored)
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            
            expanded_bounds = (
                center_x - target_size_30m * nlcd.transform.a / 2,
                center_y - target_size_30m * abs(nlcd.transform.e) / 2,
                center_x + target_size_30m * nlcd.transform.a / 2,
                center_y + target_size_30m * abs(nlcd.transform.e) / 2
            )
            
            window = rasterio.windows.from_bounds(*expanded_bounds, transform=nlcd.transform)
            window = window.round_offsets().round_lengths()
            
            x0 = nlcd.transform.c + window.col_off * nlcd.transform.a
            y0 = nlcd.transform.f + window.row_off * nlcd.transform.e
            
            mask_30m = rio_rasterize(
                [(geom_nlcd, 1)],
                out_shape=(int(window.height), int(window.width)),
                transform=Affine(nlcd.transform.a, 0, x0, 0, nlcd.transform.e, y0),
                fill=0,
                all_touched=False,
                dtype='uint8'
            ).astype(bool)
            
            h, w = mask_30m.shape
            if h > target_size_30m:
                start_h = (h - target_size_30m) // 2
                mask_30m = mask_30m[start_h:start_h + target_size_30m, :]
            if w > target_size_30m:
                start_w = (w - target_size_30m) // 2
                mask_30m = mask_30m[:, start_w:start_w + target_size_30m]
            
            if mask_30m.shape != (target_size_30m, target_size_30m):
                pad_h = target_size_30m - mask_30m.shape[0]
                pad_w = target_size_30m - mask_30m.shape[1]
                mask_30m = np.pad(mask_30m, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            mask_coarse = coarsen_mask_by_ratio(mask_30m, ratio)
            mask_ratio = mask_coarse.sum() / mask_coarse.size
            
            if mask_ratio < max_mask_ratio_fine:
                hierarchy.append((res_m, mask_ratio))
                logging.info(f"{base_name}: Added finer resolution {res_m}m (mask: {mask_ratio:.2%})")
            else:
                logging.info(f"{base_name}: Skipping {res_m}m (mask: {mask_ratio:.2%} > {max_mask_ratio_fine:.0%})")
                break
    
    return hierarchy


def compute_allowed_tokens_from_coarse(coarse_tokens: np.ndarray, decode_table: np.ndarray, D: int = 2) -> dict:
    """
    For each coarse token, compute which fine-resolution tokens are allowed.
    A fine token is allowed if its majority vote would produce the coarse token's majority class(es).
    Optimized by precomputing majority lookups.
    """
    unique_coarse = np.unique(coarse_tokens)
    logging.info(f"Computing allowed tokens for {len(unique_coarse)} unique coarse tokens...")

    # Pre-compute majority classes for every token once
    num_tokens = len(decode_table)
    majority_values_per_token: list[set[int]] = [set() for _ in range(num_tokens)]
    tokens_by_majority: dict[int, set[int]] = {}

    for token_idx in range(num_tokens):
        token_flat = decode_table[token_idx].flatten()
        values, counts = np.unique(token_flat, return_counts=True)
        max_count = counts.max()
        majority_values = values[counts == max_count]

        # Cache as a set for quick intersection checks later
        majority_values_set = set(int(v) for v in majority_values)
        majority_values_per_token[token_idx] = majority_values_set

        for majority_val in majority_values_set:
            tokens_by_majority.setdefault(majority_val, set()).add(token_idx)

    allowed_tokens = {}

    for i, coarse_idx in enumerate(unique_coarse):
        if i % 100 == 0:
            logging.info(f"  Processing coarse token {i}/{len(unique_coarse)}")

        coarse_majorities = majority_values_per_token[int(coarse_idx)]
        candidate_tokens: set[int] = set()

        for maj_val in coarse_majorities:
            candidate_tokens.update(tokens_by_majority.get(maj_val, set()))

        allowed_tokens[int(coarse_idx)] = np.array(sorted(candidate_tokens), dtype=np.int64)

    logging.info("Computed allowed tokens mapping")
    return allowed_tokens


def upsample_tokens(tokens: np.ndarray, scale: int) -> np.ndarray:
    """Upsample token grid using nearest neighbor."""
    H, W = tokens.shape
    upsampled = np.repeat(np.repeat(tokens, scale, axis=0), scale, axis=1)
    return upsampled


def generate_random_inpainting(known_tokens, known_positions, unknown_positions,
                              token_count: int,
                              disallowed_tokens=None, allowed_tokens_dict=None,
                              coarse_constraint_positions=None, coarse_constraint_values=None):
    """
    Generate random tokens for test mode, respecting constraints.
    """
    n_unknowns = len(unknown_positions)
    
    # Get valid token range derived from tokenizer size
    all_tokens = np.arange(token_count, dtype=np.int64)
    
    # Remove disallowed tokens
    if disallowed_tokens is not None:
        valid_tokens = np.setdiff1d(all_tokens, disallowed_tokens)
    else:
        valid_tokens = all_tokens
    
    # Generate random tokens
    random_tokens = np.random.choice(valid_tokens, size=n_unknowns)
    
    # Apply coarse constraints if provided
    if coarse_constraint_positions is not None and allowed_tokens_dict is not None:
        # Build a quick lookup for positions -> indices to avoid repeated searches
        position_lookup = {int(p): idx for idx, p in enumerate(coarse_constraint_positions)}
        for i, pos in enumerate(unknown_positions):
            idx = position_lookup.get(int(pos))
            if idx is None:
                continue
            coarse_val = int(coarse_constraint_values[idx])
            allowed = allowed_tokens_dict.get(coarse_val)
            if allowed is not None and len(allowed) > 0:
                random_tokens[i] = np.random.choice(allowed)
    
    # Create full token grid
    full_tokens = np.zeros(64 * 64, dtype=np.int32)
    full_tokens[known_positions] = known_tokens
    full_tokens[unknown_positions] = random_tokens
    
    return torch.tensor(full_tokens).unsqueeze(0)


def read_from_nlcd_by_geom(base_name: str, nlcd_img: Path, resolution_m: int, geojson_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    geojson_path = geojson_dir / f"base_{base_name}.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    if gdf is None or len(gdf) == 0:
        raise FileNotFoundError(f"Could not load geometry for base '{base_name}' from {geojson_path}")
    
    # geom_wgs84 = gdf.geometry.union_all()  # For future use if needed

    with rasterio.open(nlcd_img) as nlcd:
        # Transform geom to NLCD CRS
        gdf_nlcd = gdf.to_crs(nlcd.crs)
        geom_nlcd = gdf_nlcd.geometry.union_all()
        bounds = geom_nlcd.bounds  # minx, miny, maxx, maxy
        
        # Read a much larger area (10x the bounds) to ensure we get enough data
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Expand bounds by 10x
        expanded_bounds = (
            center_x - width * 5,
            center_y - height * 5,
            center_x + width * 5,
            center_y + height * 5
        )
        
        ratio = int(round(resolution_m / 30.0))
        
        # Read the expanded window
        window_expanded = rasterio.windows.from_bounds(*expanded_bounds, transform=nlcd.transform)
        window_expanded = window_expanded.round_offsets().round_lengths()
        raw_30m_large = nlcd.read(1, window=window_expanded, boundless=True, fill_value=0)
        
        # Now extract a 128x128 pixel region (at coarse resolution) centered on the mask
        # 128 pixels at coarse resolution = 128 * ratio pixels at 30m
        target_size_30m = 128 * ratio
        
        # Find center of the mask within the large window
        window_orig = rasterio.windows.from_bounds(*bounds, transform=nlcd.transform)
        rel_col = int(window_orig.col_off - window_expanded.col_off)
        rel_row = int(window_orig.row_off - window_expanded.row_off)
        rel_width = int(window_orig.width)
        rel_height = int(window_orig.height)
        
        # Center of original mask in the large array
        mask_center_x = rel_col + rel_width // 2
        mask_center_y = rel_row + rel_height // 2
        
        # Extract 128*ratio x 128*ratio window centered on mask
        x_start = max(0, mask_center_x - target_size_30m // 2)
        y_start = max(0, mask_center_y - target_size_30m // 2)
        x_end = min(raw_30m_large.shape[1], x_start + target_size_30m)
        y_end = min(raw_30m_large.shape[0], y_start + target_size_30m)
        
        # Ensure we get exactly the target size
        if x_end - x_start < target_size_30m:
            x_start = max(0, x_end - target_size_30m)
        if y_end - y_start < target_size_30m:
            y_start = max(0, y_end - target_size_30m)
            
        raw_30m = raw_30m_large[y_start:y_end, x_start:x_end]
        
        # Pad if necessary to get exact size
        if raw_30m.shape != (target_size_30m, target_size_30m):
            pad_h = target_size_30m - raw_30m.shape[0]
            pad_w = target_size_30m - raw_30m.shape[1]
            raw_30m = np.pad(raw_30m, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        logging.info(f"{base_name}: Extracted {target_size_30m}x{target_size_30m} region at 30m centered on mask")

        # Build transform for the extracted window
        # The extracted window starts at (x_start, y_start) in the expanded window
        actual_col_off = window_expanded.col_off + x_start
        actual_row_off = window_expanded.row_off + y_start
        x0 = nlcd.transform.c + actual_col_off * nlcd.transform.a
        y0 = nlcd.transform.f + actual_row_off * nlcd.transform.e

        # Rasterize fine mask at 30m resolution for the extracted window
        mask_30m = rio_rasterize(
            [(geom_nlcd, 1)],
            out_shape=raw_30m.shape,
            transform=Affine(nlcd.transform.a, 0, x0, 0, nlcd.transform.e, y0),
            fill=0,
            all_touched=False,
            dtype='uint8'
        ).astype(bool)

        # Downsample land cover by mode
        h, w = raw_30m.shape
        new_h, new_w = h // ratio, w // ratio
        raw_30m = raw_30m[:new_h * ratio, :new_w * ratio]
        reshaped = raw_30m.reshape(new_h, ratio, new_w, ratio).swapaxes(1, 2).reshape(new_h * new_w, ratio * ratio)
        from scipy.stats import mode as sp_mode
        block_modes, _ = sp_mode(reshaped, axis=1, keepdims=False)
        coarse = block_modes.reshape(new_h, new_w).astype(raw_30m.dtype)

        # Downsample mask by any
        mask_coarse = coarsen_mask_by_ratio(mask_30m, ratio)

        # Build profile for coarse grid
        coarse_transform = Affine(nlcd.transform.a * ratio, 0, x0, 0, nlcd.transform.e * ratio, y0)
        profile = {
            'driver': 'GTiff',
            'height': coarse.shape[0],
            'width': coarse.shape[1],
            'count': 1,
            'dtype': str(coarse.dtype),
            'crs': nlcd.crs,
            'transform': coarse_transform,
            'compress': 'deflate',
        }

        return coarse, mask_coarse, profile


def load_model(cfg: Config) -> torch.nn.Module:
    conf = OmegaConf.load(cfg.config_yaml)
    model = instantiate_from_config(conf.ar_model)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prefer safetensors if present
    st_path = Path(cfg.ckpt_dir) / "model.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file
        logging.info(f"Loading model weights from {st_path}")
        state_dict = load_file(str(st_path))
        model.load_state_dict(state_dict, strict=True)
        logging.info("Weights loaded (safetensors)")
    else:
        # Fallback: try Accelerate checkpoint directory
        try:
            from accelerate import Accelerator
            mixed_precision = "no" if device.type == "cpu" else "bf16"
            accelerator = Accelerator(mixed_precision=mixed_precision)
            model = accelerator.prepare(model)
            logging.info(f"Loading accelerate state from {cfg.ckpt_dir}")
            accelerator.load_state(cfg.ckpt_dir)
            model = accelerator.unwrap_model(model)
            model = model.to(device)
            logging.info("Weights loaded (accelerate)")
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint from {cfg.ckpt_dir}: {e}")

    return model


def write_geotiff(path: Path, array: np.ndarray, ref_profile: dict):
    profile = ref_profile.copy()
    profile.update({
        'count': 1,
        'dtype': array.dtype,
        'compress': 'deflate'
    })
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)


def class_proportions(arr: np.ndarray) -> dict:
    vals, counts = np.unique(arr, return_counts=True)
    total = counts.sum()
    return {int(v): float(c) / float(total) for v, c in zip(vals, counts)}


def multiresolution_inpaint(model, hierarchy, base_name, nlcd_img_path, geojson_dir,
                           decode_table, disallowed_tokens, cfg, device, sample_idx=0):
    """
    Perform multiresolution inpainting from coarse to fine.
    Returns the final inpainted result at the finest resolution.
    """
    previous_result = None
    previous_res = None
    
    for level_idx, (res_m, mask_ratio) in enumerate(hierarchy):
        logging.info(f"{base_name} Sample {sample_idx+1}: Level {level_idx+1}/{len(hierarchy)} - Resolution {res_m}m (mask: {mask_ratio:.2%})")
        
        # Load data at current resolution
        raw, mask_raw, profile = read_from_nlcd_by_geom(base_name, Path(nlcd_img_path), res_m, Path(geojson_dir))
        
        # Tokenize
        tokens_grid = tokenize_image(raw, decode_table)
        mask_tokens = coarsen_mask(mask_raw, decode_table.shape[1])
        
        Ht, Wt = tokens_grid.shape
        wt = cfg.window_tokens
        
        # If we have a coarser result, use it as constraint
        allowed_tokens_dict = None
        upsampled_prev = None
        if previous_result is not None and level_idx > 0:
            prev_h, prev_w = previous_result.shape
            curr_h, curr_w = tokens_grid.shape

            if curr_h % prev_h == 0 and curr_w % prev_w == 0:
                scale_y = curr_h // prev_h
                scale_x = curr_w // prev_w

                if scale_y != scale_x:
                    logging.warning(
                        f"{base_name}: Non-uniform scaling between resolutions; skipping constraints"
                    )
                else:
                    scale = max(scale_y, scale_x)
                    upsampled_prev = (
                        previous_result if scale == 1 else upsample_tokens(previous_result, scale)
                    )

                    # Only compute allowed tokens for tokens that appear in masked regions
                    masked_coarse_tokens = upsampled_prev[mask_tokens]
                    unique_masked = np.unique(masked_coarse_tokens)
                    logging.info(
                        f"{base_name}: Computing constraints for {len(unique_masked)} unique masked tokens"
                    )

                    if len(unique_masked) > 0:
                        allowed_tokens_dict = compute_allowed_tokens_from_coarse(unique_masked, decode_table)
                        logging.info(
                            f"{base_name}: Using coarse constraints from {previous_res}m -> {res_m}m"
                        )
            else:
                logging.warning(
                    f"{base_name}: Token grids not integer-scaled between {previous_res}m and {res_m}m; skipping constraints"
                )
        
        # Find windows to process
        ys, xs = np.where(mask_tokens)
        if len(ys) == 0:
            logging.warning(f"{base_name}: No masked tokens at {res_m}m")
            previous_result = tokens_grid
            previous_res = res_m
            continue
        
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        
        y0 = max(0, ymin - wt // 2)
        y1 = min(Ht - wt, ymax + wt // 2)
        x0 = max(0, xmin - wt // 2)
        x1 = min(Wt - wt, xmax + wt // 2)
        
        # Process windows
        result_tokens = tokens_grid.copy()
        
        # Use overlapping windows for finer resolutions
        stride = cfg.stride_tokens if level_idx == 0 else cfg.stride_tokens // 2
        
        windows_processed = 0
        for y in range(y0, y1 + 1, stride):
            for x in range(x0, x1 + 1, stride):
                if y + wt > Ht or x + wt > Wt:
                    continue
                
                window_tokens = result_tokens[y:y + wt, x:x + wt]
                window_mask = mask_tokens[y:y + wt, x:x + wt]
                
                # Check if window has enough known pixels
                known_ratio = 1.0 - (window_mask.sum() / window_mask.size)
                if known_ratio < cfg.min_filled_ratio and level_idx > 0:
                    continue  # Skip windows with too few known pixels (except at coarsest level)
                
                if not np.any(window_mask):
                    continue
                
                # Build positions
                flat_mask = window_mask.flatten()
                known_positions = np.where(~flat_mask)[0].astype(np.int64)
                unknown_positions = np.where(flat_mask)[0].astype(np.int64)
                known_tokens_vals = window_tokens.flatten()[known_positions].astype(np.int64)
                
                # Prepare constraints for fine resolution
                coarse_constraint_positions = None
                coarse_constraint_values = None
                
                if allowed_tokens_dict is not None and upsampled_prev is not None:
                    coarse_window = upsampled_prev[y:y + wt, x:x + wt]
                    coarse_constraint_positions = unknown_positions
                    coarse_constraint_values = coarse_window.flatten()[unknown_positions]
                
                windows_processed += 1
                
                if cfg.test_mode:
                    # Generate random tokens
                    gen_indices = generate_random_inpainting(
                        known_tokens_vals,
                        known_positions,
                        unknown_positions,
                        decode_table.shape[0],
                        disallowed_tokens,
                        allowed_tokens_dict,
                        coarse_constraint_positions,
                        coarse_constraint_values
                    )
                else:
                    # Use model inference
                    cond = torch.tensor([0], dtype=torch.long, device=device)
                    kt = torch.from_numpy(known_tokens_vals).to(device)
                    kp = torch.from_numpy(known_positions).to(device)
                    up = torch.from_numpy(unknown_positions).to(device)
                    
                    with torch.no_grad():
                        allowed_per_position = None
                        if allowed_tokens_dict is not None and coarse_constraint_values is not None:
                            allowed_per_position = []
                            for coarse_val in coarse_constraint_values:
                                allowed = allowed_tokens_dict.get(int(coarse_val))
                                if allowed is not None and len(allowed) > 0:
                                    filtered = np.setdiff1d(allowed, disallowed_tokens, assume_unique=False)
                                    if len(filtered) > 0:
                                        allowed_per_position.append(torch.as_tensor(filtered, dtype=torch.long, device=device))
                                        continue
                                allowed_per_position.append(None)

                        gen_indices = generate_inpainting(
                            model=model,
                            cond=cond,
                            known_tokens=kt,
                            known_positions=kp,
                            unknown_positions=up,
                            cfg_scales=cfg.cfg_scales,
                            temperature=cfg.temperature,
                            top_k=cfg.top_k,
                            top_p=cfg.top_p,
                            disallowed_classes=torch.from_numpy(disallowed_tokens).to(device),
                            allowed_token_ids_per_position=allowed_per_position,
                        )
                
                gen_flat = gen_indices[0].detach().cpu().numpy()
                gen_window = gen_flat.reshape(wt, wt)
                
                # Blend overlapping regions if using overlap
                if stride < wt:
                    # Simple averaging for overlapping regions
                    for dy in range(wt):
                        for dx in range(wt):
                            if window_mask[dy, dx]:
                                result_tokens[y + dy, x + dx] = gen_window[dy, dx]
                else:
                    result_tokens[y:y + wt, x:x + wt] = gen_window
        
        logging.info(f"{base_name}: Processed {windows_processed} windows at {res_m}m")

        if windows_processed == 0 and upsampled_prev is not None:
            logging.warning(
                f"{base_name}: No windows processed at {res_m}m; carrying forward coarse prediction"
            )
            result_tokens = upsampled_prev.copy()

        # Store result for next level
        previous_result = result_tokens
        previous_res = res_m
    
    # Return the final result and profile from finest resolution
    return previous_result, raw, mask_raw, profile


def main():
    setup_logger()
    cfg = Config()
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.info(f"Using device: {cfg.device} and checkpoint {cfg.ckpt_dir}")

    # Check for all base geometries before loading model
    logging.info("Checking availability of all base geometries...")
    missing_bases = []
    
    for base in cfg.bases:
        geojson_path = Path(cfg.geojson_dir) / f"base_{base}.geojson"
        
        if not geojson_path.exists():
            missing_bases.append(base)
            logging.error(f"Base '{base}' GeoJSON not found at {geojson_path}")
        else:
            logging.info(f"Base '{base}' GeoJSON file exists")
    
    if missing_bases:
        logging.error(f"Missing base geometries: {missing_bases}")
        logging.error("Cannot proceed without all base geometries. Please check your data.")
        sys.exit(1)
    
    logging.info(f"Feature geometries for all {len(cfg.bases)} features are available.")
    
    # Calculate resolution hierarchy for each base
    logging.info("\n" + "="*60)
    if cfg.multiresolution:
        logging.info("Calculating multiresolution hierarchies for all regions...")
    else:
        logging.info("Calculating optimal resolutions for all regions...")
    logging.info("="*60)
    
    base_hierarchies = {}
    base_resolutions = {}
    base_mask_ratios = {}
    
    for base in cfg.bases:
        logging.info(f"\nAnalyzing feature '{base}'...")
        
        if cfg.multiresolution:
            hierarchy = select_resolution_hierarchy(
                base, Path(cfg.nlcd_img_path), Path(cfg.geojson_dir),
                cfg.max_mask_ratio_coarse, cfg.max_mask_ratio_fine, cfg.finest_resolution
            )
            base_hierarchies[base] = hierarchy
            # Store the coarsest resolution for compatibility
            base_resolutions[base] = hierarchy[0][0]
            base_mask_ratios[base] = hierarchy[0][1]
            logging.info(f"Base '{base}': hierarchy = {[f'{r}m' for r, _ in hierarchy]}")
        else:
            optimal_res, mask_ratio = calculate_optimal_resolution(base, Path(cfg.nlcd_img_path), Path(cfg.geojson_dir))
            base_resolutions[base] = optimal_res
            base_mask_ratios[base] = mask_ratio
            base_hierarchies[base] = [(optimal_res, mask_ratio)]
            logging.info(f"Base '{base}': optimal resolution = {optimal_res}m")
    
    logging.info("\n" + "="*60)
    logging.info("Resolution selection complete. Summary:")
    for base in cfg.bases:
        if cfg.multiresolution:
            hierarchy = base_hierarchies[base]
            res_str = " -> ".join([f"{r}m ({m:.1%})" for r, m in hierarchy])
            logging.info(f"  {base}: {res_str}")
        else:
            res = base_resolutions[base]
            mask_pct = base_mask_ratios[base] * 100
            logging.info(f"  {base}: {res}m (mask: {mask_pct:.1f}%)")
    logging.info("="*60 + "\n")

    # Load decode table and model
    decode_table = load_decode_table(Path(cfg.data_npz))
    D = decode_table.shape[1]
    assert D == 2, f"Expected tokenizer downsample ratio D=2, found D={D}"

    if cfg.test_mode:
        logging.info("TEST MODE: Using random tokens instead of model inference")
        model = None
    else:
        model = load_model(cfg)
        model.eval()

    disallowed_tokens = compute_disallowed_token_indices(decode_table, cfg.forbidden_nlcd)

    # For plotting after processing all bases
    plot_rows = []

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    for base in cfg.bases:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Add test suffix if in test mode
        suffix = "_test" if cfg.test_mode else ""
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing base '{base}'")
        logging.info(f"{'='*60}")
        
        hierarchy = base_hierarchies[base]
        
        # Run multiple samples
        samples_list = []
        for sample_idx in range(cfg.samples_per_base):
            torch.manual_seed(cfg.seed + sample_idx)
            logging.info(f"\n{base}: Starting sample {sample_idx+1}/{cfg.samples_per_base}")
            
            if cfg.multiresolution:
                # Use multiresolution inpainting
                final_tokens, raw, mask_raw, profile = multiresolution_inpaint(
                    model, hierarchy, base, cfg.nlcd_img_path, cfg.geojson_dir,
                    decode_table, disallowed_tokens, cfg, device, sample_idx
                )
            else:
                # Use single resolution (old approach)
                res_m = base_resolutions[base]
                raw, mask_raw, profile = read_from_nlcd_by_geom(base, Path(cfg.nlcd_img_path), res_m, Path(cfg.geojson_dir))
                tokens_grid = tokenize_image(raw, decode_table)
                mask_tokens = coarsen_mask(mask_raw, D)
                
                # Simple single-resolution inpainting (simplified for brevity)
                final_tokens = tokens_grid.copy()
                # ... (would include the old sliding window logic here if needed)
            
            # Detokenize final result
            detok = detokenize(final_tokens, decode_table).astype(raw.dtype)
            
            # Ensure shape matches original
            Hc, Wc = detok.shape
            Hr, Wr = raw.shape
            if (Hc, Wc) != (Hr, Wr):
                detok_full = np.zeros_like(raw)
                detok_full[:Hc, :Wc] = detok
                detok = detok_full
            
            # Determine resolution for filename
            final_res = hierarchy[-1][0] if cfg.multiresolution else base_resolutions[base]
            
            # Save sample
            sample_path = out_dir / f"{base}_inpainted_{final_res}m_sample{sample_idx+1}{suffix}.tif"
            write_geotiff(sample_path, detok, profile)
            logging.info(f"{base}: wrote inpainted sample {sample_idx+1} to {sample_path}")
            
            samples_list.append(detok)
            
            # Collect for plotting
            plot_rows.append((f"{base} (s{sample_idx+1})", final_res, raw, mask_raw, detok))

        # Determine final resolution
        final_res = hierarchy[-1][0] if cfg.multiresolution else base_resolutions[base]
        
        # Save all samples for this base as a pickle
        with open(out_dir / f"{base}_samples_{final_res}m{suffix}.pkl", 'wb') as f:
            pickle.dump({
                'base': base,
                'resolution_m': final_res,
                'raw': raw,
                'mask': mask_raw,
                'samples': samples_list,
                'profile': profile,
            }, f)
        logging.info(f"{base}: saved {len(samples_list)} samples to pickle")
        
        # Aggregate samples to find most common pixel value at each location
        if len(samples_list) > 0:
            # Stack all samples into a 3D array (samples, height, width)
            samples_stack = np.stack(samples_list, axis=0)
            
            # For each pixel location, find the most common value across samples
            # Using scipy.stats.mode for efficiency
            from scipy.stats import mode as sp_mode
            consensus, _ = sp_mode(samples_stack, axis=0, keepdims=False)
            consensus = consensus.astype(raw.dtype)
            
            # Save consensus raster to rasters_dir
            rasters_dir = Path(cfg.rasters_dir)
            rasters_dir.mkdir(parents=True, exist_ok=True)
            consensus_path = rasters_dir / f"{base}_consensus_{final_res}m{suffix}.tif"
            write_geotiff(consensus_path, consensus, profile)
            logging.info(f"{base}: wrote consensus raster (mode of {len(samples_list)} samples) to {consensus_path}")
             
            # Log class proportions of consensus
            logging.info(f"{base}: consensus class proportions: {class_proportions(consensus)}")
            
            # Upsample consensus raster to 100m resolution using gdalwarp
            upsampled_path = rasters_dir / f"{base}_consensus_100m{suffix}.tif"
            gdal_cmd = f"gdalwarp -tr 100 100 -r near -co COMPRESS=DEFLATE {consensus_path} {upsampled_path}"
            
            import subprocess
            result = subprocess.run(gdal_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"{base}: upsampled consensus raster from {final_res}m to 100m resolution -> {upsampled_path}")
            else:
                logging.error(f"{base}: Failed to upsample raster. Error: {result.stderr}")

    # Create overview plot
    if plot_rows:
        # Group rows by base: we have original/mask + 3 samples = 5 columns
        # Build mapping base -> list of (label,res,raw,mask,pred) where pred varies
        grouped = {}
        for label, res_m, raw_i, mask_i, pred_i in plot_rows:
            base = label.split(' (s')[0]
            grouped.setdefault(base, {'res': res_m, 'raw': raw_i, 'mask': mask_i, 'preds': []})
            grouped[base]['preds'].append(pred_i)

        bases = list(grouped.keys())
        n = len(bases)
        _, axes = plt.subplots(n, 5, figsize=(15, 3 * n), squeeze=False)
        # Build LUT array for vectorized mapping
        lut = NLCDTokenizer.lut
        keys = np.array(sorted(lut.keys()), dtype=np.int64)
        rgb_vals = np.array([lut[k] for k in keys], dtype=np.float32)

        def to_rgb(a: np.ndarray) -> np.ndarray:
            out = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.float32)
            for k, rgb in zip(keys, rgb_vals):
                m = (a == k)
                if m.any():
                    out[m] = rgb
            return out

        for i, base in enumerate(bases):
            res_m = grouped[base]['res']
            raw_i = grouped[base]['raw']
            mask_i = grouped[base]['mask']
            preds = grouped[base]['preds']
            axes[i, 0].imshow(to_rgb(raw_i), interpolation='nearest')
            axes[i, 0].set_title(f"{base} original ({res_m}m)")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(mask_i.astype(np.uint8), cmap='gray', interpolation='nearest')
            axes[i, 1].set_title("mask")
            axes[i, 1].axis('off')
            for j in range(3):
                img = preds[j] if j < len(preds) else preds[-1]
                axes[i, 2 + j].imshow(to_rgb(img), interpolation='nearest')
                axes[i, 2 + j].set_title(f"sample {j+1}")
                axes[i, 2 + j].axis('off')
        plt.tight_layout()
        suffix = "_test" if cfg.test_mode else ""
        fig_path = Path(cfg.output_dir) / f"overview_mixed_res{suffix}.png"
        plt.savefig(fig_path, dpi=200)
        logging.info(f"Saved overview plot to {fig_path}")


if __name__ == "__main__":
    main()
