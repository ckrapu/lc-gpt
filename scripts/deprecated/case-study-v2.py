'''
This script uses the LC-GPT model to inpaint missing regions in NLCD land cover data.
The missing regions are defined by geojson files and are inpainted at various resolutions.
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
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
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
    output_dir: str = "../results/case_study_v2"
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
    samples_per_base: int = 20
    forbidden_nlcd: tuple[int, ...] = (11, 12, 90, 95)  # water/wetlands


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
    logging.info(f"Disallowing {len(idx)} tokens that include NLCD classes {forbidden_nlcd}")
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


def read_from_nlcd_by_geom(base_name: str, nlcd_img: Path, resolution_m: int, geojson_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    geojson_path = geojson_dir / f"base_{base_name}.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    if gdf is None or len(gdf) == 0:
        raise FileNotFoundError(f"Could not load geometry for base '{base_name}' from {geojson_path}")

    with rasterio.open(nlcd_img) as nlcd:
        gdf_nlcd = gdf.to_crs(nlcd.crs)
        geom_nlcd = gdf_nlcd.geometry.union_all()

        base_transform = nlcd.transform
        base_res = base_transform.a
        ratio = int(round(resolution_m / base_res))
        if ratio <= 0:
            raise ValueError(f"Invalid resolution {resolution_m} for base raster resolution {base_res}")

        bounds = geom_nlcd.bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        center_col = (center_x - base_transform.c) / base_transform.a
        center_row = (center_y - base_transform.f) / base_transform.e

        coarse_col = int(round(center_col / ratio))
        coarse_row = int(round(center_row / ratio))

        start_col = coarse_col - 64
        start_row = coarse_row - 64

        x0 = base_transform.c + start_col * resolution_m
        y0 = base_transform.f + start_row * resolution_m
        target_transform = Affine(resolution_m, 0.0, x0, 0.0, base_transform.e * ratio, y0)

        with WarpedVRT(
            nlcd,
            crs=nlcd.crs,
            transform=target_transform,
            width=128,
            height=128,
            resampling=Resampling.mode,
            src_nodata=0,
            dst_nodata=0,
        ) as vrt:
            coarse = vrt.read(1)

        coarse = coarse.astype(nlcd.dtypes[0], copy=False)

        mask_coarse = rio_rasterize(
            [(geom_nlcd, 1)],
            out_shape=(128, 128),
            transform=target_transform,
            fill=0,
            all_touched=False,
            dtype='uint8'
        ).astype(bool)

        profile = {
            'driver': 'GTiff',
            'height': coarse.shape[0],
            'width': coarse.shape[1],
            'count': 1,
            'dtype': str(coarse.dtype),
            'crs': nlcd.crs,
            'transform': target_transform,
            'compress': 'deflate',
            'nodata': 0,
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


def main():
    setup_logger()
    cfg = Config()
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.info(f"Using device: {cfg.device}")
    logging.info(f"Checkpoint: {cfg.ckpt_dir}")

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
    
    logging.info("All base geometries verified successfully")
    
    # Calculate optimal resolution for each base
    logging.info("\n" + "="*60)
    logging.info("Calculating optimal resolutions for all regions...")
    logging.info("="*60)
    base_resolutions = {}
    base_mask_ratios = {}
    for base in cfg.bases:
        logging.info(f"\nAnalyzing base '{base}'...")
        optimal_res, mask_ratio = calculate_optimal_resolution(base, Path(cfg.nlcd_img_path), Path(cfg.geojson_dir))
        base_resolutions[base] = optimal_res
        base_mask_ratios[base] = mask_ratio
        logging.info(f"Base '{base}': optimal resolution = {optimal_res}m")
    
    logging.info("\n" + "="*60)
    logging.info("Resolution selection complete. Summary:")
    for base in cfg.bases:
        res = base_resolutions[base]
        mask_pct = base_mask_ratios[base] * 100
        logging.info(f"  {base}: {res}m (mask: {mask_pct:.1f}%)")
    logging.info("="*60 + "\n")

    # Load decode table and model
    decode_table = load_decode_table(Path(cfg.data_npz))
    D = decode_table.shape[1]
    assert D == 2, f"Expected tokenizer downsample ratio D=2, found D={D}"

    model = load_model(cfg)
    model.eval()


    disallowed_tokens = compute_disallowed_token_indices(decode_table, cfg.forbidden_nlcd)

    # For plotting after processing all bases
    plot_rows = []

    for base in cfg.bases:
        res_m = base_resolutions[base]
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{base}_inpainted_{res_m}m.tif"

        # Load from GeoJSON geometry
        logging.info(f"Loading base '{base}' from GeoJSON")
        raw, mask_raw, profile = read_from_nlcd_by_geom(base, Path(cfg.nlcd_img_path), res_m, Path(cfg.geojson_dir))

        logging.info(f"{base}: raw shape {raw.shape}, mask shape {mask_raw.shape}")
        logging.info(f"{base}: raw class proportions: {class_proportions(raw)}")
        logging.info(f"{base}: masked pixels: {mask_raw.sum()} / {mask_raw.size} ({100*mask_raw.sum()/mask_raw.size:.1f}%)")

        # Tokenize and coarsen mask
        tokens_grid = tokenize_image(raw, decode_table)  # (Ht, Wt)
        mask_tokens = coarsen_mask(mask_raw, D)
        
        logging.info(f"{base}: masked tokens: {mask_tokens.sum()} / {mask_tokens.size} ({100*mask_tokens.sum()/mask_tokens.size:.1f}%)")

        Ht, Wt = tokens_grid.shape
        logging.info(f"{base}: token grid {tokens_grid.shape} with block window {cfg.window_tokens}x{cfg.window_tokens}")

        # Sliding-window inpainting over token grid
        wt = cfg.window_tokens
        st = cfg.stride_tokens
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Find bounding box of masked tokens to limit windows
        ys, xs = np.where(mask_tokens)
        if len(ys) == 0:
            logging.warning(f"{base}: NO MASKED TOKENS found at {res_m}m resolution!")
            logging.info(f"{base}: This means the mask doesn't overlap with valid tokens")
            logging.info(f"{base}: Skipping inpainting - writing original tokenized output")
            detok = detokenize(tokens_grid, decode_table).astype(raw.dtype)
            write_geotiff(out_path, detok, profile)
            logging.info(f"{base}: wrote unchanged output to {out_path}")
            continue
        
        logging.info(f"{base}: Found {len(ys)} masked token positions to inpaint")
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        y0 = max(0, ymin - wt // 2)
        y1 = min(Ht - wt, ymax + wt // 2)
        x0 = max(0, xmin - wt // 2)
        x1 = min(Wt - wt, xmax + wt // 2)
        
        # Count windows that will be processed
        n_windows = 0
        for y in range(y0, y1 + 1, st):
            for x in range(x0, x1 + 1, st):
                if y + wt <= Ht and x + wt <= Wt:
                    window_mask = mask_tokens[y:y + wt, x:x + wt]
                    if np.any(window_mask):
                        n_windows += 1
        
        logging.info(f"{base}: Will process {n_windows} windows with masked tokens")
        if n_windows == 0:
            logging.warning(f"{base}: No windows contain masked tokens, check if mask is properly aligned")
            logging.info(f"{base}: Skipping inpainting - writing original tokenized output")
            detok = detokenize(tokens_grid, decode_table).astype(raw.dtype)
            write_geotiff(out_path, detok, profile)
            continue

        # run multiple samples by repeating the sweep with different seeds
        for sample_idx in range(cfg.samples_per_base):
            torch.manual_seed(cfg.seed + sample_idx)
            tg = tokens_grid.copy()
            windows_processed = 0
            for y in range(y0, y1 + 1, st):
                for x in range(x0, x1 + 1, st):
                    wy, wx = y, x
                    if wy + wt > Ht or wx + wt > Wt:
                        continue

                    window_tokens = tg[wy:wy + wt, wx:wx + wt]
                    window_mask = mask_tokens[wy:wy + wt, wx:wx + wt]

                    if not np.any(window_mask):
                        continue

                    # Build known/unknown positions
                    flat_mask = window_mask.flatten()
                    known_positions = np.where(~flat_mask)[0].astype(np.int64)
                    unknown_positions = np.where(flat_mask)[0].astype(np.int64)
                    known_tokens = window_tokens.flatten()[known_positions].astype(np.int64)

                    cond = torch.tensor([0], dtype=torch.long, device=device)  # no aux conditioning
                    kt = torch.from_numpy(known_tokens).to(device)
                    kp = torch.from_numpy(known_positions).to(device)
                    up = torch.from_numpy(unknown_positions).to(device)
                    windows_processed += 1
                    logging.info(f"Sample {sample_idx+1}, window {windows_processed}/{n_windows} @ ({wy},{wx}): {len(unknown_positions)} unknowns")
                    with torch.no_grad():
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
                            disallowed_classes=torch.from_numpy(disallowed_tokens).to(device)
                        )

                    gen_flat = gen_indices[0].detach().cpu().numpy()
                    gen_window = gen_flat.reshape(wt, wt)
                    tg[wy:wy + wt, wx:wx + wt] = gen_window

            # Detokenize sample
            detok = detokenize(tg, decode_table).astype(raw.dtype)
            Hc, Wc = detok.shape
            Hr, Wr = raw.shape
            if (Hc, Wc) != (Hr, Wr):
                detok_full = np.zeros_like(raw)
                detok_full[:Hc, :Wc] = detok
                detok = detok_full
            write_geotiff(out_dir / f"{base}_inpainted_{res_m}m_sample{sample_idx+1}.tif", detok, profile)
            logging.info(f"{base}: wrote inpainted sample {sample_idx+1} (processed {windows_processed} windows)")

            # Collect for plotting
            plot_rows.append((f"{base} (s{sample_idx+1})", res_m, raw, mask_raw, detok))

        # Save all samples for this base as a pickle
        samples = []
        for i in range(cfg.samples_per_base):
            p = out_dir / f"{base}_inpainted_{res_m}m_sample{i+1}.tif"
            with rasterio.open(p) as ds:
                samples.append(ds.read(1))
        with open(out_dir / f"{base}_samples_{res_m}m.pkl", 'wb') as f:
            pickle.dump({
                'base': base,
                'resolution_m': res_m,
                'raw': raw,
                'mask': mask_raw,
                'samples': samples,
                'profile': profile,
            }, f)
        logging.info(f"{base}: saved {len(samples)} samples to pickle")
        
        # Aggregate samples to find most common pixel value at each location
        if len(samples) > 0:
            # Stack all samples into a 3D array (samples, height, width)
            samples_stack = np.stack(samples, axis=0)
            
            # For each pixel location, find the most common value across samples
            # Using scipy.stats.mode for efficiency
            from scipy.stats import mode as sp_mode
            consensus, _ = sp_mode(samples_stack, axis=0, keepdims=False)
            consensus = consensus.astype(raw.dtype)
            
            # Save consensus raster to rasters_dir
            rasters_dir = Path(cfg.rasters_dir)
            rasters_dir.mkdir(parents=True, exist_ok=True)
            consensus_path = rasters_dir / f"{base}_consensus_{res_m}m.tif"
            write_geotiff(consensus_path, consensus, profile)
            logging.info(f"{base}: wrote consensus raster (mode of {len(samples)} samples) to {consensus_path}")
            
            # Log class proportions of consensus
            logging.info(f"{base}: consensus class proportions: {class_proportions(consensus)}")
            
            # Upsample consensus raster to 100m resolution using gdalwarp
            upsampled_path = rasters_dir / f"{base}_consensus_100m.tif"
            gdal_cmd = f"gdalwarp -tr 100 100 -r near -co COMPRESS=DEFLATE {consensus_path} {upsampled_path}"
            
            import subprocess
            result = subprocess.run(gdal_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"{base}: upsampled consensus raster from {res_m}m to 100m resolution -> {upsampled_path}")
            else:
                logging.error(f"{base}: Failed to upsample raster. Error: {result.stderr}")

        # Detokenize back to raw NLCD values
        detok = detokenize(tokens_grid, decode_table).astype(raw.dtype)

        # Ensure shape matches original
        Hc, Wc = detok.shape
        Hr, Wr = raw.shape
        if (Hc, Wc) != (Hr, Wr):
            detok_full = np.zeros_like(raw)
            detok_full[:Hc, :Wc] = detok
            detok = detok_full

        logging.info(f"{base}: inpainted class proportions: {class_proportions(detok)}")

        write_geotiff(out_path, detok, profile)
        logging.info(f"{base}: wrote inpainted raster to {out_path}")

        # Collect for plotting
        plot_rows.append((base, res_m, raw, mask_raw, detok))

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
        fig_path = Path(cfg.output_dir) / f"overview_mixed_res.png"
        plt.savefig(fig_path, dpi=200)
        logging.info(f"Saved overview plot to {fig_path}")


if __name__ == "__main__":
    main()
