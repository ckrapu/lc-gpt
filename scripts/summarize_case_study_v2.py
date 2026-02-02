from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import rasterio


@dataclass
class Config:
    case_study_dir: Path = Path("results/case_study_v2")
    output_xlsx: Path = Path("results/case_study_v2_lc_proportions.xlsx")
    nlcd_classes: tuple[int, ...] = (
        11, 12, 21, 22, 23, 24, 31, 41, 42, 43,
        51, 52, 71, 72, 73, 74, 81, 82, 90, 95,
    )
    base_pkl_glob: str = "*_samples_*m.pkl"


def proportions(arr: np.ndarray, mask: np.ndarray, classes: tuple[int, ...]) -> dict[int, float]:
    masked = arr[mask]
    if masked.size == 0:
        return {c: 0.0 for c in classes}
    vals, counts = np.unique(masked, return_counts=True)
    total = counts.sum()
    count_map = {int(v): int(c) for v, c in zip(vals, counts)}
    return {c: count_map.get(c, 0) / total for c in classes}


def parse_sample_num(path: Path) -> int | None:
    m = re.search(r"_sample(\d+)\.tif$", path.name)
    return int(m.group(1)) if m else None


def main() -> None:
    cfg = Config()
    rows: list[dict] = []

    pkl_paths = sorted(cfg.case_study_dir.glob(cfg.base_pkl_glob))
    if not pkl_paths:
        raise FileNotFoundError(f"No sample pkls found in {cfg.case_study_dir}")

    for pkl_path in pkl_paths:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        base = data.get("base") or pkl_path.name.split("_samples_")[0]
        raw = data["raw"]
        mask = data["mask"].astype(bool)
        if raw.shape != mask.shape:
            raise ValueError(f"{pkl_path}: raw shape {raw.shape} != mask shape {mask.shape}")

        ref_props = proportions(raw, mask, cfg.nlcd_classes)
        rows.append({
            "base": base,
            "simulation": "reference",
            **{f"nlcd_{c}": ref_props[c] for c in cfg.nlcd_classes},
        })

        sample_paths = sorted(
            cfg.case_study_dir.glob(f"{base}_inpainted_*_sample*.tif"),
            key=lambda p: (parse_sample_num(p) is None, parse_sample_num(p) or 0, p.name),
        )
        if not sample_paths:
            continue

        for sample_path in sample_paths:
            with rasterio.open(sample_path) as ds:
                arr = ds.read(1)
            if arr.shape != mask.shape:
                raise ValueError(f"{sample_path}: shape {arr.shape} != mask shape {mask.shape}")

            sim_props = proportions(arr, mask, cfg.nlcd_classes)
            sample_num = parse_sample_num(sample_path)
            sim_label = f"sample{sample_num}" if sample_num is not None else sample_path.stem
            rows.append({
                "base": base,
                "simulation": sim_label,
                **{f"nlcd_{c}": sim_props[c] for c in cfg.nlcd_classes},
            })

    df = pd.DataFrame(rows)

    def sim_key(val: str) -> tuple[int, int]:
        if val == "reference":
            return (0, 0)
        m = re.search(r"sample(\d+)", val)
        return (1, int(m.group(1))) if m else (2, 0)

    df["__sim_order"] = df["simulation"].map(sim_key)
    df = df.sort_values(["base", "__sim_order"]).drop(columns="__sim_order")
    df.to_excel(cfg.output_xlsx, index=False)

    print(f"Wrote {len(df)} rows to {cfg.output_xlsx}")


if __name__ == "__main__":
    main()
