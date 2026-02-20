from dataclasses import dataclass
from pathlib import Path
import json
import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


@dataclass
class Config:
    boundaries_zip: Path = Path("data/Base_boundary_shp.zip")
    boundaries_shp: Path = Path("data/Base_boundary_shp/US_military_boundary.shp")
    csv_path: Path = Path("data/rec-study-installation-names.csv")
    csv_name_col: str = "FULLNAME"
    csv_tiger_col: str = "tiger_lines_data"
    shp_name_col: str = "FULLNAME"
    use_tiger_only: bool = True
    nlcd_path: Path = Path("data/nlcd_2021_land_cover_l48_20230630.img")
    finest_resolution_m: int = 240
    min_tokens: int = 128
    buffer_tokens: int = 8
    max_tokens_per_side: int = 512
    out_dir: Path = Path("results/rec_study_boundary_audit")


def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    slug = normalize_name(value).replace(" ", "_")
    if not slug:
        slug = "unknown_base"
    if slug[0].isdigit():
        slug = f"base_{slug}"
    return slug


cfg = Config()
cfg.out_dir.mkdir(parents=True, exist_ok=True)

if not cfg.boundaries_shp.exists():
    if not cfg.boundaries_zip.exists():
        raise FileNotFoundError(f"Missing shapefile zip: {cfg.boundaries_zip}")
    with zipfile.ZipFile(cfg.boundaries_zip, "r") as zf:
        zf.extractall(cfg.boundaries_zip.parent)

csv_df = pd.read_csv(cfg.csv_path)
if cfg.use_tiger_only and cfg.csv_tiger_col in csv_df.columns:
    tiger_mask = (
        csv_df[cfg.csv_tiger_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    csv_df = csv_df[tiger_mask].copy()

requested_names = (
    csv_df[cfg.csv_name_col]
    .dropna()
    .astype(str)
    .map(str.strip)
    .tolist()
)
requested_names = list(dict.fromkeys(name for name in requested_names if name))
requested_norm = [normalize_name(name) for name in requested_names]

try:
    shp_df = gpd.read_file(cfg.boundaries_shp, engine="pyogrio")
except Exception:
    shp_df = gpd.read_file(cfg.boundaries_shp)

if cfg.shp_name_col not in shp_df.columns:
    raise ValueError(f"Column '{cfg.shp_name_col}' not in shapefile columns: {list(shp_df.columns)}")

shp_df = shp_df[[cfg.shp_name_col, "geometry"]].dropna(subset=["geometry"]).copy()
shp_df["name_norm"] = shp_df[cfg.shp_name_col].astype(str).map(normalize_name)
shp_df = shp_df.dissolve(by="name_norm", as_index=False, aggfunc="first")
shp_df = shp_df.to_crs("EPSG:4326")

shape_match = {row["name_norm"]: row for _, row in shp_df.iterrows()}
missing = []
rows = []
slug_counts: dict[str, int] = {}

for requested_name, name_norm in zip(requested_names, requested_norm):
    row = shape_match.get(name_norm)
    if row is None:
        missing.append(requested_name)
        continue
    raw_slug = slugify(str(row[cfg.shp_name_col]))
    slug_counts[raw_slug] = slug_counts.get(raw_slug, 0) + 1
    base_id = raw_slug if slug_counts[raw_slug] == 1 else f"{raw_slug}_{slug_counts[raw_slug]}"
    rows.append(
        {
            "requested_name": requested_name,
            "matched_name": str(row[cfg.shp_name_col]),
            "name_norm": name_norm,
            "base_id": base_id,
            "geometry": row["geometry"],
        }
    )

if rows:
    matched_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    with rasterio.open(cfg.nlcd_path) as nlcd:
        matched_nlcd = matched_gdf.to_crs(nlcd.crs)
    bounds = matched_nlcd.geometry.bounds
    widths = (bounds["maxx"] - bounds["minx"]).to_numpy()
    heights = (bounds["maxy"] - bounds["miny"]).to_numpy()
    width_tokens = np.ceil(widths / float(cfg.finest_resolution_m)).astype(int)
    height_tokens = np.ceil(heights / float(cfg.finest_resolution_m)).astype(int)
    target_tokens = np.maximum(cfg.min_tokens, np.maximum(width_tokens, height_tokens) + cfg.buffer_tokens)

    matched_gdf["width_tokens"] = width_tokens
    matched_gdf["height_tokens"] = height_tokens
    matched_gdf["target_tokens"] = target_tokens
    matched_gdf["too_large"] = matched_gdf["target_tokens"] > cfg.max_tokens_per_side
    matched_gdf["selected"] = ~matched_gdf["too_large"]
else:
    matched_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

audit_path = cfg.out_dir / "matched_bases_audit.csv"
matched_gdf.drop(columns=["geometry"], errors="ignore").to_csv(audit_path, index=False)

missing_path = cfg.out_dir / "missing_from_shapefile.csv"
pd.DataFrame({"requested_name": missing}).to_csv(missing_path, index=False)

selected_ids = matched_gdf.loc[matched_gdf["selected"], "base_id"].tolist() if len(matched_gdf) else []
selected_names = (
    matched_gdf.loc[matched_gdf["selected"], "matched_name"].tolist() if len(matched_gdf) else []
)
selected_path = cfg.out_dir / "selected_base_ids.json"
selected_path.write_text(
    json.dumps(
        {
            "base_ids": selected_ids,
            "matched_names": selected_names,
            "count": len(selected_ids),
            "notes": {
                "finest_resolution_m": cfg.finest_resolution_m,
                "max_tokens_per_side": cfg.max_tokens_per_side,
                "use_tiger_only": cfg.use_tiger_only,
            },
        },
        indent=2,
    )
)

summary = {
    "requested_count": len(requested_names),
    "matched_count": int(len(matched_gdf)),
    "missing_count": len(missing),
    "too_large_count": int(matched_gdf["too_large"].sum()) if len(matched_gdf) else 0,
    "selected_count": int(matched_gdf["selected"].sum()) if len(matched_gdf) else 0,
    "audit_csv": str(audit_path),
    "missing_csv": str(missing_path),
    "selected_json": str(selected_path),
}
summary_path = cfg.out_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
