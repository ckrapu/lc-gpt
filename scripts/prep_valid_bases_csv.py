from dataclasses import dataclass
from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


@dataclass
class Config:
    gpkg_path: Path = Path("data/base_footprints_common.gpkg")
    gpkg_layer: str = "base_footprints"
    nlcd_path: Path = Path("data/nlcd_2021_land_cover_l48_20230630.img")
    out_csv: Path = Path("data/base_footprints_valid.csv")
    out_summary: Path = Path("data/base_footprints_valid_summary.json")
    rec_only: bool = True


cfg = Config()

try:
    gdf = gpd.read_file(cfg.gpkg_path, layer=cfg.gpkg_layer, engine="pyogrio")
except Exception:
    gdf = gpd.read_file(cfg.gpkg_path, layer=cfg.gpkg_layer)

if cfg.rec_only:
    gdf = gdf[gdf["in_rec_study"] == True].copy()

with rasterio.open(cfg.nlcd_path) as src:
    nlcd_bounds = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    gdf_nlcd = gdf.to_crs(src.crs)

gdf["intersects_nlcd"] = gdf_nlcd.geometry.intersects(nlcd_bounds).to_numpy()
valid = gdf[gdf["intersects_nlcd"]].copy()
valid["outside_nlcd"] = False
valid = valid.sort_values("base_id").reset_index(drop=True)

cols = [
    "base_id",
    "display_name",
    "source_name_rec_csv",
    "in_rec_study",
    "in_legacy_set",
    "tiger_lines_data",
    "outside_nlcd",
]
valid[cols].to_csv(cfg.out_csv, index=False)

summary = {
    "input_total": int(len(gdf)),
    "valid_total": int(len(valid)),
    "excluded_outside_nlcd": int((~gdf["intersects_nlcd"]).sum()),
    "out_csv": str(cfg.out_csv),
}
cfg.out_summary.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
