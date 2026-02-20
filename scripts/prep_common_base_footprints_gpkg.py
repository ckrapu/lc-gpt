from dataclasses import dataclass
from pathlib import Path
import json
import re
import zipfile

import geopandas as gpd
import pandas as pd


@dataclass
class Config:
    boundaries_zip: Path = Path("data/Base_boundary_shp.zip")
    boundaries_shp: Path = Path("data/Base_boundary_shp/US_military_boundary.shp")
    rec_csv: Path = Path("data/rec-study-installation-names.csv")
    rec_name_col: str = "FULLNAME"
    rec_tiger_col: str = "tiger_lines_data"
    shp_name_col: str = "FULLNAME"
    use_tiger_only: bool = True
    legacy_geojson_dir: Path = Path("data/inpaint_regions")
    legacy_glob: str = "base_*.geojson"
    out_gpkg: Path = Path("data/base_footprints_common.gpkg")
    out_layer: str = "base_footprints"
    out_ids_txt: Path = Path("data/base_footprints_common_ids.txt")
    out_ids_json: Path = Path("data/base_footprints_common_ids.json")
    out_summary_json: Path = Path("data/base_footprints_common_summary.json")


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

if not cfg.boundaries_shp.exists():
    if not cfg.boundaries_zip.exists():
        raise FileNotFoundError(f"Missing shapefile zip: {cfg.boundaries_zip}")
    with zipfile.ZipFile(cfg.boundaries_zip, "r") as zf:
        zf.extractall(cfg.boundaries_zip.parent)

rec_df = pd.read_csv(cfg.rec_csv)
if cfg.use_tiger_only and cfg.rec_tiger_col in rec_df.columns:
    tiger_mask = (
        rec_df[cfg.rec_tiger_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    rec_df = rec_df[tiger_mask].copy()

rec_df = rec_df[[cfg.rec_name_col] + ([cfg.rec_tiger_col] if cfg.rec_tiger_col in rec_df.columns else [])]
rec_df = rec_df.dropna(subset=[cfg.rec_name_col]).copy()
rec_df[cfg.rec_name_col] = rec_df[cfg.rec_name_col].astype(str).str.strip()
rec_df = rec_df[rec_df[cfg.rec_name_col] != ""].drop_duplicates(subset=[cfg.rec_name_col], keep="first")
rec_df["name_norm"] = rec_df[cfg.rec_name_col].map(normalize_name)

try:
    shp_df = gpd.read_file(cfg.boundaries_shp, engine="pyogrio")
except Exception:
    shp_df = gpd.read_file(cfg.boundaries_shp)

if cfg.shp_name_col not in shp_df.columns:
    raise ValueError(f"Shapefile missing column '{cfg.shp_name_col}'")

shp_df = shp_df[[cfg.shp_name_col, "geometry"]].dropna(subset=["geometry"]).copy()
shp_df[cfg.shp_name_col] = shp_df[cfg.shp_name_col].astype(str).str.strip()
shp_df["name_norm"] = shp_df[cfg.shp_name_col].map(normalize_name)
shp_df = shp_df.dissolve(by="name_norm", as_index=False, aggfunc={cfg.shp_name_col: "first"})
shp_df = shp_df.to_crs("EPSG:4326")

shape_by_norm = {row["name_norm"]: row for _, row in shp_df.iterrows()}

rows = []
base_idx: dict[str, int] = {}
slug_counts: dict[str, int] = {}
missing_from_shp = []

for _, rec in rec_df.iterrows():
    rec_name = rec[cfg.rec_name_col]
    name_norm = rec["name_norm"]
    row = shape_by_norm.get(name_norm)
    if row is None:
        missing_from_shp.append(rec_name)
        continue

    raw_id = slugify(str(row[cfg.shp_name_col]))
    slug_counts[raw_id] = slug_counts.get(raw_id, 0) + 1
    base_id = raw_id if slug_counts[raw_id] == 1 else f"{raw_id}_{slug_counts[raw_id]}"

    tiger_val = None
    if cfg.rec_tiger_col in rec.index:
        tiger_val = str(rec[cfg.rec_tiger_col]).strip().lower() in {"true", "1", "yes"}

    payload = {
        "base_id": base_id,
        "display_name": str(row[cfg.shp_name_col]),
        "source_name_rec_csv": rec_name,
        "source_name_shp": str(row[cfg.shp_name_col]),
        "name_norm": name_norm,
        "in_rec_study": True,
        "in_legacy_set": False,
        "legacy_id": "",
        "tiger_lines_data": tiger_val,
        "geometry": row["geometry"],
    }
    base_idx[base_id] = len(rows)
    rows.append(payload)

legacy_paths = sorted(cfg.legacy_geojson_dir.glob(cfg.legacy_glob))
for geojson_path in legacy_paths:
    stem = geojson_path.stem
    if not stem.startswith("base_"):
        continue
    legacy_id = stem[len("base_") :]

    legacy_gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    if legacy_gdf.empty:
        continue
    legacy_geom = legacy_gdf.geometry.union_all()

    if legacy_id in base_idx:
        rows[base_idx[legacy_id]]["in_legacy_set"] = True
        rows[base_idx[legacy_id]]["legacy_id"] = legacy_id
        continue

    rows.append(
        {
            "base_id": legacy_id,
            "display_name": legacy_id,
            "source_name_rec_csv": "",
            "source_name_shp": "",
            "name_norm": normalize_name(legacy_id),
            "in_rec_study": False,
            "in_legacy_set": True,
            "legacy_id": legacy_id,
            "tiger_lines_data": None,
            "geometry": legacy_geom,
        }
    )
    base_idx[legacy_id] = len(rows) - 1

out_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
out_gdf = out_gdf.sort_values("base_id").reset_index(drop=True)

cfg.out_gpkg.parent.mkdir(parents=True, exist_ok=True)
if cfg.out_gpkg.exists():
    cfg.out_gpkg.unlink()
out_gdf.to_file(cfg.out_gpkg, layer=cfg.out_layer, driver="GPKG")

base_ids = out_gdf["base_id"].tolist()
cfg.out_ids_txt.write_text("\n".join(base_ids) + "\n")
cfg.out_ids_json.write_text(json.dumps({"base_ids": base_ids, "count": len(base_ids)}, indent=2))

summary = {
    "out_gpkg": str(cfg.out_gpkg),
    "layer": cfg.out_layer,
    "count_total": int(len(out_gdf)),
    "count_rec_study": int(out_gdf["in_rec_study"].sum()),
    "count_legacy": int(out_gdf["in_legacy_set"].sum()),
    "count_rec_only": int(((out_gdf["in_rec_study"]) & (~out_gdf["in_legacy_set"])).sum()),
    "count_legacy_only": int(((~out_gdf["in_rec_study"]) & (out_gdf["in_legacy_set"])).sum()),
    "count_both": int(((out_gdf["in_rec_study"]) & (out_gdf["in_legacy_set"])).sum()),
    "missing_from_shapefile_count": len(missing_from_shp),
    "missing_from_shapefile_names": missing_from_shp,
    "ids_txt": str(cfg.out_ids_txt),
    "ids_json": str(cfg.out_ids_json),
}
cfg.out_summary_json.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
