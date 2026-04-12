"""
ISA Energia CMIP6 Dashboard — FastAPI backend
=============================================
Run:  uvicorn dashboard:app --host 0.0.0.0 --port 8000
      (or)  python dashboard.py
"""

import json
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from shapely.geometry import LineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

# ── Config ─────────────────────────────────────────────────────────────────────
ENSEMBLE_BASE = Path("../ensembles_daily")
KMZ_FILE = "../data/ISA ENERGIA BRASIL.kmz"
LINES_CACHE = Path("_lines_cache.json")

VAR_NAMES = {
    "huss":    "Umid. Específica (kg/kg)",
    "pr":      "Precipitação (kg m⁻² s⁻¹)",
    "psl":     "Pressão Nível do Mar (Pa)",
    "sfcWind": "Velocidade do Vento (m s⁻¹)",
    "tas":     "Temperatura (K)",
    "tasmax":  "Temperatura Máxima (K)",
    "tasmin":  "Temperatura Mínima (K)",
}

VAR_AGG = {
    "huss": "max", "pr": "max", "psl": "min",
    "sfcWind": "max", "tas": "max", "tasmax": "max", "tasmin": "min",
}

SCENARIO_LABELS = {
    "ssp1_2_6": "SSP1-2.6",
    "ssp2_4_5": "SSP2-4.5",
    "ssp3_7_0": "SSP3-7.0",
}

# ── In-memory state ────────────────────────────────────────────────────────────
_lines: list | None = None
_geoms: dict = {}
_da_cache: dict = {}
_catalogue: list | None = None

app = FastAPI(title="ISA Energia CMIP6 Dashboard")


# ── Line loading ───────────────────────────────────────────────────────────────
def _parse_kmz() -> tuple[list, dict]:
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    segs_by_linha: dict = {}
    attrs_by_linha: dict = {}

    print("Parsing KMZ (first run only)…")
    with zipfile.ZipFile(KMZ_FILE) as zf:
        with zf.open("doc.kml") as kml:
            root = ET.parse(kml).getroot()
            for pm in root.findall(".//kml:Placemark", ns):
                ext = {
                    d.attrib.get("name"): d.findtext("kml:value", default="", namespaces=ns)
                    for d in pm.findall(".//kml:ExtendedData//kml:Data", ns)
                }
                linha = ext.get("LINHA", "").strip()
                if not linha:
                    continue
                cn = pm.find(".//kml:LineString/kml:coordinates", ns)
                if cn is None or not cn.text:
                    continue
                coords = []
                for tok in cn.text.strip().split():
                    p = tok.split(",")
                    if len(p) >= 2:
                        coords.append((float(p[0]), float(p[1])))
                if len(coords) >= 2:
                    segs_by_linha.setdefault(linha, []).append(LineString(coords))
                    if linha not in attrs_by_linha:
                        attrs_by_linha[linha] = ext

    lines = []
    geoms = {}
    for linha, segs in segs_by_linha.items():
        a = attrs_by_linha[linha]
        geom = unary_union(segs)
        geoms[linha] = geom

        if geom.geom_type == "LineString":
            coords_l = [[c[1], c[0]] for c in geom.coords]
        else:
            coords_l = [[[c[1], c[0]] for c in g.coords] for g in geom.geoms]

        b = geom.bounds
        lines.append({
            "linha": linha,
            "empresa": a.get("EMPRESA", ""),
            "tensao": a.get("TENSAO", ""),
            "regional": a.get("REGIONAL", ""),
            "coords": coords_l,
            "bounds": [[b[1], b[0]], [b[3], b[2]]],
        })

    return lines, geoms


def load_lines() -> list:
    global _lines, _geoms
    if _lines is not None:
        return _lines

    if LINES_CACHE.exists():
        print("Loading lines from cache…")
        _lines = json.loads(LINES_CACHE.read_text())
        # Rebuild geometries from coords
        for lt in _lines:
            coords = lt["coords"]
            if coords and isinstance(coords[0][0], list):
                segs = [LineString([(c[1], c[0]) for c in seg]) for seg in coords]
                _geoms[lt["linha"]] = unary_union(segs)
            else:
                _geoms[lt["linha"]] = LineString([(c[1], c[0]) for c in coords])
        print(f"Loaded {len(_lines)} lines from cache.")
        return _lines

    _lines, _geoms = _parse_kmz()
    LINES_CACHE.write_text(json.dumps(_lines))
    print(f"Cached {len(_lines)} lines.")
    return _lines


def get_catalogue() -> list:
    global _catalogue
    if _catalogue is not None:
        return _catalogue
    records = []
    for var_dir in sorted(p for p in ENSEMBLE_BASE.iterdir() if p.is_dir()):
        for nc in sorted(var_dir.glob("ensemble_max_*.nc")):
            stem = nc.stem[len("ensemble_max_"):]
            scenario, date_range = stem.rsplit("_", 1)
            if scenario == "historical":
                continue
            records.append({
                "variable": var_dir.name,
                "scenario": scenario,
                "date_range": date_range,
                "filepath": str(nc),
            })
    _catalogue = records
    return records


def get_da(filepath: str, variable: str) -> xr.DataArray:
    if filepath not in _da_cache:
        ds = xr.open_dataset(filepath)
        _da_cache[filepath] = ds[variable].load()
        ds.close()
    return _da_cache[filepath]


# ── Spatial helpers ────────────────────────────────────────────────────────────
def normalize_lon(da: xr.DataArray) -> xr.DataArray:
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    return da


def pad_bounds(bounds, pad_km: float = 250.0):
    lon_min, lat_min, lon_max, lat_max = bounds
    lat_mid = (lat_min + lat_max) / 2
    pad_lat = pad_km / 111.0
    pad_lon = pad_km / (111.32 * max(np.cos(np.deg2rad(lat_mid)), 1e-6))
    return lon_min - pad_lon, lat_min - pad_lat, lon_max + pad_lon, lat_max + pad_lat


def subset_da(da: xr.DataArray, bounds) -> xr.DataArray:
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_min, lat_min = math.floor(lon_min), math.floor(lat_min)
    lon_max, lat_max = math.ceil(lon_max), math.ceil(lat_max)
    lats, lons = da.lat.values, da.lon.values
    lat_sl = slice(lat_min, lat_max) if lats[0] < lats[-1] else slice(lat_max, lat_min)
    lon_sl = slice(lon_min, lon_max) if lons[0] < lons[-1] else slice(lon_max, lon_min)
    return da.sel(lat=lat_sl, lon=lon_sl)


def _coordinate_edges(vals: np.ndarray) -> np.ndarray:
    """Compute cell edges from center coordinates (matches notebook logic)."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    diffs = np.diff(vals)
    edges = np.empty(vals.size + 1)
    edges[1:-1] = (vals[:-1] + vals[1:]) / 2
    edges[0]    = vals[0] - diffs[0] / 2
    edges[-1]   = vals[-1] + diffs[-1] / 2
    return edges


def build_mask(lat_c: np.ndarray, lon_c: np.ndarray, geom) -> np.ndarray:
    """
    Build a boolean mask using exact shapely polygon intersection — same approach
    as build_line_intersection_mask in aggregate.ipynb.
    Each grid cell is represented as a box; only cells whose polygon intersects
    the line geometry are marked True (no rasterization artefacts).
    """
    lat_edges = _coordinate_edges(lat_c)
    lon_edges = _coordinate_edges(lon_c)

    cells, cell_ij = [], []
    for i in range(len(lat_c)):
        y0 = min(float(lat_edges[i]), float(lat_edges[i + 1]))
        y1 = max(float(lat_edges[i]), float(lat_edges[i + 1]))
        for j in range(len(lon_c)):
            x0 = min(float(lon_edges[j]), float(lon_edges[j + 1]))
            x1 = max(float(lon_edges[j]), float(lon_edges[j + 1]))
            cells.append(box(x0, y0, x1, y1))
            cell_ij.append((i, j))

    tree = STRtree(cells)
    mask = np.zeros((len(lat_c), len(lon_c)), dtype=bool)
    for idx in tree.query(geom, predicate="intersects"):
        i, j = cell_ij[int(idx)]
        mask[i, j] = True
    return mask


# ── API ────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_lines)
    await loop.run_in_executor(None, get_catalogue)
    print("Dashboard ready.")


@app.get("/api/lines")
def api_lines():
    return JSONResponse(load_lines())


@app.get("/api/variables")
def api_variables():
    seen: dict = {}
    for r in get_catalogue():
        v = r["variable"]
        if v not in seen:
            seen[v] = {"id": v, "label": VAR_NAMES.get(v, v), "agg": VAR_AGG.get(v, "max")}
    return JSONResponse(list(seen.values()))


@app.get("/api/analysis")
def api_analysis(linha: str = Query(...), variable: str = Query(...)):
    geom = _geoms.get(linha)
    if geom is None:
        raise HTTPException(404, f"Line '{linha}' not found")

    cat = [r for r in get_catalogue() if r["variable"] == variable]
    if not cat:
        raise HTTPException(404, f"No ensemble data for '{variable}'")

    agg = VAR_AGG.get(variable, "max")
    padded = pad_bounds(geom.bounds)

    # Reference grid from first file
    first_da = normalize_lon(get_da(cat[0]["filepath"], variable))
    first_sub = subset_da(first_da, padded)
    lat_c = first_sub.lat.values
    lon_c = first_sub.lon.values

    if lat_c.size == 0 or lon_c.size == 0:
        raise HTTPException(400, "No grid data in this area")

    mask = build_mask(lat_c, lon_c, geom)
    if not mask.any():
        raise HTTPException(400, "No grid cells intersect this line")

    dlat = abs(lat_c[1] - lat_c[0]) if len(lat_c) > 1 else 1.0
    dlon = abs(lon_c[1] - lon_c[0]) if len(lon_c) > 1 else 1.0

    cell_worst = np.full((len(lat_c), len(lon_c)), np.nan)
    scenarios_out: dict = {}

    for r in cat:
        sc, dr = r["scenario"], r["date_range"]
        da = normalize_lon(get_da(r["filepath"], variable))
        # Select exact same grid coordinates as reference
        da_sub = da.sel(lat=lat_c, lon=lon_c, method="nearest", tolerance=1.0)
        vals = da_sub.values  # (T, nlat, nlon)

        # Per-cell worst-case over time
        cell_agg = (np.nanmax if agg == "max" else np.nanmin)(vals, axis=0)
        cell_worst = (np.fmax if agg == "max" else np.fmin)(cell_worst, cell_agg)

        # Time series: one value per timestep (worst masked cell)
        masked_vals = vals[:, mask]  # (T, n_cells)
        ts_vals = (np.nanmax if agg == "max" else np.nanmin)(masked_vals, axis=1)

        try:
            times = pd.to_datetime(da_sub.time.values).strftime("%Y-%m-%d").tolist()
        except Exception:
            times = [str(t)[:10] for t in da_sub.time.values]

        if sc not in scenarios_out:
            scenarios_out[sc] = {"label": SCENARIO_LABELS.get(sc, sc), "periods": {}}

        scenarios_out[sc]["periods"][dr] = {
            "times": times,
            "values": [None if np.isnan(v) else round(float(v), 5) for v in ts_vals],
        }

    # Quantiles per scenario (union of all periods)
    for sc_data in scenarios_out.values():
        combined = [
            v for pd_data in sc_data["periods"].values()
            for v in pd_data["values"] if v is not None
        ]
        if combined:
            arr = np.array(combined)
            sc_data["quantiles"] = {
                "q50": round(float(np.nanpercentile(arr, 50)), 5),
                "q90": round(float(np.nanpercentile(arr, 90)), 5),
                "q95": round(float(np.nanpercentile(arr, 95)), 5),
                "q99": round(float(np.nanpercentile(arr, 99)), 5),
            }

    # Grid cells (masked only) — worst case across all scenarios
    grid_cells = []
    for i, lat in enumerate(lat_c):
        for j, lon in enumerate(lon_c):
            if not mask[i, j]:
                continue
            val = cell_worst[i, j]
            grid_cells.append({
                "lat": float(lat), "lon": float(lon),
                "bounds": [
                    [float(lat) - dlat / 2, float(lon) - dlon / 2],
                    [float(lat) + dlat / 2, float(lon) + dlon / 2],
                ],
                "value": None if np.isnan(val) else round(float(val), 5),
            })

    return {
        "linha": linha,
        "variable": variable,
        "var_label": VAR_NAMES.get(variable, variable),
        "agg": agg,
        "grid": grid_cells,
        "scenarios": scenarios_out,
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8000, reload=False)
