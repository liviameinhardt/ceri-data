"""
Build ensemble max files for all variables and scenarios in cmip6_brazil.
One output file per (scenario, date_range) containing all variables.

Usage:
    python build_ensembles.py --freq daily
    python build_ensembles.py --freq monthly
    python build_ensembles.py --freq daily --skip-existing
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr


TARGET_LAT = np.arange(-35, 7.5, 1.0)
TARGET_LON = np.arange(285, 324.375, 1.0)


def parse_cmip6_filename(filename: str) -> Optional[Dict[str, str]]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 7:
        return None
    return {
        "variable": parts[0],
        "model": parts[2],
        "date_range": parts[-1],
    }


def format_temporal_range_label(date_range: str) -> str:
    parts = str(date_range).split("-")
    if len(parts) == 2 and len(parts[0]) >= 4 and len(parts[1]) >= 4:
        return f"{parts[0][:4]}-{parts[1][:4]}"
    return str(date_range)


def load_inventory(base_path: str) -> pd.DataFrame:
    root = Path(base_path)
    if not root.exists():
        raise FileNotFoundError(f"Data path not found: {root}")

    records = []
    for scenario_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for nc_file in sorted(scenario_dir.glob("*.nc")):
            meta = parse_cmip6_filename(nc_file.name)
            if meta:
                meta["scenario"] = scenario_dir.name
                meta["filepath"] = str(nc_file)
                records.append(meta)

    return pd.DataFrame(records)


def build_variable_ensemble(var_df: pd.DataFrame, variable: str, scenario: str,
                            date_range: str) -> tuple[xr.DataArray, float, float]:
    """Regrid all models to target grid and return ensemble max DataArray + (min, max) of raw data."""
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    model_results = []
    global_min, global_max = np.inf, -np.inf

    group = var_df[(var_df["scenario"] == scenario) & (var_df["date_range"] == date_range)]

    for _, row in group.iterrows():
        ds = xr.open_dataset(row["filepath"], decode_times=time_coder)

        if isinstance(ds.indexes["time"], xr.CFTimeIndex):
            ds = ds.convert_calendar("standard", align_on="year")

        da = ds[variable]
        global_min = min(global_min, float(da.min()))
        global_max = max(global_max, float(da.max()))

        da_interp = da.interp(lat=TARGET_LAT, lon=TARGET_LON, method="nearest")
        model_results.append(da_interp)
        ds.close()

    ensemble_cube = xr.concat(model_results, dim="model", join="outer")
    ensemble_max = ensemble_cube.max(dim="model", skipna=True)
    return ensemble_max, global_min, global_max


def main():
    parser = argparse.ArgumentParser(description="Build CMIP6 ensemble files, one per scenario.")
    parser.add_argument("--freq", choices=["daily", "monthly"], required=True,
                        help="Temporal frequency to process.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip ensemble files that already exist.")
    args = parser.parse_args()

    data_path = f"data/cmip6_brazil/{args.freq}"
    out_dir = f"data/ensembles_{args.freq}"
    extremes_path = f"{out_dir}/variables_extremes.json"

    print(f"Loading inventory from: {data_path}")
    df = load_inventory(data_path)
    df["date_range"] = df["date_range"].apply(format_temporal_range_label)

    # Load existing extremes so we can update incrementally
    if Path(extremes_path).exists():
        with open(extremes_path) as f:
            variables_extremes = json.load(f)
    else:
        variables_extremes = {}

    variables = sorted(df["variable"].unique())
    print(f"Variables found: {variables}\n")

    for (scenario, date_range), _ in df.groupby(["scenario", "date_range"]):
        out_path = f"{out_dir}/ensemble_max_{scenario}_{date_range}.nc"
        print(f"=== {scenario} / {date_range} ===")

        if args.skip_existing and Path(out_path).exists():
            print(f"  Skipping (exists): {out_path}")
            with xr.open_dataset(out_path) as ds:
                for var in ds.data_vars:
                    da = ds[var]
                    g_min, g_max = float(da.min()), float(da.max())
                    prev = variables_extremes.get(var, [np.inf, -np.inf])
                    variables_extremes[var] = [min(prev[0], g_min), max(prev[1], g_max)]
            continue

        data_vars = {}
        for variable in variables:
            var_df = df[df["variable"] == variable]
            group = var_df[(var_df["scenario"] == scenario) & (var_df["date_range"] == date_range)]
            if group.empty:
                continue
            print(f"  {variable}...")
            da, g_min, g_max = build_variable_ensemble(var_df, variable, scenario, date_range)
            data_vars[variable] = da
            prev = variables_extremes.get(variable, [np.inf, -np.inf])
            variables_extremes[variable] = [min(prev[0], g_min), max(prev[1], g_max)]

        # Group variables by their scalar (non-dimension) coordinates
        # so that e.g. height=2 and height=10 vars go into separate files.
        groups = defaultdict(dict)
        for var_name, da in data_vars.items():
            key = tuple(
                (name, float(coord.values))
                for name, coord in sorted(da.coords.items())
                if name not in da.dims and coord.ndim == 0
            )
            groups[key][var_name] = da

        os.makedirs(Path(out_path).parent, exist_ok=True)
        for coord_key, group_vars in groups.items():
            if coord_key and len(groups) > 1:
                suffix = "_" + "_".join(
                    f"{n}_{int(v)}" if v == int(v) else f"{n}_{v}"
                    for n, v in coord_key
                )
                path = out_path.replace(".nc", f"{suffix}.nc")
            else:
                path = out_path
            ds_out = xr.Dataset(group_vars)
            ds_out.to_netcdf(path)
            print(f"  Saved: {path}")

    os.makedirs(out_dir, exist_ok=True)
    with open(extremes_path, "w") as f:
        json.dump(variables_extremes, f, indent=2)
    print(f"\nExtremes saved to: {extremes_path}")


if __name__ == "__main__":
    main()
