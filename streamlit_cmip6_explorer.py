from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import math

try:
    import cartopy.crs as ccrs
    from cartopy.io import shapereader

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# def plot_model_comparison(
#     df_inventory: pd.DataFrame,
#     variable_name: str,
#     scenario: Optional[str] = None,
#     mode: str = "Temporal mean",
#     time_index: int = 0,
#     cmap: str = "coolwarm",
# ):
#     files = df_inventory[df_inventory["variable"] == variable_name].copy()
#     if scenario:
#         files = files[files["scenario"] == scenario]

#     files = files.sort_values(["model", "scenario"]).drop_duplicates("model")
#     if files.empty:
#         return None

#     n = len(files)
#     ncols = min(3, n)
#     nrows = math.ceil(n / ncols)

#     subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
#     fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), subplot_kw=subplot_kw)
#     axes = np.atleast_1d(axes).reshape(-1)

#     geoms = load_brazil_geometry() if HAS_CARTOPY else []
#     panels = []
#     vmins, vmaxs = [], []

#     for _, row in files.iterrows():
#         try:
#             with xr.open_dataset(row["filepath"]) as ds:
#                 if variable_name not in ds.data_vars:
#                     continue

#                 lat_name, lon_name = infer_lat_lon_names(ds)
#                 da = ds[variable_name]

#                 if "time" in da.dims:
#                     if mode == "Temporal mean":
#                         da = da.mean(dim="time", skipna=True)
#                     else:
#                         idx = min(time_index, da.sizes["time"] - 1)
#                         da = da.isel(time=idx)

#                 for dim in list(da.dims):
#                     if dim not in (lat_name, lon_name):
#                         da = da.isel({dim: 0})

#                 da = da.squeeze()
#                 lat = np.asarray(ds[lat_name].values)
#                 lon = np.asarray(ds[lon_name].values)
#                 values = np.asarray(da.values)

#                 grid_res = infer_grid_resolution(ds)
#                 nominal_res = ds.attrs.get("nominal_resolution", "N/A")

#             panels.append(
#                 {
#                     "model": row["model"],
#                     "scenario": row["scenario"],
#                     "lat": lat,
#                     "lon": lon,
#                     "values": values,
#                     "grid_res": grid_res,
#                     "nominal_res": nominal_res,
#                 }
#             )

#             if np.isfinite(values).any():
#                 vmins.append(float(np.nanmin(values)))
#                 vmaxs.append(float(np.nanmax(values)))
#         except Exception:
#             continue

#     if not panels:
#         plt.close(fig)
#         return None

#     vmin = min(vmins) if vmins else None
#     vmax = max(vmaxs) if vmaxs else None

#     mesh = None
#     for ax, p in zip(axes, panels):
#         if HAS_CARTOPY:
#             mesh = ax.pcolormesh(
#                 p["lon"],
#                 p["lat"],
#                 p["values"],
#                 shading="auto",
#                 cmap=cmap,
#                 vmin=vmin,
#                 vmax=vmax,
#                 transform=ccrs.PlateCarree(),
#             )
#             ax.coastlines(resolution="10m", linewidth=0.5)
#             if geoms:
#                 ax.add_geometries(
#                     geoms,
#                     crs=ccrs.PlateCarree(),
#                     facecolor="none",
#                     edgecolor="black",
#                     linewidth=1.0,
#                     zorder=5,
#                 )
#             ax.set_extent([-75, -33, -35, 7], crs=ccrs.PlateCarree())
#         else:
#             mesh = ax.pcolormesh(
#                 p["lon"],
#                 p["lat"],
#                 p["values"],
#                 shading="auto",
#                 cmap=cmap,
#                 vmin=vmin,
#                 vmax=vmax,
#             )
#             ax.set_xlabel("Longitude")
#             ax.set_ylabel("Latitude")

#         ax.set_title(f"{p['model']}\nGrid: {p['grid_res']} | Nominal: {p['nominal_res']}", fontsize=9)

#     for ax in axes[len(panels):]:
#         ax.axis("off")

#     if mesh is not None:
#         cbar = fig.colorbar(mesh, ax=axes[: len(panels)], shrink=0.88)
#         cbar.set_label(variable_name)

#     scenario_label = scenario if scenario is not None else "all scenarios"
#     fig.suptitle(f"{variable_name} | {scenario_label} | {mode}", y=1.02, fontsize=12)
#     plt.tight_layout()
#     return fig

# ...existing code...
def plot_model_comparison(
    df_inventory: pd.DataFrame,
    variable_name: str,
    scenario: Optional[str] = None,
    mode: str = "Single timestep",
    time_index: int = 0,
    cmap: str = "coolwarm",
    title_extra: Optional[str] = None,
):
    files = df_inventory[df_inventory["variable"] == variable_name].copy()
    if scenario:
        files = files[files["scenario"] == scenario]

    files = files.sort_values(["model", "scenario"]).drop_duplicates("model")
    if files.empty:
        return None

    n = len(files)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), subplot_kw=subplot_kw)
    axes = np.atleast_1d(axes).reshape(-1)

    geoms = load_brazil_geometry() if HAS_CARTOPY else []
    panels = []
    vmins, vmaxs = [], []

    for _, row in files.iterrows():
        try:
            with xr.open_dataset(row["filepath"]) as ds:
                if variable_name not in ds.data_vars:
                    continue

                lat_name, lon_name = infer_lat_lon_names(ds)
                da = ds[variable_name]

                if "time" in da.dims:
                    if mode == "Temporal mean":
                        da = da.mean(dim="time", skipna=True)
                    else:
                        idx = min(time_index, da.sizes["time"] - 1)
                        da = da.isel(time=idx)

                for dim in list(da.dims):
                    if dim not in (lat_name, lon_name):
                        da = da.isel({dim: 0})

                da = da.squeeze()
                lat = np.asarray(ds[lat_name].values)
                lon = np.asarray(ds[lon_name].values)
                values = np.asarray(da.values)

                grid_res = infer_grid_resolution(ds)
                nominal_res = ds.attrs.get("nominal_resolution", "N/A")

            panels.append(
                {
                    "model": row["model"],
                    "scenario": row["scenario"],
                    "lat": lat,
                    "lon": lon,
                    "values": values,
                    "grid_res": grid_res,
                    "nominal_res": nominal_res,
                }
            )

            if np.isfinite(values).any():
                vmins.append(float(np.nanmin(values)))
                vmaxs.append(float(np.nanmax(values)))
        except Exception:
            continue

    if not panels:
        plt.close(fig)
        return None

    vmin = min(vmins) if vmins else None
    vmax = max(vmaxs) if vmaxs else None

    mesh = None
    for ax, p in zip(axes, panels):
        if HAS_CARTOPY:
            mesh = ax.pcolormesh(
                p["lon"],
                p["lat"],
                p["values"],
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            ax.coastlines(resolution="10m", linewidth=0.5)
            if geoms:
                ax.add_geometries(
                    geoms,
                    crs=ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=5,
                )
            ax.set_extent([-75, -33, -35, 7], crs=ccrs.PlateCarree())
        else:
            mesh = ax.pcolormesh(
                p["lon"],
                p["lat"],
                p["values"],
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        ax.set_title(f"{p['model']}\nGrid: {p['grid_res']} | Nominal: {p['nominal_res']}", fontsize=9)

    for ax in axes[len(panels):]:
        ax.axis("off")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes[: len(panels)], shrink=0.88)
        cbar.set_label(variable_name)

    scenario_label = scenario if scenario is not None else "all scenarios"
    title_parts = [variable_name, scenario_label]
    if title_extra:
        title_parts.append(title_extra)
    if mode == "Single timestep":
        title_parts.append(f"Single timestep (t={time_index})")
    else:
        title_parts.append(mode)

    fig.suptitle(" | ".join(title_parts), y=1.02, fontsize=12)
    plt.tight_layout()
    return fig
# ...existing code...
# ...existing code...
def format_temporal_range_label(date_range: str) -> str:
    parts = str(date_range).split("-")
    if len(parts) == 2 and len(parts[0]) >= 4 and len(parts[1]) >= 4:
        return f"{parts[0][:4]}-{parts[1][:4]}"
    return str(date_range)
# ...existing code...

VAR_NAMES = {'hur': 'Relative Humidity (%)',
 'hurs': 'Near-Surface Relative Humidity (%)',
 'hus': 'Specific Humidity (1)',
 'huss': 'Near-Surface Specific Humidity (1)',
 'pr': 'Precipitation (kg m-2 s-1)',
 'ps': 'Surface Air Pressure (Pa)',
 'psl': 'Sea Level Pressure (Pa)',
 'sfcWind': 'Near-Surface Wind Speed (m s-1)',
 'ta': 'Air Temperature (K)',
 'tas': 'Near-Surface Air Temperature (K)',
 'tasmax': 'Daily Maximum Near-Surface Air Temperature (K)',
 'tasmin': 'Daily Minimum Near-Surface Air Temperature (K)',
 'ts': 'Surface Temperature (K)'}

st.set_page_config(page_title="CMIP6 Brazil Monthly Explorer", layout="wide")


def parse_cmip6_filename(filename: str) -> Optional[Dict[str, str]]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 7:
        return None
    return {
        "variable": parts[0],
        "table": parts[1],
        "model": parts[2],
        "experiment": "_".join(parts[3:-3]),
        "ensemble": parts[-3],
        "grid": parts[-2],
        "date_range": parts[-1],
    }


@st.cache_data(show_spinner=False)
def load_inventory(base_path: str) -> pd.DataFrame:
    root = Path(base_path)
    if not root.exists():
        return pd.DataFrame()

    records = []
    for scenario_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for nc_file in sorted(scenario_dir.glob("*.nc")):
            meta = parse_cmip6_filename(nc_file.name)
            if meta:
                meta["scenario"] = scenario_dir.name
                meta["filepath"] = str(nc_file)
                records.append(meta)

    return pd.DataFrame(records)


def infer_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat_candidates = ("lat", "latitude", "y")
    lon_candidates = ("lon", "longitude", "x")
    lat_name = next((n for n in lat_candidates if n in ds.coords or n in ds.dims), None)
    lon_name = next((n for n in lon_candidates if n in ds.coords or n in ds.dims), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Latitude/longitude coordinates were not found.")
    return lat_name, lon_name


def extract_year_range(ds: xr.Dataset) -> Tuple[Optional[int], Optional[int]]:
    if "time" not in ds.coords or ds["time"].size == 0:
        return None, None
    try:
        t = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    except Exception:
        try:
            t = ds.indexes["time"].to_datetimeindex()
        except Exception:
            return None, None
    return int(t.year.min()), int(t.year.max())


def infer_temporal_resolution(ds: xr.Dataset) -> str:
    if "time" not in ds.coords or ds["time"].size < 2:
        return "N/A"
    try:
        sample = pd.DatetimeIndex(pd.to_datetime(ds["time"].values[: min(24, ds["time"].size)]))
        freq = pd.infer_freq(sample)
        mapping = {"MS": "Monthly", "M": "Monthly", "D": "Daily", "YS": "Yearly", "AS": "Yearly"}
        return mapping.get(freq, freq if freq else "Unknown")
    except Exception:
        return "Unknown"


def infer_grid_resolution(ds: xr.Dataset) -> str:
    try:
        lat_name, lon_name = infer_lat_lon_names(ds)
        lat_vals = np.asarray(ds[lat_name].values, dtype=float)
        lon_vals = np.asarray(ds[lon_name].values, dtype=float)
        if lat_vals.ndim == 1 and lon_vals.ndim == 1 and len(lat_vals) > 1 and len(lon_vals) > 1:
            lat_res = float(np.nanmedian(np.abs(np.diff(lat_vals))))
            lon_res = float(np.nanmedian(np.abs(np.diff(lon_vals))))
            return f"{lat_res:.2f}° x {lon_res:.2f}°"
    except Exception:
        pass
    return "N/A"


@st.cache_data(show_spinner=False)
def get_file_details(filepath: str, variable_name: str) -> Dict[str, object]:
    with xr.open_dataset(filepath) as ds:
        if variable_name not in ds.data_vars:
            return {}

        da = ds[variable_name]
        y0, y1 = extract_year_range(ds)
        year_range = f"{y0}-{y1}" if y0 is not None and y1 is not None else "N/A"

        try:
            vmin = float(da.min(skipna=True).values)
            vmax = float(da.max(skipna=True).values)
            vmean = float(da.mean(skipna=True).values)
        except Exception:
            vmin = vmax = vmean = None

        return {
            "units": da.attrs.get("units", "N/A"),
            "shape": "x".join(str(s) for s in da.shape),
            "nominal_resolution": ds.attrs.get("nominal_resolution", "N/A"),
            "grid_resolution": infer_grid_resolution(ds),
            "temporal_range": year_range,
            "temporal_resolution": infer_temporal_resolution(ds),
            "time_size": int(da.sizes.get("time", 1)),
            "min": vmin,
            "max": vmax,
            "mean": vmean,
        }


def build_variable_matrix(df_inventory: pd.DataFrame) -> pd.DataFrame:
    if df_inventory.empty:
        return pd.DataFrame()
    return (
        df_inventory.groupby(["variable", "model"])["scenario"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .unstack(fill_value="-")
        .sort_index()
    )


def build_variable_table(variable_name: str, df_inventory: pd.DataFrame) -> pd.DataFrame:
    subset = (
        df_inventory[df_inventory["variable"] == variable_name]
        .sort_values(["model", "scenario", "ensemble", "grid", "date_range"])
        .reset_index(drop=True)
    )
    records = []
    for _, row in subset.iterrows():
        info = get_file_details(row["filepath"], variable_name)
        if not info:
            continue
        records.append(
            {
                "Model": row["model"],
                "Scenario": row["scenario"],
                "Temporal Range": info["temporal_range"],
                # "Temporal Resolution": info["temporal_resolution"],
                "Nominal Resolution": info["nominal_resolution"],
                "Grid Resolution": info["grid_resolution"],
                "Shape": info["shape"],
                "Units": info["units"],
                # "File": Path(row["filepath"]).name,
            }
        )
    return pd.DataFrame(records).set_index("Model")


@st.cache_resource(show_spinner=False)
def load_brazil_geometry():
    if not HAS_CARTOPY:
        return []
    shp = shapereader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
    reader = shapereader.Reader(shp)
    return [rec.geometry for rec in reader.records() if rec.attributes.get("ADMIN") == "Brazil"]


def plot_variable_map(filepath: str, variable_name: str, mode: str, time_index: int, cmap: str):
    with xr.open_dataset(filepath) as ds:
        lat_name, lon_name = infer_lat_lon_names(ds)
        da = ds[variable_name]

        if "time" in da.dims:
            if mode == "Temporal mean":
                da = da.mean(dim="time", skipna=True)
            else:
                idx = min(time_index, da.sizes["time"] - 1)
                da = da.isel(time=idx)

        for dim in list(da.dims):
            if dim not in (lat_name, lon_name):
                da = da.isel({dim: 0})

        da = da.squeeze()
        lat = np.asarray(ds[lat_name].values)
        lon = np.asarray(ds[lon_name].values)
        values = np.asarray(da.values)

    fig = plt.figure(figsize=(9, 6))
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap, transform=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", linewidth=0.6)
        geoms = load_brazil_geometry()
        if geoms:
            ax.add_geometries(
                geoms,
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
            )
        ax.set_extent([-75, -33, -35, 7], crs=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.colorbar(mesh, ax=ax, shrink=0.85, label=variable_name)
    ax.set_title(f"{variable_name} ({mode})")
    plt.tight_layout()
    return fig


def main():
    st.title("CMIP6 Brazil Monthly Explorer")

    data_path =  "cmip6_brazil/monthly"

    df_inventory = load_inventory(data_path)
    if df_inventory.empty:
        st.error("No NetCDF files found. Check the dataset path.")
        st.stop()

    st.caption(
        f"Files: {len(df_inventory)} | Variables: {df_inventory['variable'].nunique()} | "
        f"Models: {df_inventory['model'].nunique()} | Scenarios: {df_inventory['scenario'].nunique()}"
    )

    tab1, tab2 = st.tabs(
        ["General matrix", "Variable details"]
    )

    with tab1:
        st.subheader("Variable x Model matrix")
        matrix = build_variable_matrix(df_inventory)\
            .replace("historical, ssp1_2_6, ssp2_4_5, ssp3_7_0","Todos").replace("ssp1_2_6, ssp2_4_5, ssp3_7_0","SSPs").replace("historical","Histórico")
        matrix.index = [VAR_NAMES.get(v, v) for v in matrix.index]
        st.dataframe(matrix, use_container_width=True, height=460)
        st.caption("Todos = histórico + SSPs | SSPs = ssp1_2_6 + ssp2_4_5 + ssp3_7_0")

    with tab2:
        st.subheader("Detailed table per variable")
        variables = sorted(df_inventory["variable"].unique())
        
        variable_name = st.selectbox("Select variable", variables, key="details_var", format_func=lambda v: VAR_NAMES.get(v, v))
        details_table = build_variable_table(variable_name, df_inventory)
        if details_table.empty:
            st.warning("No detailed records found for the selected variable.")
        else:
            st.dataframe(details_table, use_container_width=True, height=460)

        var_df = df_inventory[df_inventory["variable"] == variable_name]
        var_df = df_inventory[df_inventory["variable"] == variable_name].copy()
        var_df["temporal_range_label"] = var_df["date_range"].apply(format_temporal_range_label)


        st.markdown("### Model comparison maps by scenario")
        # mode = st.radio("Plot mode", ["Single timestep", "Temporal mean"], horizontal=True, key="cmp_mode")

        # time_sizes = var_df["filepath"].apply(lambda fp: get_file_details(fp, variable_name).get("time_size", 1))
        # max_t = int(time_sizes.max()) if not time_sizes.empty else 1
        # t_idx = (
        #     st.slider("Time index (shared across models; clipped per file)", 0, max(max_t - 1, 0), 0, key="cmp_tidx")
        #     if mode == "Single timestep" and max_t > 1
        #     else 0
        # )

        # for scen in sorted(var_df["scenario"].unique()):
        #     st.markdown(f"#### Scenario: `{scen}`")
        #     fig = plot_model_comparison(
        #         df_inventory=var_df,
        #         variable_name=variable_name,
        #         scenario=scen,
        #         mode=mode,
        #         time_index=t_idx,
        #         cmap="coolwarm",
        #     )
        #     if fig is None:
        #         st.info(f"No plottable files for scenario `{scen}`.")
        #     else:
        #         st.pyplot(fig, use_container_width=True)
        #         plt.close(fig)
 
        for scen in sorted(var_df["scenario"].unique()):
            st.markdown(f"#### Scenario: `{scen}`")
            scen_df = var_df[var_df["scenario"] == scen]

            scen_time_sizes = scen_df["filepath"].apply(lambda fp: get_file_details(fp, variable_name).get("time_size", 1))
            scen_max_t = int(scen_time_sizes.max()) if not scen_time_sizes.empty else 1
            scen_t_idx = (
                st.slider(
                    f"Time index for `{scen}` (shared in this scenario; clipped per file)",
                    0,
                    max(scen_max_t - 1, 0),
                    0,
                    key=f"cmp_tidx_{scen}",
                )
                if scen_max_t > 1
                else 0
            )

            temporal_ranges = sorted(scen_df["temporal_range_label"].dropna().unique())
            if not temporal_ranges:
                temporal_ranges = ["N/A"]

            range_tabs = st.tabs([f"Range: {tr}" for tr in temporal_ranges])

            for tr_tab, tr in zip(range_tabs, temporal_ranges):
                with tr_tab:
                    tr_df = scen_df if tr == "N/A" else scen_df[scen_df["temporal_range_label"] == tr]
                    fig = plot_model_comparison(
                        df_inventory=tr_df,
                        variable_name=variable_name,
                        scenario=scen,
                        mode="Single timestep",
                        time_index=scen_t_idx,
                        cmap="coolwarm",
                        title_extra=f"Range: {tr}",
                    )
                    if fig is None:
                        st.info(f"No plottable files for scenario `{scen}` and range `{tr}`.")
                    else:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

if __name__ == "__main__":
    main()