from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
import zipfile
import xml.etree.ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import math
from shapely.geometry import LineString, Polygon, box

try:
    import cartopy.crs as ccrs
    from cartopy.io import shapereader

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

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
    # ncols = min(3, n)
    ncols=3
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

    # if mesh is not None:
    #     cbar = fig.colorbar(mesh, ax=axes[: len(panels)], shrink=0.88)
    #     cbar.set_label(variable_name)

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


@st.cache_resource(show_spinner=False)
def load_sao_paulo_state() -> gpd.GeoDataFrame:
    url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/brazil-states.geojson"
    try:
        states = gpd.read_file(url)
        name_col = next((c for c in states.columns if c.lower() in {"name", "nome", "nm_uf"}), None)
        if name_col is None:
            raise ValueError("State name column was not found in GeoJSON.")

        names = (
            states[name_col]
            .astype(str)
            .str.lower()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )
        sp_state = states.loc[names.str.contains("sao paulo")].copy()
        if sp_state.empty:
            raise ValueError("São Paulo geometry not found in GeoJSON.")
        return sp_state.to_crs("EPSG:4326")
    except Exception as exc:
        warnings.warn(
            f"Could not load official São Paulo boundary ({exc}). Using fallback polygon.",
            RuntimeWarning,
        )
        fallback = Polygon(
            [
                (-53.0, -25.5),
                (-44.0, -25.5),
                (-44.0, -19.7),
                (-53.0, -19.7),
                (-53.0, -25.5),
            ]
        )
        return gpd.GeoDataFrame({"name": ["Sao Paulo (fallback)"]}, geometry=[fallback], crs="EPSG:4326")


@st.cache_resource(show_spinner=False)
def parse_lines_from_kmz(kmz_file: str) -> gpd.GeoDataFrame:
    path = Path(kmz_file)
    if not path.exists():
        raise FileNotFoundError(f"KMZ file not found: {path}")

    ns_coord = "{http://www.opengis.net/kml/2.2}coordinates"
    line_geoms = []
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("doc.kml") as kml:
            for _, elem in ET.iterparse(kml, events=("end",)):
                if not elem.tag.endswith("LineString"):
                    continue
                coords_node = elem.find(ns_coord)
                if coords_node is not None and coords_node.text:
                    coords = []
                    for token in coords_node.text.strip().split():
                        parts = token.split(",")
                        if len(parts) >= 2:
                            lon, lat = float(parts[0]), float(parts[1])
                            coords.append((lon, lat))
                    if len(coords) >= 2:
                        line_geoms.append(LineString(coords))
                elem.clear()

    if not line_geoms:
        raise ValueError("No LineString geometries were found in the KMZ file.")
    return gpd.GeoDataFrame(geometry=line_geoms, crs="EPSG:4326")


@st.cache_resource(show_spinner=False)
def load_sao_paulo_context(kmz_file: str):
    sp_state = load_sao_paulo_state()
    sp_union = sp_state.union_all()
    sp_bounds = tuple(float(v) for v in sp_state.total_bounds)

    lines = parse_lines_from_kmz(kmz_file)
    lines_sp = lines[lines.intersects(sp_union)].copy()
    if lines_sp.empty:
        raise ValueError("No transmission lines intersect São Paulo.")

    line_union = lines_sp.union_all()
    return sp_state, lines_sp, sp_bounds, line_union


def coordinate_edges(coord_values: np.ndarray) -> np.ndarray:
    values = np.asarray(coord_values, dtype=float)
    if values.size == 1:
        delta = 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=float)

    diffs = np.diff(values)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    edges[0] = values[0] - diffs[0] / 2.0
    edges[-1] = values[-1] + diffs[-1] / 2.0
    return edges


def subset_da_to_bbox(da: xr.DataArray, lat_name: str, lon_name: str, bounds: Tuple[float, float, float, float]) -> xr.DataArray:
    lon_vals = da[lon_name]
    if float(lon_vals.max()) > 180:
        da = da.assign_coords({lon_name: (((lon_vals + 180) % 360) - 180)}).sortby(lon_name)

    lon_min, lat_min, lon_max, lat_max = bounds
    lon_min = math.floor(lon_min)
    lat_min = math.floor(lat_min)
    lon_max = math.ceil(lon_max)
    lat_max = math.ceil(lat_max)

    lat_vals = np.asarray(da[lat_name].values, dtype=float)
    lon_vals = np.asarray(da[lon_name].values, dtype=float)

    lat_slice = slice(lat_min, lat_max) if lat_vals[0] < lat_vals[-1] else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if lon_vals[0] < lon_vals[-1] else slice(lon_max, lon_min)

    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def build_line_intersection_mask(lat_centers: np.ndarray, lon_centers: np.ndarray, line_union) -> np.ndarray:
    lat_edges = coordinate_edges(lat_centers)
    lon_edges = coordinate_edges(lon_centers)

    cell_geometries, cell_i_lat, cell_i_lon = [], [], []
    for i in range(len(lat_centers)):
        y0, y1 = sorted((float(lat_edges[i]), float(lat_edges[i + 1])))
        for j in range(len(lon_centers)):
            x0, x1 = sorted((float(lon_edges[j]), float(lon_edges[j + 1])))
            cell_geometries.append(box(x0, y0, x1, y1))
            cell_i_lat.append(i)
            cell_i_lon.append(j)

    cells = gpd.GeoDataFrame(
        {"i_lat": cell_i_lat, "i_lon": cell_i_lon},
        geometry=cell_geometries,
        crs="EPSG:4326",
    )
    selected = cells.loc[cells.intersects(line_union)]

    mask = np.zeros((len(lat_centers), len(lon_centers)), dtype=bool)
    if not selected.empty:
        mask[selected["i_lat"].to_numpy(), selected["i_lon"].to_numpy()] = True
    return mask


def apply_temporal_statistic(da: xr.DataArray, stat_mode: str, percentile: int) -> xr.DataArray:
    if "time" not in da.dims:
        return da
    if stat_mode == "Mean":
        return da.mean(dim="time", skipna=True)
    if stat_mode == "Min":
        return da.min(dim="time", skipna=True)
    if stat_mode == "Max":
        return da.max(dim="time", skipna=True)

    out = da.quantile(percentile / 100.0, dim="time", skipna=True)
    if "quantile" in out.dims:
        out = out.squeeze("quantile", drop=True)
    return out


def build_line_filtered_panel(
    filepath: str,
    variable_name: str,
    stat_mode: str,
    percentile: int,
    sp_bounds: Tuple[float, float, float, float],
    line_union,
) -> Optional[Dict[str, object]]:
    with xr.open_dataset(filepath) as ds:
        if variable_name not in ds.data_vars:
            return None

        da = ds[variable_name]
        lat_name, lon_name = infer_lat_lon_names(ds)
        units = da.attrs.get("units", "N/A")
        nominal_res = ds.attrs.get("nominal_resolution", "N/A")
        grid_res = infer_grid_resolution(ds)

        da = subset_da_to_bbox(da, lat_name, lon_name, sp_bounds)
        if da.sizes.get(lat_name, 0) == 0 or da.sizes.get(lon_name, 0) == 0:
            return None

        da = apply_temporal_statistic(da, stat_mode, percentile)
        for dim in list(da.dims):
            if dim not in (lat_name, lon_name):
                da = da.isel({dim: 0})
        da = da.squeeze()

        lat = np.asarray(da[lat_name].values, dtype=float)
        lon = np.asarray(da[lon_name].values, dtype=float)
        if lat.size == 0 or lon.size == 0:
            return None

        mask = build_line_intersection_mask(lat, lon, line_union)
        if not np.any(mask):
            return None

        mask_da = xr.DataArray(mask, coords={lat_name: da[lat_name], lon_name: da[lon_name]}, dims=(lat_name, lon_name))
        da = da.where(mask_da)
        values = np.asarray(da.values)

    return {
        "lat": lat,
        "lon": lon,
        "values": values,
        "units": units,
        "grid_res": grid_res,
        "nominal_res": nominal_res,
    }


def plot_line_intersection_comparison(
    df_inventory: pd.DataFrame,
    variable_name: str,
    scenario: str,
    temporal_range_label: str,
    stat_mode: str,
    percentile: int,
    sp_state: gpd.GeoDataFrame,
    lines_sp: gpd.GeoDataFrame,
    sp_bounds: Tuple[float, float, float, float],
    line_union,
    cmap: str = "coolwarm",
):
    files = (
        df_inventory[df_inventory["variable"] == variable_name]
        .sort_values(["model", "scenario", "date_range"])
        .drop_duplicates("model")
    )
    if files.empty:
        return None

    panels = []
    vmins, vmaxs = [], []
    for _, row in files.iterrows():
        try:
            panel = build_line_filtered_panel(
                filepath=row["filepath"],
                variable_name=variable_name,
                stat_mode=stat_mode,
                percentile=percentile,
                sp_bounds=sp_bounds,
                line_union=line_union,
            )
            if panel is None:
                continue

            panel["model"] = row["model"]
            panels.append(panel)
            vals = panel["values"]
            if np.isfinite(vals).any():
                vmins.append(float(np.nanmin(vals)))
                vmaxs.append(float(np.nanmax(vals)))
        except Exception:
            continue

    if not panels:
        return None

    n = len(panels)
    # ncols = min(3, n)
    ncols=3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    vmin = min(vmins) if vmins else None
    vmax = max(vmaxs) if vmaxs else None

    mesh = None
    for ax, panel in zip(axes, panels):
        mesh = ax.pcolormesh(
            panel["lon"],
            panel["lat"],
            panel["values"],
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        sp_state.boundary.plot(ax=ax, color="black", linewidth=1.0)
        lines_sp.plot(ax=ax, color="crimson", linewidth=0.8, alpha=0.9)
        ax.set_xlim(sp_bounds[0], sp_bounds[2])
        ax.set_ylim(sp_bounds[1], sp_bounds[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{panel['model']}\nGrid: {panel['grid_res']} | Nominal: {panel['nominal_res']}", fontsize=9)

    for ax in axes[len(panels):]:
        ax.axis("off")

    stat_label = f"P{percentile}" if stat_mode == "Percentile" else stat_mode
    # if mesh is not None:
    #     units = panels[0].get("units", "N/A")
    #     cbar = fig.colorbar(mesh, ax=axes[: len(panels)], shrink=0.9)
    #     cbar.set_label(f"{variable_name} ({units})")

    fig.suptitle(
        f"{scenario.title()} {stat_label} {VAR_NAMES.get(variable_name, variable_name)} | {temporal_range_label}",
        y=1.02,
        fontsize=12,
    )
    plt.tight_layout()
    # Add colorbar after tight_layout to ensure it doesn't overlap
    plt.tight_layout()

    if mesh is not None:
        units = panels[0].get("units", "N/A")
        # 'ax=axes' applies the colorbar to the whole grid area
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.8, pad=0.05)
        cbar.set_label(f"{variable_name} ({units})")
        
    return fig


st.set_page_config(page_title="CMIP6 Brazil Monthly Explorer", layout="wide")
APP_ROOT = Path(__file__).resolve().parent
KMZ_PATH = APP_ROOT / "ISA ENERGIA BRASIL.kmz"


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
            .replace("historical, ssp1_2_6, ssp2_4_5, ssp3_7_0","All").replace("ssp1_2_6, ssp2_4_5, ssp3_7_0","SSPs").replace("historical","Historical")
        matrix.index = [VAR_NAMES.get(v, v) for v in matrix.index]
        st.dataframe(matrix, use_container_width=True, height=460)
        st.caption("All = Historical + SSPs | SSPs = ssp1_2_6 + ssp2_4_5 + ssp3_7_0")

    with tab2:
        st.subheader("Detailed table per variable")
        variables = sorted(df_inventory["variable"].unique())
        
        variable_name = st.selectbox("Select variable", variables, key="details_var", format_func=lambda v: VAR_NAMES.get(v, v),index=4)
        details_table = build_variable_table(variable_name, df_inventory)
        if details_table.empty:
            st.warning("No detailed records found for the selected variable.")
        else:
            st.dataframe(details_table, use_container_width=True, height=460)

        var_df = df_inventory[df_inventory["variable"] == variable_name].copy()
        var_df.loc[:, "temporal_range_label"] = var_df.loc[:, "date_range"]\
                                            .apply(format_temporal_range_label)

        st.markdown("### São Paulo maps (cells intersecting transmission lines)")
        stat_mode = st.radio("Temporal statistic", ["Mean", "Min", "Max", ], key="sp_stat_mode",horizontal=True)
        percentile = ( #"Percentile"
            st.slider("Percentile", min_value=1, max_value=99, value=90, step=1, key="sp_percentile")
            if stat_mode == "Percentile"
            else 50
        )

        try:
            sp_state, lines_sp, sp_bounds, line_union = load_sao_paulo_context(str(KMZ_PATH))
        except Exception as exc:
            st.warning(f"Could not prepare São Paulo/transmission-line layers: {exc}")
            return

        for scen in sorted(var_df["scenario"].unique()):
            st.markdown(f"#### Scenario: `{scen}`")
            scen_df = var_df[var_df["scenario"] == scen]

            temporal_ranges = sorted(scen_df["temporal_range_label"].dropna().unique())
            if not temporal_ranges:
                temporal_ranges = ["N/A"]

            range_tabs = st.tabs([f"Range: {tr}" for tr in temporal_ranges])

            for tr_tab, tr in zip(range_tabs, temporal_ranges):
                with tr_tab:
                    tr_df = scen_df if tr == "N/A" else scen_df[scen_df["temporal_range_label"] == tr]
                    fig = plot_line_intersection_comparison(
                        df_inventory=tr_df,
                        variable_name=variable_name,
                        scenario=scen,
                        temporal_range_label=tr,
                        stat_mode=stat_mode,
                        percentile=percentile,
                        sp_state=sp_state,
                        lines_sp=lines_sp,
                        sp_bounds=sp_bounds,
                        line_union=line_union,
                        cmap="coolwarm",
                    )
                    if fig is None:
                        st.info(f"No plottable maps for scenario `{scen}` and range `{tr}` with this filter.")
                    else:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

if __name__ == "__main__":
    main()