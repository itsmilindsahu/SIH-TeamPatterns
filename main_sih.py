

from fastapi import FastAPI, UploadFile, File
import xarray as xr
import numpy as np
import pandas as pd
import shutil
import os

app = FastAPI()

STATE = {"kind": None, "ds": None, "depth_name": None, "df": None, "path": None}

DEPTH_CANDIDATES = ["depth", "ZAX", "lev", "z"]


def find_depth_coord(ds):
    for c in DEPTH_CANDIDATES:
        if c in ds.coords:
            return c
    raise ValueError(f"No depth-like coordinate found. Available: {list(ds.coords)}")


def load_netcdf(path):
    ds = xr.open_dataset(path)
    depth_name = find_depth_coord(ds)
    STATE.update({"kind": "netcdf", "ds": ds, "depth_name": depth_name, "df": None, "path": path})


def load_csv(path):
    """
    Expects columns: latitude, longitude, depth, and one or more variable
    columns (e.g. temperature, salinity). Each row is one observation.
    """
    df = pd.read_csv(path)
    required = {"latitude", "longitude", "depth"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}. Found: {list(df.columns)}")
    STATE.update({"kind": "csv", "ds": None, "depth_name": "depth", "df": df, "path": path})


def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nc", ".nc4", ".netcdf"):
        load_netcdf(path)
    elif ext == ".csv":
        load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .nc or .csv")



DEFAULT_FILE = "incois_argo_10d_VAM_3530_e96d_6510_U1788916682632.nc"
if os.path.exists(DEFAULT_FILE):
    load_file(DEFAULT_FILE)


@app.post("/load_file")
async def upload_file(file: UploadFile = File(...)):
    save_path = f"uploaded_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        load_file(save_path)
    except Exception as e:
        return {"error": str(e)}
    return get_metadata()


@app.get("/metadata")
def get_metadata():
    if STATE["kind"] is None:
        return {"error": "No file loaded yet"}

    if STATE["kind"] == "netcdf":
        ds = STATE["ds"]
        return {
            "kind": "netcdf",
            "variables": list(ds.data_vars),
            "depths": sorted(ds[STATE["depth_name"]].values.tolist()),
            "file": STATE["path"],
        }
    else:  # csv
        df = STATE["df"]
        variable_cols = [c for c in df.columns if c not in ("latitude", "longitude", "depth", "time")]
        return {
            "kind": "csv",
            "variables": variable_cols,
            "depths": sorted(df["depth"].unique().tolist()),
            "file": STATE["path"],
        }


@app.get("/slice")
def get_slice(variable: str, depth: float, time_index: int = 0):
    if STATE["kind"] is None:
        return {"error": "No file loaded yet"}

    if STATE["kind"] == "netcdf":
        ds = STATE["ds"]
        depth_name = STATE["depth_name"]
        if variable not in ds.data_vars:
            return {"error": f"'{variable}' not found. Available: {list(ds.data_vars)}"}

        slice_2d = ds[variable].isel(time=time_index).sel({depth_name: depth}, method="nearest")
        raw_values = slice_2d.values
        clean_values = [[None if np.isnan(v) else float(v) for v in row] for row in raw_values]

        return {
            "variable": variable,
            "depth": float(depth),
            "latitude": ds["latitude"].values.tolist(),
            "longitude": ds["longitude"].values.tolist(),
            "values": clean_values,
        }

    else:  
        df = STATE["df"]
        tolerance = 5
        sub = df[np.abs(df["depth"] - depth) <= tolerance]
        if len(sub) == 0 or variable not in df.columns:
            return {"error": f"No data for '{variable}' near depth {depth}"}

        pivot = sub.pivot_table(index="latitude", columns="longitude", values=variable, aggfunc="mean")
        values = pivot.values
        clean_values = [[None if np.isnan(v) else float(v) for v in row] for row in values]

        return {
            "variable": variable,
            "depth": float(depth),
            "latitude": pivot.index.tolist(),
            "longitude": pivot.columns.tolist(),
            "values": clean_values,
        }


@app.get("/volume")
def get_volume(variable: str, time_index: int = 0):
    if STATE["kind"] != "netcdf":
        return {"error": "Isosurface view needs a NetCDF file (full depth range required)."}

    ds = STATE["ds"]
    depth_name = STATE["depth_name"]
    if variable not in ds.data_vars:
        return {"error": f"'{variable}' not found. Available: {list(ds.data_vars)}"}

    da = ds[variable].isel(time=time_index)

    original_depths = ds[depth_name].values
    smooth_depths = np.linspace(original_depths.min(), original_depths.max(), 40)
    da = da.interp({depth_name: smooth_depths}, method="linear")
    da = da.transpose(depth_name, "latitude", "longitude")

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    D, LA, LO = np.meshgrid(smooth_depths, lats, lons, indexing="ij")

    values = da.values

    return {
        "variable": variable,
        "x": LO.flatten().tolist(),
        "y": LA.flatten().tolist(),
        "z": (-D.flatten()).tolist(),
        "values": [None if np.isnan(v) else float(v) for v in values.flatten()],
    }
