import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Arabian Sea 3D Viewer", layout="wide")
st.title("Arabian Sea — 3D Ocean Viewer")

API_BASE = "http://127.0.0.1:8002"


@st.cache_resource
def load_model():
    gpr = joblib.load("ocean_model.pkl")
    scaler = joblib.load("ocean_scaler.pkl")
    return gpr, scaler

gpr, scaler = load_model()

def query_point(lat, lon, pressure, day_of_year):
    X_q = scaler.transform([[lat, lon, pressure, day_of_year]])
    mean, std = gpr.predict(X_q, return_std=True)
    return float(mean[0]), float(std[0])


st.sidebar.markdown("### Data source")
new_file = st.sidebar.file_uploader("Upload a NetCDF (.nc) or CSV file", type=["nc", "csv"])
if new_file is not None:
    if st.sidebar.button("Load this file"):
        files = {"file": (new_file.name, new_file.getvalue())}
        resp = requests.post(f"{API_BASE}/load_file", files=files)
        result = resp.json()
        if "error" in result:
            st.sidebar.error(result["error"])
        else:
            st.sidebar.success(f"Loaded {result['kind']} file with variables: {result['variables']}")

meta = requests.get(f"{API_BASE}/metadata").json()
if "error" in meta:
    st.error("No dataset loaded on the backend yet.")
    st.stop()

st.sidebar.markdown("---")
variable = st.sidebar.selectbox("Variable", meta["variables"])
depth = st.sidebar.select_slider("Depth (m)", options=meta["depths"], value=meta["depths"][len(meta["depths"]) // 2])

st.sidebar.markdown("---")
float_csv = st.sidebar.file_uploader("ARGO float points (CSV: lat, lon, depth, value)", type="csv", key="argo_overlay")
float_df = None
if float_csv is not None:
    float_df = pd.read_csv(float_csv)
    tolerance = 25
    float_df = float_df[np.abs(float_df["depth"] - depth) <= tolerance]

st.sidebar.markdown("---")
view_mode = st.sidebar.radio(
    "View mode",
    ["Depth slice", "Isosurface (full volume)", "Continuous Depth Sweep"]
)

lat, lon = [15.0], [65.0]


if view_mode == "Isosurface (full volume)":
    vol = requests.get(f"{API_BASE}/volume", params={"variable": variable}).json()
    if "error" in vol:
        st.error(vol["error"])
        st.stop()

    raw_values = np.array([v if v is not None else np.nan for v in vol["values"]])
    finite_vals = raw_values[~np.isnan(raw_values)]

    if finite_vals.size == 0:
        st.error(f"'{variable}' has no finite values in the returned volume — nothing to render.")
        st.stop()

    n_finite = vol.get("n_finite", finite_vals.size)
    n_total = vol.get("n_total", raw_values.size)
    if n_total:
        coverage_pct = 100.0 * n_finite / n_total
        if coverage_pct < 5:
            st.warning(
                f"Only {coverage_pct:.1f}% of the volume has finite data for '{variable}' "
                f"({n_finite}/{n_total} points). The isosurface may look sparse or empty "
                f"depending on the depth range you pick below."
            )

    x_arr = np.array(vol["x"], dtype=float)
    y_arr = np.array(vol["y"], dtype=float)
    z_arr = np.array(vol["z"], dtype=float)
    with st.expander("Diagnostics (volume payload)", expanded=True):
        st.write(
            {
                "n_finite / n_total": f"{n_finite} / {n_total}",
                "x (longitude) range": f"{np.nanmin(x_arr):.3f} to {np.nanmax(x_arr):.3f}",
                "y (latitude) range": f"{np.nanmin(y_arr):.3f} to {np.nanmax(y_arr):.3f}",
                "z (depth, negated) range": f"{np.nanmin(z_arr):.3f} to {np.nanmax(z_arr):.3f}",
                "x/y/z NaN counts": f"{np.isnan(x_arr).sum()} / {np.isnan(y_arr).sum()} / {np.isnan(z_arr).sum()}",
            }
        )
        if abs(np.nanmax(x_arr)) <= 2 and abs(np.nanmax(y_arr)) <= 2:
            st.error(
                "x/y coordinates look like a tiny placeholder range (roughly -1 to 1), not real "
                "longitude/latitude. This means the backend's /volume endpoint is likely still "
                "running old code — make sure main_SIH.py was saved and the FastAPI server was "
                "fully restarted (not just reloaded), then refresh this page."
            )

    vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))

    isomin, isomax = st.sidebar.slider(
        f"{variable} range to emphasize",
        min_value=vmin, max_value=vmax,
        value=(vmin, vmax),
    )

    render_mode = st.sidebar.radio(
        "Volume render style",
        ["Point cloud (always shows something)", "Volume (opacity-mapped voxels)"],
        help="Point cloud only needs individual finite points and can't render blank. "
             "Volume needs a well-formed grid and may render nothing if coverage is patchy.",
    )

    if render_mode == "Point cloud (always shows something)":

        point_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr) & np.isfinite(raw_values)
        if point_mask.sum() == 0:
            st.error("No points have finite x, y, z, and value all at once — nothing can be plotted.")
            st.stop()

        # Downsample if huge, so the browser doesn't choke on tens of thousands of markers.
        idx = np.where(point_mask)[0]
        if idx.size > 15000:
            idx = np.random.choice(idx, size=15000, replace=False)

        fig = go.Figure(data=go.Scatter3d(
            x=x_arr[idx], y=y_arr[idx], z=z_arr[idx],
            mode="markers",
            marker=dict(
                size=3,
                color=raw_values[idx],
                colorscale="Viridis",
                colorbar=dict(title=variable),
                opacity=0.7,
                cmin=isomin, cmax=isomax,
            ),
        ))
        fig.update_layout(
            scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth (m)"),
            height=700, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Point cloud of {variable} — {idx.size} finite points plotted (of {n_total} grid cells).")

    else:

        fig = go.Figure(data=go.Volume(
            x=vol["x"], y=vol["y"], z=vol["z"],
            value=raw_values,
            isomin=isomin, isomax=isomax,
            opacity=0.12,        
            surface_count=17,      
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        fig.update_layout(
            scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth (m)"),
            height=700, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Volume rendering of {variable} across the full water column ({isomin:.2f}–{isomax:.2f})")


elif view_mode == "Continuous Depth Sweep":
    with st.spinner("Preparing animation frames..."):
        first_d = meta["depths"][0]
        base_res = requests.get(f"{API_BASE}/slice", params={"variable": variable, "depth": first_d}).json()

        if "error" in base_res:
            st.error(base_res["error"])
            st.stop()

        lat, lon = base_res["latitude"], base_res["longitude"]
        vals = np.array(base_res["values"], dtype=float)
        z_p = np.full_like(vals, fill_value=-base_res["depth"])

        fig = go.Figure(data=[go.Surface(
            x=lon, y=lat, z=z_p,
            surfacecolor=vals,
            colorscale="Viridis",
            colorbar=dict(title=variable),
        )])

        frames = []
        for current_d in meta["depths"]:
            slice_res = requests.get(f"{API_BASE}/slice", params={"variable": variable, "depth": current_d}).json()
            if "error" not in slice_res:
                s_vals = np.array(slice_res["values"], dtype=float)
                s_zp = np.full_like(s_vals, fill_value=-slice_res["depth"])
                frames.append(go.Frame(
                    data=[go.Surface(x=slice_res["longitude"], y=slice_res["latitude"], z=s_zp, surfacecolor=s_vals)],
                    name=f"frame_{current_d}",
                ))

        fig.frames = frames
        z_min = -max(meta["depths"])
        z_max = -min(meta["depths"])

        fig.update_layout(
            scene=dict(
                xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth (m)",
                zaxis=dict(range=[z_min, z_max]),
            ),
            updatemenus=[{
                "type": "buttons", "showactive": False, "y": 1.05, "x": 0.0,
                "xanchor": "left", "yanchor": "top",
                "buttons": [
                    {"label": "▶ Play Depth Sweep", "method": "animate",
                     "args": [None, {"frame": {"duration": 300, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 100}}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }],
            height=700, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Click ▶ Play Depth Sweep to animate through all depth levels.")


else:
    response = requests.get(f"{API_BASE}/slice", params={"variable": variable, "depth": depth})
    data = response.json()

    if "error" in data:
        st.error(data["error"])
        st.stop()

    lat, lon = data["latitude"], data["longitude"]
    values = np.array(data["values"], dtype=float)
    z_plane = np.full_like(values, fill_value=-data["depth"])

    fig = go.Figure(data=go.Surface(
        x=lon, y=lat, z=z_plane,
        surfacecolor=values,
        colorscale="Viridis",
        colorbar=dict(title=variable),
    ))

    if float_df is not None and len(float_df) > 0:
        fig.add_trace(go.Scatter3d(
            x=float_df["lon"], y=float_df["lat"], z=-float_df["depth"],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="ARGO floats",
            text=[f"Observed: {v:.2f}" for v in float_df["value"]],
            hoverinfo="text",
        ))

    fig.update_layout(
        scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth (m)"),
        height=700, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing {variable} at {depth}m depth — source: {meta['kind']} ({meta['file']})")


with st.expander("Estimate where no float is nearby (GPR)"):
    c1, c2, c3, c4 = st.columns(4)
    q_lat = c1.number_input("Latitude", value=float(np.mean(lat)))
    q_lon = c2.number_input("Longitude", value=float(np.mean(lon)))
    q_depth = c3.number_input("Depth (m)", value=float(depth))
    q_day = c4.number_input("Day of year", value=150, min_value=1, max_value=365)

    if st.button("Get GPR estimate"):
        LAT_MIN, LAT_MAX = min(lat), max(lat)
        LON_MIN, LON_MAX = min(lon), max(lon)
        if not (LAT_MIN <= q_lat <= LAT_MAX and LON_MIN <= q_lon <= LON_MAX):
            st.warning(
                f"⚠️ Outside data coverage (lat {LAT_MIN:.1f}-{LAT_MAX:.1f}°N, lon {LON_MIN:.1f}-{LON_MAX:.1f}°E). "
                f"Showing the model's best guess anyway — treat this as unreliable extrapolation."
            )
        mean, std = query_point(q_lat, q_lon, q_depth, q_day)
        st.success(f"Estimated: {mean:.2f}°C ± {std:.2f}°C")
