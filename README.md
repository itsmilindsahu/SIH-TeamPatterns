# SIH-TeamPatterns — Arabian Sea 3D Ocean Viewer

A full-stack ocean data visualization tool built for **Smart India Hackathon (SIH)** by Team Patterns.

## Overview

This project provides an interactive 3D viewer for Arabian Sea oceanographic data sourced from **INCOIS ARGO float** datasets. It supports both NetCDF (`.nc`) and CSV formats, offering multiple visualization modes.

## Components

### `main_sih.py` — FastAPI Backend
A REST API server that loads and serves ocean data:
- `POST /load_file` — Upload a `.nc` or `.csv` file
- `GET /metadata` — Returns variables, depth levels, and file info
- `GET /slice` — Returns a 2D lat/lon slice at a given depth and time index
- `GET /volume` — Returns full 3D volume data for isosurface rendering

Supports auto-loading of a default NetCDF file on startup.

### `viewer_app.py` — Streamlit Frontend
An interactive dashboard built with Streamlit and Plotly:
- **Depth Slice View** — 3D surface plot at a selected depth level
- **Isosurface (Full Volume)** — Point cloud or opacity-mapped volumetric rendering
- **Continuous Depth Sweep** — Animated playback through all depth levels
- **ARGO Float Overlay** — Overlay observed float data on the map
- **GPR Estimation** — Gaussian Process Regression model to estimate values at unobserved locations

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, xarray, NumPy, pandas |
| Frontend | Streamlit, Plotly, requests |
| ML Model | scikit-learn (GPR), joblib |
| Data Format | NetCDF4, CSV |

## Getting Started

```bash
# Install dependencies
pip install fastapi uvicorn xarray numpy pandas streamlit plotly scikit-learn joblib

# Start the backend API (port 8002)
uvicorn main_sih:app --port 8002 --reload

# In a separate terminal, start the Streamlit frontend
streamlit run viewer_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Data Format (CSV)

CSV files must contain at minimum: `latitude`, `longitude`, `depth`, plus one or more variable columns (e.g., `temperature`, `salinity`).