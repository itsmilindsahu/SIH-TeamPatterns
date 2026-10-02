# SIH-TeamPatterns — Arabian Sea 3D Ocean Viewer

> **Live Demo:** [https://itsmilindsahu.github.io/SIH-TeamPatterns/](https://itsmilindsahu.github.io/SIH-TeamPatterns/)

An interactive 3D visualization tool for Arabian Sea oceanographic data, built for **Smart India Hackathon (SIH)** by Team Patterns.

## Features

| Feature | Description |
|---------|-------------|
| 🗺️ **Depth Slice** | 3D surface plot of any variable at a chosen depth level |
| 🫧 **Point Cloud** | Full-volume scatter across all depths, color-coded by value |
| 🎬 **Depth Sweep** | Animated playback sweeping through all depth levels |
| 📡 **ARGO Overlay** | Overlay observed ARGO float positions on any view |
| 📊 **CSV Upload** | Upload your own lat/lon/depth data |
| 🌊 **Sample Data** | Built-in Arabian Sea sample dataset to explore immediately |

## Usage

### Online (GitHub Pages)

Open the live link above — no installation needed.

1. Click **Load Sample Data** or upload a CSV file
2. Pick a variable (e.g. `temperature`, `salinity`)
3. Adjust the depth slider
4. Switch between view modes

### CSV Format

Your CSV must have at minimum:

```csv
latitude,longitude,depth,temperature,salinity
8.0,52.0,0,28.1,36.9
8.0,52.0,50,25.3,37.0
...
```

### Local Development (Python backend)

For NetCDF (`.nc`) support and the GPR estimation model, run the full Python stack locally:

```bash
pip install fastapi uvicorn xarray numpy pandas streamlit plotly scikit-learn joblib

# Terminal 1 — FastAPI backend (port 8002)
uvicorn main_sih:app --port 8002 --reload

# Terminal 2 — Streamlit frontend
streamlit run viewer_app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Tech Stack

| Layer | GitHub Pages | Local |
|-------|-------------|-------|
| Frontend | HTML + Plotly.js | Streamlit + Plotly |
| Backend | — (client-side JS) | FastAPI + xarray |
| Data | CSV | NetCDF / CSV |
| ML | — | scikit-learn GPR |

## Enable GitHub Pages

1. Go to **Settings → Pages** in your repository
2. Under **Source**, select **GitHub Actions**
3. Push to `main` — the workflow deploys automatically