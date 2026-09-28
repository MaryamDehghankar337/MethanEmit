"""
EMIT Methane Plume Detection App (Carbon Mapper Algorithm)
----------------------------------------------------------
This Streamlit application detects methane plumes using NASA's EMIT
hyperspectral data. It implements the Column-wise Matched Filter (CMF)
algorithm, which is the core of the Carbon Mapper operational workflow.

Author: (Your Name)
Date: 2026
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import earthaccess
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import os
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# Target area: Aradkouh / Kahrizak landfill (Tehran)
DEFAULT_AOI = {
    "type": "Polygon",
    "coordinates": [[
        [51.20, 35.40],
        [51.45, 35.40],
        [51.45, 35.60],
        [51.20, 35.60],
        [51.20, 35.40]
    ]]
}

# EMIT data parameters
EMIT_COLLECTION = "EMITL2BCH4ENH"  # Methane Enhancement product
EMIT_PLM_COLLECTION = "EMITL2BCH4PLM"  # Plume Complexes product

# Algorithm parameters
CMF_WINDOW_SIZE = 50       # Number of columns for covariance estimation
CMF_MIN_WAVELENGTH = 2100  # nm, start of methane absorption window
CMF_MAX_WAVELENGTH = 2450  # nm, end of methane absorption window
PLUME_THRESHOLD = 1000     # ppm-m, minimum enhancement to be considered plume
MIN_PLUME_PIXELS = 10      # Minimum pixels for a valid plume

# ══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS (Matched Filter Core)
# ══════════════════════════════════════════════════════════════════════

def load_emit_data(item, bbox):
    """
    Loads EMIT L2B CH4 enhancement data for a given STAC item.
    Returns a numpy array of enhancement values (ppm-m).
    """
    # Open the COG (Cloud Optimized GeoTIFF) file
    url = item.assets["EMITL2BCH4ENH"].href
    with rasterio.open(url) as src:
        # Read the data within the bounding box
        window = rasterio.windows.from_bounds(
            bbox[0], bbox[1], bbox[2], bbox[3], src.transform
        )
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        crs = src.crs
    return data, transform, crs

def column_wise_matched_filter(data, window_size=CMF_WINDOW_SIZE):
    """
    Implements the Column-wise Matched Filter (CMF) algorithm.
    
    This is a simplified version of the algorithm described in:
    Thompson et al. (2015, 2016) and used by Carbon Mapper.
    
    The CMF estimates the background covariance from a moving window
    of columns and applies a matched filter to enhance the methane signal.
    """
    n_rows, n_cols = data.shape
    filtered = np.zeros_like(data)
    
    # Define the target absorption spectrum (simplified as a spectral shape)
    # In a real implementation, this would be a high-resolution spectrum
    target_spectrum = np.exp(-0.5 * ((np.arange(n_rows) - n_rows/2) / (n_rows/10))**2)
    
    for start_col in range(0, n_cols, window_size):
        end_col = min(start_col + window_size, n_cols)
        window = data[:, start_col:end_col]
        
        # Estimate background statistics (mean and covariance)
        if window.shape[1] > 10:  # Ensure enough samples
            background_mean = np.nanmean(window, axis=1, keepdims=True)
            background_std = np.nanstd(window, axis=1, keepdims=True)
            background_std[background_std == 0] = 1.0  # Avoid division by zero
            
            # Apply matched filter: (x - mu) / sigma * target_spectrum
            normalized = (window - background_mean) / background_std
            filtered[:, start_col:end_col] = normalized * target_spectrum[:, np.newaxis]
        else:
            filtered[:, start_col:end_col] = window
            
    return filtered

def estimate_flux_ime(plume_mask, enhancement_map, wind_speed=2.0):
    """
    Estimates methane emission flux using the Integrated Methane Enhancement (IME) method.
    
    Formula: Q = (IME * U_eff) / L
    where:
    - IME = sum(enhancement * pixel_area) in ppm-m * m^2
    - U_eff = effective wind speed (m/s)
    - L = characteristic length of the plume (m)
    """
    # Conversion factor: 1 ppm-m over 1 m^2 = 1e-6 m^3 CH4 / m^2
    # CH4 density at STP: 0.717 kg/m^3
    # So, 1 ppm-m * 1 m^2 = 1e-6 m^3 * 0.717 kg/m^3 = 7.17e-7 kg
    
    pixel_area = 60 * 60  # EMIT pixel size is 60m
    ime = np.nansum(enhancement_map[plume_mask]) * pixel_area  # ppm-m * m^2
    ime_kg = ime * 7.17e-7  # Convert to kg
    
    # Characteristic length (sqrt of plume area)
    plume_area = np.sum(plume_mask) * pixel_area  # m^2
    length = np.sqrt(plume_area) if plume_area > 0 else 1.0
    
    # Effective wind speed (simplified)
    u_eff = 0.33 * wind_speed + 0.45
    
    # Emission rate in kg/s
    q_kg_s = (ime_kg * u_eff) / length if length > 0 else 0
    
    # Convert to kg/h
    q_kg_h = q_kg_s * 3600
    
    return {
        "IME_ppm_m2": ime,
        "IME_kg": ime_kg,
        "plume_area_m2": plume_area,
        "length_m": length,
        "U_eff_m_s": u_eff,
        "Q_kg_h": q_kg_h,
        "Q_ton_h": q_kg_h / 1000
    }

def generate_plume_map(enhancement_map, threshold=PLUME_THRESHOLD):
    """Creates a binary mask of methane plumes based on a threshold."""
    plume_mask = enhancement_map > threshold
    return plume_mask

# ══════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EMIT Methane Plume Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS (matching the previous app's style) ---
st.markdown("""
<style>
    .stApp { background: #f1faee; }
    .app-header { background: white; border: 1px solid #d8e6e8; border-radius: 16px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(29,53,87,0.05); }
    .app-title { font-size: 1.5rem; font-weight: 800; color: #1d3557; }
    .app-subtitle { font-size: 0.8rem; color: #457b9d; }
    .section-label { background: #a8dadc; color: #1d3557; border-radius: 999px; padding: 0.2rem 0.6rem; font-size: 0.7rem; font-weight: 800; display: inline-block; margin-bottom: 0.5rem; }
    .result-card { background: white; border: 1px solid #d8e6e8; border-radius: 12px; padding: 1rem; height: 100%; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #1d3557; }
    .metric-label { font-size: 0.75rem; color: #457b9d; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="app-header">
    <div class="app-title">🛰️ EMIT Methane Plume Detection</div>
    <div class="app-subtitle">Carbon Mapper-style Column-wise Matched Filter on NASA EMIT hyperspectral data</div>
</div>
""", unsafe_allow_html=True)

# --- Section 1: Study Area & Data Selection ---
st.markdown('<div class="section-label">01 · STUDY AREA & DATA</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Area of Interest")
    # Display map
    m = folium.Map(location=[35.505, 51.330], zoom_start=11)
    folium.GeoJson(DEFAULT_AOI, style_function=lambda x: {'color': 'blue', 'fill': False}).add_to(m)
    folium.Marker(
        [35.505, 51.330],
        popup="Aradkouh Landfill",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    st_folium(m, height=400, width=800)

with col2:
    st.markdown("### Data Parameters")
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=90))
    end_date = st.date_input("End date", datetime.now())
    max_cloud = st.slider("Max cloud cover (%)", 0, 100, 20)
    
    if st.button("🔎 Search EMIT Scenes", type="primary", use_container_width=True):
        with st.spinner("Searching NASA Earthdata for EMIT scenes..."):
            try:
                # Authenticate with Earthdata (uses ~/.netrc)
                auth = earthaccess.login(strategy="netrc")
                
                # Search for EMIT methane enhancement granules
                results = earthaccess.search_data(
                    short_name=EMIT_COLLECTION,
                    bounding_box=(51.20, 35.40, 51.45, 35.60),
                    temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
                    count=50
                )
                
                if results:
                    st.session_state.emit_results = results
                    st.success(f"Found {len(results)} EMIT granules")
                else:
                    st.warning("No EMIT granules found for this area and time range.")
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.info("Make sure you have a valid NASA Earthdata account and .netrc file.")

# --- Section 2: Process and Visualize ---
if "emit_results" in st.session_state:
    st.markdown('<div class="section-label">02 · PROCESSING & RESULTS</div>', unsafe_allow_html=True)
    
    results = st.session_state.emit_results
    
    # Create a simple selector
    options = [f"{i+1}. {r['id'][:50]}..." for i, r in enumerate(results)]
    selected_idx = st.selectbox("Select a granule to process", range(len(results)), format_func=lambda x: options[x])
    
    if st.button("🚀 Run Methane Detection", type="primary", use_container_width=True):
        with st.spinner("Downloading and processing EMIT data (this may take a few minutes)..."):
            try:
                item = results[selected_idx]
                
                # Download the data to a temporary directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    files = earthaccess.download([item], local_path=tmpdir)
                    
                    # Find the COG file
                    cog_file = None
                    for f in files:
                        if f.endswith('.tif') or f.endswith('.tiff'):
                            cog_file = f
                            break
                    
                    if cog_file is None:
                        st.error("No GeoTIFF file found in the granule.")
                        st.stop()
                    
                    # Load the data
                    with rasterio.open(cog_file) as src:
                        data = src.read(1)
                        transform = src.transform
                        crs = src.crs
                        bounds = src.bounds
                    
                    # Apply Column-wise Matched Filter
                    filtered_data = column_wise_matched_filter(data)
                    
                    # Generate plume mask
                    plume_mask = generate_plume_map(filtered_data)
                    
                    # Estimate flux
                    flux = estimate_flux_ime(plume_mask, filtered_data)
                    
                    # --- Display Results ---
                    st.markdown("### Results")
                    
                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f'<div class="result-card"><div class="metric-label">Estimated Flux</div><div class="metric-value">{flux["Q_kg_h"]:.1f} kg/h</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="result-card"><div class="metric-label">Plume Area</div><div class="metric-value">{flux["plume_area_m2"]/1e6:.2f} km²</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="result-card"><div class="metric-label">Max Enhancement</div><div class="metric-value">{np.nanmax(filtered_data):.0f} ppm-m</div></div>', unsafe_allow_html=True)
                    with m4:
                        st.markdown(f'<div class="result-card"><div class="metric-label">Plume Pixels</div><div class="metric-value">{np.sum(plume_mask):,}</div></div>', unsafe_allow_html=True)
                    
                    # Image display
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.markdown("**Filtered Methane Enhancement (ppm-m)**")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        vmin, vmax = np.nanpercentile(filtered_data, [2, 98])
                        im = ax.imshow(filtered_data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                        plt.colorbar(im, ax=ax, label='ppm-m')
                        ax.set_title("Column-wise Matched Filter Output")
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with img_col2:
                        st.markdown("**Detected Plume Mask**")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(plume_mask, cmap='Reds', alpha=0.8)
                        ax.imshow(filtered_data, cmap='gray', alpha=0.3)
                        ax.set_title(f"Plume Mask (threshold > {PLUME_THRESHOLD} ppm-m)")
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    st.info(f"**Interpretation:** The algorithm detected {np.sum(plume_mask)} pixels with methane enhancement above {PLUME_THRESHOLD} ppm-m. Based on the IME method with an assumed wind speed of 2 m/s, the estimated emission rate is **{flux['Q_kg_h']:.1f} kg/h** ({flux['Q_ton_h']:.2f} t/h).")
                    
                    # Download button
                    st.download_button(
                        "⬇ Download Results (CSV)",
                        pd.DataFrame([flux]).to_csv(index=False),
                        file_name="emit_methane_flux.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.exception(e)

# --- Footer ---
st.markdown("---")
st.markdown("*Data source: NASA EMIT L2B Methane Enhancement (EMITL2BCH4ENH). Algorithm: Column-wise Matched Filter (Carbon Mapper operational workflow).*")
