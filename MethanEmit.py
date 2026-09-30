"""EMIT Methane Plume Detection App.

Carbon Mapper-style methane detection on NASA EMIT hyperspectral data.
UI/design preserved from the Sentinel-2 app.
"""
from __future__ import annotations

import io
import os
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
from folium.plugins import Draw, MousePosition
from shapely.geometry import box, mapping, shape, Point
from shapely.ops import unary_union
from streamlit_folium import st_folium

try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

RESOLUTION = 60  # EMIT native pixel size (m)

DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)

EMIT_ENH_COLLECTION = "EMITL2BCH4ENH"   # Methane Enhancement (ppm·m)
EMIT_PLM_COLLECTION = "EMITL2BCH4PLM"   # Plume Complexes

PARAMS = {
    "plume_threshold_ppm_m": 1000.0,
    "min_plume_pixels": 10,
    "wind_speed_m_s": 2.0,
    "max_plume_area_km2": 100.0,
}

PPB_TO_KG_M2 = 5.72e-6
ALPHA_IME = 0.33
BETA_IME = 0.45
CH4_DENSITY_KG_M3 = 0.717


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════

def normalize_geometry(obj):
    if obj is None:
        return None
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "Feature":
        return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geoms = []
        for feature in obj.get("features", []):
            g = normalize_geometry(feature.get("geometry"))
            if g:
                geoms.append(shape(g))
        return mapping(unary_union(geoms)) if geoms else None
    try:
        g = shape(obj)
        return mapping(g) if not g.is_empty else None
    except Exception:
        return None


def ensure_aoi(obj):
    return normalize_geometry(obj) or mapping(DEFAULT_AOI)


def aoi_bounds(aoi):
    return shape(ensure_aoi(aoi)).bounds


def compute_zoom(bounds):
    try:
        minx, miny, maxx, maxy = bounds
        span = max(maxx - minx, maxy - miny, 1e-6)
        zoom = int(round(math.log2(360.0 / span))) - 1
        return max(3, min(15, zoom))
    except Exception:
        return 11


def create_map(aoi):
    geometry = shape(ensure_aoi(aoi))
    centroid = geometry.centroid
    zoom = compute_zoom(geometry.bounds)
    fmap = folium.Map(
        [centroid.y, centroid.x],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )
    folium.GeoJson(
        mapping(geometry),
        style_function=lambda _: {"color": "blue", "fill": False, "weight": 2},
    ).add_to(fmap)
    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
            },
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)
    MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="📍 Lat, Lon:",
        lat_formatter="function(num) {return num.toFixed(5);}",
        lng_formatter="function(num) {return num.toFixed(5);}",
    ).add_to(fmap)
    return fmap


# ══════════════════════════════════════════════════════════════════════
#  GEOCODING
# ══════════════════════════════════════════════════════════════════════

def geocode_place(query: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "polygon_geojson": 1,
        }
        headers = {"User-Agent": "EMIT-Methane-App/1.0 (streamlit)"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None
        item = data[0]
        lat = float(item["lat"])
        lon = float(item["lon"])
        label = item.get("display_name", query)

        gj = item.get("geojson")
        if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
            try:
                geom = shape(gj)
                if not geom.is_empty:
                    return geom, (lat, lon), label
            except Exception:
                pass

        bb = item.get("boundingbox")
        if bb:
            south, north, west, east = [float(x) for x in bb]
            return box(west, south, east, north), (lat, lon), label

        d = 0.02
        return box(lon - d, lat - d, lon + d, lat + d), (lat, lon), label
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════════════════
#  EARTHDATA AUTH
# ══════════════════════════════════════════════════════════════════════

def login_earthdata():
    if not EARTHACCESS_AVAILABLE:
        raise RuntimeError(
            "Package 'earthaccess' is not installed. "
            "Please check requirements.txt."
        )
    try:
        username = st.secrets["EARTHDATA_USERNAME"]
        password = st.secrets["EARTHDATA_PASSWORD"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "Earthdata credentials are not configured. "
            "Add EARTHDATA_USERNAME and EARTHDATA_PASSWORD to Streamlit secrets."
        )

    os.environ["EARTHDATA_USERNAME"] = username
    os.environ["EARTHDATA_PASSWORD"] = password

    try:
        auth = earthaccess.login(strategy="environment")
    except Exception as e:
        raise RuntimeError(f"Earthdata login failed: {e}")

    if not auth.authenticated:
        raise RuntimeError(
            "Earthdata did not accept the credentials. "
            "Check your username/password or register at urs.earthdata.nasa.gov."
        )
    return auth


# ══════════════════════════════════════════════════════════════════════
#  EMIT SEARCH & LOADING
# ══════════════════════════════════════════════════════════════════════

def search_emit_granules(aoi, start_date, end_date):
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    results = earthaccess.search_data(
        short_name=EMIT_ENH_COLLECTION,
        bounding_box=(minx, miny, maxx, maxy),
        temporal=(start_date.strftime("%Y-%m-%d"),
                  end_date.strftime("%Y-%m-%d")),
        count=200,
    )
    return list(results)


def granule_datetime(granule) -> Optional[datetime]:
    try:
        umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    except Exception:
        umm = {}
    temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
    dt_str = temporal.get("BeginningDateTime")
    if dt_str:
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    try:
        gid = granule.get("meta", {}).get("native-id", "")
        for part in gid.split("_"):
            if len(part) >= 15 and part[:8].isdigit():
                return datetime.strptime(part[:15], "%Y%m%dT%H%M%S")
    except Exception:
        pass
    return None


def granule_cloud(granule) -> float:
    try:
        umm = granule.get("umm", {})
        for attr in umm.get("AdditionalAttributes", []):
            if attr.get("Name") == "CloudCover":
                vals = attr.get("Values", [])
                if vals:
                    return float(vals[0])
    except Exception:
        pass
    return 0.0


def load_emit_enhancement(granule, aoi):
    """Load EMIT enhancement clipped to AOI, with nodata masked to NaN."""
    try:
        files = earthaccess.open([granule])
    except Exception as e:
        raise RuntimeError(f"Failed to open granule stream: {e}")

    if not files:
        raise RuntimeError("No files returned by earthaccess.open().")

    tif_path = None
    for f in files:
        name = getattr(f, "path", str(f))
        if name.lower().endswith((".tif", ".tiff")):
            tif_path = f
            break
    if tif_path is None:
        tif_path = files[0]

    minx, miny, maxx, maxy = aoi_bounds(aoi)

    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        try:
            from rasterio.windows import from_bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            crs = src.crs
        except Exception:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

    data = data.astype(np.float32)

    if nodata is not None:
        try:
            nd = float(nodata)
            data = np.where(np.isclose(data, nd, rtol=0, atol=1e-3),
                            np.nan, data)
        except Exception:
            pass

    for fv in (-9999.0, -999.0, -99999.0):
        data = np.where(np.isclose(data, fv, rtol=0, atol=1e-3),
                        np.nan, data)

    data = np.where(np.abs(data) > 1e6, np.nan, data)

    try:
        from rasterio.features import geometry_mask
        geom_mask = geometry_mask(
            [shape(ensure_aoi(aoi))],
            out_shape=data.shape,
            transform=transform,
            invert=True,
        )
        data = np.where(geom_mask, data, np.nan)
    except Exception:
        pass

    return data, transform, crs


# ══════════════════════════════════════════════════════════════════════
#  COVERAGE HELPERS
# ══════════════════════════════════════════════════════════════════════

def valid_coverage(data):
    if data is None or data.size == 0:
        return 0.0
    return float(np.isfinite(data).sum()) / float(data.size)


def coverage_badge(data):
    c = valid_coverage(data) * 100
    if c < 5:
        return f"⚠️ Very low coverage: {c:.1f}% of AOI", "#e63946"
    if c < 20:
        return f"⚠️ Partial coverage: {c:.1f}% of AOI", "#f4a261"
    return f"✓ Good coverage: {c:.1f}% of AOI", "#2a9d8f"


# ══════════════════════════════════════════════════════════════════════
#  ALGORITHM
# ══════════════════════════════════════════════════════════════════════

def detect_plume(enhancement, threshold_ppm_m, min_pixels):
    from scipy.ndimage import (
        label as nd_label,
        binary_opening,
        binary_closing,
    )

    finite = np.isfinite(enhancement)
    candidate = finite & (enhancement > threshold_ppm_m)
    plume = np.zeros_like(candidate, dtype=bool)

    if not candidate.any():
        return plume

    structure = np.ones((3, 3), dtype=np.uint8)
    candidate = binary_opening(candidate, structure=structure, iterations=1)
    candidate = binary_closing(candidate, structure=structure, iterations=1)

    if not candidate.any():
        return plume

    labeled, n = nd_label(candidate, structure=structure)
    if n == 0:
        return plume

    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    sizes[0] = 0
    keep = sizes >= min_pixels
    keep[0] = False
    if keep.any():
        plume = keep[labeled]
    return plume


def estimate_flux_ime(enhancement, plume_mask, wind_speed_m_s):
    empty = {
        "Q_kg_h": 0.0, "Q_ton_h": 0.0,
        "IME_ppm_m2": 0.0, "IME_kg": 0.0,
        "plume_area_m2": 0.0, "length_m": 0.0,
        "U_eff_m_s": 0.0, "n_pixels": 0,
        "max_enhancement": 0.0, "mean_enhancement": 0.0,
    }
    if plume_mask is None or not plume_mask.any():
        return empty

    valid_plume = plume_mask & np.isfinite(enhancement)
    n_pix = int(valid_plume.sum())
    if n_pix == 0:
        return empty

    pixel_area = RESOLUTION * RESOLUTION
    vals = np.where(valid_plume, enhancement, 0.0)
    IME_ppm_m2 = float(np.sum(vals) * pixel_area)
    IME_kg = IME_ppm_m2 * 1e-6 * CH4_DENSITY_KG_M3

    A_plume = n_pix * pixel_area
    L = float(np.sqrt(A_plume)) if A_plume > 0 else 1.0
    U_eff = ALPHA_IME * wind_speed_m_s + BETA_IME

    Q_kg_s = U_eff * IME_kg / L if L > 0 else 0.0
    Q_kg_h = Q_kg_s * 3600.0

    plume_vals = enhancement[valid_plume]

    return {
        "Q_kg_h": Q_kg_h,
        "Q_ton_h": Q_kg_h / 1000.0,
        "IME_ppm_m2": IME_ppm_m2,
        "IME_kg": IME_kg,
        "plume_area_m2": A_plume,
        "length_m": L,
        "U_eff_m_s": U_eff,
        "n_pixels": n_pix,
        "max_enhancement": float(np.nanmax(plume_vals)),
        "mean_enhancement": float(np.nanmean(plume_vals)),
    }


# ══════════════════════════════════════════════════════════════════════
#  IMAGE RENDERING
# ══════════════════════════════════════════════════════════════════════

def _compute_vrange(data):
    finite = np.isfinite(data)
    if not finite.any():
        return 0.0, 1.0
    values = data[finite]
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def enhancement_png(
    array,
    mask=None,
    colormap="turbo",
    show_outline=True,
    outline_color=(255, 255, 0),
    vmin=None,
    vmax=None,
):
    """Render enhancement with optional plume overlay + outline."""
    from PIL import Image
    import matplotlib.pyplot as plt

    data = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(data)
    rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)

    if vmin is None or vmax is None:
        vmin, vmax = _compute_vrange(data)

    if finite.any() and vmax > vmin:
        norm = np.clip(
            (np.nan_to_num(data, nan=vmin) - vmin) / (vmax - vmin), 0, 1
        )
        cmap = plt.get_cmap(colormap)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        rgb[~finite] = 255

    if mask is not None and mask.any():
        overlay = np.zeros((*data.shape, 4), dtype=np.uint8)
        overlay[..., 0] = 230
        overlay[..., 1] = 40
        overlay[..., 2] = 40
        overlay[..., 3] = np.where(mask, 160, 0).astype(np.uint8)
        base = Image.fromarray(rgb).convert("RGBA")
        over = Image.fromarray(overlay, mode="RGBA")
        rgb = np.array(Image.alpha_composite(base, over).convert("RGB"))

        if show_outline:
            try:
                from scipy.ndimage import binary_erosion, binary_dilation
                eroded = binary_erosion(mask, iterations=1)
                boundary = mask & ~eroded
                boundary = binary_dilation(boundary, iterations=1)
                rgb[boundary] = outline_color
            except Exception:
                pass

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def colorbar_png(vmin, vmax, colormap="turbo", label="CH₄ enhancement (ppm·m)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(0.9, 3.2), dpi=100)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=colormap, norm=norm, orientation="vertical"
    )
    cb.set_label(label, fontsize=8, color="#111111")
    cb.ax.tick_params(labelsize=7, colors="#111111")
    cb.outline.set_edgecolor("#555555")
    fig.patch.set_facecolor("#ffffff")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


def legend_html(kind, vmin=None, vmax=None, n_pixels=None, mean_enh=None):
    if kind == "plume":
        rows = [
            ("#e63946", "Detected plume (fill)"),
            ("#ffff00", "Plume boundary"),
            ("#ffffff", "Background / no data"),
        ]
    elif kind == "enhancement":
        lo = f"{vmin:.0f}" if vmin is not None else "low"
        hi = f"{vmax:.0f}" if vmax is not None else "high"
        rows = [
            ("#d7191c", f"High CH₄ (≈ {hi} ppm·m)"),
            ("#f7f7f7", "Near zero"),
            ("#2c7bb6", f"Low / negative (≈ {lo} ppm·m)"),
            ("#ffff00", "Plume boundary"),
        ]
    else:
        rows = [
            ("#d7191c", "High"),
            ("#ffffff", "No data"),
        ]

    items = "".join(
        f'<div class="legend-row">'
        f'<span class="legend-swatch" style="background:{c};"></span>'
        f'<span>{t}</span></div>'
        for c, t in rows
    )
    extra = ""
    if n_pixels is not None:
        extra = (
            f'<div class="legend-row" style="margin-top:0.35rem;">'
            f'<b>Plume pixels:</b> {n_pixels:,}</div>'
        )
    if mean_enh is not None:
        extra += (
            f'<div class="legend-row">'
            f'<b>Mean enh.:</b> {mean_enh:.0f} ppm·m</div>'
        )
    return (
        f'<div class="result-legend">'
        f'<div class="legend-heading">Legend</div>{items}{extra}</div>'
    )


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EMIT Methane Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --red: #e63946;
    --honeydew: #f1faee;
    --frost: #a8dadc;
    --blue: #457b9d;
    --navy: #1d3557;
    --black: #111111;
    --white: #ffffff;
    --border: #d8e6e8;
    --muted: #4f5d63;
    --dark-field: #292a33;
}
.stApp { background: #f1faee; color: #111111 !important; }
[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem; padding-left: 1.2rem; padding-right: 1.2rem; }
.app-header { position: relative; z-index: 10; display: flex; align-items: center; justify-content: space-between; background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 0.75rem 1rem; margin-top: 0.15rem; margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(29,53,87,0.05); }
.app-title { color: #111111 !important; font-size: 1.45rem; font-weight: 850; line-height: 1.1; }
.app-subtitle { color: #111111 !important; font-size: 0.78rem; margin-top: 0.15rem; }
.status-pill { background: #f1faee; color: #111111 !important; border: 1px solid #a8dadc; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem; font-weight: 750; white-space: nowrap; }
.app-card { background: #ffffff; border: 1px solid var(--border); border-radius: 15px; padding: 0.75rem; box-shadow: 0 2px 10px rgba(29,53,87,0.04); height: 100%; color: #111111 !important; }
.card-title { color: #111111 !important; font-size: 1rem; font-weight: 800; margin-bottom: 0.1rem; }
.card-caption { color: #111111 !important; font-size: 0.73rem; margin-bottom: 0.45rem; }
.section-label { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.35rem; }
.stApp p, .stApp label, .stApp small, .stApp strong, .stApp em, .stApp li, .stApp td, .stApp th, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] span, .stApp [data-testid="stMarkdownContainer"] li { color: #111111 !important; }
div[data-testid="stDateInput"] div[data-baseweb="input"], div[data-testid="stDateInput"] div[data-baseweb="input"] > div, div[data-testid="stDateInput"] input, div[data-testid="stDateInput"] input[type="text"], .stDateInput div[data-baseweb="input"], .stDateInput div[data-baseweb="input"] > div, .stDateInput input, .stDateInput input[type="text"] { background-color: var(--dark-field) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; opacity: 1 !important; }
div[data-testid="stDateInput"] input::-webkit-datetime-edit, div[data-testid="stDateInput"] input::-webkit-datetime-edit-text, div[data-testid="stDateInput"] input::-webkit-datetime-edit-month-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-day-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-year-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-fields-wrapper, .stDateInput input::-webkit-datetime-edit, .stDateInput input::-webkit-datetime-edit-text, .stDateInput input::-webkit-datetime-edit-month-field, .stDateInput input::-webkit-datetime-edit-day-field, .stDateInput input::-webkit-datetime-edit-year-field, .stDateInput input::-webkit-datetime-edit-fields-wrapper { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; opacity: 1 !important; }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, .stNumberInput input, .stTextInput input { background-color: var(--dark-field) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }
input::placeholder, textarea::placeholder { color: #bfc3cc !important; opacity: 1 !important; }
div[data-baseweb="select"] input, div[data-baseweb="select"] [role="combobox"], div[data-baseweb="select"] * { color: #111111 !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul[role="listbox"], div[data-baseweb="popover"] [role="option"], div[data-baseweb="popover"] li[role="option"] { background: #111318 !important; }
div[data-baseweb="popover"] [role="listbox"] *, div[data-baseweb="popover"] [role="option"] *, ul[role="listbox"] *, li[role="option"] * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] li[role="option"]:hover { background: #2b2e38 !important; }
.stDateInput, .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.15rem; }
.stSlider > div { padding-top: 0.05rem; padding-bottom: 0.05rem; }
.stSlider label, .stSlider [data-testid="stTickBar"] * { color: #111111 !important; }
.stSlider [data-testid="stThumbValue"], .stSlider [data-testid="stThumbValue"] * { color: #ffffff !important; }
[data-baseweb="calendar"] *, [data-baseweb="popover"] [data-baseweb="calendar"] *, [data-baseweb="calendar"] button { color: #ffffff !important; }
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus { -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }

/* BUTTONS */
.stButton > button, .stDownloadButton > button {
    border-radius: 9px;
    min-height: 2.15rem;
    font-weight: 750;
    font-size: 0.78rem;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: #e63946 !important;
    border: 1px solid #e63946 !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: #c92f3b !important;
    border-color: #c92f3b !important;
}
.stButton > button[kind="secondary"],
.stDownloadButton > button {
    background: #1d3557 !important;
    color: #ffffff !important;
    border: 1px solid #1d3557 !important;
}
.stButton > button[kind="secondary"] *,
.stDownloadButton > button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button:hover {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #1d3557 !important;
}
.stButton > button[kind="secondary"]:hover *,
.stDownloadButton > button:hover * {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled { opacity: 0.55 !important; }

[data-testid="stImage"] {
    max-width: 100% !important;
    overflow: hidden;
    border-radius: 6px;
}
[data-testid="stImage"] > img {
    max-width: 100% !important;
    height: auto !important;
    display: block;
}

.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px; padding: 0.65rem 0.75rem; margin-top: 0.45rem; }
.auth-status { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important; border-radius: 9px; padding: 0.45rem 0.6rem; font-size: 0.76rem; font-weight: 700; margin-bottom: 0.45rem; }
.auth-help { color: #111111 !important; font-size: 0.72rem; line-height: 1.45; margin: 0.2rem 0 0.45rem 0; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); }
div[data-testid="stDataFrame"] * { color: #111111 !important; }
.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem; color: #111111 !important; font-size: 0.78rem; line-height: 1.25; }
.legend-swatch { width: 18px; height: 14px; min-width: 18px; border: 1px solid #555; border-radius: 2px; display: inline-block; }
.result-card { background: #ffffff; border: 1px solid #d8e6e8; border-radius: 12px; padding: 0.6rem; }
.result-tag { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.12rem 0.45rem; font-size: 0.6rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.2rem; }
.result-name { color: #111111 !important; font-size: 0.9rem; font-weight: 800; margin-bottom: 0.35rem; }
.result-note { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.55rem 0.7rem; font-size: 0.8rem; color: #111111 !important; margin-top: 0.45rem; }
.mouse-readout { background: #f8fbfb; border: 1px dashed #a8dadc; border-radius: 9px; padding: 0.35rem 0.6rem; font-size: 0.74rem; color: #111111 !important; margin-top: 0.35rem; }
footer { visibility: hidden; }
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ EMIT Methane Plume Detection</div>
        <div class="app-subtitle">NASA EMIT hyperspectral &nbsp;|&nbsp; Carbon Mapper-style matched-filter enhancements</div>
    </div>
    <div class="status-pill">60 m native &nbsp;•&nbsp; HyperSpectral</div>
</div>
""", unsafe_allow_html=True)

if not EARTHACCESS_AVAILABLE:
    st.error(
        "⚠️ The `earthaccess` package is not installed. "
        "Add it to your `requirements.txt` and reboot the app."
    )
    st.stop()

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)


# ══════════════════════════════════════════════════════════════════════
#  01 · STUDY AREA  +  02 · SEARCH
# ══════════════════════════════════════════════════════════════════════

map_col, control_col = st.columns([1.65, 1.0], gap="small")

with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption">Search by place name, enter coordinates manually, '
        'or draw the study area directly on the map with the polygon tool. '
        'Live mouse coordinates appear in the bottom-right corner of the map.</div>',
        unsafe_allow_html=True,
    )

    ps1, ps2 = st.columns([3, 1], gap="small")
    with ps1:
        place_query = st.text_input(
            "Place name",
            placeholder="e.g. Tehran, Paris, Permian Basin, Riyadh…",
            key="place_query",
            label_visibility="collapsed",
        )
    with ps2:
        search_place_clicked = st.button(
            "🔍 Find place",
            use_container_width=True,
            key="search_place_btn",
        )

    if search_place_clicked:
        if not place_query.strip():
            st.warning("Please type a place name first.")
        else:
            with st.spinner("Geocoding place name…"):
                geom, center, label = geocode_place(place_query.strip())
            if geom is not None:
                st.session_state.aoi = mapping(geom)
                st.session_state["aoi_source"] = f"Place: {label[:90]}"
                st.session_state["_ignore_drawings_once"] = True
                st.success(f"Found: {label[:120]}")
            else:
                st.warning(
                    "Place not found. Try a more specific name or use coordinates."
                )

    with st.expander("📍 Or enter coordinates manually"):
        mc1, mc2, mc3 = st.columns(3, gap="small")
        with mc1:
            manual_lat = st.number_input(
                "Latitude",
                value=35.50, min_value=-90.0, max_value=90.0,
                step=0.01, format="%.4f", key="manual_lat",
            )
        with mc2:
            manual_lon = st.number_input(
                "Longitude",
                value=51.30, min_value=-180.0, max_value=180.0,
                step=0.01, format="%.4f", key="manual_lon",
            )
        with mc3:
            manual_size = st.number_input(
                "Half-size (°)",
                value=0.10, min_value=0.005, max_value=5.0,
                step=0.005, format="%.3f", key="manual_size",
            )
        if st.button("Apply coordinates", use_container_width=True, key="apply_coords"):
            st.session_state.aoi = mapping(box(
                manual_lon - manual_size, manual_lat - manual_size,
                manual_lon + manual_size, manual_lat + manual_size,
            ))
            st.session_state["aoi_source"] = (
                f"Manual: ({manual_lat:.4f}, {manual_lon:.4f}) ± {manual_size:.3f}°"
            )
            st.session_state["_ignore_drawings_once"] = True
            st.success("AOI set from coordinates.")

    if st.session_state.get("aoi_source"):
        st.markdown(
            f'<div class="card-caption">Current AOI: '
            f'<b>{st.session_state["aoi_source"]}</b></div>',
            unsafe_allow_html=True,
        )

    map_data = st_folium(
        create_map(st.session_state.aoi),
        height=385,
        width=1000,
        key="aoi_map",
    )

    ignore_drawings = st.session_state.pop("_ignore_drawings_once", False)
    if not ignore_drawings and map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry(
            {"type": "FeatureCollection", "features": map_data["all_drawings"]}
        )
        if new_aoi and new_aoi != st.session_state.aoi:
            st.session_state.aoi = new_aoi
            st.session_state["aoi_source"] = "Custom polygon (drawn)"
            st.rerun()

    last_clicked = map_data.get("last_clicked") if map_data else None
    if last_clicked:
        lat_c = last_clicked.get("lat")
        lon_c = last_clicked.get("lng")
        st.markdown(
            f'<div class="mouse-readout">'
            f'🖱️ Last click &nbsp;→&nbsp; '
            f'<b>Lat:</b> {lat_c:.5f} &nbsp;·&nbsp; <b>Lon:</b> {lon_c:.5f}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="mouse-readout">'
            '🖱️ Live mouse coordinates shown in the bottom-right corner of the map. '
            'Click on the map to pin a coordinate here.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">EMIT Granule Search</div>', unsafe_allow_html=True)

    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=365)

    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input("Start date", default_start, key="start_date")
    with d2:
        end_date = st.date_input("End date", default_end, key="end_date")

    st.markdown(
        '<div class="card-caption">EMIT covers ~75 km swaths, so visits to a '
        'given AOI are irregular. A wider window improves the chance of finding data.</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔎  Search EMIT granules", type="primary", use_container_width=True):
        try:
            with st.spinner("Authenticating with NASA Earthdata…"):
                login_earthdata()

            with st.spinner("Searching EMIT collection…"):
                results = search_emit_granules(
                    st.session_state.aoi, start_date, end_date
                )

            st.session_state["emit_results"] = results
            st.session_state.pop("selected_granule", None)
            st.session_state.pop("emit_result", None)

            if results:
                st.success(f"{len(results)} EMIT granule(s) found")
            else:
                st.warning(
                    "No EMIT granules found for this AOI and time range. "
                    "Try a wider date range."
                )
        except Exception as e:
            st.session_state["emit_results"] = []
            st.error(f"Search failed: {e}")

    emit_results = st.session_state.get("emit_results", [])

    if emit_results:
        rows = []
        for g in emit_results:
            dt = granule_datetime(g)
            rows.append({
                "date": dt,
                "cloud": granule_cloud(g),
                "id": g.get("meta", {}).get("native-id", "unknown")[:40],
            })
        table = pd.DataFrame(rows).sort_values("date", na_position="last")
        st.dataframe(
            table,
            use_container_width=True,
            height=112,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
            },
        )

        def format_granule(idx):
            dt = granule_datetime(emit_results[idx])
            dt_text = dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown"
            gid = emit_results[idx].get("meta", {}).get("native-id", "")
            return f"{dt_text}  ·  {gid[:50]}"

        selected_idx = st.selectbox(
            "Granule",
            list(range(len(emit_results))),
            format_func=format_granule,
            key="granule_select",
        )
        st.session_state["selected_granule"] = emit_results[selected_idx]

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  03 · DETECTION  +  04 · PROCESS
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

settings_col, action_col = st.columns([1.65, 1.0], gap="small")

with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        PARAMS["plume_threshold_ppm_m"] = st.number_input(
            "Enhancement threshold (ppm·m)",
            min_value=100.0,
            max_value=10000.0,
            value=float(PARAMS["plume_threshold_ppm_m"]),
            step=100.0,
            key="plume_threshold",
        )
    with p2:
        PARAMS["min_plume_pixels"] = st.number_input(
            "Minimum plume pixels",
            min_value=1,
            max_value=500,
            value=int(PARAMS["min_plume_pixels"]),
            step=1,
            key="min_plume_pixels",
        )
    with p3:
        PARAMS["wind_speed_m_s"] = st.number_input(
            "Wind speed (m/s)",
            min_value=0.1,
            max_value=20.0,
            value=float(PARAMS["wind_speed_m_s"]),
            step=0.1,
            key="wind_speed",
        )

    estimated_area_m2 = int(PARAMS["min_plume_pixels"]) * RESOLUTION * RESOLUTION
    st.markdown(
        f'<div class="card-caption">'
        f'Minimum plume area ≈ {estimated_area_m2:,} m² at {RESOLUTION} m resolution. '
        f'Wind speed is used for IME flux estimation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)

    selected_granule = st.session_state.get("selected_granule")
    if selected_granule is not None:
        dt = granule_datetime(selected_granule)
        dt_text = dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown date"
        st.markdown(
            f'<div class="card-title">Ready to detect</div>'
            f'<div class="card-caption">Granule: {dt_text}</div>',
            unsafe_allow_html=True,
        )

        run_detect = st.button(
            "🚀  Run Methane Detection",
            type="primary",
            use_container_width=True,
            key="run_detect",
        )

        if run_detect:
            progress = st.progress(0, text="Authenticating…")
            try:
                progress.progress(10, text="Logging in to Earthdata…")
                login_earthdata()

                progress.progress(35, text="Loading EMIT enhancement…")
                data, transform, crs = load_emit_enhancement(
                    selected_granule, st.session_state.aoi
                )

                if data is None or data.size == 0:
                    st.error("EMIT granule did not intersect the AOI.")
                    st.stop()

                progress.progress(65, text="Detecting plumes…")
                plume_mask = detect_plume(
                    data,
                    PARAMS["plume_threshold_ppm_m"],
                    int(PARAMS["min_plume_pixels"]),
                )

                progress.progress(85, text="Estimating flux…")
                flux = estimate_flux_ime(
                    data, plume_mask, PARAMS["wind_speed_m_s"]
                )

                st.session_state.emit_result = {
                    "enhancement": data,
                    "plume_mask": plume_mask,
                    "flux": flux,
                    "transform": transform,
                    "crs": crs,
                    "granule_dt": dt,
                    "threshold": PARAMS["plume_threshold_ppm_m"],
                    "wind_speed": PARAMS["wind_speed_m_s"],
                }

                progress.progress(100, text="Done")
                st.success("Detection complete")
            except Exception as e:
                st.error(f"Detection failed: {e}")
    else:
        st.markdown(
            '<div class="card-title">Select a granule first</div>'
            '<div class="card-caption">Search EMIT granules, select one, then run the detection.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  05 · RESULTS
# ══════════════════════════════════════════════════════════════════════

if "emit_result" in st.session_state:
    result = st.session_state.emit_result
    flux = result["flux"]
    enhancement = result["enhancement"]
    plume_mask = result["plume_mask"]
    transform = result.get("transform")
    crs = result.get("crs")

    vmin_enh, vmax_enh = _compute_vrange(enhancement)

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)

    metrics = st.columns(6, gap="small")
    metrics[0].metric("Flux (kg/h)", f"{flux['Q_kg_h']:.1f}")
    metrics[1].metric("Flux (t/h)", f"{flux['Q_ton_h']:.2f}")
    metrics[2].metric("Plume pixels", f"{flux['n_pixels']:,}")
    metrics[3].metric("Plume area", f"{flux['plume_area_m2']/1e6:.3f} km²")
    metrics[4].metric("Max enh. (ppm·m)", f"{flux['max_enhancement']:.0f}")
    metrics[5].metric("Mean enh. (ppm·m)", f"{flux['mean_enhancement']:.0f}")

    rc1, rc2 = st.columns(2, gap="small")
    with rc1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-tag">Enhancement</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-name">CH₄ Enhancement (ppm·m)</div>', unsafe_allow_html=True)
        img_col, legend_col = st.columns([3.4, 1.2], gap="small")
        with img_col:
            st.image(
                enhancement_png(enhancement, mask=plume_mask,
                                colormap="turbo", show_outline=True,
                                vmin=vmin_enh, vmax=vmax_enh),
                use_container_width=True,
                output_format="PNG",
            )
        with legend_col:
            st.markdown(
                '<div style="padding-top:0.3rem;"></div>',
                unsafe_allow_html=True,
            )
            st.image(
                colorbar_png(vmin_enh, vmax_enh, "turbo"),
                use_container_width=True,
            )
            st.markdown(
                legend_html("plume", vmin_enh, vmax_enh,
                            n_pixels=flux["n_pixels"],
                            mean_enh=flux["mean_enhancement"]),
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-tag">Plume mask</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-name">Detected methane plume (with outline)</div>', unsafe_allow_html=True)
        img_col, legend_col = st.columns([3.4, 1.2], gap="small")
        with img_col:
            st.image(
                enhancement_png(enhancement, mask=plume_mask,
                                colormap="turbo", show_outline=True,
                                outline_color=(255, 255, 0),
                                vmin=vmin_enh, vmax=vmax_enh),
                use_container_width=True,
                output_format="PNG",
            )
        with legend_col:
            st.markdown(
                '<div style="padding-top:0.3rem;"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                legend_html("plume", vmin_enh, vmax_enh,
                            n_pixels=flux["n_pixels"],
                            mean_enh=flux["mean_enhancement"]),
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="result-note">'
        f'<b>IME method:</b> IME = {flux["IME_ppm_m2"]:.2e} ppm·m·m² · '
        f'{flux["IME_kg"]:.2f} kg CH₄ · '
        f'U_eff = {flux["U_eff_m_s"]:.2f} m/s · '
        f'L = {flux["length_m"]:.0f} m · '
        f'Q = {flux["Q_kg_h"]:.1f} kg/h'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 📥 Download results")
    dl1, dl2, dl3, dl4 = st.columns(4, gap="small")

    dt_str = result.get("granule_dt")
    dt_tag = dt_str.strftime("%Y%m%d") if dt_str else "granule"

    with dl1:
        png_data = enhancement_png(enhancement, mask=plume_mask,
                                    colormap="turbo", show_outline=True,
                                    vmin=vmin_enh, vmax=vmax_enh)
        st.download_button(
            "⬇ Enhancement PNG",
            png_data,
            file_name=f"enhancement_{dt_tag}.png",
            mime="image/png",
            use_container_width=True,
            key="dl_enh_png",
        )

    with dl2:
        png_mask = enhancement_png(enhancement, mask=plume_mask,
                                    colormap="turbo", show_outline=True,
                                    vmin=vmin_enh, vmax=vmax_enh)
        st.download_button(
            "⬇ Plume mask PNG",
            png_mask,
            file_name=f"plume_{dt_tag}.png",
            mime="image/png",
            use_container_width=True,
            key="dl_mask_png",
        )

    with dl3:
        csv = pd.DataFrame([flux]).to_csv(index=False)
        st.download_button(
            "⬇ Flux CSV",
            csv,
            file_name=f"flux_{dt_tag}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_flux_csv",
        )

    with dl4:
        try:
            import zipfile
            geo_pkg = io.BytesIO()
            with zipfile.ZipFile(geo_pkg, "w", zipfile.ZIP_DEFLATED) as zf:
                enh_tif = io.BytesIO()
                with rasterio.open(
                    enh_tif, "w", driver="GTiff",
                    height=enhancement.shape[0], width=enhancement.shape[1],
                    count=1, dtype="float32", crs=crs, transform=transform,
                    nodata=np.nan, compress="deflate",
                ) as dst:
                    dst.write(enhancement.astype(np.float32), 1)
                zf.writestr("enhancement_ppmm.tif", enh_tif.getvalue())

                mask_tif = io.BytesIO()
                with rasterio.open(
                    mask_tif, "w", driver="GTiff",
                    height=plume_mask.shape[0], width=plume_mask.shape[1],
                    count=1, dtype="uint8", crs=crs, transform=transform,
                    nodata=0, compress="deflate",
                ) as dst:
                    dst.write(plume_mask.astype(np.uint8), 1)
                zf.writestr("plume_mask.tif", mask_tif.getvalue())

            st.download_button(
                "⬇ GeoTIFF bundle",
                geo_pkg.getvalue(),
                file_name=f"emit_{dt_tag}.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_geo_zip",
            )
        except Exception:
            st.button("⬇ GeoTIFF (unavailable)", disabled=True, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  06 · MULTI-DATE COMPARISON
# ══════════════════════════════════════════════════════════════════════

if "emit_results" in st.session_state and st.session_state.emit_results:
    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">06 · MULTI-DATE COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Compare plumes over time</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption">'
        'Processes all EMIT granules in the current search window and displays '
        'them side by side. Useful for tracking emission evolution over months.'
        '</div>',
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2 = st.columns([1, 3], gap="small")
    with btn_col1:
        run_batch = st.button(
            "🔁  Process all granules",
            type="primary",
            use_container_width=True,
            key="run_batch",
        )
    with btn_col2:
        max_granules = st.slider(
            "Max granules to process",
            min_value=2,
            max_value=20,
            value=6,
            key="max_granules",
        )

    if run_batch:
        granules = st.session_state.emit_results[: int(max_granules)]
        progress = st.progress(0, text="Processing granules…")
        batch_results = []
        for i, g in enumerate(granules):
            progress.progress(
                int(100 * (i + 1) / len(granules)),
                text=f"Processing {i+1}/{len(granules)}…",
            )
            try:
                data, tform, tcrs = load_emit_enhancement(g, st.session_state.aoi)
                if data is None or data.size == 0:
                    continue
                if valid_coverage(data) < 0.02:
                    continue
                pm = detect_plume(
                    data,
                    PARAMS["plume_threshold_ppm_m"],
                    int(PARAMS["min_plume_pixels"]),
                )
                f = estimate_flux_ime(data, pm, PARAMS["wind_speed_m_s"])
                batch_results.append({
                    "date": granule_datetime(g),
                    "enhancement": data,
                    "plume_mask": pm,
                    "flux": f,
                    "transform": tform,
                    "crs": tcrs,
                })
            except Exception:
                continue

        st.session_state.batch_results = batch_results
        progress.progress(100, text="Done")
        st.success(f"Processed {len(batch_results)} granule(s)")

    if "batch_results" in st.session_state and st.session_state.batch_results:
        batch = st.session_state.batch_results

        chart_rows = []
        for r in batch:
            if r["date"] is not None:
                chart_rows.append({
                    "date": r["date"],
                    "flux_kg_h": r["flux"]["Q_kg_h"],
                    "plume_pixels": r["flux"]["n_pixels"],
                    "plume_area_km2": r["flux"]["plume_area_m2"] / 1e6,
                })
        if chart_rows:
            chart_df = pd.DataFrame(chart_rows).sort_values("date").set_index("date")
            st.markdown("##### Estimated flux over time")
            st.line_chart(chart_df[["flux_kg_h"]], use_container_width=True, height=240)
            st.dataframe(
                chart_df,
                use_container_width=True,
                hide_index=False,
                column_config={
                    "flux_kg_h": st.column_config.NumberColumn("Flux (kg/h)", format="%.1f"),
                    "plume_pixels": st.column_config.NumberColumn("Pixels", format="%d"),
                    "plume_area_km2": st.column_config.NumberColumn("Area (km²)", format="%.3f"),
                },
            )
            st.download_button(
                "⬇ Download time series CSV",
                chart_df.to_csv(),
                file_name="emit_flux_timeseries.csv",
                mime="text/csv",
                key="dl_ts_csv",
                use_container_width=False,
            )

        st.markdown("##### Visual comparison")
        dates_labels = [
            r["date"].strftime("%Y-%m-%d") if r["date"] else f"#{i+1}"
            for i, r in enumerate(batch)
        ]
        safe_labels = {i: dates_labels[i] for i in range(len(dates_labels))}
        _batch_key = f"batch_slider_{len(batch)}"
        selected_idx = st.select_slider(
            "Select granule",
            options=list(range(len(batch))),
            format_func=lambda x: safe_labels.get(x, f"#{x}"),
            value=0,
            key=_batch_key,
        )
        chosen = batch[selected_idx]

        # shared color range across all batch results
        _all_vals = []
        for r in batch:
            e = r["enhancement"]
            if np.isfinite(e).any():
                _all_vals.append(e[np.isfinite(e)])
        if _all_vals:
            _all_vals = np.concatenate(_all_vals)
            cvmin, cvmax = np.percentile(_all_vals, [2, 98])
        else:
            cvmin, cvmax = 0.0, 1.0
        if cvmax <= cvmin:
            cvmax = cvmin + 1.0

        _has_plume = chosen["flux"]["n_pixels"] > 0
        _cov_lbl, _cov_col = coverage_badge(chosen["enhancement"])

        cc1, cc2 = st.columns(2, gap="small")
        with cc1:
            st.markdown(
                f'<div class="card-caption" style="font-weight:700;">'
                f'{dates_labels[selected_idx]} · Enhancement'
                f'</div>'
                f'<div class="card-caption" style="color:{_cov_col} !important;'
                f'font-weight:700;">{_cov_lbl}</div>',
                unsafe_allow_html=True,
            )
            st.image(
                enhancement_png(chosen["enhancement"],
                                mask=chosen["plume_mask"],
                                colormap="turbo",
                                show_outline=_has_plume,
                                vmin=cvmin, vmax=cvmax),
                use_container_width=True,
                output_format="PNG",
            )
            st.image(colorbar_png(cvmin, cvmax, "turbo"), width=90)
        with cc2:
            if _has_plume:
                status_html = (
                    f'<div class="card-caption" style="color:#2a9d8f !important;'
                    f'font-weight:700;">✓ Plume detected</div>'
                )
            else:
                status_html = (
                    f'<div class="card-caption" style="color:#e63946 !important;'
                    f'font-weight:700;">✗ No plume above threshold '
                    f'({PARAMS["plume_threshold_ppm_m"]:.0f} ppm·m)</div>'
                )
            st.markdown(
                f'<div class="card-caption" style="font-weight:700;">'
                f'{dates_labels[selected_idx]} · Plume outline'
                f'</div>{status_html}',
                unsafe_allow_html=True,
            )
            st.image(
                enhancement_png(chosen["enhancement"],
                                mask=chosen["plume_mask"],
                                colormap="turbo",
                                show_outline=_has_plume,
                                vmin=cvmin, vmax=cvmax),
                use_container_width=True,
                output_format="PNG",
            )
            st.markdown(
                legend_html("plume", cvmin, cvmax,
                            n_pixels=chosen["flux"]["n_pixels"],
                            mean_enh=chosen["flux"]["mean_enhancement"]),
                unsafe_allow_html=True,
            )
        m1, m2, m3 = st.columns(3, gap="small")
        m1.metric("Flux (kg/h)", f"{chosen['flux']['Q_kg_h']:.1f}")
        m2.metric("Plume pixels", f"{chosen['flux']['n_pixels']:,}")
        m3.metric("Plume area (km²)", f"{chosen['flux']['plume_area_m2']/1e6:.3f}")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  07 · PLUME EVOLUTION WINDOW
# ══════════════════════════════════════════════════════════════════════

if "emit_result" in st.session_state:
    _res = st.session_state.emit_result
    _ref_dt = _res.get("granule_dt")

    if _ref_dt is not None:
        st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">07 · PLUME EVOLUTION WINDOW</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card-title">Methane plume changes around the detected date</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-caption">'
            f'Searches all EMIT granules within a ±<i>N</i>-day window around '
            f'<b>{_ref_dt.strftime("%Y-%m-%d %H:%M")}</b> and shows how the plume '
            f'appears, disappears, moves, and grows or shrinks across the window.'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "💡 Methane plumes are transient. A source may appear on some "
            "overpasses and not others due to intermittent emission, cloud cover, "
            "or wind dispersion. Only some observations showing a plume is "
            "normal and expected."
        )

        ec1, ec2 = st.columns([1, 1], gap="small")
        with ec1:
            window_days = st.slider(
                "Window around detected date (± days)",
                min_value=5, max_value=45, value=15, step=1,
                key="evo_window_days",
            )
        with ec2:
            max_evo = st.slider(
                "Max granules to process",
                min_value=2, max_value=40, value=12, step=1,
                key="evo_max_granules",
            )

        run_evo = st.button(
            "🔁  Analyze plume evolution",
            type="primary",
            use_container_width=True,
            key="run_evolution",
        )

        if run_evo:
            start_d = (_ref_dt - timedelta(days=int(window_days))).date()
            end_d = (_ref_dt + timedelta(days=int(window_days))).date()

            progress = st.progress(0, text="Searching EMIT granules…")
            try:
                login_earthdata()
                evo_granules = search_emit_granules(
                    st.session_state.aoi, start_d, end_d
                )
                evo_granules = sorted(
                    evo_granules,
                    key=lambda g: granule_datetime(g) or datetime.min,
                )
                evo_granules = evo_granules[: int(max_evo)]

                if not evo_granules:
                    progress.progress(100, text="No granules found")
                    st.warning(
                        "No EMIT granules found in this window. Try a wider ± window."
                    )
                else:
                    evo_results = []
                    for i, g in enumerate(evo_granules):
                        progress.progress(
                            int(100 * (i + 1) / len(evo_granules)),
                            text=f"Processing {i+1}/{len(evo_granules)}…",
                        )
                        try:
                            data, tform, tcrs = load_emit_enhancement(
                                g, st.session_state.aoi
                            )
                            if data is None or data.size == 0:
                                continue

                            cov = valid_coverage(data)
                            if cov < 0.02:
                                continue

                            pm = detect_plume(
                                data,
                                PARAMS["plume_threshold_ppm_m"],
                                int(PARAMS["min_plume_pixels"]),
                            )
                            f = estimate_flux_ime(
                                data, pm, PARAMS["wind_speed_m_s"]
                            )

                            centroid_px = None
                            centroid_geo = None
                            if pm.any():
                                ys, xs = np.nonzero(pm)
                                cx_px = float(xs.mean())
                                cy_px = float(ys.mean())
                                centroid_px = (cx_px, cy_px)
                                try:
                                    from rasterio.transform import xy as rio_xy
                                    gx, gy = rio_xy(
                                        tform, cy_px, cx_px, offset="center"
                                    )
                                    centroid_geo = (float(gx), float(gy))
                                except Exception:
                                    pass

                            evo_results.append({
                                "date": granule_datetime(g),
                                "enhancement": data,
                                "plume_mask": pm,
                                "flux": f,
                                "transform": tform,
                                "crs": tcrs,
                                "centroid_px": centroid_px,
                                "centroid_geo": centroid_geo,
                                "coverage": cov,
                            })
                        except Exception:
                            continue

                    st.session_state.evo_results = evo_results
                    st.session_state.evo_ref_date = _ref_dt
                    st.session_state.evo_window_days_used = int(window_days)
                    progress.progress(100, text="Done")
                    st.success(
                        f"Processed {len(evo_results)} granule(s) in a "
                        f"±{int(window_days)}-day window"
                    )
            except Exception as e:
                st.error(f"Evolution analysis failed: {e}")

        if st.session_state.get("evo_results"):
            evo = st.session_state.evo_results
            used_window = st.session_state.get("evo_window_days_used", window_days)

            n_total = len(evo)
            n_with = sum(1 for r in evo if r["flux"]["n_pixels"] > 0)
            n_without = n_total - n_with
            st.markdown(
                f'<div class="result-note">'
                f'<b>{n_with}</b> of <b>{n_total}</b> observation(s) in the '
                f'±{used_window}-day window showed a detectable plume. '
                f'<b>{n_without}</b> observation(s) showed no plume above the '
                f'threshold of {PARAMS["plume_threshold_ppm_m"]:.0f} ppm·m.'
                f'</div>',
                unsafe_allow_html=True,
            )

            cov_rows = []
            for r in evo:
                lbl, col = coverage_badge(r["enhancement"])
                cov_rows.append({
                    "date": r["date"].strftime("%Y-%m-%d") if r["date"] else "-",
                    "coverage": lbl,
                    "flux_kg_h": r["flux"]["Q_kg_h"],
                })
            st.markdown("##### Data coverage per observation")
            st.dataframe(
                pd.DataFrame(cov_rows),
                use_container_width=True,
                hide_index=True,
            )

            rows = []
            for r in evo:
                rows.append({
                    "date": r["date"],
                    "flux_kg_h": r["flux"]["Q_kg_h"],
                    "plume_pixels": r["flux"]["n_pixels"],
                    "plume_area_km2": r["flux"]["plume_area_m2"] / 1e6,
                    "max_enh_ppmm": r["flux"]["max_enhancement"],
                    "mean_enh_ppmm": r["flux"]["mean_enhancement"],
                    "has_plume": int(r["flux"]["n_pixels"] > 0),
                })
            evo_df = pd.DataFrame(rows)
            if not evo_df.empty and evo_df["date"].notna().any():
                evo_df = evo_df.sort_values("date").set_index("date")

                st.markdown("##### Flux evolution")
                st.line_chart(evo_df[["flux_kg_h"]], use_container_width=True, height=220)

                st.markdown("##### Plume area evolution")
                st.line_chart(evo_df[["plume_area_km2"]], use_container_width=True, height=200)

                st.dataframe(
                    evo_df,
                    use_container_width=True,
                    hide_index=False,
                    column_config={
                        "flux_kg_h": st.column_config.NumberColumn("Flux (kg/h)", format="%.1f"),
                        "plume_pixels": st.column_config.NumberColumn("Pixels", format="%d"),
                        "plume_area_km2": st.column_config.NumberColumn("Area (km²)", format="%.3f"),
                        "max_enh_ppmm": st.column_config.NumberColumn("Max enh.", format="%.0f"),
                        "mean_enh_ppmm": st.column_config.NumberColumn("Mean enh.", format="%.0f"),
                        "has_plume": st.column_config.NumberColumn("Plume?", format="%d"),
                    },
                )

                st.download_button(
                    "⬇ Download evolution CSV",
                    evo_df.to_csv(),
                    file_name="emit_plume_evolution.csv",
                    mime="text/csv",
                    key="dl_evo_csv",
                    use_container_width=False,
                )

            geo_pts = [
                (r["date"], r["centroid_geo"])
                for r in evo
                if r.get("centroid_geo") is not None and r.get("date") is not None
            ]
            if len(geo_pts) >= 2:
                st.markdown("##### Plume centroid movement")
                crows = []
                for d, (gx, gy) in geo_pts:
                    crows.append({"date": d, "x": gx, "y": gy})
                cdf = pd.DataFrame(crows).sort_values("date")
                x0, y0 = cdf.iloc[0]["x"], cdf.iloc[0]["y"]
                cdf["dx_px"] = (cdf["x"] - x0) / RESOLUTION
                cdf["dy_px"] = (cdf["y"] - y0) / RESOLUTION
                st.dataframe(
                    cdf[["date", "dx_px", "dy_px"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "dx_px": st.column_config.NumberColumn("ΔX (px)", format="%.2f"),
                        "dy_px": st.column_config.NumberColumn("ΔY (px)", format="%.2f"),
                    },
                )

            # shared color range for evolution
            all_valid = []
            for r in evo:
                e = r["enhancement"]
                if np.isfinite(e).any():
                    all_valid.append(e[np.isfinite(e)])
            if all_valid:
                all_valid = np.concatenate(all_valid)
                shared_vmin, shared_vmax = np.percentile(all_valid, [2, 98])
            else:
                shared_vmin, shared_vmax = 0.0, 1.0
            if shared_vmax <= shared_vmin:
                shared_vmax = shared_vmin + 1.0

            st.markdown("##### Visual evolution")
            dates_labels = [
                r["date"].strftime("%Y-%m-%d") if r["date"] else f"#{i+1}"
                for i, r in enumerate(evo)
            ]
            safe_labels = {i: dates_labels[i] for i in range(len(dates_labels))}
            _evo_key = f"evo_slider_{len(evo)}"
            sel_idx = st.select_slider(
                "Select observation",
                options=list(range(len(evo))),
                format_func=lambda x: safe_labels.get(x, f"#{x}"),
                value=0,
                key=_evo_key,
            )
            chosen = evo[sel_idx]
            _cov_lbl, _cov_col = coverage_badge(chosen["enhancement"])
            _has_plume = chosen["flux"]["n_pixels"] > 0

            cc1, cc2 = st.columns(2, gap="small")
            with cc1:
                st.markdown(
                    f'<div class="card-caption" style="font-weight:700;">'
                    f'{dates_labels[sel_idx]} · Enhancement'
                    f'</div>'
                    f'<div class="card-caption" style="color:{_cov_col} !important;'
                    f'font-weight:700;">{_cov_lbl}</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    enhancement_png(
                        chosen["enhancement"],
                        mask=chosen["plume_mask"],
                        colormap="turbo",
                        show_outline=_has_plume,
                        vmin=shared_vmin,
                        vmax=shared_vmax,
                    ),
                    use_container_width=True,
                    output_format="PNG",
                )
                st.image(colorbar_png(shared_vmin, shared_vmax, "turbo"), width=90)
            with cc2:
                if _has_plume:
                    status_html = (
                        f'<div class="card-caption" style="color:#2a9d8f !important;'
                        f'font-weight:700;">✓ Plume detected</div>'
                    )
                else:
                    status_html = (
                        f'<div class="card-caption" style="color:#e63946 !important;'
                        f'font-weight:700;">✗ No plume above threshold '
                        f'({PARAMS["plume_threshold_ppm_m"]:.0f} ppm·m)</div>'
                    )
                st.markdown(
                    f'<div class="card-caption" style="font-weight:700;">'
                    f'{dates_labels[sel_idx]} · Plume outline'
                    f'</div>{status_html}',
                    unsafe_allow_html=True,
                )
                st.image(
                    enhancement_png(
                        chosen["enhancement"],
                        mask=chosen["plume_mask"],
                        colormap="turbo",
                        show_outline=_has_plume,
                        vmin=shared_vmin,
                        vmax=shared_vmax,
                    ),
                    use_container_width=True,
                    output_format="PNG",
                )
                st.markdown(
                    legend_html("plume",
                                n_pixels=chosen["flux"]["n_pixels"],
                                mean_enh=chosen["flux"]["mean_enhancement"]),
                    unsafe_allow_html=True,
                )
            em1, em2, em3, em4 = st.columns(4, gap="small")
            em1.metric("Flux (kg/h)", f"{chosen['flux']['Q_kg_h']:.1f}")
            em2.metric("Plume pixels", f"{chosen['flux']['n_pixels']:,}")
            em3.metric("Plume area (km²)", f"{chosen['flux']['plume_area_m2']/1e6:.3f}")
            em4.metric("Max enh. (ppm·m)", f"{chosen['flux']['max_enhancement']:.0f}")

            st.markdown("##### Plume mask gallery (all observations)")
            n_cols = 4
            n_obs = len(evo)
            grid_rows = (n_obs + n_cols - 1) // n_cols
            for gr in range(grid_rows):
                gcols = st.columns(n_cols, gap="small")
                for gc in range(n_cols):
                    idx = gr * n_cols + gc
                    if idx >= n_obs:
                        break
                    r = evo[idx]
                    label = (
                        r["date"].strftime("%Y-%m-%d")
                        if r["date"] else f"#{idx+1}"
                    )
                    _has = r["flux"]["n_pixels"] > 0
                    with gcols[gc]:
                        st.markdown(
                            f'<div class="card-caption" style="font-weight:700; '
                            f'text-align:center; margin-bottom:0.15rem;">'
                            f'{label}<br/>'
                            f'<span style="font-weight:400;">'
                            f'{r["flux"]["Q_kg_h"]:.0f} kg/h · '
                            f'{r["flux"]["n_pixels"]} px</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            enhancement_png(
                                r["enhancement"],
                                mask=r["plume_mask"],
                                colormap="turbo",
                                show_outline=_has,
                                vmin=shared_vmin,
                                vmax=shared_vmax,
                            ),
                            use_container_width=True,
                            output_format="PNG",
                        )

        st.markdown('</div>', unsafe_allow_html=True)
