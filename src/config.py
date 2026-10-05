"""
ReLoop WAI project - shared configuration, paths, plotting style and helpers.
Author: Himanshu Rai (XW, IIM Ranchi EMBA 2025-27) | Course: Logistics & Warehousing Management
All synthetic data is generated with a fixed seed so every number in the report is reproducible.
"""
from pathlib import Path
import math
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW, SYN = ROOT / "data" / "raw", ROOT / "data" / "synthetic"
FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
RESULTS = ROOT / "outputs" / "results.json"
SEED = 42

# Palette: circular-economy greens with one amber accent for "problem/baseline"
C = dict(dark="#1B4332", green="#2D6A4F", mid="#52B788", light="#B7E4C7",
         accent="#E76F51", amber="#F4A261", grey="#6C757D", ink="#212529")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

ROAD_CIRCUITY = 1.30          # road km / great-circle km (typical for Indian highways)

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def road_km(a, b):
    """Approximate road distance between two (lat, lon) tuples."""
    return haversine_km(a[0], a[1], b[0], b[1]) * ROAD_CIRCUITY

def save_result(key, value):
    """Append a block of headline numbers to outputs/results.json (single source of truth for report)."""
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=2, default=float))

CITY = {  # lat, lon (city centroids, approx.)
    "Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49), "Pune": (18.52, 73.86),
    "Mumbai": (19.08, 72.88), "Chennai": (13.08, 80.27), "Gurugram": (28.46, 77.03),
    "Noida": (28.54, 77.39), "Kolkata": (22.57, 88.36), "Ahmedabad": (23.02, 72.57),
    "Kochi": (9.93, 76.27), "Jaipur": (26.91, 75.79), "Bhubaneswar": (20.30, 85.82),
    "Nagpur": (21.15, 79.09), "Roorkee": (29.87, 77.89), "Greater Noida": (28.47, 77.50),
    "Manesar": (28.36, 76.94), "Vapi": (20.37, 72.90), "Chakan": (18.76, 73.86),
}
