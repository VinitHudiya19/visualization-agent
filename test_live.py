"""
Live test script - hit all viz-agent endpoints with real sample data.
Run: python test_live.py  (while server is running on port 8003)
"""

import sys
import os
# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
import json
import base64
from pathlib import Path

BASE = "http://127.0.0.1:8003"

# -- Sample Dataset: Monthly Sales Data --
COLUMNS = [
    {"name": "date",           "semantic": "datetime"},
    {"name": "region",         "semantic": "categorical", "unique": 5},
    {"name": "city",           "semantic": "categorical", "unique": 10},
    {"name": "category",       "semantic": "categorical", "unique": 4},
    {"name": "product",        "semantic": "categorical", "unique": 8},
    {"name": "channel",        "semantic": "categorical", "unique": 3},
    {"name": "orders",         "semantic": "numeric"},
    {"name": "units_sold",     "semantic": "numeric"},
    {"name": "revenue",        "semantic": "numeric"},
    {"name": "cost",           "semantic": "numeric"},
    {"name": "profit",         "semantic": "numeric"},
    {"name": "discount_pct",   "semantic": "numeric"},
    {"name": "customer_rating","semantic": "numeric"}
]

ROWS = [
    {"date": "2024-01-01", "region": "North", "city": "Delhi", "category": "Electronics", "product": "Smartphone", "channel": "Online", "orders": 120, "units_sold": 150, "revenue": 750000, "cost": 520000, "profit": 230000, "discount_pct": 10, "customer_rating": 4.5},

    {"date": "2024-01-05", "region": "West", "city": "Mumbai", "category": "Fashion", "product": "Shoes", "channel": "Offline", "orders": 90, "units_sold": 110, "revenue": 220000, "cost": 140000, "profit": 80000, "discount_pct": 15, "customer_rating": 4.2},

    {"date": "2024-01-10", "region": "South", "city": "Bangalore", "category": "Electronics", "product": "Laptop", "channel": "Online", "orders": 70, "units_sold": 80, "revenue": 960000, "cost": 700000, "profit": 260000, "discount_pct": 8, "customer_rating": 4.6},

    {"date": "2024-01-15", "region": "East", "city": "Kolkata", "category": "Home", "product": "Mixer", "channel": "Online", "orders": 60, "units_sold": 75, "revenue": 150000, "cost": 90000, "profit": 60000, "discount_pct": 12, "customer_rating": 4.1},

    {"date": "2024-01-20", "region": "Central", "city": "Nagpur", "category": "Grocery", "product": "Organic Pack", "channel": "Offline", "orders": 110, "units_sold": 140, "revenue": 98000, "cost": 60000, "profit": 38000, "discount_pct": 5, "customer_rating": 4.3},

    {"date": "2024-02-01", "region": "North", "city": "Delhi", "category": "Fashion", "product": "Jacket", "channel": "Online", "orders": 130, "units_sold": 160, "revenue": 320000, "cost": 210000, "profit": 110000, "discount_pct": 18, "customer_rating": 4.4},

    {"date": "2024-02-07", "region": "West", "city": "Pune", "category": "Electronics", "product": "Tablet", "channel": "Online", "orders": 85, "units_sold": 95, "revenue": 475000, "cost": 330000, "profit": 145000, "discount_pct": 9, "customer_rating": 4.5},

    {"date": "2024-02-12", "region": "South", "city": "Chennai", "category": "Home", "product": "Air Purifier", "channel": "Offline", "orders": 75, "units_sold": 85, "revenue": 255000, "cost": 180000, "profit": 75000, "discount_pct": 10, "customer_rating": 4.2},

    {"date": "2024-02-18", "region": "East", "city": "Patna", "category": "Grocery", "product": "Daily Essentials", "channel": "Online", "orders": 140, "units_sold": 180, "revenue": 126000, "cost": 78000, "profit": 48000, "discount_pct": 6, "customer_rating": 4.0},

    {"date": "2024-02-25", "region": "Central", "city": "Bhopal", "category": "Fashion", "product": "T-shirt", "channel": "Offline", "orders": 100, "units_sold": 130, "revenue": 130000, "cost": 85000, "profit": 45000, "discount_pct": 20, "customer_rating": 4.3},

    {"date": "2024-03-02", "region": "North", "city": "Chandigarh", "category": "Electronics", "product": "Smartwatch", "channel": "Online", "orders": 95, "units_sold": 120, "revenue": 360000, "cost": 250000, "profit": 110000, "discount_pct": 11, "customer_rating": 4.6},

    {"date": "2024-03-08", "region": "West", "city": "Ahmedabad", "category": "Home", "product": "Microwave", "channel": "Offline", "orders": 60, "units_sold": 70, "revenue": 210000, "cost": 150000, "profit": 60000, "discount_pct": 13, "customer_rating": 4.1},

    {"date": "2024-03-15", "region": "South", "city": "Hyderabad", "category": "Electronics", "product": "Headphones", "channel": "Online", "orders": 150, "units_sold": 200, "revenue": 400000, "cost": 260000, "profit": 140000, "discount_pct": 7, "customer_rating": 4.5},

    {"date": "2024-03-20", "region": "East", "city": "Ranchi", "category": "Grocery", "product": "Snacks Pack", "channel": "Offline", "orders": 120, "units_sold": 160, "revenue": 96000, "cost": 60000, "profit": 36000, "discount_pct": 5, "customer_rating": 4.2},

    {"date": "2024-03-28", "region": "Central", "city": "Indore", "category": "Fashion", "product": "Jeans", "channel": "Online", "orders": 110, "units_sold": 140, "revenue": 280000, "cost": 180000, "profit": 100000, "discount_pct": 16, "customer_rating": 4.4}
]

DATA_PAYLOAD = {"columns": COLUMNS, "rows": ROWS}


def save_png(b64_str, filename):
    """Decode base64 PNG and save to disk."""
    if b64_str:
        img = base64.b64decode(b64_str)
        path = Path(f"./test_outputs/{filename}")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(img)
        print(f"   [SAVED] {path} ({len(img)} bytes)")
    else:
        print("   [WARN] No PNG returned")


def test_health():
    print("\n" + "="*60)
    print("[1] GET /health")
    print("="*60)
    r = httpx.get(f"{BASE}/health")
    print(f"   Status: {r.status_code}")
    print(f"   {json.dumps(r.json(), indent=2)}")


def test_recommend():
    print("\n" + "="*60)
    print("[2] POST /recommend  (rule-based, no LLM)")
    print("="*60)
    r = httpx.post(f"{BASE}/recommend", json={
        "columns": COLUMNS,
        "task": "Show revenue trend over time"
    })
    print(f"   Status: {r.status_code}")
    print(f"   {json.dumps(r.json(), indent=2)}")


def test_chart():
    print("\n" + "="*60)
    print("[3] POST /chart  (Groq generates Plotly spec + PNG)")
    print("="*60)
    r = httpx.post(f"{BASE}/chart", json={
        "task": "Compare total revenue across regions",
        "data": DATA_PAYLOAD,
        "color_scheme": "vibrant",
        "render_png": True,
    }, timeout=30)
    print(f"   Status: {r.status_code}")
    body = r.json()
    print(f"   Chart type: {body.get('chart_type')}")
    print(f"   Spec keys: {list(body.get('spec', {}).keys())}")
    print(f"   Traces count: {len(body.get('spec', {}).get('data', []))}")
    save_png(body.get("png_base64"), "chart_bar.png")


def test_auto_insights():
    print("\n" + "="*60)
    print("[4] POST /auto-insights  (Groq auto-picks best visuals)")
    print("="*60)
    r = httpx.post(f"{BASE}/auto-insights", json={
        "data": DATA_PAYLOAD,
        "color_scheme": "corporate",
        "render_png": True,
        "max_insights": 2,
    }, timeout=60)
    print(f"   Status: {r.status_code}")
    body = r.json()
    print(f"   Requested: {body.get('total_requested')}")
    print(f"   Generated: {body.get('total_generated')}")
    print()
    for i, chart in enumerate(body.get("charts", [])):
        status = chart.get("status", "?")
        icon = "[OK]" if status == "success" else "[FAIL]"
        print(f"   {icon} Chart {i+1}: {chart.get('chart_type', '?')} -- {chart.get('task', '?')}")
        if chart.get("png_base64"):
            save_png(chart["png_base64"], f"insight_{i+1}_{chart['chart_type']}.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Viz-Agent Live Test -- Real Sales Data")
    print("Server must be running: python -m uvicorn app.main:app --port 8003")
    print("=" * 60)

    test_health()
    test_recommend()
    test_chart()
    test_auto_insights()

    print("\n" + "="*60)
    print("[DONE] All tests complete! Check ./test_outputs/ for PNGs and HTML")
    print("="*60)
