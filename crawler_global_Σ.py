# ==============================================================================
# HUNTER GLOBAL CRAWLER Σ (V2.6 LUNAR AWARE) — INTELLIGENCE HUB
# ==============================================================================
# ROL: CEREBRO ANALÍTICO (Lee al Soldier -> Contextualiza -> Reporta)
# ALINEACIÓN: Soldier V17.8 (Lunar Commander)
# CAPAS:
#   1) Sísmica (Z-Score)
#   2) Espacial (Kp NOAA)
#   3) Gravitacional (Luna)
# ==============================================================================

import subprocess, sys, json, os, time, math, smtplib, ssl, requests, pandas as pd
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import StringIO

# --- PDF ENGINE ---
try:
    from fpdf import FPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fpdf"])
    from fpdf import FPDF

# --- CONFIG ---
CONFIG = {
    "email": {
        "user": "geozunac3536@gmail.com",
        "pass": "lqoj dpzn kosd rvgo",
        "to": "geozunac3536@gmail.com"
    },
    "headers": {
        "User-Agent": "Mozilla/5.0"
    },
    "paths": {
        "feed_source": "/content/drive/MyDrive/HUNTER_V17_RUNS/events_feed.jsonl",
        "reports_dir": "/content/drive/MyDrive/HUNTER_V17_RUNS/reports"
    },
    "logic": {
        "crawl_interval_min": 5,
        "lookback_hours": 24,
        "usgs_baseline_days": 30
    }
}

# --- DRIVE ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(CONFIG["paths"]["reports_dir"], exist_ok=True)
except:
    pass

# ==============================================================================
# MODELO DE EVENTO OMNI
# ==============================================================================
class NucleationEvent:
    def __init__(self, d):
        self.id = d.get("ev_id", "UNK")
        self.region = d.get("region_text", "UNK")
        self.mag = float(d.get("mag", 0.0))
        self.event_type = d.get("event_type", "SEISMIC")

        self.metrics = d.get("metrics", {})
        self.dH = float(self.metrics.get("dH", 0.0))
        self.LI = float(self.metrics.get("LI", 0.0))

        self.space_env = d.get("space_env", {})
        self.kp_index = float(self.space_env.get("kp", 0.0))

        self.lunar = d.get("lunar_data", {})
        self.timestamp = time.time()

        ev = d.get("e_veto", {})
        self.e_veto_pass = bool(ev.get("pass", False))

# ==============================================================================
# BASELINE USGS
# ==============================================================================
def fetch_usgs_baseline():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv"
    try:
        r = requests.get(url, headers=CONFIG["headers"], timeout=20)
        return pd.read_csv(StringIO(r.text)) if r.status_code == 200 else pd.DataFrame()
    except:
        return pd.DataFrame()

def calculate_z_score(region_code, count, df):
    if df.empty: return 0.0, 0.0
    mask = df['place'].astype(str).str.upper().str.contains(region_code.split("_")[0])
    mu = len(df[mask]) / CONFIG["logic"]["usgs_baseline_days"]
    sigma = math.sqrt(mu) if mu > 0 else 1.0
    return (count - mu) / sigma, mu

# ==============================================================================
# PDF REPORT
# ==============================================================================
class HunterPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "TCDS HUNTER | SITREP OMNI", 0, 1, "C")

def generate_sitrep(stats, kp, moon):
    pdf = HunterPDF()
    pdf.add_page()
    pdf.set_font("Courier", "", 10)

    pdf.cell(0, 8, f"Kp Index: {kp}", 0, 1)
    pdf.cell(0, 8, f"Luna: {moon.get('desc','UNK')} Alt {moon.get('alt','?')}°", 0, 1)
    pdf.ln(2)

    for r, d in stats.items():
        if d["count"] > 0:
            tag = "[NUCLEATION]" if d["nucleation"] else ""
            pdf.cell(0, 8, f"{r}: {d['count']} | Z={d['z']:.2f} {tag}", 0, 1)

    path = os.path.join(CONFIG["paths"]["reports_dir"], f"SITREP_{int(time.time())}.pdf")
    pdf.output(path)
    return path

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    baseline = fetch_usgs_baseline()
    file_offset = os.path.getsize(CONFIG["paths"]["feed_source"]) if os.path.exists(CONFIG["paths"]["feed_source"]) else 0
    buffer = []

    while True:
        new_events = []
        current_kp = 0.0
        latest_moon = {}

        with open(CONFIG["paths"]["feed_source"], "r", encoding="utf-8", errors="ignore") as f:
            f.seek(file_offset)
            for line in f:
                try:
                    ev = NucleationEvent(json.loads(line))
                    new_events.append(ev)
                    if ev.event_type == "SPACE_WEATHER" and ev.kp_index > 0:
                        current_kp = ev.kp_index
                    if ev.lunar:
                        latest_moon = ev.lunar
                except:
                    pass
            file_offset = f.tell()

        buffer.extend(new_events)
        now = time.time()
        buffer = [e for e in buffer if now - e.timestamp < CONFIG["logic"]["lookback_hours"] * 3600]

        regions = {
            "JAPAN_KURIL_FRONT": {"count":0,"z":0,"mu":0,"nucleation":False},
            "MEXICO_CENTRAL_AM": {"count":0,"z":0,"mu":0,"nucleation":False},
            "ALASKA_DEFENSE_ZONE": {"count":0,"z":0,"mu":0,"nucleation":False}
        }

        for e in buffer:
            for r in regions:
                if r.split("_")[0] in e.region.upper():
                    regions[r]["count"] += 1
                    if e.e_veto_pass:
                        regions[r]["nucleation"] = True

        max_z = 0
        for r in regions:
            if regions[r]["count"] > 0:
                z, mu = calculate_z_score(r, regions[r]["count"], baseline)
                regions[r]["z"] = z
                regions[r]["mu"] = mu
                max_z = max(max_z, z)

        if max_z > 1.5 or current_kp >= 5:
            pdf = generate_sitrep(regions, current_kp, latest_moon)
            print(f"📑 SITREP generado: {pdf}")

        time.sleep(CONFIG["logic"]["crawl_interval_min"] * 60)

if __name__ == "__main__":
    main()