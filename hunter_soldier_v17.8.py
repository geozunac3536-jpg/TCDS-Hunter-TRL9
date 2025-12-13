# ==============================================================================
# TCDS HUNTER V17.8 — LUNAR COMMANDER (OMNI SENTINEL + TIDAL STRESS)
# ==============================================================================
# ESTADO: TRL-9 (FULL STACK + ASTRONOMY)
# MÓDULOS ACTIVOS:
#   1. SOLDIER SÍSMICO: Detección de Ruptura (Entropía ΔH + LI)
#   2. SENTINEL VOLCÁNICO: Smithsonian GVP Feed
#   3. SPACE OPS: NOAA SWPC (Kp Index)
#   4. LUNAR OPS: Cálculo de Altitud/Fase Lunar (Tidal Stress)
#   5. REPORTING: Radar HD + Email Táctico
#   6. CLOUD LINK: Persistencia en Google Drive
# ==============================================================================

import subprocess
import sys
import os

# --- 0. GESTOR DE DEPENDENCIAS (AUTO-INSTALL) ---
def install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"🔧 INSTALANDO {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

modules = ["obspy", "geopy", "matplotlib", "scipy", "requests", "imageio", "ephem"]
for mod in modules:
    install(mod)

import time
import json
import smtplib
import ssl
import threading
import hashlib
import requests
import ephem
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from scipy.stats import entropy

import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from geopy.distance import geodesic

# --- 1. CONFIGURACIÓN MAESTRA ---
CONFIG = {
    "email": {
        "user": "",
        "pass": "",
        "to": [""]
    },
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    },
    "seismic": {
        "window_hours": 24,
        "poll_interval": 5,  # minutos
        "min_mag": 0.0,
        "max_radius_km": 3500,
        "e_veto_dh": -0.18,
        "window_pre": 7,   # minutos
        "window_post": 15, # minutos
        "providers": ["IRIS", "GEOFON", "USGS", "ETH"]
    },
    "volcano": {
        "source_url": "https://volcano.si.edu/news/WeeklyVolcanoRSS.xml",
        "poll_interval": 900
    },
    "space": {
        "kp_source": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
        "poll_interval": 900,
        "storm_threshold": 4.0
    },
    "paths": {
        "root": "/content/drive/MyDrive/HUNTER_V17_RUNS",
        "feed": "events_feed.jsonl",
        "imgs": "evidencia_hd",
        "memory": "tcds_memory.json"
    }
}

NEON_CYAN = '#00F3FF'
NEON_MAGENTA = '#FF00FF'
NEON_GREEN = '#39FF14'
BG_COLOR = '#050505'

# MEMORIA COMPARTIDA (CLIMA ESPACIAL)
_CURRENT_ENV = {"kp_index": 0.0, "status": "UNKNOWN", "last_update": 0}

# --- 2. CONEXIÓN CLOUD (DRIVE) ---
def activar_drive():
    try:
        from google.colab import drive
        print("🔌 CONECTANDO A INFRAESTRUCTURA CLOUD (DRIVE)...")
        drive.mount('/content/drive')

        base = CONFIG["paths"]["root"]
        CONFIG["paths"]["feed"] = os.path.join(base, "events_feed.jsonl")
        CONFIG["paths"]["imgs"] = os.path.join(base, "evidencia_hd")
        CONFIG["paths"]["memory"] = os.path.join(base, "state", "tcds_memory.json")

        os.makedirs(CONFIG["paths"]["imgs"], exist_ok=True)
        os.makedirs(os.path.dirname(CONFIG["paths"]["memory"]), exist_ok=True)

        # Asegura que el feed exista
        os.makedirs(os.path.dirname(CONFIG["paths"]["feed"]), exist_ok=True)

        print(f"🚀 SISTEMA PERSISTENTE: {base}")
    except Exception as e:
        print(f"⚠️ MODO LOCAL (EFÍMERO). ({e})")
        base = "/content/HUNTER_LOCAL"
        os.makedirs(base, exist_ok=True)
        CONFIG["paths"]["feed"] = os.path.join(base, "events_feed.jsonl")
        CONFIG["paths"]["imgs"] = os.path.join(base, "evidencia_hd")
        CONFIG["paths"]["memory"] = os.path.join(base, "state", "tcds_memory.json")
        os.makedirs(CONFIG["paths"]["imgs"], exist_ok=True)
        os.makedirs(os.path.dirname(CONFIG["paths"]["memory"]), exist_ok=True)

# --- 3. SPACE OPS (NOAA) ---
def space_weather_loop():
    print("🌌 [THREAD] Space Weather Sentinel: ACTIVO (NOAA SWPC)")
    while True:
        try:
            r = requests.get(CONFIG["space"]["kp_source"], timeout=20)
            if r.status_code == 200:
                data = r.json()
                latest = data[-1]
                kp = float(latest[1])

                status = "QUIET"
                if kp >= 4: status = "UNSETTLED"
                if kp >= 5: status = "STORM"

                _CURRENT_ENV["kp_index"] = kp
                _CURRENT_ENV["status"] = status
                _CURRENT_ENV["last_update"] = time.time()

                if kp >= float(CONFIG["space"]["storm_threshold"]):
                    rec = {
                        "ev_id": f"SPACE_{int(time.time())}",
                        "event_type": "SPACE_WEATHER",      # ✅ clave para tu Crawler
                        "region_text": "GLOBAL IONOSPHERE",
                        "mag": kp,
                        "metrics": {"dH": 0.0, "LI": 0.0},
                        "space_env": {"kp": kp, "status": status},
                        "t_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "desc": f"GEOMAGNETIC STORM: Kp {kp} ({status})"
                    }
                    try:
                        with open(CONFIG["paths"]["feed"], "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(f"      🌌 ALERTA CLIMA ESPACIAL: Kp {kp} ({status})")
                    except:
                        pass
        except:
            pass

        time.sleep(int(CONFIG["space"]["poll_interval"]))

# --- 4. VOLCANO OPS (SMITHSONIAN) ---
def volcano_loop():
    print("🔥 [THREAD] Sentinel Volcánico: ACTIVO (Smithsonian)")
    processed = set()
    while True:
        try:
            r = requests.get(CONFIG["volcano"]["source_url"], headers=CONFIG["headers"], timeout=20)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                for item in root.findall('./channel/item'):
                    title = (item.find('title').text or "").strip()
                    desc = (item.find('description').text or "").strip()
                    uid = hashlib.sha256((title + desc[:50]).encode("utf-8")).hexdigest()[:8]

                    if uid in processed:
                        continue

                    severity = "ACTIVITY"
                    dH_est = -0.5
                    dlow = desc.lower()

                    if "lava" in dlow or "eruption" in dlow:
                        severity = "ERUPTION"; dH_est = -4.5
                    elif "ash" in dlow or "explosion" in dlow:
                        severity = "EXPLOSION"; dH_est = -2.0

                    rec = {
                        "ev_id": f"VOLC_{int(time.time())}",
                        "event_type": "VOLCANIC",
                        "region_text": f"VOLCANO: {title}",
                        "mag": 0.0,
                        "metrics": {"dH": dH_est, "LI": 0.8},
                        "t_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "desc": f"[{severity}] {desc[:200]}..."
                    }
                    try:
                        with open(CONFIG["paths"]["feed"], "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(f"      🌋 ACTIVIDAD VOLCÁNICA: {title} [{severity}]")
                    except:
                        pass

                    processed.add(uid)
        except:
            pass

        time.sleep(int(CONFIG["volcano"]["poll_interval"]))

# --- 5. LUNAR OPS (TIDAL STRESS ENGINE) ---
def calcular_luna(lat, lon, utc_time):
    """Calcula la posición de la Luna relativa al epicentro."""
    try:
        obs = ephem.Observer()
        obs.lat = str(lat)
        obs.lon = str(lon)

        # utc_time puede venir como datetime
        if isinstance(utc_time, datetime):
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            obs.date = ephem.Date(utc_time)
        else:
            obs.date = utc_time  # fallback

        moon = ephem.Moon()
        moon.compute(obs)

        alt_deg = np.degrees(float(moon.alt))
        az_deg = np.degrees(float(moon.az))
        phase = float(moon.phase)

        pos_desc = "HORIZONTE (AXIAL)"
        if alt_deg > 60:
            pos_desc = "CENIT (VERTICAL PULL)"
        elif alt_deg < -60:
            pos_desc = "NADIR (OPPOSITE PULL)"
        elif -30 < alt_deg < 30:
            pos_desc = "HORIZONTE (TANGENTIAL)"

        return {
            "alt": round(alt_deg, 2),
            "az": round(az_deg, 2),
            "phase": round(phase, 1),
            "desc": pos_desc
        }
    except Exception as e:
        return {"alt": 0.0, "az": 0.0, "phase": 0.0, "desc": "ERROR", "error": str(e)}

# --- 6. CORE SÍSMICO ---
CLIENTS = {}

def init_federation():
    print(" 🔗 Conectando Federación Sísmica...")
    for prov in CONFIG["seismic"]["providers"]:
        try:
            CLIENTS[prov] = Client(prov, timeout=8)
            print(f"   ✅ {prov:<6} [ON]")
        except:
            pass

def calcular_fisica_v17(tr):
    try:
        data = tr.data.astype(float)
        if np.std(data) == 0:
            return None, 0.0, 0.0
        data = (data - np.mean(data)) / np.std(data)

        energy = data**2
        prob = energy / np.sum(energy)
        dH = entropy(prob) - np.log(len(data))

        mid = len(data) // 2
        LI = abs(np.corrcoef(data[:mid], data[mid:2*mid])[0, 1]) if mid > 10 else 0.0

        idx_max = np.argmax(np.abs(data))
        tC = tr.stats.starttime + (idx_max * tr.stats.delta)

        return tC, float(LI), float(dH)
    except:
        return None, 0.0, 0.0

def generar_radar_hd(tr, info, tC, LI, dH, ev_id, lunar_data):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    times = tr.times()
    data = tr.data
    env = np.abs(data)

    if len(data) > 50000:
        factor = 2
        times = times[::factor]
        data = data[::factor]
        env = env[::factor]

    ax1.plot(times, data, color=NEON_CYAN, lw=0.6)
    ax1.set_title(f"TCDS SIGNAL TRACE | {info}", color='white', loc='left',
                  fontsize=10, fontweight='bold')
    ax1.set_ylabel("Amplitude (σ)", color='gray')
    ax1.grid(True, which='major', color='#333333', linestyle='--')
    ax1.set_facecolor(BG_COLOR)

    ax2.plot(times, env, color=NEON_MAGENTA, lw=0.8)
    ax2.fill_between(times, env, color=NEON_MAGENTA, alpha=0.1)

    env_str = f"SPACE WX: Kp {_CURRENT_ENV['kp_index']} ({_CURRENT_ENV['status']})"
    moon_str = f"MOON: Alt {lunar_data.get('alt', 0)}° ({lunar_data.get('desc', 'UNK')})"

    stats_box = (
        "PHYSICS METRICS\n"
        "----------------\n"
        f"LI (Locking): {LI:.4f}\n"
        f"dH (Entropy): {dH:.4f}\n"
        f"{env_str}\n"
        f"{moon_str}"
    )

    ax2.text(
        0.02, 0.85, stats_box, transform=ax2.transAxes,
        fontsize=9, color=NEON_GREEN, fontfamily='monospace',
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='gray')
    )

    ax2.set_xlabel("Time (s)", color='gray')
    ax2.set_facecolor(BG_COLOR)

    path = os.path.join(CONFIG["paths"]["imgs"], f"RADAR_HD_{ev_id}.png")
    plt.savefig(path, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    return path

def buscar_traza(ev, master):
    try:
        origin = ev.preferred_origin()
        lat = origin.latitude
        lon = origin.longitude
        t0 = origin.time

        t_pre = int(CONFIG["seismic"]["window_pre"]) * 60
        t_post = int(CONFIG["seismic"]["window_post"]) * 60

        inv = master.get_stations(
            latitude=lat, longitude=lon,
            maxradius=float(CONFIG["seismic"]["max_radius_km"]) / 111.1,
            level="station"
        )

        best_sta = None
        min_dist = 999999.0
        for net in inv:
            for sta in net:
                d = geodesic((lat, lon), (sta.latitude, sta.longitude)).km
                if d < min_dist:
                    min_dist = d
                    best_sta = (net.code, sta.code)
                if min_dist < 50:
                    break
            if min_dist < 50:
                break

        if not best_sta:
            return None, "NO_EST"

        for prov in CONFIG["seismic"]["providers"]:
            if prov not in CLIENTS:
                continue
            try:
                st = CLIENTS[prov].get_waveforms(
                    best_sta[0], best_sta[1], "*", "BHZ",
                    t0 - t_pre, t0 + t_post
                )
                if len(st) > 0:
                    tr = st[0]
                    tr.detrend("linear")
                    tr.filter("bandpass", freqmin=0.5, freqmax=8.0)
                    return tr, f"{prov}::{best_sta[0]}.{best_sta[1]}"
            except:
                continue
    except:
        pass

    return None, "NO_DATA"

def enviar_email_hd(data, img_path):
    cfg = CONFIG["email"]
    if not cfg["user"] or not cfg["pass"] or not cfg["to"]:
        print("⚠️ Email no configurado (user/pass/to). Se omite envío.")
        return

    msg = MIMEMultipart()
    msg['From'] = cfg["user"]
    msg['To'] = ", ".join(cfg["to"])

    icon = "🚨" if data.get('nucleation') else "⚠️"
    moon_alt = data.get('moon', {}).get('alt', 0)
    msg['Subject'] = f"{icon} HUNTER LUNAR: {data.get('region','UNK')[:60]} | M{data.get('mag',0)} | Moon {moon_alt}°"

    body = f"""REPORTE DE INTELIGENCIA TCDS LUNAR
=======================================
OBJETIVO
Región: {data.get('region','UNK')}
Magnitud: {data.get('mag',0)}
Física: LI={data.get('LI',0):.4f} | dH={data.get('dH',0):.4f}

CONTEXTO GRAVITACIONAL (TIDAL STRESS)
Posición Luna: {data.get('moon',{}).get('desc','UNK')}
Altitud: {data.get('moon',{}).get('alt',0)}°
Fase: {data.get('moon',{}).get('phase',0)}%

CONTEXTO IONOSFÉRICO (SPACE OPS)
Kp Index: {data.get('env',{}).get('kp',0)} ({data.get('env',{}).get('status','UNK')})

EVIDENCIA
Radar HD adjunto.
=======================================
"""
    msg.attach(MIMEText(body, 'plain'))

    if img_path and os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(img_path))
            msg.attach(img)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ctx) as s:
            s.login(cfg["user"], cfg["pass"])
            s.send_message(msg)
        print("      ✉️ Reporte LUNAR HD enviado.")
    except Exception as e:
        print(f"      ❌ Error SMTP: {e}")

# --- 7. MAIN LOOP ---
def main():
    print("=== TCDS HUNTER V17.8 LUNAR COMMANDER ===")
    activar_drive()
    init_federation()

    threading.Thread(target=volcano_loop, daemon=True).start()
    threading.Thread(target=space_weather_loop, daemon=True).start()

    master = CLIENTS.get("IRIS") or CLIENTS.get("USGS")
    if not master:
        print("💀 SIN CONEXIÓN GLOBAL")
        return

    memoria = set()
    if os.path.exists(CONFIG["paths"]["memory"]):
        try:
            with open(CONFIG["paths"]["memory"], "r", encoding="utf-8") as f:
                memoria = set(json.load(f))
        except:
            pass

    print(f" 🧠 Memoria: {len(memoria)} eventos.")
    print(" 📡 Escaneando cielo y tierra...")

    while True:
        try:
            t_end = UTCDateTime.now()
            t_start = t_end - (float(CONFIG["seismic"]["window_hours"]) * 3600)

            cat = master.get_events(
                starttime=t_start,
                endtime=t_end,
                minmagnitude=float(CONFIG["seismic"]["min_mag"])
            )

            nuevos = [ev for ev in cat if str(ev.resource_id) not in memoria]

            if nuevos:
                print(f"\n🔎 Analizando {len(nuevos)} objetivos...")
                for ev in nuevos:
                    eid = str(ev.resource_id)
                    memoria.add(eid)

                    try:
                        mag = float(ev.preferred_magnitude().mag) if ev.preferred_magnitude() else 0.0
                        reg = ev.event_descriptions[0].text if ev.event_descriptions else "UNKNOWN"
                        origin = ev.preferred_origin()
                        if origin is None:
                            continue

                        tr, src_info = buscar_traza(ev, master)
                        if not tr:
                            continue

                        tC, li, dH = calcular_fisica_v17(tr)
                        if tC is None:
                            continue

                        is_nuc = (dH <= float(CONFIG["seismic"]["e_veto_dh"]))
                        env_now = {"kp": float(_CURRENT_ENV["kp_index"]), "status": str(_CURRENT_ENV["status"])}

                        moon_data = calcular_luna(origin.latitude, origin.longitude, origin.time.datetime)

                        print(f"      🧠 {reg[:30]}... | dH={dH:.3f} | Kp={env_now['kp']} | Moon={moon_data.get('alt',0)}°")

                        if is_nuc:
                            safe_id = "".join(x for x in eid if x.isalnum())[-12:] or f"EV{int(time.time())}"
                            img_path = generar_radar_hd(tr, f"M{mag} {reg}", tC, li, dH, safe_id, moon_data)

                            data = {
                                "id": eid,
                                "region": reg,
                                "mag": mag,
                                "nucleation": True,
                                "dH": dH,
                                "LI": li,
                                "env": env_now,
                                "moon": moon_data
                            }

                            enviar_email_hd(data, img_path)

                            rec = {
                                "ev_id": eid,
                                "event_type": "SEISMIC",
                                "region_text": reg,
                                "mag": mag,
                                "metrics": {"dH": dH, "LI": li},
                                "space_env": env_now,
                                "lunar_data": moon_data,
                                "e_veto": {"pass": True},
                                "t_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            }

                            with open(CONFIG["paths"]["feed"], "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec) + "\n")

                    except:
                        pass

                try:
                    with open(CONFIG["paths"]["memory"], "w", encoding="utf-8") as f:
                        json.dump(list(memoria), f)
                except:
                    pass

        except Exception as e:
            print(f"Error Loop: {e}")

        time.sleep(int(CONFIG["seismic"]["poll_interval"]) * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 SISTEMA DETENIDO.")