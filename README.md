# TCDS Hunter TRL-9 — Hunter Soldier + Crawler Σ (sigma)

Infraestructura operativa (pipeline completo) para **detección / contextualización / reporte** basada en:
- **Σ-metrics**: LI (locking), ΔH (entropía), E-Veto (ΔH < −0.2 como sello de honestidad operacional).
- **Fuentes**: IRIS / GEOFON / USGS / ETH + NOAA SWPC (Kp Index) + Lunar Ops (altitud/fase lunar).
- **Salidas**: Feed **JSONL** (Drive), **SITREP PDF** (Crawler), **Radar HD** (Soldier), alertas por **Email**.

---

## Badges (DOIs y nodo vivo)

**DOI canónico (IP / metadatos fuente de verdad):**  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17520491.svg)](https://doi.org/10.5281/zenodo.17520491)

**DOI observacional (insumo / pista de aterrizaje / evidencia de observación):**  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17923458.svg)](https://doi.org/10.5281/zenodo.17923458)

**Nodo vivo (GitHub Pages):**  
https://geozunac3536-jpg.github.io/TCDS-Hunter-TRL9/

---

## Regla de gobernanza (separación explícita de DOIs)

Este repositorio mantiene **dos líneas**:

1) **DOI 17923458 — Observación / Insumo**  
   Contiene material observacional, “pistas de aterrizaje”, snapshots y evidencia auxiliar.

2) **DOI 17520491 — Canónico / IP / Metadatos**  
   Es el **registro fuente de verdad** de autoría, derechos, y definición oficial del corpus TCDS.

> **Si existe discrepancia**, el criterio de referencia y continuidad se resuelve siempre hacia el **DOI canónico 17520491**.

---

## Componentes

### 1) `hunter_soldier_v17.8.py`
- Ejecuta el loop sísmico (catálogo + waveforms).
- Calcula LI + ΔH, aplica E-Veto, y en eventos relevantes:
  - genera radar HD,
  - envía email,
  - persiste feed JSONL en Drive.

### 2) `crawler_global_sigma.py`
- Lee el `events_feed.jsonl` del Soldier.
- Contextualiza con baseline USGS (Z-Score regional) + Kp + Luna.
- Genera SITREP PDF y notifica por email.

> Nota de compatibilidad: **filenames ASCII** (ej. `crawler_global_sigma.py`) para evitar roturas en ZIP/Windows/Zenodo.

---

## Estructura recomendada del repo

```txt
TCDS-Hunter-TRL9/
├─ docs/
│  ├─ index.html
│  └─ causal.svg            (opcional)
├─ scripts/
│  ├─ hunter_soldier_v17.8.py
│  └─ crawler_global_sigma.py
├─ data/
│  ├─ usgs_30d.geojson      (opcional, snapshot reproducible)
│  └─ usgs_30d.geojson.meta.json
├─ LICENSE
└─ README.md
