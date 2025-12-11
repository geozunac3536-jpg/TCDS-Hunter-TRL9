# TCDS Hunter/Crawler — TRL-9 Evidence Package

> **Teoría Cromodinámica Sincrónica (TCDS)**  
> Núcleo sísmico de campo: Hunter (soldado) + Crawler (cerebro global)  
> Evidencia de operación en condiciones cercanas a **TRL-9** con colapso entrópico profundo (ΔH ≪ 0).

---

## 1. Descripción

Este repositorio contiene el **paquete técnico de evidencia** del sistema:

- `Hunter` — nodo soldado de borde, que captura y analiza trazas sísmicas en tiempo real.
- `Crawler` — nodo cerebro global, que agrega eventos, aplica la Ley de Balance Coherencial Universal (LBCU) y emite alertas Σ.
- `Σ-metrics` — conjunto de métricas de coherencia que discriminan entre ruido y señal peligrosa:
  - LI (Locking Index)
  - R(t) (correlación temporal del locking)
  - RMSE_SL (error cuadrático medio entre señal y locking)
  - κΣ (curvatura coherencial)
  - ΔH (variación entrópica)

Este paquete demuestra el funcionamiento del sistema TCDS en condiciones de campo, incluyendo corridas reales con anomalías profundas en:

- Región Texas (ΔH ≈ −6.58)
- Región Puerto Rico (ΔH ≈ −5.17)

Los resultados están alineados con el criterio anti-apofenia del **E-Veto TCDS**:

> **E-Veto:** una señal no es válida aunque LI y R sean altos, a menos que cumpla  
> ΔH ≤ −0.20 en ventanas de coherencia Σ.

---

## 2. Referencia citable

Este repositorio está vinculado al registro en Zenodo:

- **DOI del paquete TRL-9:** `10.5281/zenodo.17885562`  
- **Página Zenodo:** https://doi.org/10.5281/zenodo.17885562

Por favor cite este trabajo como:

> Genaro Carrasco Ozuna (2025).  
> *TCDS Hunter/Crawler — TRL-9 Evidence Package (Seismic Causal Coherence Engine).*  
> Zenodo. https://doi.org/10.5281/zenodo.17885562

---

## 3. Estructura del repositorio

Resumen de las carpetas principales:

```text
.
├── src/               # Código fuente del Hunter, Crawler y núcleo Σ
├── data/              # Datos crudos/procesados (events_feed.jsonl, resúmenes)
├── docs/              # Reporte TRL-9 (LaTeX/PDF), figuras y tablas
├── web/               # Portal 4K con Reloj Causal Σ y Usage Registry
├── metadata/          # JSON-LD, esquemas de métricas y de uso
├── audit/             # Registros Σ-Trace de corridas específicas
├── notebooks/         # Notebooks de exploración y generación de figuras
└── zenodo_bundle/     # ZIP preparado para publicación en Zenodo
