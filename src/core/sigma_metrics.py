"""
sigma_metrics.py

Cálculo de Σ-metrics para TCDS Hunter/Crawler en versión σ-céntrica:

- LI  → Locking Index (firma directa del Sincronón en la ventana).
- R   → correlación de Pearson entre señal y patrón de locking.
- RMSE_SL → error cuadrático medio entre señal y componente de locking.
- κΣ  → curvatura coherencial (dinámica de LI en el tiempo).

Aquí adoptamos explícitamente:

    Q ≡ σ_intensidad ≡ LI

Es decir: LI no es solo “una métrica más”; es la forma observable mínima
del Sincronón en cada ventana. El resto de métricas ayudan a validar
si esa LI es robusta, significativa y consistente con ΔH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class SigmaMetrics:
    """
    Contenedor de métricas Σ en una ventana.

    Attributes
    ----------
    LI : float
        Locking Index (0–1). Intensidad observable del Sincronón σ.
    R : float
        Correlación de Pearson entre señal y patrón de locking.
    RMSE_SL : float
        Root Mean Squared Error entre señal normalizada y locking normalizado.
    kappa_sigma : Optional[float]
        Curvatura coherencial κΣ, derivada de la serie temporal LI(t).
    extra : Dict[str, Any]
        Campo para adjuntar detalles (LI_series, tiempos, etc.).
    """
    LI: float
    R: float
    RMSE_SL: float
    kappa_sigma: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normaliza un array restando la media y dividiendo por la desviación estándar.

    Si la desviación es casi cero, retorna un array de ceros para evitar
    divisiones numéricamente inestables.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(1, dtype=float)
    m = np.mean(x)
    s = np.std(x)
    if s < eps:
        return np.zeros_like(x)
    return (x - m) / s


def _corr_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Correlación de Pearson robusta a constantes y longitudes pequeñas.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(x.size, y.size)
    if n < 2:
        return 0.0

    x = x[:n]
    y = y[:n]

    x_norm = _normalize(x, eps=eps)
    y_norm = _normalize(y, eps=eps)

    if np.allclose(x_norm, 0.0) or np.allclose(y_norm, 0.0):
        return 0.0

    r = np.mean(x_norm * y_norm)
    r = float(np.clip(r, -1.0, 1.0))
    return r


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Root Mean Squared Error entre dos arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    diff = x[:n] - y[:n]
    return float(np.sqrt(np.mean(diff * diff)))


def compute_LI(
    signal: np.ndarray,
    locking_pattern: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    Calcula el Locking Index (LI) en una ventana.

    Definición operativa:

        LI = |r|

    donde r es la correlación de Pearson entre:

        - señal normalizada, y
        - patrón de locking normalizado.

    Interpretación σ-céntrica:

    - LI ≈ 0   → σ prácticamente ausente (domina φ).
    - LI ≈ 1   → σ domina el comportamiento de la señal (locking máximo).

    Parameters
    ----------
    signal : np.ndarray
        Señal de la ventana.
    locking_pattern : np.ndarray
        Patrón de locking Σ (envolvente, plantilla, modo, etc.).
    eps : float, optional
        Término de regularización numérica.

    Returns
    -------
    float
        LI en [0, 1].
    """
    r = _corr_pearson(signal, locking_pattern, eps=eps)
    LI = float(abs(r))
    return LI


def estimate_kappa_sigma(
    li_series: np.ndarray,
    times: Optional[np.ndarray] = None
) -> float:
    """
    Estima la curvatura coherencial κΣ a partir de LI(t).

    Aproximación mínima reproducible:

    - Se toma la segunda derivada discreta de LI(t).
    - κΣ = max |d²LI/dt²| en la ventana.

    Interpretación:

    - κΣ alto indica cambios rápidos en la intensidad de σ (nucleación, transición).
    - κΣ bajo indica régimen estacionario.

    Parameters
    ----------
    li_series : np.ndarray
        Serie temporal LI(t), tamaño >= 3 recomendado.
    times : np.ndarray, optional
        Tiempos asociados a LI(t) (no se usan aún, pero se conserva el parámetro).

    Returns
    -------
    float
        κΣ estimada. Si la serie es muy corta, retorna 0.0.
    """
    li_series = np.asarray(li_series, dtype=float)
    n = li_series.size
    if n < 3:
        return 0.0

    d2 = li_series[:-2] - 2 * li_series[1:-1] + li_series[2:]
    kappa = float(np.max(np.abs(d2)))
    return kappa


def compute_sigma_metrics(
    signal: np.ndarray,
    locking_pattern: np.ndarray,
    *,
    li_series: Optional[np.ndarray] = None,
    li_times: Optional[np.ndarray] = None
) -> SigmaMetrics:
    """
    Calcula Σ-metrics en una ventana de señal.

    Parameters
    ----------
    signal : np.ndarray
        Señal 1D de la ventana.
    locking_pattern : np.ndarray
        Patrón de locking Σ (misma resolución temporal que `signal`).
    li_series : np.ndarray, optional
        Serie LI(t) de subventanas internas (si ya está calculada).
    li_times : np.ndarray, optional
        Tiempos asociados a li_series.

    Returns
    -------
    SigmaMetrics
        LI, R, RMSE_SL, κΣ (si disponible) y extras.
    """
    signal = np.asarray(signal, dtype=float)
    locking_pattern = np.asarray(locking_pattern, dtype=float)

    # R: correlación de Pearson
    R = _corr_pearson(signal, locking_pattern)

    # LI: módulo de R → intensidad observable de σ
    LI = float(abs(R))

    # RMSE_SL: error entre señal y locking, ambos estandarizados
    signal_norm = _normalize(signal)
    locking_norm = _normalize(locking_pattern)
    RMSE_SL = _rmse(signal_norm, locking_norm)

    # κΣ: dinámica de LI(t)
    kappa_sigma = None
    extra: Dict[str, Any] = {}

    if li_series is not None:
        kappa_sigma = estimate_kappa_sigma(li_series, li_times)
        extra["li_series"] = np.asarray(li_series, dtype=float).tolist()
        if li_times is not None:
            extra["li_times"] = np.asarray(li_times, dtype=float).tolist()

    return SigmaMetrics(
        LI=float(LI),
        R=float(R),
        RMSE_SL=float(RMSE_SL),
        kappa_sigma=None if kappa_sigma is None else float(kappa_sigma),
        extra=extra if extra else None,
    )

