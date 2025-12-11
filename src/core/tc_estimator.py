"""
tc_estimator.py

Estimación del Tiempo Causal t_C a partir de métricas de coherencia.

En TCDS:
    - t_C marca el momento en que σ domina la dinámica.
    - Puede asociarse al máximo de LI(t), al máximo de κΣ, o a una combinación.

Aquí implementamos una versión mínima reproducible:
    - t_C = tiempo del máximo de LI(t) en la ventana.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def estimate_t_c(
    li_series: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
    """
    Estima t_C a partir de una serie LI(t).

    Parameters
    ----------
    li_series : np.ndarray
        Serie LI(t) (Locking Index por subventanas).
    times : np.ndarray, optional
        Tiempos asociados a LI(t). Si es None, se asume índice entero.

    Returns
    -------
    t_c : float
        Tiempo estimado de máxima coherencia (dominio de σ).
    idx_max : int
        Índice del máximo en li_series.
    """
    li_series = np.asarray(li_series, dtype=float)
    if li_series.size == 0:
        return 0.0, -1

    idx_max = int(np.argmax(li_series))

    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.size > idx_max:
            t_c = float(times[idx_max])
        else:
            t_c = float(idx_max)
    else:
        t_c = float(idx_max)

    return t_c, idx_max
