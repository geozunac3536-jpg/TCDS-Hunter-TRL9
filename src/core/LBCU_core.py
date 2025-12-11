"""
lBCU_core.py

Implementación básica de la Ley de Balance Coherencial Universal (LBCU)
y utilidades de entropía para el cálculo de ΔH en el contexto TCDS.

La idea operativa aquí es simple y auditable:

- H(x): entropía de Shannon de la distribución de amplitudes de la señal.
- H(ref): entropía de una referencia "sin estructura" (ruido o señal barajada).
- ΔH = H(ref) - H(x)

Si ΔH << 0, la señal x es mucho más ordenada que la referencia: hay colapso
entrópico coherencial (candidato a nucleación).

Este módulo no sabe nada de sismología per se: sólo hace cálculo de H y ΔH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LBCUResult:
    """
    Resultado del balance LBCU en una ventana.

    A nivel operativo, interpretamos:

    - Q:  "empuje coherencial" (puede ser una medida de inyección de orden).
    - Sigma: "coherencia medida" (ej. LI o combinación de métricas Σ).
    - phi: "fricción efectiva" (lo que queda para balancear Q·Sigma).

    En esta implementación, phi es simplemente Q * Sigma, pero dejamos el
    dataclass listo para extensiones más elaboradas.
    """
    Q: float
    Sigma: float
    phi: float
    delta_H: float


def compute_entropy(
    x: np.ndarray,
    n_bins: int = 64,
    eps: float = 1e-12
) -> float:
    """
    Calcula la entropía de Shannon de la distribución de amplitudes de `x`.

    Parameters
    ----------
    x : np.ndarray
        Señal (1D). Se ignorarán NaNs e infinitos.
    n_bins : int, optional
        Número de bins para el histograma, por defecto 64.
    eps : float, optional
        Pequeño término para evitar log(0), por defecto 1e-12.

    Returns
    -------
    float
        Entropía de Shannon H(x) en nats.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0

    # Normalizamos amplitudes a rango [0, 1] para estabilidad
    xmin, xmax = np.min(x), np.max(x)
    if xmax <= xmin:
        # Toda la señal es constante → entropía casi cero
        return 0.0

    x_norm = (x - xmin) / (xmax - xmin)

    hist, _ = np.histogram(x_norm, bins=n_bins, range=(0.0, 1.0), density=True)
    p = hist / np.sum(hist)  # normalizamos a distribución de probabilidad
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p))
    return float(H)


def _make_surrogate(
    x: np.ndarray,
    random_state: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Construye un surrogate barajando la señal (preserva histograma, destruye orden temporal).
    """
    x = np.asarray(x, dtype=float)
    if random_state is None:
        random_state = np.random.default_rng()
    surrogate = x.copy()
    random_state.shuffle(surrogate)
    return surrogate


def compute_delta_H(
    x: np.ndarray,
    reference: Optional[np.ndarray] = None,
    n_bins: int = 64,
    random_state: Optional[np.random.Generator] = None
) -> Tuple[float, float, float]:
    """
    Calcula ΔH = H(ref) - H(x) para una ventana.

    - Si `reference` es None, se usa un surrogate barajado de `x` como referencia.
    - Si se pasa un array en `reference`, se usa su entropía como H(ref).

    Parameters
    ----------
    x : np.ndarray
        Señal de la ventana.
    reference : np.ndarray, optional
        Señal de referencia (ruido / baseline) de la misma naturaleza que `x`.
        Si es None, se genera un surrogate barajado a partir de `x`.
    n_bins : int, optional
        Número de bins para el histograma, por defecto 64.
    random_state : np.random.Generator, optional
        Generador aleatorio para el surrogate.

    Returns
    -------
    delta_H : float
        Diferencia H(ref) - H(x).
    H_x : float
        Entropía de la señal.
    H_ref : float
        Entropía de la referencia.
    """
    x = np.asarray(x, dtype=float)

    H_x = compute_entropy(x, n_bins=n_bins)

    if reference is None:
        ref = _make_surrogate(x, random_state=random_state)
    else:
        ref = np.asarray(reference, dtype=float)

    H_ref = compute_entropy(ref, n_bins=n_bins)
    delta_H = H_ref - H_x
    return float(delta_H), float(H_x), float(H_ref)


def lBCU_balance(
    Q: float,
    Sigma: float,
    delta_H: float
) -> LBCUResult:
    """
    Aplica la forma operativa mínima de la LBCU en una ventana:

        Q · Σ = φ

    donde Q se interpreta como "empuje coherencial" (por ejemplo, intensidad
    del precursor), Σ como grado de coherencia (ej. LI) y φ como fricción
    efectiva.

    Esta función no inventa Q ni Σ: tú decides qué alimentar; aquí solo
    se registra el balance y se adjunta el ΔH calculado.

    Parameters
    ----------
    Q : float
        Medida de empuje coherencial (escala libre, pero consistente en tu análisis).
    Sigma : float
        Medida de coherencia Σ (por ejemplo, LI).
    delta_H : float
        Variación entrópica ΔH asociada a la ventana.

    Returns
    -------
    LBCUResult
        Objeto con Q, Sigma, phi = Q·Sigma y ΔH.

    Notes
    -----
    En análisis posteriores podrías revisar condiciones como:

    - Q·Σ suficientemente grande y ΔH << 0 → fuerte candidato a nucleación.
    - Q·Σ grande pero ΔH ≈ 0 → posible apofenia (señal coherente pero sin
      reducción entrópica real, el E-Veto debería bloquear).
    """
    phi = float(Q) * float(Sigma)
    return LBCUResult(Q=float(Q), Sigma=float(Sigma), phi=phi, delta_H=float(delta_H))
