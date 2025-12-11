"""
lBCU_core.py

Núcleo de la Ley de Balance Coherencial Universal (LBCU) en versión σ-céntrica.

    Q · Σ = φ

En el marco TCDS aplicado al Hunter/Crawler TRL-9:

- Q ≡ σ_intensidad  → el Sincronón como empuje cuántico mínimo.
- Σ ≡ coherencia Σ  → respuesta macroscópica de la misma excitación σ.
- φ ≡ fricción      → resistencia del sustrato χ a la organización de σ.
- ΔH                → huella entrópica del Sincronón en la ventana.

Operacionalmente:

- LI (Locking Index) es la firma directa de σ en una ventana de señal.
- ΔH < 0 confirma que σ no es apofenia: hay colapso entrópico real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LBCUResult:
    """
    Estado de balance LBCU en una ventana.

    Attributes
    ----------
    Q : float
        Intensidad del Sincronón (σ) en la ventana (aquí Q = LI).
    Sigma : float
        Coherencia Σ observada (aquí Σ = LI).
    phi : float
        Fricción efectiva del sustrato χ (φ = Q · Σ = LI²).
    delta_H : float
        Variación entrópica ΔH = H(ref) - H(signal).
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

    No asume nada de sismología: sólo mide desorden vs. orden en amplitudes.

    Parameters
    ----------
    x : np.ndarray
        Señal 1D (se ignorarán NaNs e infinitos).
    n_bins : int, optional
        Número de bins para el histograma, por defecto 64.
    eps : float, optional
        Offset para evitar log(0), por defecto 1e-12.

    Returns
    -------
    float
        Entropía H(x) en nats.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0

    xmin, xmax = np.min(x), np.max(x)
    if xmax <= xmin:
        # Señal constante → prácticamente sin información
        return 0.0

    x_norm = (x - xmin) / (xmax - xmin)

    hist, _ = np.histogram(x_norm, bins=n_bins, range=(0.0, 1.0), density=True)
    p = hist / np.sum(hist)  # distribución de probabilidad
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p))
    return float(H)


def _make_surrogate(
    x: np.ndarray,
    random_state: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Construye un surrogate barajando la señal (preserva histograma, destruye orden temporal).
    Esto representa el estado φ-driven (ruido incoherente) sin acción de σ.
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

    Interpretación TCDS:

    - H(x)  → entropía de la señal actual (posible acción de σ).
    - H(ref)→ entropía de un estado φ-driven (surrogate barajado o baseline).
    - ΔH    → huella directa del Sincronón:
              si ΔH << 0, σ ha impuesto orden sobre el ruido.

    Parameters
    ----------
    x : np.ndarray
        Señal de la ventana.
    reference : np.ndarray, optional
        Señal de referencia. Si es None, se usa un surrogate barajado de `x`.
    n_bins : int, optional
        Número de bins del histograma, por defecto 64.
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


def lBCU_balance_from_LI(
    LI: float,
    delta_H: float
) -> LBCUResult:
    """
    Aplica la LBCU usando la interpretación σ-céntrica:

        Q = σ_intensidad = LI
        Σ = LI
        φ = Q · Σ = LI²

    donde LI es el Locking Index medido en la ventana y ΔH es la huella
    entrópica calculada para esa misma ventana.

    Parameters
    ----------
    LI : float
        Locking Index en la ventana (0–1). Firma directa de σ.
    delta_H : float
        Variación entrópica ΔH asociada a la ventana.

    Returns
    -------
    LBCUResult
        Estado de balance σ–Σ–φ para esa ventana.

    Notas
    -----
    - Si LI es alto pero ΔH ≈ 0, el E-Veto debería bloquear la interpretación
      como acción verdadera de σ (caso apofenia).
    - Si LI es alto y ΔH << 0, tenemos un candidato fuerte a nucleación causal.
    """
    Q = float(LI)       # σ como empuje cuántico
    Sigma = float(LI)   # misma coherencia medida
    phi = Q * Sigma     # φ = LI²

    return LBCUResult(
        Q=Q,
        Sigma=Sigma,
        phi=float(phi),
        delta_H=float(delta_H),
    )
