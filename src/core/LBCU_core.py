"""
lBCU_core.py

Núcleo de la Ley de Balance Coherencial Universal (LBCU) en versión σ-céntrica.

    Q · Σ = φ

En TCDS, tras el cierre ontológico:

    Q ≡ σ  (Sincronón)
    σ_obs ≡ LI (Locking Index)

Por tanto, en una ventana:

    Q = LI
    Σ = LI
    φ = Q · Σ = LI²

Este módulo sólo registra ese balance y adjunta ΔH; no inventa Q ni Σ.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LBCUResult:
    """
    Estado de balance LBCU en una ventana.

    Attributes
    ----------
    Q : float
        Intensidad del Sincronón (σ) observada (Q = LI).
    Sigma : float
        Coherencia Σ (Sigma = LI).
    phi : float
        Fricción efectiva del sustrato χ (φ = LI²).
    delta_H : float
        Huella entrópica del Sincronón en la ventana.
    """
    Q: float
    Sigma: float
    phi: float
    delta_H: float


def lBCU_balance_from_LI(
    LI: float,
    delta_H: float,
) -> LBCUResult:
    """
    Aplica la LBCU usando la interpretación σ-céntrica:

        Q = LI
        Σ = LI
        φ = Q · Σ = LI²

    donde LI es el Locking Index de la ventana y ΔH su huella entrópica.

    Parameters
    ----------
    LI : float
        Locking Index en la ventana (0–1).
    delta_H : float
        ΔH de la ventana.

    Returns
    -------
    LBCUResult
        Resultado del balance σ–Σ–φ.
    """
    Q = float(LI)
    Sigma = float(LI)
    phi = Q * Sigma

    return LBCUResult(
        Q=Q,
        Sigma=Sigma,
        phi=float(phi),
        delta_H=float(delta_H),
    )
