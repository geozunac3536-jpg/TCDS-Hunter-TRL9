"""
e_veto.py

Implementación del filtro E-Veto de TCDS para bloquear apofenia:

Una señal coherente (LI alto, R alto) NO se considera válida a menos que
muestre también un colapso entrópico significativo:

    ΔH ≤ threshold   (por defecto threshold = -0.20)

Este módulo proporciona funciones simples para aplicar el E-Veto por
ventana y para resumir conjuntos de ventanas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class EVetoDecision:
    """
    Resultado de evaluar el E-Veto en una ventana.

    Attributes
    ----------
    delta_H : float
        Variación entrópica ΔH en la ventana.
    threshold : float
        Umbral usado (típicamente -0.20).
    passed : bool
        True si ΔH ≤ threshold (es decir, colapso entrópico suficiente).
    """
    delta_H: float
    threshold: float
    passed: bool


def apply_e_veto(
    delta_H: float,
    threshold: float = -0.20
) -> EVetoDecision:
    """
    Evalúa el criterio E-Veto en una sola ventana.

    Parameters
    ----------
    delta_H : float
        Variación entrópica ΔH (H(ref) - H(signal)).
        Valores muy negativos implican mayor orden en la señal.
    threshold : float, optional
        Umbral de colapso entrópico. Por defecto -0.20.

    Returns
    -------
    EVetoDecision
        Objeto con ΔH, threshold y si pasa o no el filtro.
    """
    passed = bool(delta_H <= threshold)
    return EVetoDecision(delta_H=float(delta_H), threshold=float(threshold), passed=passed)


@dataclass
class EVetoSummary:
    """
    Resumen estadístico de E-Veto sobre múltiples ventanas.

    Attributes
    ----------
    threshold : float
        Umbral usado.
    n : int
        Número total de ventanas evaluadas.
    n_pass : int
        Número de ventanas que pasan el E-Veto.
    fraction_pass : float
        Fracción de ventanas que pasan el E-Veto.
    deep_events : int
        Número de ventanas con ΔH mucho menor que el umbral (e.g., ΔH ≤ threshold - 1.0).
    """
    threshold: float
    n: int
    n_pass: int
    fraction_pass: float
    deep_events: int


def summarize_e_veto(
    deltas_H: Iterable[float],
    threshold: float = -0.20,
    deep_margin: float = 1.0
) -> EVetoSummary:
    """
    Resume el comportamiento del E-Veto sobre una colección de ΔH.

    Parameters
    ----------
    deltas_H : Iterable[float]
        Colección de valores ΔH.
    threshold : float, optional
        Umbral principal del E-Veto, por defecto -0.20.
    deep_margin : float, optional
        Margen para contar "deep events", por defecto 1.0.
        Es decir, se considera deep_event si ΔH ≤ threshold - deep_margin.

    Returns
    -------
    EVetoSummary
        Resumen con conteos y fracciones.
    """
    vals: List[float] = [float(v) for v in deltas_H]
    n = len(vals)
    if n == 0:
        return EVetoSummary(
            threshold=float(threshold),
            n=0,
            n_pass=0,
            fraction_pass=0.0,
            deep_events=0,
        )

    n_pass = sum(1 for v in vals if v <= threshold)
    deep_events = sum(1 for v in vals if v <= (threshold - deep_margin))
    fraction_pass = n_pass / n

    return EVetoSummary(
        threshold=float(threshold),
        n=n,
        n_pass=n_pass,
        fraction_pass=float(fraction_pass),
        deep_events=deep_events,
    )
