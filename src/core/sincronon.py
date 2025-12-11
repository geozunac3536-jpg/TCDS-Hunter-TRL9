"""
sincronon.py

Análisis integrado de ventana para el Sincronón (σ) en TCDS.

Este módulo encapsula el pipeline completo:

    señal → Σ-metrics → ΔH → E-Veto → LBCU → t_C

De modo que cada ventana de análisis queda representada por un solo
objeto auditable (SincrononWindowResult) con:

    - LI, R, RMSE_SL
    - ΔH, H_signal, H_ref
    - κΣ
    - t_C (si se proporciona LI(t))
    - decisión E-Veto
    - balance LBCU σ·Σ = φ

Pensado para ser llamado desde Hunter_Soldier o Crawler_Global, pero
sin depender de sismología explícita.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .sigma_metrics import SigmaMetrics, compute_sigma_metrics
from .entropy_core import compute_delta_H
from .e_veto import EVetoDecision, apply_e_veto
from .lBCU_core import LBCUResult, lBCU_balance_from_LI
from .tc_estimator import estimate_t_c


@dataclass
class SincrononWindowResult:
    """
    Resultado completo del análisis de una ventana bajo el paradigma σ-céntrico.

    Attributes
    ----------
    sigma_metrics : SigmaMetrics
        Métricas Σ de coherencia (incluye LI, R, RMSE_SL, κΣ).
    delta_H : float
        Huella entrópica del Sincronón en la ventana.
    H_signal : float
        Entropía de la señal.
    H_ref : float
        Entropía de la referencia.
    e_veto : EVetoDecision
        Decisión del filtro E-Veto.
    lBCU : LBCUResult
        Resultado del balance σ·Σ = φ.
    t_c : Optional[float]
        Tiempo causal estimado (si hay LI(t) y tiempos).
    t_c_index : Optional[int]
        Índice del máximo LI(t) usado para t_C.
    """

    sigma_metrics: SigmaMetrics
    delta_H: float
    H_signal: float
    H_ref: float
    e_veto: EVetoDecision
    lBCU: LBCUResult
    t_c: Optional[float] = None
    t_c_index: Optional[int] = None

    @property
    def LI(self) -> float:
        """Conveniencia: intensidad observable del Sincronón (σ)."""
        return self.sigma_metrics.LI


def analyze_window(
    signal: np.ndarray,
    locking_pattern: np.ndarray,
    *,
    li_series: Optional[Sequence[float]] = None,
    li_times: Optional[Sequence[float]] = None,
    reference: Optional[np.ndarray] = None,
    entropy_bins: int = 64,
    e_veto_threshold: float = -0.20,
) -> SincrononWindowResult:
    """
    Analiza una ventana de señal bajo el marco TCDS σ-céntrico.

    Pipeline:

        1) Σ-metrics → LI, R, RMSE_SL, κΣ.
        2) ΔH, H_signal, H_ref.
        3) E-Veto: valida colapso entrópico.
        4) LBCU: Q = σ = LI, Σ = LI, φ = LI².
        5) t_C estimado si se proporciona LI(t).

    Parameters
    ----------
    signal : np.ndarray
        Señal de la ventana.
    locking_pattern : np.ndarray
        Patrón de locking Σ para esta ventana.
    li_series : Sequence[float], optional
        Serie LI(t) (si ya se calculó por subventanas).
    li_times : Sequence[float], optional
        Tiempos asociados a LI(t).
    reference : np.ndarray, optional
        Señal de referencia para ΔH. Si es None, se usa surrogate de `signal`.
    entropy_bins : int, optional
        Bins del histograma para entropía.
    e_veto_threshold : float, optional
        Umbral E-Veto para ΔH.

    Returns
    -------
    SincrononWindowResult
        Objeto con todos los resultados del análisis de ventana.
    """
    signal = np.asarray(signal, dtype=float)
    locking_pattern = np.asarray(locking_pattern, dtype=float)

    # 1) Σ-metrics
    sigma_metrics = compute_sigma_metrics(
        signal,
        locking_pattern,
        li_series=None if li_series is None else np.asarray(li_series, dtype=float),
        li_times=None if li_times is None else np.asarray(li_times, dtype=float),
    )

    # 2) ΔH
    delta_H, H_signal, H_ref = compute_delta_H(
        signal,
        reference=reference,
        n_bins=entropy_bins,
    )

    # 3) E-Veto
    e_decision = apply_e_veto(delta_H, threshold=e_veto_threshold)

    # 4) LBCU (Q=σ=LI)
    l_result = lBCU_balance_from_LI(sigma_metrics.LI, delta_H)

    # 5) t_C (si hay LI(t))
    t_c: Optional[float] = None
    idx_max: Optional[int] = None
    if li_series is not None:
        t_c, idx_max = estimate_t_c(
            np.asarray(li_series, dtype=float),
            None if li_times is None else np.asarray(li_times, dtype=float),
        )

    return SincrononWindowResult(
        sigma_metrics=sigma_metrics,
        delta_H=float(delta_H),
        H_signal=float(H_signal),
        H_ref=float(H_ref),
        e_veto=e_decision,
        lBCU=l_result,
        t_c=t_c,
        t_c_index=idx_max,
    )
