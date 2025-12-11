"""
Core TCDS — Núcleo σ-céntrico para Hunter TRL-9.

Este paquete reúne las herramientas fundamentales para detectar y
caracterizar la acción del Sincronón (σ) en una señal:

- sigma_metrics: LI, R, RMSE_SL, κΣ.
- entropy_core: entropía y ΔH.
- e_veto: filtro entrópico anti-apofenia.
- lBCU_core: balance σ·Σ = φ.
- tc_estimator: estimación de t_C.
- sincronon: análisis integrado de ventana.

Diseñado para ser isomórfico entre dominios:
sismología, biología, ΣFET, clima espacial, etc.
"""

from .sigma_metrics import SigmaMetrics, compute_sigma_metrics, compute_LI
from .entropy_core import compute_entropy, compute_delta_H
from .e_veto import EVetoDecision, EVetoSummary, apply_e_veto, summarize_e_veto
from .lBCU_core import LBCUResult, lBCU_balance_from_LI
from .tc_estimator import estimate_t_c
from .sincronon import SincrononWindowResult, analyze_window
