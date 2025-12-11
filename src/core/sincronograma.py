"""
sincronograma.py

Generación de sincronogramas Σ para TCDS Hunter/Crawler.

Esta versión asume que trabajas con un objeto `Trace` estilo ObsPy
(o compatible) que expone:

- tr.times() → array de tiempos (segundos relativos)
- tr.data   → array de amplitudes

Si no usas ObsPy, puedes:
- adaptar una pequeña clase con esos dos métodos/atributos, o
- escribir un wrapper antes de llamar a esta función.

El sincronograma muestra:

- Arriba: señal sísmica cruda.
- Abajo: envolvente absoluta y anotaciones Σ:
  - t_C (tiempo causal estimado)
  - LI y ΔH de la ventana
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def generar_sincronograma(
    tr,
    info_str: str,
    ev_id: str,
    t_c: float,
    LI: float,
    delta_H: float,
    output_dir: str = "docs/report_trl9/figures",
    prefix: Optional[str] = None,
    dpi: int = 200
) -> str:
    """
    Genera el sincronograma y lo guarda como PNG (y opcionalmente PDF).

    Parameters
    ----------
    tr : obspy.Trace-like
        Traza sísmica con métodos/atributos:
          - tr.times() → np.ndarray de tiempos (s)
          - tr.data    → np.ndarray de amplitudes
    info_str : str
        Texto descriptivo de la señal (estación, red, etc.).
    ev_id : str
        Identificador del evento (ej. USGS/IRIS ID o ID interno).
    t_c : float
        Tiempo causal estimado en segundos desde el inicio de la traza.
    LI : float
        Locking Index de la ventana de interés.
    delta_H : float
        Variación entrópica ΔH de la ventana.
    output_dir : str, optional
        Carpeta donde se guardarán las figuras.
    prefix : str, optional
        Prefijo de nombre de archivo. Si es None, se usa ev_id.
    dpi : int, optional
        Resolución del PNG, por defecto 200.

    Returns
    -------
    str
        Ruta del archivo PNG generado.

    Notes
    -----
    - El archivo se nombra como:
        {prefix or ev_id}_sincronograma.png
    - También se genera un PDF con el mismo nombre si lo necesitas
      para el reporte TRL-9.
    """
    # Prepara directorio
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{prefix or ev_id}_sincronograma"
    png_path = os.path.join(output_dir, base_name + ".png")
    pdf_path = os.path.join(output_dir, base_name + ".pdf")

    # Extrae datos
    try:
        times = np.asarray(tr.times(), dtype=float)
        data = np.asarray(tr.data, dtype=float)
    except Exception as exc:  # pragma: no cover
        raise TypeError("El objeto 'tr' debe exponer 'times()' y 'data' compatibles con NumPy.") from exc

    if times.size != data.size:
        n = min(times.size, data.size)
        times = times[:n]
        data = data[:n]

    # Envolvente simple: valor absoluto suavizado
    abs_data = np.abs(data)
    # Suavizado por convolución simple para que se vea agradable
    kernel_size = max(5, int(0.01 * times.size))
    kernel = np.ones(kernel_size) / kernel_size
    env = np.convolve(abs_data, kernel, mode="same")

    # Figura
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    # Panel superior: señal
    ax1.plot(times, data, linewidth=0.5)
    ax1.set_title(f"Señal sísmica | {info_str}", fontsize=11)
    ax1.set_ylabel("Amplitud", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(
        t_c,
        color="tabred",
        linewidth=1.0,
        linestyle="--",
        label=r"$t_C$"
    )
    ax1.legend(loc="upper right", fontsize=8)

    # Panel inferior: envolvente + anotaciones Σ
    ax2.plot(times, env, linewidth=0.6)
    ax2.fill_between(times, env, alpha=0.12)
    ax2.set_ylabel("Envolvente |Σ|", fontsize=9)
    ax2.set_xlabel("Tiempo (s)", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Línea vertical en t_C
    ax2.axvline(
        t_c,
        color="tabred",
        linewidth=1.0,
        linestyle="--"
    )

    # Texto con LI y ΔH
    text_str = rf"$t_C = {t_c:.2f}\,\mathrm{{s}}$" + "\n" + rf"$LI = {LI:.3f}$" + "\n" + rf"$\Delta H = {delta_H:.2f}$"
    bbox_props = dict(
        boxstyle="round,pad=0.3",
        fc="black",
        ec="white",
        alpha=0.7
    )
    ax2.text(
        0.02,
        0.95,
        text_str,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=bbox_props
    )

    fig.suptitle(
        f"TCDS — Sincronograma Σ\nEvento: {ev_id}",
        fontsize=12,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Guardado
    fig.savefig(png_path, dpi=dpi)
    try:
        fig.savefig(pdf_path)
    except Exception:
        # No es crítico si falla el PDF en algunos entornos
        pass

    plt.close(fig)
    return png_path
