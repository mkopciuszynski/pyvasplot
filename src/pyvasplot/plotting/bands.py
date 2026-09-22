from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from pyvasplot.pyvasp import PyVASP
from pyvasplot.plotting.kpath import generate_labels, generate_kpath_bands
from pyvasplot.plotting._bands import generate_line_bands, shifted_energy


def plot_bands(
    dft: PyVASP,
    k_section: int | None = None,
    k_norm: float | None = None,
    k_mirror: bool = False,
    k_flip: bool = False,
    k_start: float = 0.0,
    k_labels: bool = True,
    e_min: float = -3.0,
    e_max: float = 1.0,
    ax: plt.Axes | None = None,
    attr: str = "k-",
    **kwargs,
) -> plt.Axes:
    """Plot band energies.

    Path calculations use the actual reciprocal-space k-path.
    Non-path calculations use a linear k-point coordinate.

    Parameters
    ----------
    dft
        Loaded PyVASP calculation.
    k_section
        Optional path section to plot.
    k_norm
        Optional k-axis normalization.
    k_mirror
        Plot a mirrored copy around k=0.
    k_flip
        Reverse the order of k-points.
    k_start
        Starting position of the plotted k-axis.
    e_min, e_max
        Energy-axis limits relative to the Fermi level.
    k_labels
        Show k-point labels for complete path plots.
    ax
        Matplotlib axes.
    attr
        Matplotlib line style.
    **kwargs
        Additional arguments passed to ``ax.plot``.
    """
    x_labels: list[str] = []
    x_label_pos: list[float] = []

    if dft.calculation_type.is_path:
        kx, bands = generate_kpath_bands(
            dft,
            k_norm=k_norm,
            k_section=k_section,
        )

        x_label_pos, x_labels = generate_labels(dft, kx)

        if k_section is not None:
            k_labels = False

    else:
        if k_section is not None:
            raise ValueError("k_section can only be used with a path calculation.")

        kx, bands = generate_line_bands(dft, k_norm)
        k_labels = False

    if k_flip:
        bands = np.flip(bands, axis=0)

    if "color" in kwargs:
        attr = "-"

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(k_start + kx, bands - dft.efermi - dft.eshift, attr, **kwargs)

    if k_mirror:
        ax.plot(k_start - kx, bands - dft.efermi - dft.eshift, attr, **kwargs)

    # ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_ylim(e_min, e_max)

    if k_labels:
        ax.set_xlim(kx.min(), kx.max())
        ax.set_xticks(x_label_pos, x_labels)

    return ax
