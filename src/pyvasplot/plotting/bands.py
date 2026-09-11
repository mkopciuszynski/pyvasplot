from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from pyvasplot import PyVASP
from pyvasplot.plotting.kpath import generate_labels, generate_kpath_bands


def plot_bands(
    dft: PyVASP,
    k_section: int | None = None,
    k_norm: float | None = None,
    k_mirror: bool = False,
    k_flip: bool = False,
    k_start: float = 0.0,
    k_labels: bool = True,
    ek_min: float = -3.0,
    ek_max: float = 1.0,
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
    ek_min, ek_max
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

        kx, bands = _generate_line_bands(dft, k_norm)
        k_labels = False

    if k_flip:
        bands = np.flip(bands, axis=0)

    if "color" in kwargs:
        attr = "-"

    if ax is None:
        _, ax = plt.subplots()

    print(kx.shape)

    ax.plot(k_start + kx, bands - dft.efermi - dft.eshift, attr, **kwargs)

    if k_mirror:
        ax.plot(k_start - kx, bands - dft.efermi - dft.eshift, attr, **kwargs)

    #ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_ylim(ek_min, ek_max)

    if k_labels:
        ax.set_xlim(kx.min(), kx.max())
        ax.set_xticks(x_label_pos, x_labels)

    return ax


def _generate_line_bands(
    dft: PyVASP,
    k_norm: float | None,
) -> tuple[NDArray, NDArray]:
    """Generate a linear k-axis for non-path calculations."""
    if k_norm is None:
        k_norm = dft.knorm

    kx = np.linspace(0, k_norm, dft.nkpoints)

    return kx, dft.eigenvalues
