from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pyvasplot.pyvasp import PyVASP


def generate_line_bands(
    dft: PyVASP,
    k_norm: float | None,
) -> tuple[NDArray, NDArray]:
    """Generate a linear k-axis for non-path calculations."""
    if k_norm is None:
        k_norm = dft.knorm

    kx = np.linspace(0, k_norm, dft.nkpoints)

    return kx, dft.eigenvalues


def shifted_energy(
    dft: PyVASP,
    bands: NDArray,
) -> NDArray:
    """Convert absolute band energies to energy relative to the Fermi level."""
    if dft.efermi is None:
        raise ValueError("Fermi energy was not loaded.")

    return bands - dft.efermi - dft.eshift
