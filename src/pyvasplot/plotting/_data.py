from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pyvasplot import PyVASP
from pyvasplot.plotting.kpath import generate_kpath_bands


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


def prepare_projection_data(
    dft: PyVASP,
    k_section: int | None,
    k_norm: float | None,
    k_flip: bool,
) -> tuple[NDArray, NDArray, NDArray]:
    """Prepare coordinates, bands, and PROCAR data for projection plots."""
    procar_data = dft.procar_data

    if dft.calculation_type.is_path:
        kx, bands = generate_kpath_bands(
            dft,
            k_norm=k_norm,
            k_section=k_section,
        )

        if k_section is not None:
            nk = len(kx)
            start = k_section * nk
            stop = start + nk
            procar_data = procar_data[start:stop, :]
    else:
        kx, bands = generate_line_bands(dft, k_norm)

    if k_flip:
        bands = np.flip(bands, axis=0)
        procar_data = np.flip(procar_data, axis=0)

    return kx, bands, procar_data
