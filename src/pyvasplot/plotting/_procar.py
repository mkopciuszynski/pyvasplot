from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from pyvasplot.plotting._bands import (
    generate_line_bands,
    shifted_energy,
)
from pyvasplot.plotting.kpath import generate_kpath_bands
from pyvasplot.pyvasp import PyVASP


def prepare_projection_data(
    dft: PyVASP,
    k_section: int | None = None,
    k_norm: float | None = None,
    k_flip: bool = False,
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


def project_procar(
    procar_data: NDArray,
    orbitals: Sequence[str],
    ions: int | Sequence[int] | np.ndarray | None,
    selected_orbitals: str | Sequence[str] | None,
    ion_reduction: Literal["sum", "mean"] = "mean",
) -> NDArray:
    """Sum selected orbitals and reduce selected ions by sum or mean."""
    if ions is None:
        selected_ions = list(range(procar_data.shape[2]))
    elif isinstance(ions, (int, np.integer)):
        selected_ions = [int(ions)]
    else:
        selected_ions = list(ions)
        if not selected_ions:
            selected_ions = list(range(procar_data.shape[2]))

    orbital_indices = orbital_indices_for(orbitals, selected_orbitals)

    data = np.sum(
        procar_data[:, :, :, orbital_indices],
        axis=3,
    )

    selected_data = data[:, :, selected_ions]
    if ion_reduction == "sum":
        return np.sum(selected_data, axis=2)
    if ion_reduction == "mean":
        return np.mean(selected_data, axis=2)
    raise ValueError("ion_reduction must be 'sum' or 'mean'")


def make_energy_map(
    dft: PyVASP,
    kx: NDArray,
    bands: NDArray,
    projection: NDArray,
    e_min: float,
    e_max: float,
    e_size: int,
    lorentz_width: float,
) -> NDArray:
    """Convert discrete bands into a Lorentzian-broadened map."""
    energy_vec = np.linspace(e_min, e_max, e_size)
    result = np.zeros((len(kx), e_size))
    shifted_bands = shifted_energy(dft, bands)

    for ik in range(len(kx)):
        for ib in range(dft.nbands):
            energy = shifted_bands[ik, ib]

            if e_min < energy < e_max:
                result[ik, :] += projection[ik, ib] / (
                    1 + ((energy_vec - energy) / lorentz_width) ** 2
                )

    return result


def interpolate_map(
    kx: NDArray,
    data: NDArray,
    kx_interp: NDArray,
) -> NDArray:
    """Interpolate an energy map along the k direction."""
    result = np.zeros((len(kx_interp), data.shape[1]))

    for ie in range(data.shape[1]):
        spline = CubicSpline(kx, data[:, ie])
        result[:, ie] = spline(kx_interp)

    return result


def orbital_indices_for(
    orbitals: Sequence[str],
    selection: str | Sequence[str] | None,
) -> tuple[int, ...]:
    """Convert selected orbital names into PROCAR orbital indices."""
    if selection is None:
        return tuple(list(range(len(orbitals))))

    if isinstance(selection, str):
        selected = [selection]
    else:
        selected = list(selection)

    if not selected:
        return tuple(list(range(len(orbitals))))

    return tuple([orbitals.index(orbital) for orbital in selected])
