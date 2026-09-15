from __future__ import annotations

import numpy as np
from collections.abc import Sequence
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from pyvasplot import PyVASP
from pyvasplot.plotting._data import shifted_energy


def project_procar(
    dft: PyVASP,
    procar_data: NDArray,
    ions: int | Sequence[int] | np.ndarray | None,
    orbitals: str | Sequence[str] | None,
) -> NDArray:
    """Sum selected orbitals and average over selected ions."""
    if ions is None:
        selected_ions = list(range(dft.nions))
    elif isinstance(ions, (int, np.integer)):
        selected_ions = [int(ions)]
    else:
        selected_ions = list(ions)
        if not selected_ions:
            selected_ions = list(range(dft.nions))

    orbital_indices = orbital_indices_for(dft.procar.orbitals, orbitals)

    data = np.sum(
        procar_data[:, :, :, orbital_indices],
        axis=3,
    )

    return np.mean(data[:, :, selected_ions], axis=2)


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
) -> list[int]:
    """Convert selected orbital names into PROCAR orbital indices."""
    if selection is None:
        return list(range(len(orbitals)))

    if isinstance(selection, str):
        selected = [selection]
    else:
        selected = list(selection)

    if not selected:
        return list(range(len(orbitals)))

    return [orbitals.index(orbital) for orbital in selected]
