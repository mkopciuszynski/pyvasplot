from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from numpy.typing import NDArray

from pyvasplot import PyVASP
from pyvasplot.plotting.bands import _generate_line_bands, _shifted_energy
from pyvasplot.plotting.path import generate_path_bands


def plot_bands_procar(
    dft: PyVASP,
    ion_list: list[int],
    orb_list: list[str],
    k_section: int | None = None,
    k_norm: float | None = None,
    k_mirror: bool = False,
    k_flip: bool = False,
    ek_min: float = -3.0,
    ek_max: float = 1.0,
    min_val: float = 0.01,
    norm_val: float = 0.1,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot projected band structure using PROCAR weights."""
    kx, bands, procar_data = _prepare_projection_data(
        dft,
        k_section=k_section,
        k_norm=k_norm,
        k_flip=k_flip,
    )

    projection = _project_procar(
        dft,
        procar_data,
        ion_list,
        orb_list,
    )

    weight = (projection - min_val) * 1000 * norm_val
    valid = weight > 0

    energy = _shifted_energy(dft, bands)

    valid &= energy > ek_min
    valid &= energy < ek_max

    kx_mat = np.repeat(kx, dft.nbands).reshape(
        len(kx),
        dft.nbands,
    )

    if ax is None:
        _, ax = plt.subplots()

    if k_mirror:
        ax.scatter(
            kx_mat[valid],
            energy[valid],
            s=weight[valid],
            **kwargs,
        )
        ax.scatter(
            -kx_mat[valid],
            energy[valid],
            s=weight[valid],
            **kwargs,
        )
    else:
        ax.scatter(
            kx_mat[valid],
            energy[valid],
            s=weight[valid],
            **kwargs,
        )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_ylim(ek_min, ek_max)

    return ax


def plot_procar_map(
    dft: PyVASP,
    ion_list: list[int],
    orb_list: list[str],
    k_section: int | None = None,
    k_norm: float = 1.0,
    k_mirror: bool = False,
    k_flip: bool = False,
    ek_min: float = -3.0,
    ek_max: float = 1.0,
    ek_size: int = 256,
    lorentz_width: float = 0.04,
    interp_k: int = 2,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Create an ARPES-like map from PROCAR projections."""
    kx, bands, procar_data = _prepare_projection_data(
        dft,
        k_section=k_section,
        k_norm=k_norm,
        k_flip=k_flip,
    )

    projection = _project_procar(
        dft,
        procar_data,
        ion_list,
        orb_list,
    )

    procar_map = _make_energy_map(
        dft,
        kx,
        bands,
        projection,
        ek_min,
        ek_max,
        ek_size,
        lorentz_width,
    )

    kx_interp = np.linspace(
        kx.min(),
        kx.max(),
        len(kx) * interp_k,
    )

    procar_map_interp = _interpolate_map(
        kx,
        procar_map,
        kx_interp,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if k_mirror:
        procar_map_interp = np.concatenate(
            (
                np.flip(procar_map_interp, axis=0),
                procar_map_interp,
            ),
            axis=0,
        )

        ax.imshow(
            procar_map_interp.T,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=(-k_norm, k_norm, ek_min, ek_max),
            **kwargs,
        )
    else:
        ax.imshow(
            procar_map_interp.T,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=(0, k_norm, ek_min, ek_max),
            **kwargs,
        )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E - E_F$ (eV)")

    return ax


def plot_procar_scatter(
    dft: PyVASP,
    ion_list: list[int],
    orb_list: list[str],
    k_section: int | None = None,
    k_norm: float = 1.0,
    k_mirror: bool = False,
    k_flip: bool = False,
    ek_min: float = -3.0,
    ek_max: float = 1.0,
    ek_size: int = 64,
    min_val: float = 0.01,
    norm_val: float = 0.5,
    lorentz_width: float = 0.04,
    interp_k: int = 1,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Create a scatter-style ARPES map from PROCAR projections."""
    kx, bands, procar_data = _prepare_projection_data(
        dft,
        k_section=k_section,
        k_norm=k_norm,
        k_flip=k_flip,
    )

    projection = _project_procar(
        dft,
        procar_data,
        ion_list,
        orb_list,
    )

    procar_map = _make_energy_map(
        dft,
        kx,
        bands,
        projection,
        ek_min,
        ek_max,
        ek_size,
        lorentz_width,
    )

    kx_interp = np.linspace(
        kx.min(),
        kx.max(),
        len(kx) * interp_k,
    )

    procar_map_interp = _interpolate_map(
        kx,
        procar_map,
        kx_interp,
    )

    energy_vec = np.linspace(
        ek_min,
        ek_max,
        ek_size,
    )

    k_mat, energy_mat = np.meshgrid(
        kx_interp,
        energy_vec,
    )

    scatter_data = (
        procar_map_interp - min_val
    ) * 1000 * norm_val

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        k_mat,
        energy_mat,
        s=scatter_data.T,
        **kwargs,
    )

    if k_mirror:
        ax.scatter(
            -k_mat,
            energy_mat,
            s=scatter_data.T,
            **kwargs,
        )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E - E_F$ (eV)")

    return ax


def _prepare_projection_data(
    dft: PyVASP,
    k_section: int | None,
    k_norm: float | None,
    k_flip: bool,
) -> tuple[NDArray, NDArray, NDArray]:
    """Prepare k coordinates, bands and PROCAR data."""
    procar_data = dft.procar_data

    if dft.calculation_type.is_path:
        kx, bands = generate_path_bands(
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
        kx, bands = _generate_line_bands(
            dft,
            k_norm,
        )

    if k_flip:
        bands = np.flip(bands, axis=0)
        procar_data = np.flip(procar_data, axis=0)

    return kx, bands, procar_data


def _project_procar(
    dft: PyVASP,
    procar_data: NDArray,
    ion_list: list[int],
    orb_list: list[str],
) -> NDArray:
    """Sum selected orbitals and average over selected ions."""
    orbitals = dft.procar.orbitals
    orbital_indices = _orb_list_to_num(
        orbitals,
        orb_list,
    )

    data = np.sum(
        procar_data[:, :, :, orbital_indices],
        axis=3,
    )

    data = np.mean(
        data[:, :, ion_list],
        axis=2,
    )

    return data


def _make_energy_map(
    dft: PyVASP,
    kx: NDArray,
    bands: NDArray,
    projection: NDArray,
    ek_min: float,
    ek_max: float,
    ek_size: int,
    lorentz_width: float,
) -> NDArray:
    """Convert discrete bands into a Lorentzian-broadened map."""
    energy_vec = np.linspace(
        ek_min,
        ek_max,
        ek_size,
    )

    result = np.zeros(
        (len(kx), ek_size),
    )

    shifted_bands = _shifted_energy(
        dft,
        bands,
    )

    for ik in range(len(kx)):
        for ib in range(dft.nbands):
            energy = shifted_bands[ik, ib]

            if ek_min < energy < ek_max:
                result[ik, :] += projection[ik, ib] / (
                    1 + ((energy_vec - energy) / lorentz_width) ** 2
                )

    return result


def _interpolate_map(
    kx: NDArray,
    data: NDArray,
    kx_interp: NDArray,
) -> NDArray:
    """Interpolate an energy map along the k direction."""
    result = np.zeros(
        (len(kx_interp), data.shape[1]),
    )

    for ie in range(data.shape[1]):
        spline = CubicSpline(
            kx,
            data[:, ie],
        )
        result[:, ie] = spline(kx_interp)

    return result


def _orb_list_to_num(
    orbitals: list[str],
    orb_list: list[str] | str,
) -> list[int]:
    """Convert orbital names into PROCAR orbital indices."""
    if isinstance(orb_list, str):
        orb_list = [orb_list]

    if not orb_list:
        return list(range(len(orbitals)))

    return [
        orbitals.index(orbital)
        for orbital in orb_list
    ]
