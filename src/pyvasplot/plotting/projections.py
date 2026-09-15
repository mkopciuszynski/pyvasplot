from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
from pyvasplot.types import CalculationType


from pyvasplot import PyVASP
from pyvasplot.plotting._data import (
    prepare_projection_data,
    shifted_energy,
)
from pyvasplot.plotting._procar import (
    interpolate_map,
    make_energy_map,
    project_procar,
)


def plot_procar_bands(
    dft: PyVASP,
    ions: int | Sequence[int] | np.ndarray | None = None,
    orbitals: str | Sequence[str] | None = None,
    k_section: int | None = None,
    k_norm: float | None = None,
    k_mirror: bool = False,
    k_flip: bool = False,
    e_min: float = -3.0,
    e_max: float = 1.0,
    min_val: float = 0.01,
    norm_val: float = 0.1,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot projected band structure using PROCAR weights."""
    kx, bands, procar_data = prepare_projection_data(
        dft,
        k_section=k_section,
        k_norm=k_norm,
        k_flip=k_flip,
    )

    projection = project_procar(
        dft,
        procar_data,
        ions,
        orbitals,
    )

    weight = (projection - min_val) * 1000 * norm_val
    valid = weight > 0

    energy = shifted_energy(dft, bands)

    valid &= energy > e_min
    valid &= energy < e_max

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
    ax.set_ylim(e_min, e_max)

    return ax


def plot_procar_map(
    dft: PyVASP,
    ions: int | Sequence[int] | np.ndarray | None = None,
    orbitals: str | Sequence[str] | None = None,
    k_section: int | None = None,
    k_norm: float | None = None,
    k_mirror: bool = False,
    k_flip: bool = False,
    e_min: float = -3.0,
    e_max: float = 1.0,
    ek_size: int = 256,
    lorentz_width: float = 0.04,
    interp_k: int = 2,
    cmap: str = "inferno",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Create an ARPES-like map from PROCAR projections."""
    kx, bands, procar_data = prepare_projection_data(
        dft,
        k_section=k_section,
        k_norm=k_norm,
        k_flip=k_flip,
    )

    projection = project_procar(
        dft,
        procar_data,
        ions,
        orbitals,
    )

    procar_map = make_energy_map(
        dft,
        kx,
        bands,
        projection,
        e_min,
        e_max,
        ek_size,
        lorentz_width,
    )

    kx_interp = np.linspace(
        kx.min(),
        kx.max(),
        len(kx) * interp_k,
    )

    procar_map_interp = interpolate_map(
        kx,
        procar_map,
        kx_interp,
    )
    k_extent = k_norm if k_norm is not None else float(kx.max())

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
            extent=(-k_extent, k_extent, e_min, e_max),
            **kwargs,
        )
    else:
        ax.imshow(
            procar_map_interp.T,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=(0, k_extent, e_min, e_max),
            **kwargs,
        )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E - E_F$ (eV)")

    return ax


def plot_procar_scatter(
    dft: PyVASP,
    ions: int | Sequence[int] | np.ndarray | None = None,
    orbitals: str | Sequence[str] | None = None,
    k_section: int | None = None,
    k_norm: float | None = None,
    k_mirror: bool = False,
    k_flip: bool = False,
    e_min: float = -3.0,
    e_max: float = 1.0,
    ek_size: int = 64,
    min_val: float = 0.01,
    norm_val: float = 0.5,
    lorentz_width: float = 0.04,
    interp_k: int = 1,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Create a scatter-style ARPES map from PROCAR projections."""
    kx, bands, procar_data = prepare_projection_data(
        dft,
        k_section=k_section,
        k_norm=k_norm,
        k_flip=k_flip,
    )

    projection = project_procar(
        dft,
        procar_data,
        ions,
        orbitals,
    )

    procar_map = make_energy_map(
        dft,
        kx,
        bands,
        projection,
        e_min,
        e_max,
        ek_size,
        lorentz_width,
    )

    kx_interp = np.linspace(
        kx.min(),
        kx.max(),
        len(kx) * interp_k,
    )

    procar_map_interp = interpolate_map(
        kx,
        procar_map,
        kx_interp,
    )

    energy_vec = np.linspace(
        e_min,
        e_max,
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

def plot_procar_kxky(
    dft: PyVASP,
    ions: int | Sequence[int] | np.ndarray | None = None,
    orbitals: str | Sequence[str] | None = None,
    kx_norm: float | None = None,
    ky_norm: float | None = None,
    k_mirror: bool = False,
    e_cut: float = 0.0,
    e_cut_width: float = 0.1,
    ek_size: int = 256,
    lorentz_width: float = 0.04,
    interp_k: int = 1,
    cmap: str = "inferno",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Create a KXKY spectral map from PROCAR projections."""

    if dft.calculation_type is not CalculationType.KXKY:
        raise ValueError(
            "plot_procar_kxky requires KXKY data."
        )

    if dft.data.kxky is None:
        raise RuntimeError(
            "KXKY data has not been loaded."
        )

    if lorentz_width <= 0:
        raise ValueError(
            "lorentz_width must be greater than zero."
        )

    if e_cut_width <= 0:
        raise ValueError(
            "e_cut_width must be greater than zero."
        )

    if ek_size < 2:
        raise ValueError(
            "ek_size must be at least 2."
        )

    if interp_k < 1:
        raise ValueError(
            "interp_k must be at least 1."
        )

    original_ky = dft.selected_ky

    try:
        # Determine the kx coordinate from the first ky slice.
        dft.selected_ky = 0

        kx, bands, procar_data = prepare_projection_data(
            dft,
            k_norm=kx_norm,
        )

        projection = project_procar(
            dft,
            procar_data,
            ions,
            orbitals,
        )

        n_kx = len(kx)
        n_ky = dft.nky

        # Energy window around e_cut.
        e_min = e_cut - e_cut_width * 5
        e_max = e_cut + e_cut_width * 5

        map_kxky = np.zeros((n_kx, n_ky))

        for ky in range(n_ky):
            dft.selected_ky = ky

            kx, bands, procar_data = prepare_projection_data(
                dft,
                k_norm=kx_norm,
            )

            projection = project_procar(
                dft,
                procar_data,
                ions,
                orbitals,
            )

            procar_map = make_energy_map(
                dft,
                kx,
                bands,
                projection,
                e_min,
                e_max,
                ek_size,
                lorentz_width,
            )

            # Average the central 10% of the energy window.
            center = ek_size // 2
            half_width = max(
                1,
                int(ek_size * 0.05),
            )

            map_kxky[:, ky] = procar_map[
                :,
                center - half_width : center + half_width,
            ].mean(axis=1)

    finally:
        dft.selected_ky = original_ky

    # Interpolate along kx, consistent with plot_procar_map().
    kx_interp = np.linspace(
        kx.min(),
        kx.max(),
        len(kx) * interp_k,
    )

    map_kxky_interp = np.empty(
        (len(kx_interp), n_ky),
    )

    for ky in range(n_ky):
        map_kxky_interp[:, ky] = interpolate_map(
            kx,
            map_kxky[:, ky, None],
            kx_interp,
        ).ravel()

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6, 4),
        )

    # Physical extent.
    kx_extent = (
        kx_norm
        if kx_norm is not None
        else float(kx.max())
    )

    ky_extent = (
        ky_norm
        if ky_norm is not None
        else dft.kynorm
    )

    if k_mirror:
        map_mirrored = np.concatenate(
            (
                np.flip(map_kxky_interp, axis=0),
                map_kxky_interp,
            ),
            axis=0,
        )

        map_mirrored = np.concatenate(
            (
                np.flip(map_mirrored, axis=1),
                map_mirrored,
            ),
            axis=1,
        )

        ax.imshow(
            map_mirrored.T,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=(
                -kx_extent,
                kx_extent,
                -ky_extent,
                ky_extent,
            ),
            **kwargs,
        )

    else:
        ax.imshow(
            map_kxky_interp.T,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=(
                0,
                kx_extent,
                0,
                ky_extent,
            ),
            **kwargs,
        )

    ax.set_xlabel(
        r"$k_x$"
        if kx_norm is not None
        else r"$k_x$ (1/A)"
    )
    ax.set_ylabel(
        r"$k_y$"
        if ky_norm is not None
        else r"$k_y$ (1/A)"
    )

    return ax