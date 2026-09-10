from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from pyvasplot import PyVASP


def plot_kpath(
    dft: PyVASP,
    ax: plt.Axes | None = None,
    inv_space_size: int = 3,
    attr: str = "-b",
    **kwargs,
) -> plt.Axes:
    """Plot the k-point path in reciprocal space.

    Parameters
    ----------
    dft
        Loaded PyVASP calculation.
    ax
        Matplotlib axes. If None, new axes are created.
    inv_space_size
        Number of reciprocal lattice translations to display.
    attr
        Matplotlib line style for the k-path.
    **kwargs
        Additional arguments passed to ``ax.plot``.
    """
    kpts = np.asarray(dft.kpoints.kpts)

    reciprocal_lattice = dft.structure.lattice.reciprocal_lattice.matrix
    kpts_cart = np.dot(kpts, reciprocal_lattice)

    if ax is None:
        _, ax = plt.subplots()

    reciprocal_x, reciprocal_y = _reciprocal_lattice_points(
        reciprocal_lattice,
        inv_space_size,
    )

    ax.plot(reciprocal_x, reciprocal_y, "ok")
    ax.plot(0, 0, "or")
    ax.plot(kpts_cart[:, 0], kpts_cart[:, 1], attr, **kwargs)

    max_limit = np.max(np.abs(kpts_cart[:, :2])) * 1.5

    if max_limit > 0:
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)

    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$k_x$ (1/A)")
    ax.set_ylabel(r"$k_y$ (1/A)")

    return ax


def generate_path_bands(
    dft: PyVASP,
    k_norm: float | None = None,
    k_section: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Generate x coordinates and band energies for a k-path calculation.

    Parameters
    ----------
    dft
        Loaded PyVASP calculation.
    k_norm
        Optional normalization of an individual path section.
    k_section
        Optional section to extract.

    Returns
    -------
    kx, bands
        One-dimensional k coordinates and corresponding band energies.
    """
    bands = dft.eigenvalues

    kpts = np.asarray(dft.kpoints.kpts)
    reciprocal_lattice = dft.structure.lattice.reciprocal_lattice.matrix
    kpts_cart = np.dot(kpts, reciprocal_lattice)

    nk_section = _get_nkpoints_section(dft)

    kx = _kx_for_sections(kpts_cart, nk_section)

    if k_section is None:
        if k_norm is not None:
            raise ValueError("k_norm cannot be used with a complete path.")

        return kx, bands

    start = k_section * nk_section
    stop = start + nk_section

    bands = bands[start:stop, :]

    if k_norm is None:
        kx = kx[start:stop]
        kx = kx - kx.min()
    else:
        kx = np.linspace(0, k_norm, nk_section)

    return kx, bands


def generate_labels(
    dft: PyVASP,
    kx: NDArray,
) -> tuple[list[float], list[str]]:
    """Generate x-axis positions and labels for a k-path plot."""
    nk_section = _get_nkpoints_section(dft)

    x_labels_pos = list(np.append(kx[::nk_section], kx[-1]))

    labels = list(dft.kpoints.labels)

    labels = [labels[0]] + labels[1::2]
    labels = _convert_to_tex(labels)

    return x_labels_pos, labels


def _get_nkpoints_section(dft: PyVASP) -> int:
    """Return the number of k-points in one path section."""
    # For now this is derived from the KPOINTS labels rather than
    # requiring nkpoints_section to be another PyVASP property.
    labels = dft.kpoints.labels

    if not labels:
        raise ValueError("KPOINTS does not contain path labels.")

    n_sections = (len(labels) - 1) // 2

    if n_sections <= 0:
        raise ValueError("Unable to determine k-path sections.")

    if dft.nkpoints % n_sections != 0:
        raise ValueError(
            "Number of k-points is not divisible by the number "
            "of k-path sections."
        )

    return dft.nkpoints // n_sections


def _kx_for_sections(
    kpts_cart: NDArray,
    nk_section: int,
) -> NDArray:
    """Generate cumulative distance along the k-path."""
    if kpts_cart.shape[0] % 2 != 0:
        kpts_cart = np.vstack((kpts_cart, kpts_cart[-1]))

    kpts_diff = kpts_cart[1::2] - kpts_cart[0::2]
    kpts_diff_len = np.linalg.norm(kpts_diff, axis=1)

    k_len = np.cumsum(kpts_diff_len)

    sections = zip(
        [0.0] + list(k_len[:-1]),
        k_len,
    )

    return np.concatenate(
        [
            np.linspace(
                start,
                end,
                nk_section,
                endpoint=False,
            )
            for start, end in sections
        ]
    )


def _convert_to_tex(labels: list[str]) -> list[str]:
    """Convert special k-point labels to matplotlib-friendly labels."""
    tex_labels = {
        "Gamma": r"$\Gamma$",
        "\\Gamma": r"$\Gamma$",
    }

    return [tex_labels.get(label, label) for label in labels]


def _reciprocal_lattice_points(
    reciprocal_lattice: NDArray,
    inv_space_size: int,
) -> tuple[NDArray, NDArray]:
    """Generate reciprocal lattice translation points."""
    size = inv_space_size * 2 + 1

    x = np.zeros(size**2)
    y = np.zeros(size**2)

    ind = 0

    for i in range(-inv_space_size, inv_space_size + 1):
        for j in range(-inv_space_size, inv_space_size + 1):
            x[ind] = (
                reciprocal_lattice[0, 0] * i
                + reciprocal_lattice[1, 0] * j
            )
            y[ind] = (
                reciprocal_lattice[0, 1] * i
                + reciprocal_lattice[1, 1] * j
            )
            ind += 1

    return x, y
