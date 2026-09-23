from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pyvasplot.pyvasp import PyVASP


def plot_kpath(
    dft: PyVASP,
    ax: plt.Axes | None = None,
    inv_space_size: int = 3,
    attr: str = "-b",
    axis_limit: float = 2.0,
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
    axis_limit
        Symmetric axis limit in reciprocal-space units. If None, it is
        determined automatically from the plotted points.
    **kwargs
        Additional arguments passed to ``ax.plot``.
    """
    kpts = np.asarray(dft.kpoints.kpts)

    reciprocal_lattice = dft.structure.lattice.reciprocal_lattice.matrix
    kpts_cart = np.dot(kpts, reciprocal_lattice)

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    reciprocal_x, reciprocal_y = _reciprocal_lattice_points(
        reciprocal_lattice,
        inv_space_size,
    )

    ax.plot(reciprocal_x, reciprocal_y, "ok")
    ax.plot(0, 0, "or")
    ax.plot(kpts_cart[:, 0], kpts_cart[:, 1], attr, **kwargs)

    if axis_limit > 0:
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)

    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$k_x$ (1/A)")
    ax.set_ylabel(r"$k_y$ (1/A)")

    return ax


def generate_kpath_bands(
    dft: PyVASP,
    k_norm: float | None = None,
    k_section: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Generate x coordinates and band energies for a k-path calculation."""

    bands = dft.eigenvalues

    # These k-points correspond one-to-one with the eigenvalues.
    kpts = np.asarray(dft.procar.kpoints)

    reciprocal_lattice = dft.structure.lattice.reciprocal_lattice.matrix
    kpts_cart = np.dot(kpts, reciprocal_lattice)

    if len(kpts_cart) != len(bands):
        raise ValueError(
            "Number of PROCAR k-points does not match eigenvalues: "
            f"{len(kpts_cart)} != {len(bands)}"
        )

    # Calculate cumulative distance along the actual PROCAR path.
    distances = np.linalg.norm(
        np.diff(kpts_cart, axis=0),
        axis=1,
    )

    kx = np.concatenate(([0.0], np.cumsum(distances)))

    if k_section is None:
        if k_norm is not None:
            raise ValueError("k_norm cannot be used with a complete path.")

        return kx, bands

    # For now, determine the section boundaries from the actual
    # PROCAR k-points.
    sections = _get_kpath_sections(dft)

    if k_section >= len(sections):
        raise ValueError(
            f"k_section={k_section} is out of range. "
            f"Available sections: {len(sections)}"
        )

    start, stop = sections[k_section]

    bands = bands[start:stop]
    kx = kx[start:stop]

    # Start the selected section at zero.
    kx = kx - kx[0]

    if k_norm is not None:
        kx = np.linspace(0.0, k_norm, len(bands))

    return kx, bands


def _get_kpath_sections(
    dft: PyVASP,
) -> list[tuple[int, int]]:
    """
    Determine k-path sections for a line-mode calculation.

    The KPOINTS file defines the number of sections and the number
    of requested points per section. PROCAR contains the actual
    electronic-structure data and may contain fewer points because
    repeated boundary points can be removed by the parser.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, stop)`` indices suitable for slicing PROCAR data.
    """
    procar_kpts = np.asarray(
        dft.procar.kpoints,
        dtype=float,
    )

    n_procar = len(procar_kpts)

    if n_procar == 0:
        return []

    kpoints = np.asarray(
        dft.kpoints.kpts,
        dtype=float,
    )

    if len(kpoints) == 0:
        return [(0, n_procar)]

    if len(kpoints) % 2 != 0:
        raise ValueError("Line-mode KPOINTS must contain pairs of k-points.")

    # In VASP line mode:
    #
    #     A B
    #     B C
    #     C D
    #
    # each pair defines one section.
    n_sections = len(kpoints) // 2

    # KPOINTS.num_kpts is the number of points requested per section.
    points_per_section = int(dft.kpoints.num_kpts)

    if points_per_section <= 0:
        raise ValueError("KPOINTS.num_kpts must be positive.")

    expected_points = n_sections * points_per_section

    # Normally this should be the number written in PROCAR before
    # pymatgen removes repeated boundary points.
    #
    # We do not require it to match exactly because different VASP /
    # parser combinations can treat boundary points differently.
    n_removed = expected_points - n_procar

    if n_removed < 0:
        raise ValueError(
            "PROCAR contains more k-points than expected from KPOINTS: "
            f"{n_procar} > {expected_points}."
        )

    # If nothing was removed, the section boundaries are trivial.
    if n_removed == 0:
        return [
            (
                section * points_per_section,
                (section + 1) * points_per_section,
            )
            for section in range(n_sections)
        ]

    sections: list[tuple[int, int]] = []

    # Build approximately equal sections first.
    base_length = n_procar // n_sections
    remainder = n_procar % n_sections

    start = 0

    for section in range(n_sections):
        length = base_length

        if section < remainder:
            length += 1

        stop = start + length

        sections.append((start, stop))

        start = stop

    return sections


def generate_labels(
    dft: PyVASP,
    kx: NDArray,
) -> tuple[list[float], list[str]]:
    """Generate x-axis positions and labels for a k-path plot."""
    num_kpts = dft.num_kpts

    x_labels_pos = list(np.append(kx[::num_kpts], kx[-1]))

    labels = list(dft.kpoints.labels)

    labels = [labels[0]] + labels[1::2]
    labels = _convert_to_tex(labels)

    return x_labels_pos, labels


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
            x[ind] = reciprocal_lattice[0, 0] * i + reciprocal_lattice[1, 0] * j
            y[ind] = reciprocal_lattice[0, 1] * i + reciprocal_lattice[1, 1] * j
            ind += 1

    return x, y
