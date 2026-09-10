from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from pyvasplot import PyVASP


def plot_path(
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

    kx = np.concatenate(
        ([0.0], np.cumsum(distances))
    )

    if k_section is None:
        if k_norm is not None:
            raise ValueError(
                "k_norm cannot be used with a complete path."
            )

        return kx, bands

    # For now, determine the section boundaries from the actual
    # PROCAR k-points.
    sections = _get_kpath_sections(dft.procar.kpoints)

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
    kpts: NDArray,
    atol: float = 1e-8,
) -> list[tuple[int, int]]:
    """
    Find k-path sections from repeated k-points.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, stop)`` indices for each path section.
    """
    kpts = np.asarray(kpts)

    if len(kpts) < 2:
        return [(0, len(kpts))]

    repeated = np.all(
        np.isclose(
            kpts[1:],
            kpts[:-1],
            atol=atol,
        ),
        axis=1,
    )

    boundaries = np.flatnonzero(repeated) + 1

    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(kpts)]))

    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
    ]


def _match_kpoints_to_bands(
    kpts_cart: NDArray,
    nkpoints: int,
) -> NDArray:
    """
    Remove repeated consecutive k-points until their number matches
    the number of k-points parsed from PROCAR.

    Parameters
    ----------
    kpts_cart
        K-points from KPOINTS in Cartesian reciprocal coordinates.
    nkpoints
        Number of k-points available in the parsed PROCAR data.

    Returns
    -------
    NDArray
        K-points aligned with the eigenvalue array.
    """
    if len(kpts_cart) < nkpoints:
        raise ValueError(
            f"KPOINTS contains {len(kpts_cart)} points, but PROCAR "
            f"contains {nkpoints} electronic k-points."
        )

    if len(kpts_cart) == nkpoints:
        return kpts_cart

    keep = np.ones(len(kpts_cart), dtype=bool)

    # Repeated points are normally the high-symmetry points at the
    # boundaries between path sections.
    duplicate_indices = np.flatnonzero(
        np.all(
            np.isclose(
                np.diff(kpts_cart, axis=0),
                0.0,
                atol=1e-8,
            ),
            axis=1,
        )
    ) + 1

    n_remove = len(kpts_cart) - nkpoints

    if len(duplicate_indices) < n_remove:
        raise ValueError(
            f"KPOINTS contains {len(kpts_cart)} points while PROCAR "
            f"contains {nkpoints}, but only {len(duplicate_indices)} "
            f"repeated k-points were found."
        )

    keep[duplicate_indices[:n_remove]] = False

    return kpts_cart[keep]

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
