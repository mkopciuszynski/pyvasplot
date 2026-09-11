from __future__ import annotations

import numpy as np

from pyvasplot import PyVASP


def by_name(dft: PyVASP, ion_name: str) -> np.ndarray:
    """
    Return ion indices matching a chemical element symbol.

    Parameters
    ----------
    dft
        Loaded PyVASP calculation.
    ion_name
        Element symbol, e.g. ``"Sb"`` or ``"Bi"``.

    Returns
    -------
    numpy.ndarray
        Array containing the indices of the selected ions.
    """
    return dft.structure.indices_from_symbol(ion_name)


def by_layer(
    dft: PyVASP,
    layer: int = 1,
    min_layer_distance: float = 1.5,
) -> np.ndarray:
    """
    Return ion indices belonging to a selected atomic layer.

    Layers are determined from the Cartesian z coordinates of the ions.
    ``layer=1`` corresponds to the topmost layer.

    Parameters
    ----------
    dft
        Loaded PyVASP calculation.
    layer
        Layer number, starting from 1 for the topmost layer.
    min_layer_distance
        Minimum z-distance used to distinguish separate layers.

    Returns
    -------
    numpy.ndarray
        Array containing the ion indices belonging to the selected layer.

    Raises
    ------
    ValueError
        If ``layer`` is less than 1 or ``min_layer_distance`` is not positive.
    """
    if layer < 1:
        raise ValueError("layer must be >= 1")

    if min_layer_distance <= 0:
        raise ValueError("min_layer_distance must be > 0")

    ion_z = np.asarray(dft.structure.cart_coords[:, 2])

    ion_indices = np.argsort(ion_z)
    z_sorted = np.sort(ion_z)

    z_diff = np.diff(z_sorted)

    if len(z_sorted) == 0:
        return np.array([], dtype=int)

    # Detect a large gap between the bottom and top parts of a slab.
    if len(z_diff) > 0 and z_diff.max() > 5 * min_layer_distance:
        max_index = int(np.argmax(z_diff))

        z_sorted[: max_index + 1] += (
            z_sorted[-1] + min_layer_distance
        )

        z_sorted = np.roll(
            z_sorted,
            -(max_index + 1),
        )

        ion_indices = np.roll(
            ion_indices,
            -(max_index + 1),
        )

        z_diff = np.diff(z_sorted)

    # Mark the end of the final layer.
    z_diff = np.append(z_diff, min_layer_distance)

    layer_boundaries = np.flatnonzero(
        z_diff >= min_layer_distance
    )

    if layer > len(layer_boundaries):
        raise ValueError(
            f"Layer {layer} does not exist. "
            f"Found {len(layer_boundaries)} layers."
        )

    start = layer_boundaries[-layer - 1] + 1
    end = layer_boundaries[-layer] + 1

    return ion_indices[start:end]
