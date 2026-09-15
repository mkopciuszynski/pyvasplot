from __future__ import annotations

import numpy as np

from pyvasplot import PyVASP


def by_name(dft: PyVASP, ion_name: str) -> tuple[int, ...]:
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
    tuple[int, ...]
        Tuple containing the indices of the selected ions.
    """
    return dft.structure.indices_from_symbol(ion_name)



def by_layer(dft: PyVASP, layer=1, min_layer_distance=1.5):
    """
       Get ion list by selecting monolayer 1 - means top layer
       :param layer_num: means top layer
       :return:
       """

    structure = dft.structure
    ion_z = structure.cart_coords[:, 2]

    ion_n_sorted = np.argsort(ion_z)
    ion_z_sorted = np.sort(ion_z)
    z_diff = np.diff(ion_z_sorted)

    if z_diff.max() > 5 * min_layer_distance:

        max_ind = max((v, i) for i, v in enumerate(z_diff))[1]
        ion_z_sorted[0:max_ind + 1] += ion_z_sorted[-1] + min_layer_distance

        ion_z_sorted = np.roll(ion_z_sorted, -(max_ind + 1))
        ion_n_sorted = np.roll(ion_n_sorted, -(max_ind + 1))

        z_diff = np.diff(ion_z_sorted)
        print('Gap in the center')

    z_diff = np.append(z_diff, min_layer_distance)
    first_element_in_ml = np.argwhere(z_diff >= min_layer_distance)

    return np.sort(ion_n_sorted[first_element_in_ml[-1 - layer][0] + 1:first_element_in_ml[0 - layer][0] + 1])

    