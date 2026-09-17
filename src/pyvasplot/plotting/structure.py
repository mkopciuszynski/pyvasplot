from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor

from pyvasplot.pyvasp import PyVASP


_VIEW_ROTATIONS = {
    "xy": "0x,0y,0z",
    "xz": "-90x,0y,0z",
    "yz": "90x,90y,180z",
}

_VIEW_TITLES = {
    "xy": "X-Y",
    "xz": "X-Z",
    "yz": "Y-Z",
}


def plot_structure(
    dft: PyVASP,
    views: str | Sequence[str] = ("xy", "xz", "yz"),
    ax: plt.Axes | Sequence[plt.Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    radii: float = 0.7,
    **kwargs,
) -> plt.Axes | list[plt.Axes]:
    """Plot the crystal structure in one or more static projections."""

    if dft.structure is None:
        raise RuntimeError("Structure has not been loaded.")

    if isinstance(views, str):
        views = (views,)

    views = tuple(views)

    invalid_views = set(views) - _VIEW_ROTATIONS.keys()
    if invalid_views:
        raise ValueError(
            f"Unknown structure view(s): {sorted(invalid_views)}. "
            f"Valid views are: {tuple(_VIEW_ROTATIONS)}."
        )

    atoms = AseAtomsAdaptor.get_atoms(dft.structure)

    if ax is None:
        if len(views) == 1:
            _, axes = plt.subplots(figsize=figsize)
            axes = [axes]
        else:
            if figsize is None:
                figsize = (3 * len(views), 5)

            _, axes = plt.subplots(
                1,
                len(views),
                figsize=figsize,
                squeeze=False,
            )
            axes = list(axes[0])

    elif isinstance(ax, plt.Axes):
        if len(views) != 1:
            raise ValueError(
                "A single Axes can only be used with one view."
            )
        axes = [ax]

    else:
        axes = list(ax)

        if len(axes) != len(views):
            raise ValueError(
                f"Expected {len(views)} axes, got {len(axes)}."
            )

    for axis, view in zip(axes, views):
        plot_atoms(
            atoms,
            axis,
            rotation=_VIEW_ROTATIONS[view],
            radii=radii,
            **kwargs,
        )

        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(_VIEW_TITLES[view])

    # Use the same physical scale for all projections.
    if len(axes) > 1:
        _equalize_structure_scale(axes)

    axes[0].figure.tight_layout()

    if len(axes) == 1:
        return axes[0]

    return axes


def _equalize_structure_scale(
    axes: Sequence[plt.Axes],
) -> None:
    """Use the same physical scale for all structure projections."""

    max_extent = 0.0

    for axis in axes:
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()

        max_extent = max(
            max_extent,
            xmax - xmin,
            ymax - ymin,
        )

    for axis in axes:
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()

        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2

        axis.set_xlim(
            xcenter - max_extent / 2,
            xcenter + max_extent / 2,
        )
        axis.set_ylim(
            ycenter - max_extent / 2,
            ycenter + max_extent / 2,
        )

        axis.set_aspect("equal", adjustable="box")