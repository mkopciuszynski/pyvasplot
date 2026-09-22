from pathlib import Path

from pyvasplot.data import VASPData
from pyvasplot.io.bs import load_bs


def load_hse(
    path: Path,
    calculation_type,
) -> VASPData:

    data = load_bs(
        path,
        calculation_type,
    )

    # HSE-specific processing here

    return data
