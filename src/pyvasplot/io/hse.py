from pathlib import Path

from pyvasplot.data import VASPData
from pyvasplot.io.standard import load_standard


def load_hse(
    path: Path,
    calculation_type,
) -> VASPData:

    data = load_standard(
        path,
        calculation_type,
    )

    # HSE-specific processing here

    return data