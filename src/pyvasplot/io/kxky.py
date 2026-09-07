from pathlib import Path

from pyvasplot.data import VASPData


def load_kxky(path: Path) -> VASPData:

    # Find PROCAR.*, OUTCAR.*, KPOINTS.*
    # Load corresponding datasets
    # Build VASPData.calculations

    ...