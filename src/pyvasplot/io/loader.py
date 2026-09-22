from pathlib import Path

import dill

from pyvasplot.data import VASPData
from pyvasplot.io.source import DataSource
from pyvasplot.io.hse import load_hse
from pyvasplot.io.kxky import load_kxky
from pyvasplot.io.bs import load_bs
from pyvasplot.types import CalculationType


def load(
    path: str | Path,
    calculation_type: CalculationType,
    cache_path: Path,
    subpath: str | Path | None = None,
    reload: bool = False,
    show_progress: bool = False,
) -> VASPData:
    """Load a VASP calculation from a directory or ZIP archive."""

    if not reload and cache_path.exists():
        return _load_cache(cache_path)

    with DataSource(path, subpath=subpath) as calculation_path:
        data = _load_calculation(
            calculation_path,
            calculation_type,
            show_progress=show_progress,
        )

    _save_cache(cache_path, data)

    return data


def _load_calculation(
    path: Path,
    calculation_type: CalculationType,
    show_progress: bool = False,
) -> VASPData:
    """Load a calculation using the appropriate loader."""

    if calculation_type.is_kxky:
        return load_kxky(path, show_progress=show_progress)

    if calculation_type.is_hse:
        return load_hse(path, calculation_type)

    return load_bs(path, calculation_type)


def _load_cache(cache_path: Path) -> VASPData:
    """Load VASP data from a local cache file."""

    with cache_path.open("rb") as file:
        data = dill.load(file)

    if not isinstance(data, VASPData):
        raise TypeError(
            f"Invalid cache format in {cache_path}: "
            f"expected VASPData, got {type(data).__name__}"
        )

    return data


def _save_cache(
    cache_path: Path,
    data: VASPData,
) -> None:
    """Save VASP data to a local cache file."""

    with cache_path.open("wb") as file:
        dill.dump(data, file)
