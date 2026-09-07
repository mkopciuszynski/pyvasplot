from pathlib import Path

import dill

from pyvasplot.data import VASPData
from pyvasplot.types import CalculationType
from pyvasplot.io.source import DataSource
from pyvasplot.io.standard import load_standard
from pyvasplot.io.hse import load_hse
from pyvasplot.io.kxky import load_kxky


def load(
    path: str | Path,
    calculation_type: CalculationType,
    cache_path: Path,
    dataset: str | None = None,
    parent: str | None = None,
    reload: bool = False,
) -> VASPData:
    """Load a VASP calculation from a directory or ZIP archive."""

    if not reload and cache_path.exists():
        return _load_cache(cache_path)

    with DataSource(path) as root:
        calculation_path = _resolve_directory(
            root=root,
            calculation_type=calculation_type,
            dataset=dataset,
            parent=parent,
        )

        if calculation_type.is_kxky:
            data = load_kxky(calculation_path)

        elif calculation_type.is_hse:
            data = load_hse(
                calculation_path,
                calculation_type,
            )

        else:
            data = load_standard(
                calculation_path,
                calculation_type,
            )

    _save_cache(cache_path, data)

    return data


def _resolve_directory(
    root: Path,
    calculation_type: CalculationType,
    dataset: str | None = None,
    parent: str | None = None,
) -> Path:
    """Resolve a calculation type to the directory containing VASP files."""

    root = _resolve_parent(root, parent)

    if calculation_type.is_path:
        return _resolve_path_calculation(
            root,
            calculation_type,
            dataset,
        )

    return _resolve_standard_calculation(
        root,
        calculation_type,
        dataset,
    )


def _resolve_parent(
    root: Path,
    parent: str | None,
) -> Path:
    """Resolve an optional parent calculation directory."""

    if parent is None:
        return root

    parent_path = root / parent

    if not parent_path.is_dir():
        raise FileNotFoundError(
            f"Parent calculation not found: {parent_path}"
        )

    return parent_path


def _resolve_standard_calculation(
    root: Path,
    calculation_type: CalculationType,
    dataset: str | None,
) -> Path:
    """Resolve a standard calculation.

    Standard calculations normally have a directory with the same
    name as the calculation type.
    """

    calculation_dir = root / calculation_type.value

    if calculation_dir.is_dir():
        if dataset is None:
            return calculation_dir

        dataset_dir = calculation_dir / dataset

        if dataset_dir.is_dir():
            return dataset_dir

        raise FileNotFoundError(
            f"Dataset '{dataset}' not found in {calculation_dir}"
        )

    # BS is commonly represented directly by a BS_* directory.
    if calculation_type is CalculationType.BS:
        return _resolve_path_calculation(
            root,
            calculation_type,
            dataset,
        )

    raise FileNotFoundError(
        f"Could not find directory for calculation type "
        f"'{calculation_type.value}' in {root}"
    )


def _resolve_path_calculation(
    root: Path,
    calculation_type: CalculationType,
    dataset: str | None,
) -> Path:
    """Resolve a band-path calculation.

    Path calculations are identified by directories beginning with
    'BS_', for example BS_MGKM, BS_GKMG, or BS_GKMG_Z3.
    """

    if dataset is not None:
        path = root / dataset

        if path.is_dir():
            return path

        raise FileNotFoundError(
            f"Dataset '{dataset}' not found in {root}"
        )

    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("BS_")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No BS_* path calculation found in {root}"
        )

    if len(candidates) > 1:
        names = "\n".join(
            f"    {path.name}"
            for path in candidates
        )

        raise ValueError(
            f"Multiple band-path datasets found in {root}:\n"
            f"{names}\n\n"
            f"Specify dataset='...' to select one."
        )

    return candidates[0]


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
