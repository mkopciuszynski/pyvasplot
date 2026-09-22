import re
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pymatgen.electronic_structure.core import Spin

from pyvasplot.data import KXKYData, VASPData
from pyvasplot.types import CalculationType, infer_calculation_type


class PyVASP:
    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
        calculation_type: CalculationType | str | None = None,
        subpath: str | Path | None = None,
        local_dir: str | Path = "dft_local",
    ) -> None:
        self.path = Path(path)
        self.subpath = Path(subpath) if subpath is not None else None

        self.name = name or _generate_name(
            str(path),
            str(subpath) if subpath is not None else None,
        )

        if calculation_type is None:
            calculation_type = infer_calculation_type(self.subpath or self.path)

        self.calculation_type = (
            CalculationType(calculation_type) if calculation_type is not None else None
        )

        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self._data = VASPData()
        self._eshift = 0.0
        self._selected_ky = 0

    def load(
        self,
        reload: bool = False,
        show_progress: bool = False,
    ) -> None:
        """Load calculation data, optionally showing progress for long loads."""
        if self.calculation_type is None:
            raise ValueError(
                "Could not determine the calculation type. "
                "Please provide calculation_type explicitly."
            )

        from pyvasplot.io.loader import load

        self._data = load(
            path=self.path,
            calculation_type=self.calculation_type,
            cache_path=self.cache_path,
            subpath=self.subpath,
            reload=reload,
            show_progress=show_progress,
        )

    def __str__(self) -> str:
        efermi = self.outcar.efermi if self.outcar is not None else "-"
        lines = [
            f"PyVASP: {self.name}",
            f"  data path: {self.full_data_path}",
            f"  calculation type: {self.calculation_type or '-'}",
            f"  model number: {self.model_number}",
            f"  cache: {self.full_cache_path}",
            f"  e-fermi: {efermi} eV",
            f"  eshift: {self.eshift} eV",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"data_path={self.full_data_path!r}, "
            f"calculation_type={self.calculation_type!r}, "
            f"cache_path={self.full_cache_path!r})"
        )

    @property
    def full_path(self) -> Path:
        """Absolute path to the input directory or ZIP archive."""
        return self.path.resolve()

    @property
    def full_data_path(self) -> Path:
        """Absolute path to the calculation, including its subpath."""
        if self.subpath is None:
            return self.full_path

        return (self.full_path / self.subpath).resolve()

    @property
    def full_local_dir(self) -> Path:
        """Absolute path to the local cache directory."""
        return self.local_dir.resolve()

    @property
    def cache_path(self) -> Path:
        """Path to the local cache file."""
        return self.local_dir / f"{self.name}.pkl"

    @property
    def full_cache_path(self) -> Path:
        """Absolute path to the local cache file."""
        return self.cache_path.resolve()

    @property
    def eshift(self) -> float:
        return self._eshift

    @eshift.setter
    def eshift(self, value: float) -> None:
        if np.abs(value) > 1.0:
            warnings.warn(
                "The energy shift is bigger than 1 eV!",
                stacklevel=2,
            )
        self._eshift = value

    @property
    def procar(self):
        if self._data.kxky is not None:
            return self._data.kxky.slices[self._selected_ky].procar
        return self._data.procar

    @property
    def outcar(self):
        if self._data.kxky is not None:
            return self._data.kxky.slices[self._selected_ky].outcar
        return self._data.outcar

    @property
    def kpoints(self):
        if self._data.kxky is not None:
            return self._data.kxky.slices[self._selected_ky].kpoints
        return self._data.kpoints

    @property
    def structure(self):
        if self._data.structure is None:
            raise RuntimeError("Structure has not been loaded.")
        return self._data.structure

    @property
    def kxky(self) -> KXKYData | None:
        return self._data.kxky

    @property
    def eigenvalues(self):
        return self.procar.eigenvalues[Spin.up]

    @property
    def procar_data(self):
        return self.procar.data[Spin.up]

    @property
    def efermi(self):
        return self.outcar.efermi

    @property
    def nbands(self) -> int:
        return self.eigenvalues.shape[1]

    @property
    def nkpoints(self) -> int:
        """Number of k-points represented by the parsed electronic data."""
        return self.eigenvalues.shape[0]

    @property
    def nions(self) -> int:
        """Number of ions represented in the loaded projection data."""
        return self.procar.nions

    @property
    def num_kpts(self) -> int:
        """Number of k-points in the current k-path section."""

        return self.kpoints.num_kpts

    @property
    def kpts(self) -> NDArray:
        """K-point coordinates from the KPOINTS file."""
        return np.asarray(self.kpoints.kpts)

    @property
    def cell_real(self) -> NDArray:
        """Real-space lattice vectors."""
        return np.asarray(self.structure.lattice.matrix)

    @property
    def cell_inverse(self) -> NDArray:
        """Reciprocal-space lattice vectors including 2π."""
        return np.asarray(self.structure.lattice.reciprocal_lattice.matrix)

    @property
    def knorm(self) -> float:
        """Total cumulative k-path distance."""

        kpts = self.kpts

        if len(kpts) < 2:
            return 0.0

        kpts_cart = np.dot(kpts, self.cell_inverse)

        distances = np.linalg.norm(
            np.diff(kpts_cart, axis=0),
            axis=1,
        )

        return float(np.sum(distances))

    @property
    def nky(self) -> int:
        """Number of ky slices in KXKY data."""
        if not self.calculation_type.is_kxky:
            raise ValueError("nky is only available for KXKY calculations.")

        if self.kxky is None:
            raise RuntimeError("KXKY data has not been loaded.")

        return self.kxky.nky

    @property
    def selected_ky(self) -> int:
        """Currently selected ky slice."""
        if not self.calculation_type.is_kxky:
            raise ValueError("selected_ky is only available for KXKY calculations.")
        return self._selected_ky

    @selected_ky.setter
    def selected_ky(self, ky: int) -> None:
        """Select a ky slice and expose its data through the main properties."""
        if not self.calculation_type.is_kxky:
            raise ValueError("selected_ky is only available for KXKY calculations.")

        if self.kxky is None:
            raise RuntimeError("KXKY data has not been loaded.")

        if not 0 <= ky < self.kxky.nky:
            raise IndexError(
                f"ky index {ky} is out of range. Valid range is 0-{self.kxky.nky - 1}."
            )

        self._selected_ky = ky

    @property
    def kynorm(self) -> float:
        """Maximum ky coordinate across the KXKY grid."""

        if not self.calculation_type.is_kxky:
            raise ValueError("kynorm is only available for KXKY data.")

        if self.kxky is None:
            raise RuntimeError("KXKY data has not been loaded.")

        ky_values = []

        for slice_data in self.kxky.slices:
            kpts_cart = np.dot(
                slice_data.kpoints.kpts,
                self.cell_inverse,
            )
            ky_values.append(np.max(kpts_cart[:, 1]))

        if not ky_values:
            return 0.0
        return float(np.max(ky_values))

    @property
    def model_number(self) -> str:
        """Return the four-digit model number from the calculation name."""
        match = re.search(r"\d{4}", self.name)

        if match:
            return match.group()

        raise ValueError(f"Could not find a four-digit model number in '{self.name}'.")


def _generate_name(path: str, subpath: str | None = None) -> str:
    normalized = path.replace("\\", "_").replace("/", "_")
    normalized = normalized.replace(":", "_")

    if subpath:
        subpath = subpath.replace("\\", "_").replace("/", "_")
        normalized = f"{normalized}_{subpath}"

    pattern = r"(\d{4})_(.*)"
    match = re.search(pattern, normalized)

    if match:
        return f"{match.group(1)}_{match.group(2)}"

    return normalized
