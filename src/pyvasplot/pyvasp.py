from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import IStructure
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.outputs import Outcar

from pyvasplot.data import VASPData
from pyvasplot.types import CalculationType


class PyVASP:
    """Interface to VASP calculation data."""

    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
        calculation_type: CalculationType | str = CalculationType.BS,
        local_dir: str | Path = "dft_local",
    ) -> None:
        self.path = Path(path)
        self.name = name or self._generate_name()
        self.calculation_type = CalculationType(calculation_type)

        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.data = VASPData()

        self._eshift = 0.0
        self._selected_ky = 0

        if not self.path.is_dir():
            raise FileNotFoundError(
                f"VASP directory not found: {self.path}"
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, reload: bool = False) -> None:
        """Load the VASP calculation data."""
        from pyvasplot.io.loader import load

        self.data = load(
            path=self.path,
            calculation_type=self.calculation_type,
            cache_path=self.cache_path,
            reload=reload,
        )

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------

    @property
    def cache_path(self) -> Path:
        """Path to the local cache file."""
        return self.local_dir / f"{self.name}.pkl"

    @property
    def model_number(self) -> int | None:
        """Return the four-digit model number from the calculation name."""
        import re

        match = re.search(r"\d{4}", self.name)

        if match is None:
            return None

        return int(match.group())

    # ------------------------------------------------------------------
    # VASP / pymatgen objects
    # ------------------------------------------------------------------

    @property
    def procar(self):
        """Loaded PROCAR data."""
        return self.data.procar

    @property
    def outcar(self) -> Outcar:
        """Loaded OUTCAR data."""
        return self.data.outcar

    @property
    def kpoints(self) -> Kpoints:
        """Loaded KPOINTS data."""
        return self.data.kpoints

    @property
    def structure(self) -> IStructure:
        """Structure loaded from CONTCAR."""
        return self.data.structure

    # ------------------------------------------------------------------
    # Electronic structure
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> NDArray:
        """Band eigenvalues from PROCAR."""
        return self.procar.eigenvalues

    @property
    def procar_data(self) -> NDArray:
        """Raw projection data from PROCAR."""
        return self.procar.data

    @property
    def efermi(self) -> float:
        """Fermi energy."""
        return self.outcar.efermi

    @property
    def nbands(self) -> int:
        """Number of bands."""
        return self.eigenvalues.shape[1]

    @property
    def nkpoints(self) -> int:
        """Number of k-points."""
        return self.eigenvalues.shape[0]

    @property
    def nions(self) -> int:
        """Number of ions."""
        return self.procar.nions

    # ------------------------------------------------------------------
    # K-points
    # ------------------------------------------------------------------

    @property
    def kpts(self) -> NDArray:
        """K-points as a NumPy array."""
        return np.asarray(self.kpoints.kpts)

    @property
    def nkpoints_section(self) -> int:
        """Number of k-points in a KPOINTS section."""
        return self.kpoints.num_kpts

    # ------------------------------------------------------------------
    # Structure / lattice
    # ------------------------------------------------------------------

    @property
    def lattice(self) -> NDArray:
        """Real-space lattice vectors."""
        return np.asarray(self.structure.lattice.matrix)

    @property
    def reciprocal_lattice(self) -> NDArray:
        """Reciprocal lattice vectors."""
        return np.asarray(
            self.structure.lattice.reciprocal_lattice.matrix
        )

    # ------------------------------------------------------------------
    # Energy shift
    # ------------------------------------------------------------------

    @property
    def eshift(self) -> float:
        """Energy shift applied for plotting."""
        return self._eshift

    @eshift.setter
    def eshift(self, value: float) -> None:
        self._eshift = float(value)

    # ------------------------------------------------------------------
    # kx-ky calculations
    # ------------------------------------------------------------------

    @property
    def selected_ky(self) -> int:
        """Currently selected ky index."""
        return self._selected_ky

    @selected_ky.setter
    def selected_ky(self, value: int) -> None:
        if not self.calculation_type.is_kxky:
            raise AttributeError(
                "selected_ky is only available for kxky calculations."
            )

        if self.data.calculations is None:
            raise RuntimeError("Calculation data has not been loaded.")

        if not 0 <= value < len(self.data.calculations):
            raise IndexError(
                f"ky index {value} is out of range."
            )

        self._selected_ky = value

    @property
    def current(self) -> VASPData:
        """Currently active dataset."""
        if self.calculation_type.is_kxky:
            if self.data.calculations is None:
                raise RuntimeError("Calculation data has not been loaded.")

            return self.data.calculations[self._selected_ky]

        return self.data

    @property
    def nky(self) -> int:
        """Number of ky calculations."""
        if not self.calculation_type.is_kxky:
            raise AttributeError(
                "nky is only available for kxky calculations."
            )

        if self.data.calculations is None:
            raise RuntimeError("Calculation data has not been loaded.")

        return len(self.data.calculations)

    # ------------------------------------------------------------------
    # Derived k-space quantities
    # ------------------------------------------------------------------

    @property
    def knorm(self) -> NDArray:
        """Distance between consecutive k-points."""
        reciprocal_kpts = self.kpts @ self.reciprocal_lattice

        return np.linalg.norm(
            np.diff(reciprocal_kpts, axis=0),
            axis=1,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PyVASP("
            f"path={str(self.path)!r}, "
            f"name={self.name!r}, "
            f"calculation_type={self.calculation_type.value!r}"
            f")"
        )

    def __str__(self) -> str:
        return (
            f"PyVASP\n"
            f"  path: {self.path}\n"
            f"  type: {self.calculation_type.value}\n"
            f"  name: {self.name}\n"
            f"  model: {self.model_number}\n"
            f"  eshift: {self.eshift}\n"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_name(self) -> str:
        """Generate a name from the VASP directory path."""
        import re

        path_string = str(self.path)

        match = re.search(r"\d{4}.*", path_string)

        if match:
            return match.group()

        return self.path.name