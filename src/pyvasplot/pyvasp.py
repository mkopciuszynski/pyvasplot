from pathlib import Path

import re
import numpy as np
from numpy.typing import NDArray

from pymatgen.core import IStructure

from pyvasplot.data import VASPData
from pyvasplot.types import CalculationType
from pymatgen.electronic_structure.core import Spin


from pathlib import Path

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

        self.name = name or _generate_name(str(path), str(subpath))

        if calculation_type is None:
            calculation_type = infer_calculation_type(
                self.subpath or self.path
            )

        self.calculation_type = (
            CalculationType(calculation_type)
            if calculation_type is not None
            else None
        )

        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.data = VASPData()
        self._eshift = 0.0





    def load(self, reload: bool = False) -> None:
        if self.calculation_type is None:
            raise ValueError(
                "Could not determine the calculation type. "
                "Please provide calculation_type explicitly."
            )

        from pyvasplot.io.loader import load

        self.data = load(
            path=self.path,
            calculation_type=self.calculation_type,
            cache_path=self.cache_path,
            subpath=self.subpath,
            reload=reload,
        )


    def __str__(self):
        return f"\n \
                data path: {str(self.path)} \n \
                calculation type: {str(self.calculation_type)} \n \
                name: {self.name} \n \
                eshift: {self.eshift} eV \n \
                e-fermi: {self.efermi} eV"

    @property
    def cache_path(self) -> Path:
        """Path to the local cache file."""
        return self.local_dir / f"{self.name}.pkl"



    @property
    def eshift(self) -> float:
        return self._eshift
    
    @eshift.setter
    def eshift(self, value):
        if np.abs(value) > 1.0:
            Warning("The energy shift is bigger than 1 eV!")
        self._eshift = value

    @property
    def procar(self):
        return self.data.procar

    @property
    def outcar(self):
        return self.data.outcar

    @property
    def kpoints(self):
        return self.data.kpoints

    @property
    def structure(self):
        return self.data.structure

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
        return self.eigenvalues.shape[0]

    @property
    def nions(self) -> int:
        return self.procar.nions

        
    @property
    def nkpoints(self) -> int:
        """Number of k-points represented by the parsed electronic data."""
        return self.eigenvalues.shape[0]


    @property
    def num_kpts(self) -> int:
        """Number of k-points in the current k-path section."""

        return self.kpoints.num_kpts


    @property
    def kpts(self) -> NDArray:
        """K-point coordinates from the KPOINTS file."""
        return np.asarray(self.kpoints.kpts)


    @property
    def nions(self) -> int:
        """Number of ions."""
        return self.procar.nions


    @property
    def structure(self) -> IStructure:
        """Atomic structure loaded from the calculation."""
        if self.data.structure is None:
            raise RuntimeError("Structure has not been loaded.")

        return self.data.structure


    @property
    def cell_real(self) -> NDArray:
        """Real-space lattice vectors."""
        return np.asarray(self.structure.lattice.matrix)


    @property
    def cell_inverse(self) -> NDArray:
        """Reciprocal-space lattice vectors including 2π."""
        return np.asarray(
            self.structure.lattice.reciprocal_lattice.matrix
        )


    @property
    def knorm(self) -> NDArray:
        """
        Cumulative k-path distance.

        The returned array has one value for each k-point in the KPOINTS file.
        """
        kpts = self.kpts

        if len(kpts) == 0:
            return np.array([], dtype=float)

        if len(kpts) == 1:
            return np.array([0.0])

        kpts_cart = np.dot(kpts, self.cell_inverse)

        distances = np.linalg.norm(
            np.diff(kpts_cart, axis=0),
            axis=1,
        )

        return np.concatenate(
            ([0.0], np.cumsum(distances))
        )


    @property
    def nky(self) -> int:
        """Number of ky calculations for KXKY data."""
        if not self.calculation_type.is_kxky:
            raise ValueError("nky is only available for KXKY data.")

        if self.data.calculations is None:
            raise RuntimeError("KXKY calculations have not been loaded.")

        return len(self.data.calculations)


    @property
    def kynorm(self) -> float:
        """Maximum ky coordinate for KXKY data."""
        if not self.calculation_type.is_kxky:
            raise ValueError("kynorm is only available for KXKY data.")

        kpts_cart = np.dot(self.kpts, self.cell_inverse)

        return float(np.max(kpts_cart[:, 1]))


    @property
    def model_number(self) -> str:
        """Return the four-digit model number from the calculation name."""
        match = re.search(r"\d{4}", self.name)

        if match:
            return match.group()

        raise ValueError(
            f"Could not find a four-digit model number in '{self.name}'."
        )

    
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