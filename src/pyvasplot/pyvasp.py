from pathlib import Path

from pyvasplot.data import VASPData
from pyvasplot.types import CalculationType
from pymatgen.electronic_structure.core import Spin



class PyVASP:
    """Interface to data produced by a VASP calculation."""

    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
        calculation_type: CalculationType | str = CalculationType.BS,
        dataset: str | None = None,
        parent: str | None = None,
        local_dir: str | Path = "dft_local",
    ) -> None:
        self.path = Path(path)
        self.name = name or self._generate_name()

        self.calculation_type = CalculationType(calculation_type)
        self.dataset = dataset
        self.parent = parent

        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.data = VASPData()

        self._eshift = 0.0
        self._selected_ky = 0

        if not self.path.exists():
            raise FileNotFoundError(
                f"VASP data not found: {self.path}"
            )

        if not self.path.is_dir() and self.path.suffix.lower() != ".zip":
            raise ValueError(
                "VASP data must be a directory or a ZIP archive."
            )

    @property
    def cache_path(self) -> Path:
        """Path to the local cache file."""
        return self.local_dir / f"{self.name}.pkl"

    def load(self, reload: bool = False) -> None:
        """Load the selected VASP calculation."""

        from pyvasplot.io.loader import load

        self.data = load(
            path=self.path,
            calculation_type=self.calculation_type,
            dataset=self.dataset,
            parent=self.parent,
            cache_path=self.cache_path,
            reload=reload,
        )

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
    def eshift(self) -> float:
        return self._eshift

    @eshift.setter
    def eshift(self, value: float) -> None:
        self._eshift = float(value)

    def _generate_name(self) -> str:
        """Generate a name suitable for the cache file."""
        if self.path.suffix.lower() == ".zip":
            return self.path.stem

        return self.path.name

    def __str__(self):
        return f"\n \
                origin dir: {str(self.path)} \n \
                type: {str(self.calculation_type)} \n \
                name: {self.name} \n \
                eshift: {self.eshift}eV \n \
                e-fermi: {self.efermi}eV"