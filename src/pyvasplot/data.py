from dataclasses import dataclass

from pymatgen.core.structure import IStructure
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.outputs import Outcar, Procar


@dataclass
class KXKYSlice:
    ky: int
    procar: Procar
    outcar: Outcar
    kpoints: Kpoints


@dataclass
class KXKYData:
    slices: list[KXKYSlice]

    @property
    def nky(self) -> int:
        return len(self.slices)


@dataclass
class VASPData:
    """Container for data loaded from a VASP calculation."""

    procar: Procar | None = None
    outcar: Outcar | None = None
    kpoints: Kpoints | None = None
    structure: IStructure | None = None

    kxky: KXKYData | None = None
