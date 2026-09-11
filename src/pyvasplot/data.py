from dataclasses import dataclass, field
from pathlib import Path


from pymatgen.core.structure import IStructure
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.outputs import Outcar, Procar


@dataclass
class VASPData:
    """Container for data loaded from a VASP calculation."""

    procar: Procar | None = None
    outcar: Outcar | None = None
    kpoints: Kpoints | None = None
    structure: IStructure | None = None

    # Used by calculations containing multiple related datasets,
    # such as kxky calculations.
    calculations: list["VASPData"] | None = field(default=None)