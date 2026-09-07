from dataclasses import dataclass, field

from pymatgen.core.structure import IStructure
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.outputs import Outcar

from pyvasplot.procar import ProcarSO


@dataclass
class VASPData:
    """Container for data loaded from a VASP calculation."""

    procar: ProcarSO | None = None
    outcar: Outcar | None = None
    kpoints: Kpoints | None = None
    structure: IStructure | None = None

    # Used by calculations containing multiple related datasets,
    # such as kxky calculations.
    calculations: list["VASPData"] | None = field(default=None)