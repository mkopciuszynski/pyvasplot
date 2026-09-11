from pathlib import Path

from pymatgen.core.structure import IStructure
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.io.vasp.outputs import Procar


from pyvasplot.data import VASPData


def load_bs(path: Path, calculation_type) -> VASPData:

    procar = Procar(path / "PROCAR")

    outcar = Outcar(path / "OUTCAR")
    kpoints = Kpoints.from_file(path / "KPOINTS")
    structure = IStructure.from_file(path / "CONTCAR")

    return VASPData(
        procar=procar,
        outcar=outcar,
        kpoints=kpoints,
        structure=structure,
    )