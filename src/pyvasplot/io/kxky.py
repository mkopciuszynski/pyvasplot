from pathlib import Path

from pymatgen.core import IStructure
from pymatgen.io.vasp.outputs import Kpoints, Outcar, Procar
from tqdm import tqdm

from pyvasplot.data import KXKYData, KXKYSlice, VASPData


def load_kxky(path: Path, show_progress: bool = False) -> VASPData:
    """Load a KXKY calculation consisting of indexed VASP files."""

    procar_files = _sort_indexed_files(path, "PROCAR.*")
    outcar_files = _sort_indexed_files(path, "OUTCAR.*")
    kpoints_files = _sort_indexed_files(path, "KPOINTS.*")

    if not procar_files:
        raise FileNotFoundError(f"No PROCAR.* files found in KXKY directory: {path}")

    if not (len(procar_files) == len(outcar_files) == len(kpoints_files)):
        raise ValueError(
            "KXKY file count mismatch: "
            f"{len(procar_files)} PROCAR files, "
            f"{len(outcar_files)} OUTCAR files, "
            f"{len(kpoints_files)} KPOINTS files."
        )

    slices = []

    for ky, (procar_file, outcar_file, kpoints_file) in enumerate(
        tqdm(
            zip(procar_files, outcar_files, kpoints_files),
            total=len(procar_files),
            desc="Loading KXKY",
            unit="ky",
            disable=not show_progress,
        )
    ):
        slices.append(
            KXKYSlice(
                ky=ky,
                procar=Procar(procar_file),
                outcar=Outcar(outcar_file),
                kpoints=Kpoints.from_file(kpoints_file),
            )
        )

    structure = IStructure.from_file(path / "CONTCAR")

    return VASPData(
        procar=slices[0].procar,
        outcar=slices[0].outcar,
        kpoints=slices[0].kpoints,
        structure=structure,
        kxky=KXKYData(slices=slices),
    )


def _sort_indexed_files(path: Path, pattern: str) -> list[Path]:
    files = list(path.glob(pattern))

    try:
        return sorted(
            files,
            key=lambda file: int(file.name.rsplit(".", 1)[1]),
        )
    except ValueError as exc:
        raise ValueError(f"Expected indexed files matching {pattern!r}") from exc
