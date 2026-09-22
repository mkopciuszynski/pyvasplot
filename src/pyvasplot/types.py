from enum import StrEnum
from pathlib import Path


class CalculationType(StrEnum):
    """Types of VASP calculations supported by pyvasplot."""

    BS = "BS"
    BS_KPATH = "BS_KPATH"

    HSE06 = "HSE06"
    HSE06_KPATH = "HSE06_KPATH"

    SO_STATIC = "SO_STATIC"
    SO_STATIC_KPATH = "SO_STATIC_KPATH"

    KXKY = "KXKY"
    SO_STATIC_KXKY = "SO_STATIC_KXKY"

    @property
    def is_path(self) -> bool:
        """Whether this calculation contains a k-path."""
        return self in {
            CalculationType.BS_KPATH,
            CalculationType.HSE06_KPATH,
            CalculationType.SO_STATIC_KPATH,
        }

    @property
    def is_kxky(self) -> bool:
        """Whether this is a kxky calculation."""
        return self in {CalculationType.KXKY, 
                        CalculationType.SO_STATIC_KXKY}

    @property
    def is_hse(self) -> bool:
        """Whether this is an HSE calculation."""
        return self in {
            CalculationType.HSE06,
            CalculationType.HSE06_KPATH,
        }

    @property
    def is_so(self) -> bool:
        """Whether this calculation includes spin-orbit coupling."""
        return self in {
            CalculationType.SO_STATIC,
            CalculationType.SO_STATIC_KPATH,
            CalculationType.SO_STATIC_KXKY,
        }


def infer_calculation_type(path: str | Path) -> CalculationType | None:
    path = Path(path)

    parts = {part.upper() for part in path.parts}

    if "SO_STATIC" in parts:
        return CalculationType.SO_STATIC

    if "HSE06" in parts:
        return CalculationType.HSE06

    if "BZ" in parts:
        return CalculationType.KXKY

    return None
