from enum import StrEnum


class CalculationType(StrEnum):
    """Types of VASP calculations supported by pyvasplot."""

    BS = "BS"
    BS_PATH = "BS-path"
    HSE = "HSE"
    HSE_PATH = "HSE-path"
    SO = "SO"
    SO_PATH = "SO-path"
    SOXK = "SOxK"
    KXKY = "kxky"

    @property
    def is_path(self) -> bool:
        """Whether this calculation contains a k-path."""
        return self in {
            CalculationType.BS_PATH,
            CalculationType.HSE_PATH,
            CalculationType.SO_PATH,
        }

    @property
    def is_kxky(self) -> bool:
        """Whether this is a kxky calculation."""
        return self is CalculationType.KXKY

    @property
    def is_hse(self) -> bool:
        """Whether this is an HSE calculation."""
        return self in {
            CalculationType.HSE,
            CalculationType.HSE_PATH,
        }

    @property
    def is_so(self) -> bool:
        """Whether this calculation includes spin-orbit coupling."""
        return self in {
            CalculationType.SO,
            CalculationType.SO_PATH,
            CalculationType.SOXK,
        }