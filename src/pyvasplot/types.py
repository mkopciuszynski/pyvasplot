from enum import StrEnum


class CalculationType(StrEnum):
    """Types of VASP calculations supported by pyvasplot."""

    BS = "BS"
    BS_PATH = "BS_PATH"

    HSE06 = "HSE06"
    HSE06_PATH = "HSE06_PATH"

    SO = "SO"
    SO_PATH = "SO_PATH"

    KXKY = "KXKY"
    SO_KXKY = "SO_KXKY"


    @property
    def is_path(self) -> bool:
        """Whether this calculation contains a k-path."""
        return self in {
            CalculationType.BS_PATH,
            CalculationType.HSE06_PATH,
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
            CalculationType.HSE06,
            CalculationType.HSE06_PATH,
        }

    @property
    def is_so(self) -> bool:
        """Whether this calculation includes spin-orbit coupling."""
        return self in {
            CalculationType.SO,
            CalculationType.SO_PATH,
            CalculationType.SO_KXKY,
        }