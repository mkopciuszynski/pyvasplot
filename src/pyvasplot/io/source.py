from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


class DataSource:
    """Provide filesystem access to calculation data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._temporary_directory: TemporaryDirectory[str] | None = None
        self.root: Path | None = None

        if self.path.is_dir():
            self.root = self.path

        elif self.path.is_file() and self.path.suffix.lower() == ".zip":
            self._temporary_directory = TemporaryDirectory()
            self._extract_zip()

        else:
            raise ValueError(
                f"Data source must be a directory or ZIP archive: {self.path}"
            )

    def _extract_zip(self) -> None:
        with ZipFile(self.path, "r") as archive:
            archive.extractall(self._temporary_directory.name)

        extracted_root = Path(self._temporary_directory.name)

        entries = list(extracted_root.iterdir())

        # ZIP contains one top-level directory.
        if len(entries) == 1 and entries[0].is_dir():
            self.root = entries[0]
        else:
            # ZIP contains the calculation directories directly.
            self.root = extracted_root

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> Path:
        if self.root is None:
            raise RuntimeError("Data source has not been initialized.")
        return self.root

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()