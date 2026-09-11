from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


class DataSource:
    def __init__(
        self,
        path: str | Path,
        subpath: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.subpath = Path(subpath) if subpath is not None else None
        self._temporary_directory: TemporaryDirectory[str] | None = None

        try:
            if self.path.is_dir():
                root = self.path

            elif self.path.is_file() and self.path.suffix.lower() == ".zip":
                self._temporary_directory = TemporaryDirectory()
                root = self._extract_zip()

            else:
                raise ValueError(
                    f"Data source must be a directory or ZIP archive: "
                    f"{self.path}"
                )

            if self.subpath is None:
                self.root = root
            else:
                self.root = root / self.subpath

                if not self.root.is_dir():
                    raise FileNotFoundError(
                        f"Calculation directory '{self.subpath}' "
                        f"not found in {self.path}"
                    )

        except Exception:
            self.close()
            raise

    def _extract_zip(self) -> Path:
        extraction_root = Path(self._temporary_directory.name)

        with ZipFile(self.path, "r") as archive:
            archive.extractall(extraction_root)

        entries = list(extraction_root.iterdir())

        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]

        return extraction_root

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> Path:
        return self.root

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()