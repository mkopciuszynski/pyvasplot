import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyvasplot import PyVASP


DATA_DIR = Path(__file__).parent / "data"


class TestPyVASP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_zip = DATA_DIR / "Sb_111_GGA_0014.zip"
        cls.test_data_dir = DATA_DIR / "Sb_111_GGA_0014"

    def setUp(self):

        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp_dir.name)
        self.cache_dir = self.tmp_dir / "dft_local"

    def tearDown(self):
        self._tmp_dir.cleanup()

    def assert_bs_path_data(self, dft: PyVASP):
        """Check the expected data loaded from the Sb KPATH calculation."""

        efermi = dft.efermi
        if efermi is None:
            self.fail("Fermi energy was not loaded")

        self.assertAlmostEqual(efermi, -2.3627, places=3)

        self.assertEqual(dft.nkpoints, 93)
        self.assertEqual(dft.nbands, 12)
        self.assertEqual(dft.nions, 2)

        self.assertEqual(dft.eigenvalues.shape, (93, 12))
        self.assertEqual(dft.procar_data.shape, (93, 12, 2, 9))

        np.testing.assert_allclose(
            dft.eigenvalues[0, :2],
            [-11.81701326, -10.23635854],
        )

        self.assertAlmostEqual(
            dft.eigenvalues[1, 1],
            -10.22156422,
        )

        self.assertAlmostEqual(
            dft.eigenvalues[-1, -1],
            4.76539969,
        )

        self.assertAlmostEqual(
            dft.procar_data[0, 0, 0, 0],
            0.274,
        )

        self.assertAlmostEqual(
            dft.procar_data[1, 1, 0, 3],
            0.021,
        )

        self.assertAlmostEqual(
            dft.procar_data[-1, 11, 0, 3],
            0.014,
        )

    def test_load_kpath(self):
        """Load a KPATH calculation from its concrete directory."""

        dft = PyVASP(
            self.test_data_dir / "BS_MGKM",
            calculation_type="BS_KPATH",
            local_dir=self.cache_dir,
        )

        dft.load(reload=True)

        self.assert_bs_path_data(dft)

    def test_load_kpath_from_zip(self):
        """Load a KPATH calculation from a ZIP archive."""

        dft = PyVASP(
            self.test_data_zip,
            subpath="BS_MGKM",
            calculation_type="BS_KPATH",
            local_dir=self.cache_dir,
        )

        dft.load(reload=True)
        self.assertEqual(dft.cache_path.parent, self.cache_dir)
        self.assert_bs_path_data(dft)

    def test_load_so_static_from_directory(self):
        """Load an SO calculation from its concrete directory."""

        dft = PyVASP(
            self.test_data_dir / "SO_STATIC" / "BS_MGKM",
            calculation_type="SO_STATIC_KPATH",
            local_dir=self.cache_dir,
        )

        dft.load(reload=True)

        self.assertEqual(dft.cache_path.parent, self.cache_dir)
        self.assertIsNotNone(dft.procar)
        self.assertIsNotNone(dft.outcar)
        self.assertIsNotNone(dft.kpoints)
        self.assertIsNotNone(dft.structure)

    def test_load_so_static_from_zip(self):
        """Load an SO calculation nested inside a ZIP archive."""

        dft = PyVASP(
            self.test_data_zip,
            subpath="SO_STATIC/BS_MGKM",
            calculation_type="SO_STATIC_KPATH",
            local_dir=self.cache_dir,
        )

        dft.load(reload=True)

        self.assertIsNotNone(dft.procar)
        self.assertIsNotNone(dft.outcar)
        self.assertIsNotNone(dft.kpoints)
        self.assertIsNotNone(dft.structure)

    def test_load_from_cache(self):
        """Load the calculation from the local cache."""

        dft = PyVASP(
            self.test_data_zip,
            subpath="BS_MGKM",
            calculation_type="BS_KPATH",
            local_dir=self.cache_dir,
        )

        dft.load(reload=True)

        self.assertTrue(dft.cache_path.exists())

        cached_dft = PyVASP(
            self.test_data_zip,
            subpath="BS_MGKM",
            calculation_type="BS_KPATH",
            local_dir=self.cache_dir,
        )

        cached_dft.load()

        self.assert_bs_path_data(cached_dft)

    def test_paths_and_representations(self):
        """Expose absolute paths and safe object representations."""

        dft = PyVASP(
            self.test_data_zip,
            subpath="BS_MGKM",
            calculation_type="BS_KPATH",
            local_dir=self.cache_dir,
        )

        self.assertTrue(dft.full_path.is_absolute())
        self.assertEqual(
            dft.full_data_path,
            dft.full_path / "BS_MGKM",
        )
        self.assertTrue(dft.full_local_dir.is_absolute())
        self.assertTrue(dft.full_cache_path.is_absolute())
        self.assertNotIn("_None", dft.name)
        self.assertIn(str(dft.full_data_path), str(dft))
        self.assertIn(str(dft.full_cache_path), str(dft))
        self.assertIn("PyVASP(", repr(dft))

    def test_invalid_path(self):
        """An invalid input path should raise an appropriate error."""

        dft = PyVASP(
            self.tmp_dir / "does_not_exist",
            calculation_type="BS_KPATH",
            local_dir=self.cache_dir,
        )

        with self.assertRaises((FileNotFoundError, ValueError)):
            dft.load(reload=True)

    def test_invalid_zip_subpath(self):
        """An invalid calculation path inside a ZIP should fail."""

        dft = PyVASP(
            self.test_data_zip,
            subpath="does_not_exist",
            calculation_type="BS_KPATH",
            local_dir=self.cache_dir,
        )

        with self.assertRaises(FileNotFoundError):
            dft.load(reload=True)

    def test_invalid_calculation_type(self):
        """An unsupported calculation type should be rejected."""

        with self.assertRaises(ValueError):
            PyVASP(
                self.test_data_zip,
                subpath="BS_MGKM",
                calculation_type="invalid",
                local_dir=self.cache_dir,
            )


if __name__ == "__main__":
    unittest.main()