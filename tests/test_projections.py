import unittest
import tempfile

import numpy as np
from pathlib import Path

from pyvasplot import PyVASP
from pymatgen.util.typing import Spin

from pyvasplot.plotting._procar import project_procar, orbital_indices_for
from pyvasplot.plotting.projections import prepare_projection_data
from pyvasplot.plotting._data import shifted_energy


DATA_DIR = Path(__file__).parent / "data"


class TestProjections(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp_dir.name)
        self.cache_dir = self.tmp_dir / "dft_local"

        test_data_dir = DATA_DIR / "WSb_110_GGA_0012"
        self.dft = PyVASP(
            test_data_dir,
            subpath="BS_001",
            calculation_type="BS",
            local_dir=self.cache_dir,
        )
        self.dft.load(reload=True)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_procar_data(self):
        """Expose absolute paths and safe object representations."""

        dft = self.dft

        self.assertTrue(dft.full_path.is_absolute())

        self.assertEqual(dft.nkpoints, 64)
        self.assertEqual(dft.nbands, 128)
        self.assertEqual(dft.nions, 19)

        self.assertAlmostEqual(dft.efermi, 4.5587, places=4)
        self.assertAlmostEqual(dft.eshift, 0.0)

        procar = dft.data.procar

        self.assertEqual(procar.data[Spin.up].shape, (64, 128, 19, 9))
        self.assertEqual(procar.nbands, 128)
        self.assertEqual(procar.nions, 19)
        self.assertEqual(
            procar.orbitals, ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]
        )
        self.assertEqual(procar.nkpoints, 64)

        eigenvalues = dft.eigenvalues
        self.assertIsNotNone(eigenvalues)

        self.assertEqual(eigenvalues.shape, (64, 128))
        self.assertAlmostEqual(float(eigenvalues[0, 0]), -34.55425867, places=5)
        self.assertAlmostEqual(float(eigenvalues[0, 1]), -34.47250960, places=5)
        self.assertAlmostEqual(float(eigenvalues[1, 1]), -34.47232214, places=5)
        self.assertAlmostEqual(float(eigenvalues[-1, -1]), 6.46746462, places=5)
        self.assertAlmostEqual(float(eigenvalues[-1, -2]), 6.44046041, places=5)

        procar_data = procar.data[Spin.up]
        self.assertAlmostEqual(procar_data[0, 0, 0, 0], 0.000, places=3)
        self.assertAlmostEqual(procar_data[0, 0, 0, 1], 0.013, places=3)
        self.assertAlmostEqual(procar_data[0, 0, 10, 1], 0.145, places=3)
        self.assertAlmostEqual(procar_data[-1, -1, 10, 0], 0.172, places=3)
        self.assertAlmostEqual(procar_data[-1, -1, 11, 0], 0.009, places=3)

    def test_projections(self):
        orbitals = self.dft.procar.orbitals
        selection = ("px", "py")

        orbital_indices = orbital_indices_for(orbitals, selection)

        self.assertEqual(orbital_indices, (3, 1))

        kx, bands, procar_data = prepare_projection_data(self.dft)

        projected = project_procar(self.dft, procar_data, orbitals="s", ions=0)
        self.assertAlmostEqual(projected[0, 0], 0.000, places=3)
        self.assertAlmostEqual(projected[-1, -1], 0.163, places=3)

        projected = project_procar(self.dft, procar_data, orbitals=("px", "py"), ions=0)
        self.assertAlmostEqual(projected[0, 0], (0.013 + 0.0), places=3)
        self.assertAlmostEqual(projected[-1, -1], (0.006 + 0.0), places=3)

        projected = project_procar(self.dft, procar_data, orbitals=("s", "py"), ions=0)
        self.assertAlmostEqual(projected[0, 0], (0.000 + 0.013), places=3)
        self.assertAlmostEqual(projected[-1, -1], (0.163 + 0.006), places=3)

        projected = project_procar(self.dft, procar_data, orbitals=None, ions=0)

        self.assertAlmostEqual(projected[0, 0], (0.013), places=3)
        self.assertAlmostEqual(projected[-1, -1], (0.180), places=3)

        projected = project_procar(
            self.dft, procar_data, orbitals="py", ions=None, ion_reduction="sum"
        )

        self.assertAlmostEqual(projected[0, 0], (0.972), places=2)

        projected = project_procar(self.dft, procar_data, orbitals="py", ions=11)

        self.assertAlmostEqual(projected[0, 2], (0.060), places=3)

        self.assertAlmostEqual(bands[0, 0], -34.55425867, places=5)
        self.assertAlmostEqual(bands[-1, -1], 6.46746462, places=5)

        self.assertAlmostEqual(kx[0], 0.0)
        self.assertAlmostEqual(kx[-1], self.dft.knorm)

        shifted_bands = shifted_energy(self.dft, bands)

        self.assertAlmostEqual(shifted_bands[0, 0], -34.5543 - 4.5587, places=4)


if __name__ == "__main__":
    unittest.main()
