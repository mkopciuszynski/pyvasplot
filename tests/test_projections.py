import tempfile
import unittest
from pathlib import Path
from typing import cast

from pymatgen.io.vasp.outputs import Outcar, Procar
from pymatgen.util.typing import Spin

from pyvasplot import PyVASP
from pyvasplot.plotting._bands import shifted_energy
from pyvasplot.plotting._procar import orbital_indices_for, project_procar
from pyvasplot.plotting.projections import prepare_projection_data

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = DATA_DIR / "WSb_110_GGA_0012"

EXPECTED_ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]

# Values copied from selected records in the fixture's PROCAR file.
EXPECTED_BAND_ENERGIES = {
    (0, 0): -34.55425867,
    (0, 1): -34.47250960,
    (1, 1): -34.47232214,
    (-1, -2): 6.44046041,
    (-1, -1): 6.46746462,
}

EXPECTED_PROJECTIONS = {
    (0, 0, 0, 0): 0.000,
    (0, 0, 0, 1): 0.013,
    (0, 0, 10, 1): 0.145,
    (-1, -1, 10, 0): 0.172,
    (-1, -1, 11, 0): 0.009,
}


class TestProjections(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmp_dir.name) / "dft_local"

        self.dft = PyVASP(
            MODEL_DIR,
            sub_path="BS_001",
            calculation_type="BS",
            local_dir=self.cache_dir,
        )
        self.dft.load(reload=True)
        self.procar = cast(Procar, self.dft.procar)
        self.outcar = cast(Outcar, self.dft.outcar)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_procar_data(self):
        """Verify loaded metadata and selected values from the PROCAR fixture."""

        dft = self.dft

        self.assertTrue(dft.full_path.is_absolute())

        self.assertEqual(dft.nkpoints, 64)
        self.assertEqual(dft.nbands, 128)
        self.assertEqual(dft.nions, 19)

        efermi = cast(float, self.outcar.efermi)
        self.assertAlmostEqual(efermi, 4.5587, places=4)
        self.assertAlmostEqual(dft.eshift, 0.0)

        procar = self.procar

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

        for (kpoint, band), expected in EXPECTED_BAND_ENERGIES.items():
            with self.subTest(kpoint=kpoint, band=band):
                self.assertAlmostEqual(eigenvalues[kpoint, band], expected, places=5)

        procar_data = procar.data[Spin.up]
        for (kpoint, band, ion, orbital), expected in EXPECTED_PROJECTIONS.items():
            with self.subTest(kpoint=kpoint, band=band, ion=ion, orbital=orbital):
                self.assertAlmostEqual(
                    procar_data[kpoint, band, ion, orbital], expected, places=3
                )

    def test_projections(self):
        """Verify orbital selection, ion reduction, and energy preparation."""
        orbitals = cast(list[str], self.procar.orbitals)
        self.assertEqual(orbitals, EXPECTED_ORBITALS)
        self.assertEqual(orbital_indices_for(orbitals, ("px", "py")), (3, 1))

        kx, bands, procar_data = prepare_projection_data(self.dft)

        projected = project_procar(
            procar_data,
            orbitals,
            ions=0,
            selected_orbitals="s",
        )
        self.assertAlmostEqual(projected[0, 0], 0.000, places=3)
        self.assertAlmostEqual(projected[-1, -1], 0.163, places=3)

        projected = project_procar(
            procar_data,
            orbitals,
            ions=0,
            selected_orbitals=("px", "py"),
        )
        self.assertAlmostEqual(projected[0, 0], 0.013, places=3)
        self.assertAlmostEqual(projected[-1, -1], 0.006, places=3)

        projected = project_procar(
            procar_data,
            orbitals,
            ions=0,
            selected_orbitals=("s", "py"),
        )
        self.assertAlmostEqual(projected[0, 0], 0.013, places=3)
        self.assertAlmostEqual(projected[-1, -1], 0.169, places=3)

        projected = project_procar(
            procar_data,
            orbitals,
            ions=0,
            selected_orbitals=None,
        )
        self.assertAlmostEqual(projected[0, 0], 0.013, places=3)
        self.assertAlmostEqual(projected[-1, -1], 0.180, places=3)

        projected = project_procar(
            procar_data,
            orbitals,
            ions=None,
            selected_orbitals="py",
            ion_reduction="sum",
        )

        self.assertAlmostEqual(projected[0, 0], (0.972), places=2)

        projected = project_procar(
            procar_data,
            orbitals,
            ions=11,
            selected_orbitals="py",
        )

        self.assertAlmostEqual(projected[0, 2], 0.060, places=3)

        self.assertAlmostEqual(bands[0, 0], EXPECTED_BAND_ENERGIES[0, 0], places=5)
        self.assertAlmostEqual(bands[-1, -1], EXPECTED_BAND_ENERGIES[-1, -1], places=5)

        self.assertAlmostEqual(kx[0], 0.0)
        self.assertAlmostEqual(kx[-1], self.dft.knorm)

        shifted_bands = shifted_energy(self.dft, bands)

        self.assertAlmostEqual(
            shifted_bands[0, 0], EXPECTED_BAND_ENERGIES[0, 0] - 4.5587, places=4
        )


if __name__ == "__main__":
    unittest.main()
