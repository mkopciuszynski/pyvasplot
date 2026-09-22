import unittest
from pathlib import Path
from pymatgen.io.vasp.outputs import Kpoints, Outcar, Procar
from pymatgen.util.typing import Spin

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = DATA_DIR / "WSb_110_GGA_0012" / "BS_001"

EXPECTED_ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]

# Values copied from the first and last records in the fixture's PROCAR file.
EXPECTED_BAND_ENERGIES = {
    (0, 0): -34.55425867,
    (0, 1): -34.47250960,
    (1, 1): -34.47232214,
    (-1, -2): 6.44046041,
    (-1, -1): 6.46746462,
}

# Values copied from selected ion/orbital records in the fixture's PROCAR file.
EXPECTED_PROJECTIONS = {
    (0, 0, 0, 0): 0.000,
    (0, 0, 0, 1): 0.013,
    (0, 0, 10, 1): 0.145,
    (-1, -1, 10, 0): 0.172,
    (-1, -1, 11, 0): 0.009,
}


class TestPymatgen(unittest.TestCase):
    def setUp(self):
        self.outcar_path = MODEL_DIR / "OUTCAR"
        self.procar_path = MODEL_DIR / "PROCAR"
        self.kpoints_path = MODEL_DIR / "KPOINTS"

    def test_load_outcar(self):
        outcar = Outcar(self.outcar_path)

        self.assertIsNotNone(outcar)
        self.assertTrue(hasattr(outcar, "efermi"))
        self.assertTrue(hasattr(outcar, "run_stats"))
        self.assertTrue(hasattr(outcar, "magnetization"))
        self.assertAlmostEqual(float(outcar.efermi), 4.5587, places=4)

    def test_load_procar(self):
        """Load PROCAR metadata and verify selected source-file values."""
        procar = Procar(self.procar_path)

        self.assertIsNotNone(procar)
        self.assertTrue(hasattr(procar, "nkpoints"))
        self.assertTrue(hasattr(procar, "nbands"))
        self.assertTrue(hasattr(procar, "nions"))

        self.assertEqual(procar.data[Spin.up].shape, (64, 128, 19, 9))
        self.assertEqual(procar.nbands, 128)
        self.assertEqual(procar.nions, 19)
        self.assertEqual(procar.orbitals, EXPECTED_ORBITALS)
        self.assertEqual(procar.nkpoints, 64)

        eigenvalues = procar.eigenvalues[Spin.up]
        self.assertIsNotNone(eigenvalues)
        self.assertEqual(eigenvalues.shape, (64, 128))

        for (kpoint, band), expected in EXPECTED_BAND_ENERGIES.items():
            with self.subTest(kpoint=kpoint, band=band):
                self.assertAlmostEqual(eigenvalues[kpoint, band], expected, places=5)

        projections = procar.data[Spin.up]
        for (kpoint, band, ion, orbital), expected in EXPECTED_PROJECTIONS.items():
            with self.subTest(kpoint=kpoint, band=band, ion=ion, orbital=orbital):
                self.assertAlmostEqual(
                    projections[kpoint, band, ion, orbital], expected, places=3
                )

    def test_load_kpoints(self):
        """Load the fixture's line-mode KPOINTS definition."""
        kpoints = Kpoints(self.kpoints_path)

        self.assertEqual(kpoints.num_kpts, 0)
        self.assertEqual(kpoints.kpts, [(1, 1, 1)])


if __name__ == "__main__":
    unittest.main()
