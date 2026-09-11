import unittest
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.outputs import Outcar, Eigenval, Procar, Vasprun
from pymatgen.util.typing import Spin

data_dir = Path(__file__).parent / "data/Sb_111_GGA_0014/BS_MGKM"  # vasp 6.4

class TestPymatgen(unittest.TestCase):

    def setUp(self):
        self.outcar_path = data_dir / "OUTCAR"
        self.eigenval_path = data_dir / "EIGENVAL"
        self.procar_path = data_dir / "PROCAR"
        self.vasprun_path = data_dir / "Vasprun.xml"

    def test_load_outcar(self):
        outcar = Outcar(self.outcar_path)
        self.assertIsNotNone(outcar)
        self.assertTrue(hasattr(outcar, 'efermi'))
        self.assertTrue(hasattr(outcar, 'run_stats'))
        self.assertTrue(hasattr(outcar, 'magnetization'))
        self.assertAlmostEqual(float(outcar.efermi), -2.3627, places=4)

    def test_load_eigenvalues(self):
        eigenval = Eigenval(self.eigenval_path)

        eigenvalues = eigenval.eigenvalues[Spin.up]
        self.assertIsNotNone(eigenvalues)
        self.assertEqual(eigenvalues.shape, (96, 12, 2))
        self.assertAlmostEqual(float(eigenvalues[0, 0, 0]), -11.817013, places=5)
        self.assertAlmostEqual(float(eigenvalues[0, 10, 0]), 4.453938, places=5)
        self.assertAlmostEqual(float(eigenvalues[1, 10, 0]), 4.486040, places=5)

    def test_load_vasprun(self):
        vasprun = Vasprun(self.vasprun_path, parse_projected_eigen=True)

        self.assertIsNotNone(vasprun)
        self.assertTrue(hasattr(vasprun, 'efermi'))
        self.assertTrue(hasattr(vasprun, 'final_energy'))

        self.assertAlmostEqual(float(vasprun.efermi), -2.36265145, places=5)

        self.assertIsNotNone(vasprun.eigenvalues)
        eigenvalues = vasprun.eigenvalues[Spin.up]
        self.assertIsNotNone(eigenvalues)

        self.assertEqual(eigenvalues.shape, (96, 12, 2))
        self.assertAlmostEqual(float(eigenvalues[0, 0, 0]), -11.8170, places=3)
        self.assertAlmostEqual(float(eigenvalues[0, 10, 0]), 4.4539, places=3)
        self.assertAlmostEqual(float(eigenvalues[1, 10, 0]), 4.4860, places=3)

    def test_load_procar(self):
        procar = Procar(self.procar_path)

        self.assertIsNotNone(procar)
        self.assertTrue(hasattr(procar, 'nkpoints'))
        self.assertTrue(hasattr(procar, 'nbands'))
        self.assertTrue(hasattr(procar, 'nions'))

        self.assertEqual(procar.data[Spin.up].shape, (93, 12, 2, 9))
        self.assertEqual(procar.nbands, 12)
        self.assertEqual(procar.orbitals, ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2'])
        self.assertEqual(procar.nkpoints, 93)

        eigenvalues = procar.eigenvalues[Spin.up]
        self.assertIsNotNone(eigenvalues)

        self.assertEqual(eigenvalues.shape, (93, 12))
        self.assertAlmostEqual(float(eigenvalues[0, 0]), -11.81701326, places=5)
        self.assertAlmostEqual(float(eigenvalues[0, 1]), -10.23635854, places=5)
        self.assertAlmostEqual(float(eigenvalues[1, 1]), -10.22156422, places=5)

    def test_compare_eigenvalues(self):
        eigenval = Eigenval(self.eigenval_path)
        vasprun = Vasprun(self.vasprun_path)
        procar = Procar(self.procar_path)

        eigenval_eigenvalues = eigenval.eigenvalues[Spin.up][:, :, 0]
        vasprun_eigenvalues = vasprun.eigenvalues[Spin.up][:, :, 0]
        procar_eigenvalues = procar.eigenvalues[Spin.up]

        self.assertIsNotNone(eigenval_eigenvalues)
        self.assertIsNotNone(procar_eigenvalues)

        # Vasprun vs Eigenval (Both are shape (96, 12))
        self.assertEqual(eigenval_eigenvalues.shape, vasprun_eigenvalues.shape)
        np.testing.assert_allclose(eigenval_eigenvalues, vasprun_eigenvalues, rtol=1e-2)
        

        # # Procar deduplicates adjacent k-points (93 k-points vs 96 k-points).
        # # We compare unique k-points to account for this behavior.
        # unique_eigenval_eigenvalues = np.unique(eigenval_eigenvalues, axis=0)
        # unique_procar_eigenvalues = np.unique(procar_eigenvalues, axis=0)

        # np.testing.assert_allclose(
        #     unique_eigenval_eigenvalues, 
        #     unique_procar_eigenvalues, 
        #     rtol=1e-2,
        #     atol=1e-2
        # )


if __name__ == '__main__':
    unittest.main()