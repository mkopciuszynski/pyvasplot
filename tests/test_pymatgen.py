import unittest
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.outputs import Outcar, Eigenval, Procar, Kpoints
from pymatgen.util.typing import Spin

DATA_DIR = Path(__file__).parent / "data"

class TestPymatgen(unittest.TestCase):

    def setUp(self):
        model_path = DATA_DIR / "WSb_110_GGA_0012/BS_001"
        self.outcar_path = model_path / "OUTCAR"
        self.procar_path = model_path / "PROCAR"
        self.contcar_path = model_path / "CONTCAR"
        self.kpoints_path = model_path / "KPOINTS"



    def test_load_outcar(self):
        outcar = Outcar(self.outcar_path)

        self.assertIsNotNone(outcar)
        self.assertTrue(hasattr(outcar, 'efermi'))
        self.assertTrue(hasattr(outcar, 'run_stats'))
        self.assertTrue(hasattr(outcar, 'magnetization'))
        self.assertAlmostEqual(float(outcar.efermi), 4.5587, places=4)


    def test_load_procar(self):
        procar = Procar(self.procar_path)

        self.assertIsNotNone(procar)
        self.assertTrue(hasattr(procar, 'nkpoints'))
        self.assertTrue(hasattr(procar, 'nbands'))
        self.assertTrue(hasattr(procar, 'nions'))

        self.assertEqual(procar.data[Spin.up].shape, (64, 128, 19, 9))
        self.assertEqual(procar.nbands, 128)
        self.assertEqual(procar.nions, 19)
        self.assertEqual(procar.orbitals, ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2'])
        self.assertEqual(procar.nkpoints, 64)

        eigenvalues = procar.eigenvalues[Spin.up]
        self.assertIsNotNone(eigenvalues)

        self.assertEqual(eigenvalues.shape, (64, 128))
        self.assertAlmostEqual(float(eigenvalues[0, 0]), -34.55425867, places=5)
        self.assertAlmostEqual(float(eigenvalues[0, 1]), -34.47250960, places=5)
        self.assertAlmostEqual(float(eigenvalues[1, 1]), -34.47232214, places=5)
        self.assertAlmostEqual(float(eigenvalues[-1, -1]), 6.46746462, places=5)
        self.assertAlmostEqual(float(eigenvalues[-1, -2]), 6.44046041, places=5)
 
        procar_data = procar.data[Spin.up]
        self.assertAlmostEqual(procar_data[0, 0, 0, 0 ], 0.000, places=3)
        self.assertAlmostEqual(procar_data[0, 0, 0, 1 ],  0.013, places=3)
        self.assertAlmostEqual(procar_data[0, 0, 10, 1 ],  0.145, places=3)
        self.assertAlmostEqual(procar_data[-1,-1, 10, 0 ], 0.172, places=3)
        self.assertAlmostEqual(procar_data[-1,-1, 11, 0 ], 0.009, places=3)


    
    def test_load_kpoints(self):
        kpoints = Kpoints(self.kpoints_path)

        self.assertEqual(kpoints.num_kpts, 0)
        self.assertEqual(kpoints.kpts, [(1, 1, 1)])
        


if __name__ == '__main__':
    unittest.main()