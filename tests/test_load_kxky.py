import unittest
from pathlib import Path
from pyvasplot import PyVASP

# Define the data directory
data_dir = Path(__file__).parent / "data"

class TestPyVASP(unittest.TestCase):

    def setUp(self):
        self.dft_model_dir = data_dir / "AuSi2_hex_GGA_1171"

    
    def test_load_kxky_data(self):
        
        dft_dir = self.dft_model_dir / "BS_BZ"
        dft = PyVASP(dft_dir=dft_dir, dft_type="kxky")
        dft.load(reload=True)

        self.assertIsInstance(dft.selected_ky, int)
        self.assertEqual(dft.selected_ky, 0)
        self.assertIsInstance(dft.nky, int)
        self.assertEqual(dft.nky, 17)

        self.assertEqual(dft.efermi, -1.2935)
        self.assertEqual(dft.nkpoints, 32)
        self.assertEqual(dft.nbands, 78)
        self.assertEqual(dft.nions, 26)

        self.assertEqual(dft.name, "1171_BS_BZ")

        self.assertEqual(dft.eigenvalues.shape, (32, 78))
        self.assertEqual(dft.procar_data.shape, (32, 78, 26, 9))

        dft.selected_ky = 5
        self.assertEqual(dft.efermi, -1.2137)
        self.assertEqual(dft.nkpoints, 32)
        self.assertEqual(dft.nbands, 78)
        self.assertEqual(dft.nions, 26)

        self.assertEqual(dft.procar_data.shape, (32, 78, 26, 9))

