import unittest
from pathlib import Path
import matplotlib.pyplot as plt
from pyvasplot.plotting import plot_bands
from pyvasplot import PyVASP

# Define the data directory
data_dir = Path("D:/vasp")

class TestPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup any state specific to the execution of the given class (shared setup)."""
        cls.data_dir = data_dir

    def setUp(self):
        """Setup any state specific to the execution of the given method."""
        self.dft_dir_bs = self.data_dir / "WSb_110_GGA_0061/BS_001"
        self.lorentz_width = 0.1
        self.kx_norm = 1.0
        self.cmap = 'viridis'

    def sample_dft_bs(self):
        """Fixture-like method for loading BS data from a specific directory."""
        dft = PyVASP(dft_dir=self.dft_dir_bs, dft_type="BS")
        dft.load(reload=True)
        return dft
         

    def test_plot_bands(self):
        # Call the function

        dft = self.sample_dft_bs()
        ax = plot_bands(dft, ek_min=-50,ek_max=50)

        # Check if the plot is created
        self.assertIsInstance(ax, plt.Axes)

        # Extract the xy data from the plot
        xy_data = ax.lines[0].get_xydata()
        assert xy_data.shape == (dft.nkpoints, 2)
        self.assertEqual(xy_data[0, 1], dft.eigenvalues[0, 0] - dft.efermi)
        self.assertEqual(xy_data[10, 1], dft.eigenvalues[10, 0] - dft.efermi)



if __name__ == '__main__':
    unittest.main()