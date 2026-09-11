import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyvasplot import PyVASP
from pyvasplot.plotting import plot_bands


DATA_DIR = Path(__file__).parent / "data"


class TestPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = DATA_DIR / "Sb_111_GGA_0014"

    def setUp(self):
        self.dft = PyVASP(
            self.test_data_dir,
            calculation_type="BS_KPATH",
            subpath="BS_MGKM",
        )
        self.dft.load(reload=True)

    def tearDown(self):
        plt.close("all")


    def test_plot_bands_contains_all_bands(self):
        """The plot should contain one line for every band."""
        ax = plot_bands(
            self.dft,
            ek_min=-50,
            ek_max=50,
        )

        self.assertEqual(
            len(ax.lines),
            self.dft.nbands,
        )

    def test_plot_bands_x_coordinates(self):
        """The x coordinates should contain all k-points."""
        ax = plot_bands(
            self.dft,
            ek_min=-50,
            ek_max=50,
        )

        x_data = ax.lines[0].get_xdata()

        self.assertEqual(
            len(x_data),
            self.dft.nkpoints,
        )

        np.testing.assert_allclose(
            x_data,
            np.linspace(
                0,
                self.dft.knorm,
                self.dft.nkpoints,
            ),
        )

    def test_plot_bands_energy_values(self):
        """The plotted energies should equal E - EF - eshift."""
        ax = plot_bands(
            self.dft,
            ek_min=-50,
            ek_max=50,
        )

        y_data = ax.lines[0].get_ydata()

        expected = (
            self.dft.eigenvalues[:, 0]
            - self.dft.efermi
            - self.dft.eshift
        )

        np.testing.assert_allclose(
            y_data,
            expected,
        )

    def test_plot_bands_specific_points(self):
        """Check several individual band energies."""
        ax = plot_bands(
            self.dft,
            ek_min=-50,
            ek_max=50,
        )

        y_data = ax.lines[0].get_ydata()

        for index in [0, 1, 10, -1]:
            expected = (
                self.dft.eigenvalues[index, 0]
                - self.dft.efermi
                - self.dft.eshift
            )

            self.assertAlmostEqual(
                y_data[index],
                expected,
            )

    def test_plot_bands_energy_limits(self):
        """The requested energy limits should be applied to the axes."""
        ax = plot_bands(
            self.dft,
            ek_min=-50,
            ek_max=50,
        )

        self.assertEqual(
            ax.get_ylim(),
            (-50, 50),
        )


if __name__ == "__main__":
    unittest.main()