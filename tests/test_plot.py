import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyvasplot import PyVASP
from pyvasplot.plotting import plot_bands, plot_procar_bands
from pyvasplot.plotting.kpath import generate_kpath_bands
from pyvasplot.plotting.structure import plot_structure

DATA_DIR = Path(__file__).parent / "data"


class TestPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = DATA_DIR / "Sb_111_GGA_0014"

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmp_dir.name) / "dft_local"
        self.dft = PyVASP(
            self.test_data_dir,
            calculation_type="BS_KPATH",
            subpath="BS_MGKM",
            local_dir=self.cache_dir,
        )
        self.dft.load(reload=True)

    def tearDown(self):
        plt.close("all")

    def test_plot_structure(self):
        axes = plot_structure(self.dft)

        self.assertEqual(len(axes), 3)

        axes[0].figure.clf()

    def test_plot_bands_contains_all_bands(self):
        """The plot should contain one line for every band."""
        ax = plot_bands(
            self.dft,
            e_min=-50,
            e_max=50,
        )

        self.assertEqual(
            len(ax.lines),
            self.dft.nbands,
        )

    def test_plot_bands_uses_loaded_kpath_and_all_band_data(self):
        """Every plotted line should use the loaded k-path and energies."""
        ax = plot_bands(
            self.dft,
            e_min=-50,
            e_max=50,
        )

        expected_kx, expected_bands = generate_kpath_bands(self.dft)

        self.assertEqual(len(ax.lines), expected_bands.shape[1])
        for band_index, line in enumerate(ax.lines):
            np.testing.assert_allclose(np.asarray(line.get_xdata()), expected_kx)
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()),
                expected_bands[:, band_index] - self.dft.efermi - self.dft.eshift,
            )

    def test_plot_bands_energy_values(self):
        """The plotted energies should equal E - EF - eshift."""
        ax = plot_bands(
            self.dft,
            e_min=-50,
            e_max=50,
        )

        y_data = np.asarray(ax.lines[0].get_ydata())

        expected = self.dft.eigenvalues[:, 0] - self.dft.efermi - self.dft.eshift

        np.testing.assert_allclose(
            y_data,
            expected,
        )

    def test_plot_bands_specific_points(self):
        """Check several individual band energies."""
        ax = plot_bands(
            self.dft,
            e_min=-50,
            e_max=50,
        )

        y_data = np.asarray(ax.lines[0].get_ydata())

        for index in [0, 1, 10, -1]:
            expected = (
                self.dft.eigenvalues[index, 0] - self.dft.efermi - self.dft.eshift
            )

            self.assertAlmostEqual(
                y_data[index],
                expected,
            )

    def test_plot_bands_energy_limits(self):
        """The requested energy limits should be applied to the axes."""
        ax = plot_bands(
            self.dft,
            e_min=-50,
            e_max=50,
        )

        self.assertEqual(
            ax.get_ylim(),
            (-50, 50),
        )

    def test_plot_procar_bands_accepts_none_or_empty_selection(self):
        """Default or empty ion/orbital selections should mean all entries."""
        ax = plot_procar_bands(self.dft, ions=None, orbitals=None)
        self.assertIsNotNone(ax)

        ax = plot_procar_bands(self.dft, ions=[], orbitals=[])
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
