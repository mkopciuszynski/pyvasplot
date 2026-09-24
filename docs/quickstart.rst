Quickstart
==========

Loading a VASP calculation
--------------------------

pyVASPlot provides the :class:`pyvasplot.PyVASP` container for working with
VASP calculations.

For example:

.. code-block:: python

   from pyvasplot import PyVASP

   dft = PyVASP(
       "path/to/Sb_111_GGA_0014",
       calculation_type="BS_KPATH",
       sub_path="BS_MGKM",
   )

   dft.load()

Plotting a band structure
-------------------------

Band structures can be visualized using the plotting functions provided by
``pyvasplot.plotting``:

.. code-block:: python

   from pyvasplot.plotting import plot_bands, plot_kpath

   plot_kpath(dft)
   plot_bands(dft)

Working with projections
------------------------

Orbital- and ion-projected data can be selected and visualized as k-space
maps.

For example:

.. code-block:: python

   from pyvasplot.select.ions import by_name
   from pyvasplot.plotting import plot_procar_map

   orbitals = ("s", "py", "pz", "px")
   ions = by_name(dft, "Sb")

   sec_MGM = {
       "k_section": 0,
       "k_mirror": True,
       "k_flip": True,
   }

   plot_procar_map(
       dft,
       ions=ions,
       orbitals=orbitals,
       **sec_MGM,
   )

More examples
-------------

More detailed workflows, including interactive analysis and comparison with
ARPES data, will be added as Jupyter notebooks.