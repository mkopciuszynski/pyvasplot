# pyVASPlot

Python tools for loading VASP calculations and visualizing electronic
structure data.

## Features

- Load VASP calculations from directories or ZIP archives.
- Read band structures, spin-orbit calculations, HSE calculations, and
  KXKY data through `pymatgen`-based loaders.
- Plot band structures and reciprocal-space k-paths.
- Plot orbital and ion projections from PROCAR data.
- Cache parsed data locally for faster subsequent loads.

## Installation

pyVASPlot requires Python 3.12 or newer.

```bash
pip install pyvasplot
```

For development:

```bash
git clone <repository-url>
cd pyvasplot
pip install -e ".[dev]"
```

## Loading Data

Load a calculation from a directory by providing its calculation type:

```python
from pyvasplot import PyVASP

dft = PyVASP(
	"path/to/Sb_111_GGA_0014/BS_MGKM",
	calculation_type="BS_KPATH",
)
dft.load()

print(dft.nbands)
print(dft.nkpoints)
print(dft.efermi)
```

ZIP archives can be loaded by specifying the directory inside the archive:

```python
dft = PyVASP(
	"path/to/Sb_111_GGA_0014.zip",
	subpath="BS_MGKM",
	calculation_type="BS_KPATH",
)
dft.load()
```

Supported calculation types include:

```text
BS, BS_KPATH
HSE06, HSE06_KPATH
SO_STATIC, SO_STATIC_KPATH
KXKY, SO_STATIC_KXKY
```

## Plotting

```python
import matplotlib.pyplot as plt

from pyvasplot.plotting import plot_bands, plot_kpath

plot_bands(dft, e_min=-5, e_max=5)
plt.show()
```

For a reciprocal-space k-path:

```python
plot_kpath(dft)
plt.show()
```

## PROCAR Projections

Ion and orbital selections accept either a single value or a collection:

```python
from pyvasplot.plotting import plot_procar_bands

# Single ion and orbital
plot_procar_bands(dft, ions=0, orbitals="pz")

# Multiple ions and orbitals
plot_procar_bands(
	dft,
	ions=[0, 1],
	orbitals=["px", "py", "pz"],
)
plt.show()
```

Ion selectors are available for common structure-based selections:

```python
from pyvasplot.select import by_layer, by_name

top_layer = by_layer(dft, layer=1)
antimony = by_name(dft, "Sb")
```

## Paths and Cache

Parsed data is cached in `dft_local` by default. A different cache directory
can be supplied when creating `PyVASP`:

```python
dft = PyVASP(
	"path/to/calculation",
	calculation_type="BS_KPATH",
	local_dir=".cache/pyvasplot",
)
```

Useful path properties are available on every `PyVASP` instance:

```python
dft.full_path         # Absolute path to the input directory or ZIP archive
dft.full_data_path    # Absolute path combined with subpath
dft.full_local_dir    # Absolute cache directory
dft.full_cache_path   # Absolute cache file path
```

`str(dft)` provides a readable summary, while `repr(dft)` provides a compact
developer-oriented representation.

## Development

Run the test suite with:

```bash
python -m unittest discover
```

The test data is stored under `tests/data`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
