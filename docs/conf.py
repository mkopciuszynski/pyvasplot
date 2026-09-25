from datetime import datetime

project = "pyVASPlot"
author = "Marek Kopciuszynski"
year = datetime.now().year
copyright = f"{year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "myst_nb",
]


templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

nb_execution_mode = "auto"

