from datetime import datetime

project = "pyVASPlot"
author = "Marek Kopciuszynski"
year = datetime.now().year
copyright = f"{year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_nb",
]


templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

nb_execution_mode = "auto"

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": True,
    "special-members": "__init__",
}
