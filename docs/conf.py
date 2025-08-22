# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "QuaCCAToo"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx_copybutton",
]
source_suffix = ".rst"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "class_diagram.svg",
    "QuaCCAToo_logo.svg",
    # "example_notebooks/*",
]
numpydoc_class_members_toctree = False

autoclass_content = "both"
autodoc_inherit_docstrings = False

# -- Options for HTML output -------------------------------------------------

html_baseurl = "https://qiss-hzb.github.io/quaccatoo"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_show_copyright = False
