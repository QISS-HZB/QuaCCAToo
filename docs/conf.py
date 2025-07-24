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
copyright = "2025, Lucas Tsunaki, Anmol Singh, Sergei Trofimov"
author = "Lucas Tsunaki, Anmol Singh, Sergei Trofimov"


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
# html_logo ='./QuaCCAToo_v3.png'
# html_theme_options = {'full_logo': False}
html_static_path = ["_static"]
