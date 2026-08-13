import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "BayesianLearning"
author = "Matteo Piscitello, Leonardo Premutico"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

html_theme = "alabaster"
