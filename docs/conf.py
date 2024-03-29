# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import torch_bsf

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "PyTorch-BSF"
copyright = "2021, FUJITSU LIMITED and RIKEN"
author = "RIKEN AIP-FUJITSU Collaboration Center (RAFCC)"

# The full version, including alpha/beta/rc tags
release = version = torch_bsf.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_multiversion",
]

# -- autodoc -----------------------------------------------------------------
autodoc_type_aliases = {
    "Index": "torch_bsf.bezier_simplex.Index",
}
autodoc_inherit_docstrings = False

# -- doctest -----------------------------------------------------------------
# https://www.sphinx-doc.org/ja/master/usage/extensions/doctest.html#confval-doctest_global_setup
doctest_global_setup = """
import torch
import torch_bsf
"""
doctest_test_doctest_blocks = "default"

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
    "pl": ("https://pytorch-lightning.readthedocs.io/en/latest/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for sphinx-multiversion -----------------------------------------

smv_tag_whitelist = r"^v\d+\.\d+.\d+$"  # v1.2.3
smv_branch_whitelist = r"^(?!gh-pages|release-please).+$"  # all branches except gh-pages and release-please


# https://www.sphinx-doc.org/ja/master/usage/extensions/linkcode.html
# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(torch_bsf.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "torch_bsf/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "master"  # if 'dev' in release else ('v' + release)
    return "https://github.com/rafcc/pytorch-bsf/blob/%s/%s" % (tag, filename)
