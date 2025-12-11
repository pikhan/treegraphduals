# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import json
from datetime import datetime

# Read coverage data if available
coverage_data = {}
coverage_file = os.path.join(os.path.dirname(__file__), '..', 'coverage.json')
if os.path.exists(coverage_file):
    with open(coverage_file, 'r') as f:
        coverage_data = json.load(f)

# Read test results
test_results = "No test results available"
test_file = os.path.join(os.path.dirname(__file__), 'test_results.txt')
if os.path.exists(test_file):
    with open(test_file, 'r') as f:
        test_results = f.read()

# Make data available to templates
html_context = {
    'coverage_percent': coverage_data.get('totals', {}).get('percent_covered', 0),
    'test_results': test_results,
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

# -- Project information -----------------------------------------------------
project = 'treegraphduals'
copyright = '2025, Ibraheem Khan'
author = 'Ibraheem Khan'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',       # Support for NumPy/Google style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.doctest',        # Test code snippets in docstrings
    'sphinx.ext.intersphinx',    # Link to other project docs
    'sphinx.ext.mathjax',        # Math support
    'sphinx.ext.coverage',       # Coverage of docstrings
    'myst_parser',
    'matplotlib.sphinxext.plot_directive',
]

# Plot directive configuration
plot_include_source = True  # Show the code
plot_html_show_source_link = False  # Hide "Source code" link
plot_formats = [('png', 100)]  # Output format and DPI
plot_html_show_formats = False  # Don't show format links


# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Doctest configuration
doctest_global_setup = '''
import sys
sys.path.insert(0, '/home/pikhan/PycharmProjects/treegraphduals')
from core import Tree, BinaryTree
import numpy as np
'''

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
}
