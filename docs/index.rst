Welcome to TreeGraphDuals Documentation
=========================================

|ci| |coverage| |ruff| |pypi| |pyversions| |license| |docs| |doi|

|pre-commit| |typed| |hypothesis|

.. |ci| image:: https://github.com/pikhan/treegraphduals/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/pikhan/treegraphduals/actions/workflows/ci.yml
   :alt: CI
.. |coverage| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pikhan/treegraphduals/badges/coverage.json
   :target: https://github.com/pikhan/treegraphduals/actions/workflows/ci.yml
   :alt: Coverage
.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff
.. |pypi| image:: https://img.shields.io/pypi/v/treegraphduals
   :target: https://pypi.org/project/treegraphduals/
   :alt: PyPI
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/treegraphduals
   :target: https://pypi.org/project/treegraphduals/
   :alt: Python versions
.. |license| image:: https://img.shields.io/pypi/l/treegraphduals
   :target: https://github.com/pikhan/treegraphduals/blob/main/LICENSE
   :alt: License
.. |docs| image:: https://readthedocs.org/projects/treegraphduals/badge/?version=latest
   :target: https://treegraphduals.readthedocs.io/en/latest/
   :alt: Docs
.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.22839827.svg
   :target: https://doi.org/10.5281/zenodo.22839827
   :alt: DOI
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |typed| image:: https://img.shields.io/badge/typed-Pyrefly-blue
   :target: https://pyrefly.org/
   :alt: Typed: Pyrefly
.. |hypothesis| image:: https://img.shields.io/badge/tested%20with-Hypothesis-blue
   :target: https://hypothesis.readthedocs.io/
   :alt: Tested with Hypothesis

An extension of my Master's thesis: [The Horizontal Tunnelability Graph is Dual to Level Set Trees](https://scholarworks.unr.edu/handle/11714/10548).
A package for working with trees, their duals, graphs, time series, and more.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install numpy scipy networkx

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from treegraphduals.core import Tree, BinaryTree
   import numpy as np

   # Create a tree
   tree = Tree(n_nodes=5, root=0)
   tree.add_edge(0, 1, length=1.0)
   tree.add_edge(0, 2, length=1.5)

   # Perform operations
   leaves = tree.get_leaves()
   dfs_order = tree.depth_first_search()

   # Convert to NetworkX
   nx_graph = tree.to_networkx()

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial
   examples
   definitions

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/timeseries
   api/visualizations

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   testing
   ai_provenance
   changelog
   adr/README

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
