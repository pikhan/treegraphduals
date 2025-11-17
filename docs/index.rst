Welcome to TreeGraphDuals Documentation
=========================================

An extension of my Master's thesis: [The Horizontal Tunnelability Graph is Dual to Level Set Trees](https://scholarworks.unr.edu//handle/11714/10548).
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

   from core import Tree, BinaryTree
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

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   testing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
