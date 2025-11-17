Testing & Coverage
==================

.. note::
   This page is auto-generated from the latest test run.
   Last updated: |today|

Test Results
------------

.. include:: test_results.txt
   :literal:

Coverage Report
---------------

.. include:: coverage_report.md
   :parser: myst_parser.sphinx_

Detailed Coverage
-----------------

`View interactive HTML coverage report <htmlcov/index.html>`_

Running Tests Yourself
----------------------

.. code-block:: bash

   # Run tests with coverage
   pytest --cov=core --cov-report=html

   # View coverage
   python -m http.server 8000 --directory htmlcov

Rebuilding Documentation
------------------------

To rebuild docs with fresh test results:

.. code-block:: bash

   # From project root
   cd docs
   make docs-with-tests