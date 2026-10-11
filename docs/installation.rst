Installation
============

Coming soon...

.. code-block:: bash

   pip install treegraphduals

Verifying a release
-------------------

For releases after v0.1.0, the sdist and wheel come with Sigstore-signed
build provenance, which records that the file was built by this repository's
release workflow from the tagged commit. To check a downloaded file, use the
`GitHub CLI <https://cli.github.com/>`_:

.. code-block:: bash

   gh attestation verify treegraphduals-<version>-py3-none-any.whl --repo pikhan/treegraphduals

The same provenance is attached to each GitHub release as
``treegraphduals-<version>.provenance.sigstore.json``. Pass it with
``gh attestation verify --bundle`` to verify without querying GitHub's
attestation API. PyPI also shows a publish attestation for each file, under
"Provenance" on the file's page.
