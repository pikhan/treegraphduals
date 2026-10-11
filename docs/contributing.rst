Contributing
============

Coming soon...

SPEC 0 floor bumps
------------------

The project follows `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_
for its supported Python and dependency versions. :doc:`adr/0001-spec-0-support-policy`
sets the policy, and :doc:`adr/0002-spec-0-windows-per-minor-version` says how
the dependency windows are counted.

Where the versions live
~~~~~~~~~~~~~~~~~~~~~~~

``pyproject.toml`` is the single source of truth:

- ``requires-python`` (``>=X.Y``) is the **floor**;
- the ``Programming Language :: Python :: X.Y`` classifiers are the **full list**
  of supported versions.

``scripts/sync_python_versions.py`` copies the floor into ``.readthedocs.yaml``,
``README.md`` and ``CLAUDE.md``, drops classifiers below the floor, and sets
``.python-version`` (the local development interpreter) to the newest
supported version (ADR 0001, decision 7). The CI workflows don't hold a copy: they ask the script at run
time, so the test matrix is the classifiers plus any non-blocking prerelease
listed in ``.github/workflows/ci.yml``. A pre-commit hook and CI both run the
script with ``--check``, so a hand edit can't leave the copies inconsistent.
After editing ``requires-python`` or the classifiers by hand, run::

    uv run python scripts/sync_python_versions.py

Adding a new Python version (ADR 0001, decision 5) means adding its classifier
and removing it from the non-blocking list in ``ci.yml``.

The quarterly pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~

On day 3 of January, April, July and October, ``.github/workflows/spec0.yml``
runs `spec0-action <https://github.com/scientific-python/spec0-action>`_. If any
floor is out of the SPEC 0 window, it raises it, syncs the copies, re-locks
``uv.lock``, and opens a pull request as the project's GitHub App. GitHub asks
the code owners (``.github/CODEOWNERS``) to review it, which sends them a
notification. To handle it:

1. Wait for CI. The ``lowest-direct`` job installs the new floors on the floor
   Python and runs the test suite, so it is the one that shows whether the new
   floors work.
2. If CI is red, push fixes to the PR's branch. A dependency that has to stay
   higher than SPEC 0 requires gets raised by hand; a dependency that must not
   be bumped can be listed in the action's ``excluded_packages``.
3. If a Python version was dropped, remove workarounds for it.
4. Merge, and add a ``CHANGELOG.md`` entry for the dropped versions.

To run the workflow outside the schedule, open the Actions tab, pick
**SPEC 0 floor bump**, and click **Run workflow**.

When something goes wrong
~~~~~~~~~~~~~~~~~~~~~~~~~

- **The workflow fails.** It opens an issue that mentions the code owners, with
  a link to the failed run. A common cause is the GitHub App: its secrets
  (``PR_APP_ID``, ``PR_APP_PRIVATE_KEY``) must exist, and the App must be
  installed on the repository with read and write access to contents and pull
  requests.
- **The workflow stops running.** GitHub disables scheduled workflows in public
  repositories after 60 days without repository activity, and emails before it
  does. The quarterly run is more than 60 days after the previous one, so this
  will happen in a quiet period. Re-enable it in the Actions tab (one click),
  then run it by hand if a quarter was missed.
- **The maintainers change.** Update ``.github/CODEOWNERS``. Nothing else in
  this automation names a person.

Security
--------

Please don't report security vulnerabilities in public issues. Use GitHub's
`private vulnerability reporting form
<https://github.com/pikhan/treegraphduals/security/advisories/new>`_ (the
**Report a vulnerability** button under the repository's Security tab), which
only the maintainers can see.

When editing ``.github/workflows/``:

- Pin every action to a full commit SHA, with its version in a trailing
  comment (``uses: actions/checkout@<sha> # v7.0.1``). Dependabot keeps the
  SHAs and the comments current.
- Start from a read-only default (``permissions: contents: read``) and grant
  write permissions as narrowly as practical, preferably to the jobs that need
  them.
- Check out with ``persist-credentials: false`` unless a later step pushes
  with those credentials.

The `zizmor <https://docs.zizmor.sh/>`_ pre-commit hook checks these rules on
every commit that touches a workflow or ``.github/dependabot.yml``.
