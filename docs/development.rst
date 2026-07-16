Development
===========

Repository layout
-----------------

* ``rolloutlib/`` — runtime package code.
* ``tests/`` — unit tests and opt-in external-service integration tests.
* ``docs/`` — this Sphinx documentation source.

Quality checks
--------------

Run the complete local check set before opening a pull request:

.. code-block:: console

   $ uv run pytest -q
   $ uv run ruff check rolloutlib tests
   $ uv run pyright
   $ uv build
   $ uv run --group docs sphinx-build -W docs docs/_build/html

Backend-specific policy implementations should remain outside the core
package. They should implement ``Policy`` or ``AsyncPolicy`` and return an
action or ``PolicyOutput``.

Releases
--------

The release workflow in ``.github/workflows/release.yml`` builds and publishes
to PyPI when a version tag is pushed. Update ``project.version`` in
``pyproject.toml``, verify the checks above, and push a matching tag:

.. code-block:: console

   $ git tag vX.Y.Z
   $ git push origin vX.Y.Z

GitHub Actions uses PyPI Trusted Publishing. The repository's ``pypi``
environment and trusted publisher must be configured in PyPI before release.
