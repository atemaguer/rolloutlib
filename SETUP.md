# Development setup

Rolloutlib uses [uv](https://docs.astral.sh/uv/) for environment management,
dependency locking, testing, and builds. Python 3.11 or newer is required.

## Clone and install

```bash
git clone https://github.com/atemaguer/rolloutlib.git
cd rolloutlib
uv sync
```

`uv sync` creates the project environment, installs the locked runtime
dependencies, and installs the development dependency group.

For the optional benchmark dataset loaders:

```bash
uv sync --extra benchmarks
```

## Verify the checkout

Run the same checks used by the release workflow:

```bash
uv run pytest -q
uv run ruff check rolloutlib tests
uv run pyright
uv build
```

The default test suite skips paid or external-service integrations.

## Tinker integration test

The Tinker smoke and AIME parity tests are opt-in. Install the Tinker SDK,
Tinker Cookbook, and benchmark datasets in the project environment, configure
your Tinker credentials, and run:

```bash
uv run pip install tinker tinker-cookbook datasets
export RUN_TINKER_INTEGRATION=1
uv run pytest tests/test_tinker_policy.py tests/test_tinker_aime.py -q
```

Useful overrides include `TINKER_MODEL_NAME`, `TINKER_MODEL_PATH`,
`TINKER_RENDERER`, `TINKER_MAX_TOKENS`, and `TINKER_AIME_LIMIT`. These tests
make model/API requests and may incur provider costs.

## Working on the package

Runtime code lives under `rolloutlib/`. Tests live under `tests/`. Keep
backend-specific clients outside the core package; policies should implement
the public `Policy` or `AsyncPolicy` contract and return actions or
`PolicyOutput` values.

After changing dependencies, update the lockfile with:

```bash
uv lock
uv sync
```

## Release

The package is published by `.github/workflows/release.yml` when a matching
version tag is pushed. Update `project.version` in `pyproject.toml`, verify the
repository is clean, then run:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

GitHub Actions uses PyPI Trusted Publishing. The repository's `pypi`
environment and trusted publisher must be configured in PyPI before a release.
