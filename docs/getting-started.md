# Getting started

## Requirements

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

## Install from PyPI

```console
pip install rolloutlib
```

## Install from source

```console
git clone https://github.com/atemaguer/rolloutlib.git
cd rolloutlib
uv sync
```

For the optional benchmark dataset loaders:

```console
uv sync --extra benchmarks
```

## Verify the checkout

```console
uv run pytest -q
uv run ruff check rolloutlib tests
uv run pyright
uv build
```

Build the documentation locally:

```console
uv sync --group docs
uv run --group docs mkdocs build --strict
```

The generated site is written to `site/`. Serve it locally with:

```console
uv run --group docs mkdocs serve
```

## Tinker integration

The Tinker smoke and AIME parity tests are opt-in. Install the Tinker SDK,
Tinker Cookbook, and benchmark datasets, configure your Tinker credentials,
and run:

```console
uv run pip install tinker tinker-cookbook datasets
export RUN_TINKER_INTEGRATION=1
uv run pytest tests/test_tinker_policy.py tests/test_tinker_aime.py -q
```

Useful overrides include `TINKER_MODEL_NAME`, `TINKER_MODEL_PATH`,
`TINKER_RENDERER`, `TINKER_MAX_TOKENS`, and `TINKER_AIME_LIMIT`. These tests
make model/API requests and may incur provider costs.
