# Development

## Repository layout

- `rolloutlib/` — runtime package code.
- `tests/` — unit tests and opt-in external-service integration tests.
- `docs/` — the MkDocs documentation source.
- `mkdocs.yml` — the Material for MkDocs site configuration.

## Quality checks

Run the complete local check set before opening a pull request:

```console
uv run pytest -q
uv run ruff check rolloutlib tests
uv run pyright
uv build
uv run --group docs mkdocs build --strict
```

Backend-specific policy implementations should remain outside the core
package. They should implement `Policy` or `AsyncPolicy` and return an action
or `PolicyOutput`.

## Releases

The release workflow in `.github/workflows/release.yml` builds and publishes to
PyPI when a version tag is pushed. Update `project.version` in `pyproject.toml`,
verify the checks above, and push a matching tag:

```console
git tag vX.Y.Z
git push origin vX.Y.Z
```

GitHub Actions uses PyPI Trusted Publishing. The repository's `pypi`
environment and trusted publisher must be configured in PyPI before release.

## Documentation deployment

The `docs.yml` workflow builds the MkDocs site with `--strict` and deploys the
generated `site/` artifact to GitHub Pages whenever `main` changes. Enable
GitHub Pages in the repository settings with **Source: GitHub Actions**. The
published site is served at:

```text
https://atemaguer.github.io/rolloutlib/
```
