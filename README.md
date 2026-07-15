# rolloutlib

Environments and post-training for language agents.

## Development

Install the project:

```bash
uv sync
```

## Release

Releases are published to PyPI by GitHub Actions when a tag matching the package
version is pushed:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The repository uses PyPI Trusted Publishing, so the `pypi` GitHub environment must be
registered as a trusted publisher for this repository's `release.yml` workflow before
the first release.
