# Install

"Installed" means the `lbx_to_avery_5167.py` command-line script runs and its
runtime dependencies are importable. This repo is a flat-layout Python project
built around one entry-point script plus the `brickarchitect_label_converter`
package.

## Requirements
- Python 3.12 (the `Brewfile` pins `python@3.12`; no `pyproject.toml` exists since this is an app, not a published package).
- Runtime packages: `defusedxml`, `pillow`, `pymupdf`, `pypdf`, `reportlab`.
- Developer tooling (tests and lint): `bandit`, `packaging`, `pyflakes`, `pytest`, `rich`.

## Install steps
- Obtain the source (clone the repository).
- Install Python 3.12 if needed. On macOS with Homebrew: `brew bundle` (reads the `Brewfile`).
- Install runtime dependencies: `python3 -m pip install -r pip_requirements.txt`.
- For running tests, also install developer tools: `python3 -m pip install -r pip_requirements-dev.txt`.

## Verify install
- Confirm the CLI loads and prints usage:
```bash
python3 lbx_to_avery_5167.py --help
```

## Known gaps
- [ ] Confirm supported platforms beyond macOS Homebrew (Linux, Windows) with a fresh-environment install.

Editable install (`pip install -e .`) does not apply: there is no `pyproject.toml`
or `setup.py` build backend. Run the entry-point script directly from a source
checkout instead.
