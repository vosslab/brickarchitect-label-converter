# File structure

This document summarizes the current repository layout.

## Top-level layout

```text
brickarchitect-label-converter/
+- brickarchitect_label_converter/   Core Python package
+- LEGO_BRICK_LABELS-v40/            Input LBX dataset (v40)
+- tests/                            Pytest suite and fixtures
+- devel/                           Developer and release tooling
+- docs/                            Project documentation
+- output/                          Generated PDFs and tiles (git ignored)
+- lbx_to_avery_5167.py             Executable CLI wrapper
+- refresh_v40_baseline.py          Regenerate test count baselines
+- run.sh                           Convenience runner
+- pip_requirements.txt             Runtime dependencies
+- pip_requirements-dev.txt         Developer dependencies
+- Brewfile                         Homebrew dependencies
+- VERSION                          Repo version string
+- REPO_TYPE                        Project type marker (python)
+- README.md / AGENTS.md / CLAUDE.md / LICENSE
```

- [../brickarchitect_label_converter](../brickarchitect_label_converter):
  core package modules (see below).
- [../lbx_to_avery_5167.py](../lbx_to_avery_5167.py): executable entry point; a
  thin wrapper that re-exports package names and calls the package `main()`.
- [../refresh_v40_baseline.py](../refresh_v40_baseline.py): regenerates the v40
  label-count baselines used by the tests.
- [../run.sh](../run.sh): runs the pipeline over `LEGO_BRICK_LABELS-v40/Labels`.

## Package modules

- [../brickarchitect_label_converter/__init__.py](../brickarchitect_label_converter/__init__.py):
  package docstring only.
- [../brickarchitect_label_converter/cli.py](../brickarchitect_label_converter/cli.py):
  CLI parsing, config assembly, and pipeline orchestration.
- [../brickarchitect_label_converter/config.py](../brickarchitect_label_converter/config.py):
  constants, dataclasses, and font and unit helpers.
- [../brickarchitect_label_converter/lbx_lib.py](../brickarchitect_label_converter/lbx_lib.py):
  LBX XML parsing and text normalization.
- [../brickarchitect_label_converter/segment.py](../brickarchitect_label_converter/segment.py):
  label collection and boundary or cluster detection.
- [../brickarchitect_label_converter/render.py](../brickarchitect_label_converter/render.py):
  tile rendering and Avery sheet imposition.

## Input dataset

[../LEGO_BRICK_LABELS-v40](../LEGO_BRICK_LABELS-v40) holds the tracked `.lbx`
inputs (304 files):

- `Labels/`: 16 numbered category folders (`1.BASIC` through `16.DUPLO`).
- `Colors/`: color reference labels.
- `Groups/`: group and retired-group index labels.
- `New_in_v40/`: labels new in the v40 set.
- `ABOUT.txt`: dataset provenance.

## Tests

[../tests](../tests) holds the pytest suite, shared helpers
(`file_utils.py`, `conftest.py`), and regression fixtures under
[../tests/fixtures](../tests/fixtures). Tests cover geometry invariants, label
counts, LBX integrity, render smoke pages, and repo-wide lint and style gates.
See [PYTEST_STYLE.md](PYTEST_STYLE.md) and [E2E_TESTS.md](E2E_TESTS.md).

## Developer tooling

[../devel](../devel) holds version and changelog tooling
(`bump_version.py`, `changelog_lib.py`, `commit_changelog.py`,
`query_changelog.py`, `rotate_changelog.py`), build and cleanup scripts
(`clean_build.sh`, `dist_clean.sh`), PyPI submission (`submit_to_pypi.py`), and
`DEVEL_README.md`.

## Generated artifacts

- `output/` and `output*/` directories hold generated PDFs, tile PDFs under
  `tiles/`, and manifest JSON. Ignored by
  [../.gitignore](../.gitignore); not tracked.
- `report_*.txt` lint report files are also git ignored.

## Documentation map

Documentation lives in this `docs/` folder. Root-level docs are `README.md`,
`AGENTS.md`, `CLAUDE.md`, and `LICENSE`. Architecture and layout live in
[CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) and this file. Setup and usage live
in [INSTALL.md](INSTALL.md) and [USAGE.md](USAGE.md). Boundary detection is
documented in [LABEL_BOUNDARIES.md](LABEL_BOUNDARIES.md).

## Where to add new work

- Code: add or extend modules under
  [../brickarchitect_label_converter](../brickarchitect_label_converter).
- Tests: add `test_*.py` files under [../tests](../tests); fixtures under
  [../tests/fixtures](../tests/fixtures).
- Docs: add or update files in this `docs/` folder.
- Scripts: keep single-purpose scripts at the repo root or under
  [../devel](../devel) for release and maintenance tooling.
