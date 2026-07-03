# Changelog

## 2026-07-03

### Additions and New Features
- Add `docs/FILE_FORMATS.md` documenting the LBX input container, the Avery 5167 PDF output geometry, tile PDFs, the manifest JSON schema, and the label-count and multi-image log files, linking to `docs/LABEL_BOUNDARIES.md` for segmentation detail.

### Fixes and Maintenance
- Add `docs/screenshots/avery_5167_label_sheet.png` (rendered from page 1 of the generated `output/avery_5167.pdf`) and embed it in the managed screenshot block of `README.md` so the GitHub landing page shows a real Avery 5167 label sheet of LEGO brick labels.
- Standardize `README.md`: add core doc links (`docs/CODE_ARCHITECTURE.md`, `docs/FILE_STRUCTURE.md`), curate the documentation list, and verify the quick-start and testing commands.
- Reserve a managed screenshots sentinel block in `README.md` for the screenshot-docs skill.
- Refresh `docs/INSTALL.md` from `pyproject.toml`, `pip_requirements.txt`, `pip_requirements-dev.txt`, and `Brewfile`: add Python 3.12 requirement, runtime and dev package lists, `brew bundle` step, and a `--help` verify command.
- Refresh `docs/USAGE.md` against the current `brickarchitect_label_converter.cli` argparse: correct the full flag set (including on/off toggles and `--stop-before-rendering`), add worked examples, and document inputs and outputs (PDF, `tiles/`, manifest JSON, label count log).
- Refresh `docs/RELATED_PROJECTS.md` from repo evidence and bounded web discovery: add confirmed entries for the Brick Architect label source, the Brother P-touch LBX input format, the Avery 5167 output target, and the five `pyproject.toml` dependencies, plus possible same-domain tools (`jdlien/lbx-utils`, `Alecto3-D/brother-p-touch-editor-format`, `brickventory/lego-labels`) and a commonly-confused list.
- Refresh `docs/CODE_ARCHITECTURE.md` from current source: document the five package modules, the `LabelObject` / `LabelCluster` / config dataclasses, the six-step `run_pipeline` data flow, outputs (PDF, `tiles/`, manifest JSON), testing, and extension points.
- Refresh `docs/FILE_STRUCTURE.md` from current layout: add an ASCII top-level tree, the entry-point wrapper and root scripts, the `LEGO_BRICK_LABELS-v40` dataset breakdown (304 `.lbx` files, 16 category folders), the `devel/` tooling, tests and fixtures, and git-ignored generated artifacts.
- Fix failing `pytest tests/`: add the `fitz` -> `pymupdf` import alias in `tests/test_import_requirements.py` so `import fitz` resolves to the declared `pymupdf` dependency, and correct broken same-folder markdown links in `docs/CHANGELOG.md` and `tests/TESTS_README.md`.
- Refresh `docs/ROADMAP.md` from `docs/REFACTOR_PLAN.md` (Phases 0-4 all done) and `config.py` evidence: list near-term work (shrink the five whitelist sets, retire `GROUP_SPLIT_TEXT_THRESHOLD`, document printer calibration) and longer-term work (overlapping boxes, non-polyline separators, release plan).
- Refresh `docs/TODO.md` as a real backlog derived from planning docs and config whitelists; note that no source `TODO`/`FIXME` markers exist.
- Refresh `docs/TROUBLESHOOTING.md` with issues grounded in `docs/CHANGELOG.md` history and code behavior: setup and empty-input checks, label-splitting logs, ASCII glyph normalization, image bleed shrink, overlapping-label limitation, and calibration alignment.
- Trim `AGENTS.md` to a bare-path pointer file: replace prose "See X in docs/..." lines with grouped pointers into the style, orientation, and test docs; preserve the repo-specific rules (changelog logging, implement-rather-than-wait, always-run-tests, agents may run tests/) and the Codex Python 3.12 interpreter and site-packages notes.
- Correct doc references left stale by the `pyproject.toml` removal: `docs/FILE_STRUCTURE.md` no longer lists it in the top-level tree; `docs/INSTALL.md` no longer cites it for the Python 3.12 requirement or an unconfirmed editable install; `docs/RELATED_PROJECTS.md`'s five dependency citations now point to `pip_requirements.txt`; `docs/ROADMAP.md` no longer claims it exists for packaging. Also linked the `LABEL_BOUNDARIES.md` bare-filename reference in `docs/TROUBLESHOOTING.md` and pointed `docs/RELATED_PROJECTS.md`'s Brick Architect entry at the specific legacy (2023, v40) download page.
- Rewrite `docs/USAGE.md` for a novice, terminal-nervous reader: plain-language intro, an explicit "before you start" check, a numbered first-run walkthrough with a single copy-paste command, a plain-language breakdown of each command part, and a "few options worth trying" section ahead of the full flag reference table.

### Decisions and Failures
- Removed `pyproject.toml`: this is a standalone app, not a published PyPI package, so packaging metadata and a build backend serve no purpose. `VERSION` remains the sole version source of truth. Known follow-up: `devel/submit_to_pypi.py` now hard-fails on any invocation (it requires `pyproject.toml` at the repo root) and its whole purpose (PyPI publishing) is moot under this decision; left in place pending a decision to delete it and its now-orphaned `packaging` dev dependency.
- `pyproject.toml`'s removal also dropped its `[tool.pytest.ini_options] filterwarnings` suppression of three PyMuPDF SWIG-binding `DeprecationWarning`s (`SwigPyPacked`, `SwigPyObject`, `swigvarlink`). Decided not to re-suppress them in `tests/conftest.py`: they originate from an external dependency, not repo code, and are informational only (`pytest tests/` still passes 911/911 with them visible).

## 2026-03-10
- Tighten cluster bounds to visual content for larger rendered labels.
- Fix multiline text rendering order so lines display top-to-bottom instead of reversed.

## 2026-02-05
- Add tile and imposition limits for faster test runs (`--max-labels`, `--max-pages`).
- Record imposition and tile limits in manifests.
- Fix text vertical alignment calculations for label rendering.
- Add tile options to override text alignment, weight, and size, plus cluster alignment controls.
- Add a one-step `run` command that renders tiles and imposes in one invocation.
- Add `tests/conftest.py` to ensure repo imports resolve in pytest.
- Add deterministic geometry tests for label grid placement and scale-to-fit math.
- Add LBX integrity tests for zip structure, asset references, namespaces, and bounds.
- Remove CLI overrides for vertical alignment to keep argparse minimal.
- Add row-aware label clustering to avoid merging labels across rows.
- Add LBX tests to ensure label clusters do not span multiple rows.
- Remove CLI subcommands and simplify to a single LBX-to-Avery pipeline.
- Switch XML parsing to defusedxml to satisfy Bandit security checks.
- Add verbose progress output for the single-command pipeline.
- Normalize label text rendering to a fixed 8 pt bold style for consistency.
- Adjust verbose output to use progress counts instead of listing every file.
- Disable auto-shrink for text so label fonts stay consistent.
- Use gap clustering to pick a more stable label split threshold.
- Throttle progress logging to every 10 LBX files plus final completion.
- Re-enable auto-shrink and reduce default text size to 7.5 pt for more consistent labels.
- Add text overlap detection while rendering tiles.
- Use separator lines and per-row gap thresholds to split labels more reliably.
- Add label boundary documentation in `docs/LABEL_BOUNDARIES.md`.
- Add background grid clustering for multi-label LBX sheets.
- Prefer separator-based splitting over background grid when lines exist.
- Improve background grid clustering by testing row multipliers and merging image-only clusters.
- Enforce a 5 pt minimum font size when auto-shrinking text and report clamp counts.
- Add a progress bar for tile rendering.
- Trim extreme gaps when computing clustering thresholds to better split labels.
- Defer overlap warnings until after the tile progress bar completes.
- Add label validation warnings for missing text and image-after-text ordering.
- Merge loose text into image-only groups when they share a row.
- Insert category labels based on LBX file names (black background with white text).
- Restrict multi-row background grid grouping to a small whitelist.
- Add a per-row pairing pass (single row height) to fix image/text mismatches.
- Add a whitelist for pairing image/text order in `MINIFIG-accessories-all`.
- Wrap category label text and replace hyphens with spaces.
- Sort LBX processing order by numeric category prefixes.
- Add `MINIFIG-weapon_3` to the pairing whitelist.
- Impose labels in column-first order instead of row-first.
- Log low label-count LBX files to `output/label_counts.log`.
- Use PATH `rm` in run.sh for the preferred coreutils variant.
- Update label-count test to account for category labels.
- Shrink all images to 95% of their box and center them to reduce bleed risk.
- Keep multi-row grouping for `CLIP-flexible` but restrict loose text merging to a whitelist.
- Center clusters using visual text bounds instead of full text boxes.
- Cap image upscaling at 2x while keeping the 95% shrink.
- Split large LBX groups into multiple label clusters when they contain many text entries.
- Expand the loose text merge whitelist to cover MINIFIG clothing/hair, NATURE flowers, and OTHER shooter labels.
- Merge text-only groups into image-only groups within the same row for whitelisted LBX files.
- Fix text-merge whitelist entries to use the actual LBX stems (MINIFIG category/hair, NATURE-flower, OTHER-shooter_1).
- Add `--stop-before-rendering` to stop after label collection for faster checks.
- Log multi-image labels to `output/multi_image_labels.log`.
- Print timing stats after the pipeline completes.
- Use scaled visual bounds (including image scale caps) to improve centering.
- Split multi-image clusters for `CLIP-clip_3` and `ANGLE-wedge_plate_63` using text pairing.
- Merge text-only clusters into image-only clusters for multi-image split files.
- Allow multi-image splitting even when only one text is present in a cluster.
- Document label boundary exceptions and whitelists.
- Add v40 baseline fixtures and tests for label counts and multi-image labels.
- Add `refresh_v40_baseline.py` to regenerate v40 baseline fixtures.
- Add periodicity-based row and column clustering with gap-based fallback.
- Skip periodicity when splitting large XML groups to avoid oversplitting.
- Add general image-text matching with score-based fallback to pairing.
- Start Phase 3 with recursive segmentation of oversized clusters.
- Mark Phase 3 (recursive segmentation) as done in [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
- Add label bbox invariant and rendered page smoke tests.
- Document PyMuPDF dependency in [INSTALL.md](INSTALL.md).
- Mark Phase 2 (general image-text matching) as done in [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
- Add [REFACTOR_PLAN.md](REFACTOR_PLAN.md) to track refactor phases.
- Split the pipeline into `brickarchitect_label_converter` modules and keep `lbx_to_avery_5167.py` as a wrapper entry point.
- Mark Phase 4 (modularization) as done in [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
- Add `pip_requirements.txt` with the core runtime dependencies.
- Use a `balc` import alias for `brickarchitect_label_converter` modules.
- Merge adjacent numeric text fragments for `INDEX-slope` to form `SLOPE 45-55-75`.
- Add `pyproject.toml` and `VERSION` for packaging metadata.
- Update [INSTALL.md](INSTALL.md) to install dependencies via `pip_requirements.txt`.
- Refresh README documentation links and update the testing command.
- Add documentation stubs for code architecture, file structure, roadmap, and troubleshooting.
- Filter PyMuPDF swig deprecation warnings in pytest config.
- Update render smoke test to use `get_flattened_data` when available.

## 2026-02-04
- Add plan document for the LBX to Avery 5167 conversion workflow.
- Add LBX to Avery 5167 imposition script with calibration and manifest output.
- Add a test covering label clustering and PDF output for a sample LBX file.
- Fix polyline rendering for draw:poly objects in the PDF output.
- Refresh README and add install and usage documentation.
- Fix label rendering orientation so text and images are upright.
- Add an optional flag to draw sticker outlines on output pages.
- Add run.sh helper script to generate Avery 5167 output.
- Remove Avery template file and bake geometry defaults into the script.
- Add tile rendering and PDF imposition pipeline using vector tiles.
- Normalize label text to ASCII to avoid missing glyphs.
