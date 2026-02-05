# Changelog

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
