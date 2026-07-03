# Release history

## v26.02.0 - 2026-03-10
### Highlights
- Convert Brick Architect LBX label files into Avery 5167 PDF sheets through a
  single LBX-to-Avery pipeline.
- Add a one-step `run` command that renders tiles and imposes them in one
  invocation.
- Render labels as vector tiles and impose them onto pages in column-first
  order.
- Cluster multi-label LBX sheets with row-aware grouping, background grid
  clustering, separator-based splitting, and periodicity-based row and column
  detection with gap-based fallback.
- Match images to text with a score-based general matcher plus pairing
  fallback, and recursively segment oversized clusters.
- Insert category labels from LBX file names as black labels with white text,
  sorted by numeric category prefix.
- Add tile and imposition limits (`--max-labels`, `--max-pages`) and a
  `--stop-before-rendering` flag for faster checks, and record the limits in
  manifests.
- Normalize label text to a fixed bold style with auto-shrink, a 5 pt minimum
  font size, and clamp-count reporting.
- Shrink images to 95% of their box, center them on scaled visual bounds, and
  cap upscaling at 2x to reduce bleed.
- Report progress with a tile-rendering progress bar, throttled logging, and
  end-of-run timing stats.
- Split the pipeline into `brickarchitect_label_converter` modules and keep
  `lbx_to_avery_5167.py` as the wrapper entry point.
- Add packaging metadata with `pyproject.toml`, `VERSION`, and
  `pip_requirements.txt`.

### Notable fixes
- Render multiline text top-to-bottom instead of reversed.
- Correct text vertical alignment calculations for label rendering.
- Tighten cluster bounds to visual content for larger rendered labels.
- Fix polyline rendering for `draw:poly` objects in the PDF output.
- Fix label rendering orientation so text and images stay upright.
- Normalize label text to ASCII to avoid missing glyphs.
- Detect and warn on text overlap and missing-text or image-after-text
  ordering while rendering tiles.

### Compatibility notes
- Removed CLI subcommands in favor of a single LBX-to-Avery pipeline.
- Removed CLI overrides for vertical alignment to keep argparse minimal.
- Removed the Avery template file and baked geometry defaults into the script.
- Requires the PyMuPDF dependency; see [INSTALL.md](INSTALL.md).

### Validation
- Added deterministic geometry tests for grid placement and scale-to-fit math.
- Added LBX integrity tests for zip structure, asset references, namespaces,
  bounds, and row spanning.
- Added label bbox invariant and rendered page smoke tests.
- Added v40 baseline fixtures and tests for label counts and multi-image
  labels, with `refresh_v40_baseline.py` to regenerate them.
- Added `tests/conftest.py` so repo imports resolve in pytest.
