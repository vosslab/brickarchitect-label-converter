# Code architecture

This document describes the modules and data flow that convert Brother P-touch
LBX label sets into Avery 5167 label sheets as PDFs.

## Overview

The tool reads `.lbx` files (zipped Brother P-touch label XML), groups the
drawing objects on each label into clusters, renders each cluster to a single
label tile PDF, then imposes the tiles onto Avery 5167 sheets (4 columns by 20
rows, 80 labels per page). A JSON manifest records inputs, hashes, and results.

All logic lives in the [../brickarchitect_label_converter](../brickarchitect_label_converter)
package. The executable
[../lbx_to_avery_5167.py](../lbx_to_avery_5167.py) is a thin wrapper: it
re-exports package names for backward compatibility and calls the package
`main()`.

## Major components

- [../brickarchitect_label_converter/cli.py](../brickarchitect_label_converter/cli.py):
  Argument parsing (`parse_args`), config assembly (`build_config`,
  `build_tile_config`), and pipeline orchestration (`run_pipeline`, `main`).
- [../brickarchitect_label_converter/lbx_lib.py](../brickarchitect_label_converter/lbx_lib.py):
  LBX XML parsing and text normalization. Parses text, image, and poly elements
  into `LabelObject` instances. Uses `defusedxml` for safe XML parsing.
- [../brickarchitect_label_converter/segment.py](../brickarchitect_label_converter/segment.py):
  Label collection and boundary detection. Gathers `.lbx` paths, clusters
  drawing objects into per-label `LabelCluster` groups (grid, gap, periodicity,
  and separator heuristics), and merges or splits clusters using named
  whitelists.
- [../brickarchitect_label_converter/render.py](../brickarchitect_label_converter/render.py):
  Tile rendering and sheet imposition with ReportLab and pypdf. Draws text,
  image, rect, and poly objects, renders each cluster to a tile PDF, imposes
  tiles onto Avery sheets, and writes the manifest.
- [../brickarchitect_label_converter/config.py](../brickarchitect_label_converter/config.py):
  Shared constants (page geometry, fonts, thresholds, heuristic whitelists), the
  `ImpositionConfig`, `TileConfig`, and `ImpositionResult` dataclasses, plus
  `map_font_name` and `inches_to_points` helpers.

## Data model

- `LabelObject` in
  [../brickarchitect_label_converter/lbx_lib.py](../brickarchitect_label_converter/lbx_lib.py):
  one drawing object (`kind` is text, image, or poly) with position, size, and
  styling fields.
- `LabelCluster` in
  [../brickarchitect_label_converter/segment.py](../brickarchitect_label_converter/segment.py):
  a group of `LabelObject` instances forming one printable label, with source
  path, index, and bounding box.
- `ImpositionConfig`, `TileConfig`, `ImpositionResult` in
  [../brickarchitect_label_converter/config.py](../brickarchitect_label_converter/config.py):
  page layout, tile layout, and imposition summary.

## Data flow

The primary path, driven by `run_pipeline` in
[../brickarchitect_label_converter/cli.py](../brickarchitect_label_converter/cli.py):

1. `segment.gather_lbx_paths` expands input files and directories into a sorted
   list of `.lbx` paths.
2. `segment.collect_labels` parses each file, builds `LabelCluster` groups, and
   returns clusters, per-file counts, SHA-256 hashes, and gap thresholds.
3. `segment.write_label_count_log` writes a per-file label count log next to the
   output. With `--stop-before-rendering`, the pipeline stops here.
4. `render.build_image_cache` loads embedded images; `render.render_tiles`
   writes one tile PDF per cluster under the `tiles/` directory.
5. `render.impose_tiles` places tiles onto Avery 5167 pages and returns an
   `ImpositionResult`.
6. `render.write_manifest` writes the JSON manifest.

## Outputs

- Avery sheet PDF at the `--output` path.
- Tile PDFs under a `tiles/` directory beside the output PDF.
- Manifest JSON at the `--manifest` path, or `<output>.json` by default.

See [LABEL_BOUNDARIES.md](LABEL_BOUNDARIES.md) for the boundary-detection
heuristics and [USAGE.md](USAGE.md) for CLI flags.

## Testing and verification

Run the pytest suite with `python3 -m pytest tests/`. The suite covers geometry
invariants, label counts against fixtures in
[../tests/fixtures](../tests/fixtures), LBX integrity, render smoke pages, and
repo-wide lint and style gates. See [PYTEST_STYLE.md](PYTEST_STYLE.md).

## Extension points

- New CLI flags: extend `parse_args` and thread values through `build_config` or
  `build_tile_config` in
  [../brickarchitect_label_converter/cli.py](../brickarchitect_label_converter/cli.py).
- New clustering or merge heuristics: add functions in
  [../brickarchitect_label_converter/segment.py](../brickarchitect_label_converter/segment.py)
  and tune constants or whitelists in
  [../brickarchitect_label_converter/config.py](../brickarchitect_label_converter/config.py).
- New object kinds or draw styles: add a parser in
  [../brickarchitect_label_converter/lbx_lib.py](../brickarchitect_label_converter/lbx_lib.py)
  and a draw helper in
  [../brickarchitect_label_converter/render.py](../brickarchitect_label_converter/render.py).

## Known gaps

- Add a diagram of the parsing, clustering, and rendering pipeline.
- Document the LBX XML schema details that the parser relies on.
