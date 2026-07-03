# File formats

This document describes the input the converter reads and the output it writes.
For how the converter decides where one label ends and the next begins, see
[LABEL_BOUNDARIES.md](LABEL_BOUNDARIES.md).

## Input: LBX files

- Each `.lbx` file is a zip container exported by Brother P-touch / Brick
  Architect.
- The container holds a `label.xml` describing page size, label size, objects,
  and object transforms, plus referenced image assets.
- The XML is parsed with `defusedxml.ElementTree.fromstring` for safety.
- Objects used for labels are text and image objects; polyline (`draw:poly`)
  objects can act as separators.
- Inputs may be individual `.lbx` files or directories; directories are searched
  for `.lbx` files and processing order is sorted by numeric category prefix.

## Output: Avery 5167 PDF

- The main output is a single PDF imposed for Avery 5167 return-address labels.
- Sheet layout is 4 columns by 20 rows on US Letter, 80 labels per page.
- Label size is 1.75 in by 0.5 in. Geometry defaults live in
  `brickarchitect_label_converter/config.py` (margins, gaps, and a content
  inset so print never touches the sticker edge).
- By default only full 80-label pages are written. Pass `-p` or
  `--include-partial` to allow a partial final sheet.
- A category label is inserted per LBX file using the file name (black
  background, bold white text).

## Output: tile PDFs

- Each label is first rendered as a vector tile PDF under a `tiles/` folder next
  to the output PDF.
- Tiles are then placed into the Avery grid during imposition.

## Output: manifest JSON

- A manifest JSON is written next to the output PDF (default `OUTPUT.pdf.json`,
  or the path given with `-m`).
- Keys are sorted and the file is indented for readable diffs.
- Recorded fields include:

  | Field | Description |
  | --- | --- |
  | `inputs` | List of input LBX file paths. |
  | `label_counts` | Label count per input file. |
  | `lbx_hashes` | SHA256 hash per input file. |
  | `gap_thresholds` | Computed gap threshold per file. |
  | `labels_per_page` | Labels per imposed page. |
  | `total_labels` | Total labels collected. |
  | `printed_labels` | Labels placed on full pages. |
  | `leftover_labels` | Labels not placed (partial page excluded). |
  | `pages` | Number of pages written. |
  | `layout` | Label size, columns, rows, margins, gaps, inset, and scales. |
  | `fonts` | Regular, bold, italic, and bold-italic PDF font names. |

## Output: log files

- `output/label_counts.log`: LBX files that produced a low label count, for
  reviewing possible under-splits.
- `output/multi_image_labels.log`: labels that contain multiple images.
