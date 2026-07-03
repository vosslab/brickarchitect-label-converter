# Troubleshooting

This page collects common issues when running the LBX conversion pipeline.

## Setup and input

- Missing module errors: install dependencies with
  `python3 -m pip install -r pip_requirements.txt`. PyMuPDF is required for
  rendering (see [INSTALL.md](INSTALL.md)).
- Output is empty: confirm the input path points at a folder containing `.lbx`
  files and re-run. The run prints `LBX files found:` so a count of `0` means
  the input path is wrong.

## Label splitting

- Labels merged together or split too much: the pipeline picks boundaries from
  XML groups, separator lines, then gap or periodicity clustering. See
  [LABEL_BOUNDARIES.md](LABEL_BOUNDARIES.md) for how each step decides.
- Too few labels for a file: low-count files are logged to
  `output/label_counts.log`. Check that file against the source LBX to see if a
  cluster was under-split.
- A label shows multiple images: multi-image labels are logged to
  `output/multi_image_labels.log`. A few files are split by image-to-text
  pairing (see the multi-image split notes in
  [LABEL_BOUNDARIES.md](LABEL_BOUNDARIES.md)).
- Overlapping labels: if the LBX data has overlapping object boxes, overlaps can
  still occur. This is a known limitation.

## Rendering

- Missing or wrong glyphs: text is normalized to ASCII by default to avoid
  missing glyphs. Pass `-N` or `--no-normalize-text` to preserve original text,
  but expect gaps if the font lacks those characters.
- Text lines in the wrong order: multiline text renders top-to-bottom. This was
  fixed after an earlier reversed-order bug.
- Images bleeding past sticker edges: images are shrunk to 95 percent of their
  box and centered, with upscaling capped at 2x, to reduce bleed.

## Printing and calibration

- Labels do not line up on the Avery 5167 sheet: add a calibration page with
  `-c` or `--calibration` to print label boxes and a 1 inch ruler mark, then
  adjust the printer to "Actual size" with no scaling.

## Known gaps

- A full step-by-step printer calibration guide (measuring and correcting left
  and top offsets) is not written yet.
