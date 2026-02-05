# Usage

## Run
```bash
python3 lbx_to_avery_5167.py LEGO_BRICK_LABELS-v40/Labels \
  --output output/avery_5167.pdf \
  --draw-outlines
```

This command always runs the full pipeline (LBX to Avery PDF).

Tile PDFs are written under the output directory in `tiles/`.

The script also inserts a category label per LBX file using the file name
(black background with bold white text). Hyphens are converted to spaces
and long names are wrapped across lines.

## Options
- `-o`, `--output`: Output PDF path.
- `-m`, `--manifest`: Output manifest JSON path (defaults to `OUTPUT.pdf.json`).
- `-d`, `--draw-outlines`: Draw light outlines for each sticker.
- `-c`, `--calibration`: Add a calibration page with label boxes and a 1 inch ruler mark.
- `-p`, `--include-partial`: Allow a partial final sheet (default is full 80-label pages only).
- `-g`, `--max-pages`: Limit the number of label pages imposed.
- `-l`, `--max-labels`: Limit the number of labels imposed (combine with `--include-partial` for partial sheets).
- `-n`, `--normalize-text`: Normalize label text to ASCII (use `-N` or `--no-normalize-text` to disable).
- `--stop-before-rendering`: Stop after collecting labels (skip rendering and imposition).
