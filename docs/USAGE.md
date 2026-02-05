# Usage

## Render tile PDFs
```bash
python3 lbx_to_avery_5167.py tiles LEGO_BRICK_LABELS-v40/Labels \
  --tiles-dir output/tiles \
  --manifest output/tiles.json
```

## Impose tiles onto Avery 5167 sheets
```bash
python3 lbx_to_avery_5167.py impose output/tiles \
  --output output/avery_5167.pdf \
  --draw-outlines
```

## Tile options
- `--gap-threshold`: Override object clustering gap threshold in points.
- `--label-width`: Label width in inches.
- `--label-height`: Label height in inches.
- `--inset`: Safe inset in inches.

## Imposition options
- `--calibration`: Add a calibration page with label boxes and a 1 inch ruler mark.
- `--draw-outlines`: Draw light outlines for each sticker on every page.
- `--include-partial`: Allow a partial final sheet (default is full 80-label pages only).
- `--manifest`: Write a JSON manifest (defaults to `OUTPUT.pdf.json`).
- `--tiles-manifest`: Tile manifest JSON path for ordering.
- `--x-scale`: Horizontal scale for placement.
- `--y-scale`: Vertical scale for placement.
