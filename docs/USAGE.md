# Usage

This tool reads Brother P-touch LBX label files (the LEGO Brick Labels set
uses these) and produces a printable PDF sheet laid out for Avery 5167
labels, 80 labels per page. You run it once per batch of LBX files; it does
not modify your LBX files or install anything.

If you have never used a terminal before: a "terminal" (also called a
command line or CLI) is a text window where you type commands and press
Enter. Nothing here deletes your files or changes your system - it only
reads the LBX files you point it at and writes a new PDF file. If a command
below does not do what you expect, nothing is lost; just try again.

## Before you start

Make sure the tool is installed first - see [INSTALL.md](INSTALL.md). You
should be able to run the following without an error before continuing:

```bash
python3 lbx_to_avery_5167.py --help
```

If that prints a list of options (not an error), you are ready.

## Your first run, step by step

1. Open a terminal and move into the repository folder (the folder that
   contains `lbx_to_avery_5167.py`):
   ```bash
   cd path/to/brickarchitect-label-converter
   ```
2. Run the converter against the bundled sample label set. Copy this whole
   command, paste it into the terminal, and press Enter:
   ```bash
   python3 lbx_to_avery_5167.py LEGO_BRICK_LABELS-v40/Labels --output output/avery_5167.pdf
   ```
3. Watch the terminal print progress as it reads LBX files and renders
   pages. This can take a little while for the full label set; that is
   normal.
4. When it finishes, open the new file it created:
   `output/avery_5167.pdf`. That is your printable label sheet.

Nothing else on your computer changes. If you want to try different
options, just re-run the command with a different `--output` path so you
do not overwrite the first PDF while comparing results.

## What each part of the command means

- `python3 lbx_to_avery_5167.py` - runs the tool.
- `LEGO_BRICK_LABELS-v40/Labels` - where to read `.lbx` files from. This can
  be a single `.lbx` file or a folder of them.
- `--output output/avery_5167.pdf` - where to write the resulting PDF. This
  option is required every time.

## A few options worth trying

These are optional flags you add after the command above. You do not need
any of them for a first run.

- Draw a thin outline around each sticker (helpful for checking alignment
  before printing on real label paper): add `--draw-outlines`.
- Print just one page first, to check alignment before using a full sheet
  of labels: add `--max-pages 1`.
- Add a calibration page (a ruler mark you can measure against, to fix
  printer offset): add `--calibration`.

Example combining the first-run command with these:

```bash
python3 lbx_to_avery_5167.py LEGO_BRICK_LABELS-v40/Labels \
  --output output/preview.pdf \
  --max-pages 1 \
  --calibration
```

If your printed labels do not line up with the sticker sheet, see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for calibration steps.

## Inputs and outputs

- Input: `.lbx` files, or a directory containing them (for example
  `LEGO_BRICK_LABELS-v40/Labels`). Your input files are never modified.
- Output PDF: written to the path you passed with `--output`.
- Tile PDFs: written to a `tiles/` folder next to the output PDF (one
  intermediate file per label category; safe to ignore for normal use).
- Manifest JSON: written next to the output PDF, describing what was
  rendered. Defaults to `OUTPUT.pdf.json` if you do not pass `--manifest`.
- Label count log: written alongside the output, listing how many labels
  were found per input file.

## Full CLI reference

The script lives at the repo root and delegates to
`brickarchitect_label_converter.cli`. Run `python3 lbx_to_avery_5167.py
--help` at any time to see this same list from the tool itself.

- `-o`, `--output`: Output PDF path (required).
- `-m`, `--manifest`: Output manifest JSON path (defaults to `OUTPUT.pdf.json`).
- `-d`, `--draw-outlines` / `-D`, `--no-draw-outlines`: Draw or disable per-sticker outlines (default off).
- `-c`, `--calibration` / `-C`, `--no-calibration`: Add or disable a calibration page (default off).
- `-p`, `--include-partial` / `-P`, `--no-include-partial`: Allow or exclude a partial final sheet (default full 80-label pages only).
- `-n`, `--normalize-text` / `-N`, `--no-normalize-text`: Normalize label text to ASCII (default on).
- `-g`, `--max-pages`: Limit the number of label pages imposed.
- `-l`, `--max-labels`: Limit the number of labels imposed (combine with `--include-partial` for partial sheets).
- `--stop-before-rendering`: Stop after collecting labels (skip rendering and imposition).

## Maintenance

To refresh the v40 baseline fixtures used by tests:
```bash
./refresh_v40_baseline.py
```

## Known gaps

- [ ] No true `--dry-run` flag exists; `--stop-before-rendering` is the closest partial preview. Confirm whether a full dry-run is wanted.
