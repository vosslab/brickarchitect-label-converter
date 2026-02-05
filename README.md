# brickarchitect-label-converter

Convert Brother P-touch LBX label sets into Avery 5167 label sheets as PDFs. This is for batch printing LEGO storage labels without wasting partial sticker sheets.

## Documentation
- [docs/INSTALL.md](docs/INSTALL.md): Setup requirements and dependencies.
- [docs/USAGE.md](docs/USAGE.md): CLI usage and examples.
- [docs/PLAN.md](docs/PLAN.md): Implementation plan and workflow notes.
- [docs/CHANGELOG.md](docs/CHANGELOG.md): Timeline of changes.
- [docs/PYTHON_STYLE.md](docs/PYTHON_STYLE.md): Python style rules for this repo.
- [docs/MARKDOWN_STYLE.md](docs/MARKDOWN_STYLE.md): Markdown style rules for this repo.
- [docs/REPO_STYLE.md](docs/REPO_STYLE.md): Repo organization and conventions.

## Quick start
```bash
python3 lbx_to_avery_5167.py tiles LEGO_BRICK_LABELS-v40/Labels \
  --tiles-dir output/tiles \
  --manifest output/tiles.json

python3 lbx_to_avery_5167.py impose output/tiles \
  --output output/avery_5167.pdf \
  --draw-outlines
```

Add `--calibration` to prepend a calibration page. Omit `--include-partial` to only generate full 80-label pages.

## Testing
```bash
python3 -m pytest tests/test_lbx_to_avery_5167.py
```
