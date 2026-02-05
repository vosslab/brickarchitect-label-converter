# File structure

This document summarizes the current repository layout.

## Top-level
- `brickarchitect_label_converter/`: Core Python package modules.
- `docs/`: Project documentation.
- `LEGO_BRICK_LABELS-v40/`: Input LBX dataset for v40 labels.
- `output/`: Generated PDFs and tiles (not tracked).
- `tests/`: Pytest suite and regression fixtures.
- `lbx_to_avery_5167.py`: Wrapper entry point for the CLI.
- `run.sh`: Convenience script to run the pipeline.
- `pip_requirements.txt`: Runtime dependency list.
- `pyproject.toml`: Packaging metadata.
- `VERSION`: Repo version string.

## Package modules
- `brickarchitect_label_converter/cli.py`: CLI parsing and orchestration.
- `brickarchitect_label_converter/config.py`: Constants and dataclasses.
- `brickarchitect_label_converter/lbx_lib.py`: LBX XML parsing helpers.
- `brickarchitect_label_converter/segment.py`: Label boundary logic.
- `brickarchitect_label_converter/render.py`: PDF rendering and imposition.

## Known gaps
- Add a short note about generated artifacts in `output/`.
