# Code architecture

This document describes the main modules and data flow for the LBX to Avery 5167
conversion pipeline.

## Pipeline
- CLI orchestration lives in `brickarchitect_label_converter/cli.py` and is
  exposed by `lbx_to_avery_5167.py`.
- LBX parsing and normalization live in `brickarchitect_label_converter/lbx_lib.py`.
- Segmentation and label boundary logic live in
  `brickarchitect_label_converter/segment.py`.
- Rendering and imposition live in `brickarchitect_label_converter/render.py`.

## Data model
- `LabelObject` (text, image, rect, poly) is defined in
  `brickarchitect_label_converter/lbx_lib.py`.
- `LabelCluster` is defined in `brickarchitect_label_converter/segment.py`.
- Rendering and imposition configs are defined in
  `brickarchitect_label_converter/config.py`.

## Outputs
- Label tiles are written under `output/tiles/` during the render step.
- Final Avery sheets are written as a PDF along with a JSON manifest.

## Known gaps
- Add a diagram of the parsing and rendering pipeline.
- Document the LBX XML schema details that the parser relies on.
