#!/usr/bin/env bash

set -euo pipefail

/opt/homebrew/opt/python@3.12/bin/python3.12 lbx_to_avery_5167.py tiles \
	LEGO_BRICK_LABELS-v40/Labels \
	--tiles-dir output/tiles \
	--manifest output/tiles.json

/opt/homebrew/opt/python@3.12/bin/python3.12 lbx_to_avery_5167.py impose \
	output/tiles \
	--output output/avery_5167.pdf \
	--draw-outlines
