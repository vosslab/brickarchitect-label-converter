#!/usr/bin/env bash

set -euo pipefail

/opt/homebrew/opt/python@3.12/bin/python3.12 lbx_to_avery_5167.py \
	LEGO_BRICK_LABELS-v40/Labels \
	--output output/avery_5167.pdf \
	--draw-outlines
