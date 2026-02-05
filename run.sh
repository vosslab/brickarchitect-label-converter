#!/usr/bin/env bash

set -euo pipefail

outfile="output/avery_5167.pdf"

rm -rf "$outfile" "output/tiles/"

/opt/homebrew/opt/python@3.12/bin/python3.12 lbx_to_avery_5167.py \
	LEGO_BRICK_LABELS-v40/Labels \
	--output $outfile \
	--draw-outlines

echo "output written to: $outfile"
