# Plan

## Step 1: Lock down geometry
- Avery 5167: 1.75 in x 0.5 in, 4 columns x 20 rows on Letter (8.5 in x 11 in).
- Define calibration constants to tune per printer: left margin, top margin, horizontal gap, vertical gap.
- Add a calibration test mode that draws label boxes and a 1 inch ruler mark.

## Step 2: Treat LBX as source of truth
- Unzip the .lbx files.
- Parse label.xml for page size, label size, object list, and object transforms.
- Extract referenced media assets (BMP, PNG, JPG if present) from the container.
- Use lbx-utils as scaffolding where compatible, but do not assume it covers every object type.

## Step 3: Build a minimal renderer for a single label
- Decide output format: per-label PDF (vector text, images placed as-is) or per-label PNG at fixed DPI (600 recommended).
- Implement object drawing for needed types: text, image, optional border.
- Normalize fonts: register known TTFs and map LBX font names to chosen fonts.
- Accept that Brother font matching will be approximate, but keep consistent.

## Step 4: Normalize each label asset for imposition
- Define an inset (0.02 to 0.04 in) so content never touches sticker edges.
- If PNG: render at fixed DPI; trim whitespace cautiously (alpha-aware if needed); keep consistent padding after trimming.
- If PDF: use a fixed media box equal to the label logical size, not a tight bounding box.

## Step 5: Impose to Avery 5167 using ReportLab
- Create a Letter PDF with a 4 x 20 grid in points.
- For each slot: compute x and y using calibrated margins and gaps; place the label asset scaled to fit inside the inset and centered.
- Paginate every 80 labels.

## Step 6: Printer calibration loop
- Print the calibration page on plain paper with Actual size and no scaling.
- Overlay with an Avery 5167 sheet and adjust left and top offsets.
- Optional: add separate x-scale and y-scale nudges.

## Step 7: CLI and reproducibility
- Provide one script with two subcommands: extract (LBX to per-label assets plus manifest JSON) and impose (assets to avery_5167.pdf).
- The manifest should record source LBX hash, render DPI, font mapping, and calibration constants.
- Add a small test: one known LBX input, assert the expected label count and a non-empty output PDF.

## Step 8: Risk management
- Start with a single LBX that represents the real content mix.
- Implement only object types that appear; add more when a label fails to render.
- Keep a diff view: render one label as PNG and compare it to the Brother-exported PDF.
