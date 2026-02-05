# LABEL_BOUNDARIES.md

This document explains how the script decides where one label ends and the next begins.

## Summary
- The script uses XML group containers when they exist.
- Otherwise it uses geometry from object positions to assign objects to labels.
- Separator lines (draw:poly) are used as hard boundaries when present.
- If there are no separator lines, the script groups objects by gaps in X and Y.

## Step 1: Parse the LBX XML
- Each LBX is a zip that contains `label.xml`.
- The XML is parsed with `defusedxml.ElementTree.fromstring`.
- All objects are collected from `<pt:objects>`.

## Step 2: Use XML groups when present
- If `<group>` elements exist, each group becomes one label.
- A group must contain at least one text or image object to be used.
- Some LBX files place multiple logical labels inside a single group. If a group has
  at least `GROUP_SPLIT_TEXT_THRESHOLD` text objects, the group is split using the
  same separator/gap logic used for loose objects (periodicity is not used for
  group splits).

## Step 3: Detect separator lines
- Polyline objects are inspected for separator lines.
- A vertical separator is a poly with small width and tall height.
- A horizontal separator is a poly with small height and long width.
- Separator positions are taken from the poly object `x` or `y` plus half the line width or height.
- Nearby separator positions are merged with a small tolerance.

## Step 4: Background grid grouping
- The `<style:backGround>` width and height are treated as a single label size.
- Grid grouping is only used when no separator lines are present.
- If objects span more than 1.6 times the background width or height, the script bins objects into a row and column grid using the background size.
- The grid is evaluated with a few row height multipliers and origin offsets to reduce image-only labels.
- Multi-row grid sizes are only used for a small whitelist of known stacked-row LBX files.
- The best grid candidate is chosen by scoring how many labels contain both text and images.
- Image-only labels are merged into the nearest text label to keep images paired with text.
- Multi-row grid whitelist:
  - `CLIP-flexible`
  - `OTHER-chain_string`
  - `TECHNIC-mechanical_1`
  - `TECHNIC-mechanical_2`

## Step 5: Geometry grouping when groups do not exist
- Only text and image objects are used for grouping.
- If horizontal separators exist, objects are split into row bins using those positions.
- If vertical separators exist, objects are split into column bins using those positions.
- Each row and column bin becomes a label.

## Step 6: Gap-based grouping when separators do not exist
When there are no separator lines, the script tries a periodic layout first:
- Object centers are used to detect a dominant step size on Y (rows) and X (columns).
- If the periodicity confidence is high enough, objects are snapped to that step.
- If periodicity is weak or ambiguous, the script falls back to gap-based grouping.

Gap-based fallback:
- Rows are found by grouping objects on the Y axis.
- The Y gap threshold is computed from the distribution of Y gaps.
- Columns are found by grouping each row on the X axis.
- The X gap threshold is computed per row.
- The gap thresholds use a 2-group split of gap sizes when possible.
- A minimum threshold is enforced to avoid tiny noise splits.

## Step 7: Per-row pairing pass (whitelisted)
- A per-row pairing pass can be enabled for specific LBX files that routinely swap
  image/text order (for example, `MINIFIG-accessories-all`).
- The pass uses a single row height and pairs images to the next text within that row.
- Pairing whitelist:
  - `MINIFIG-accessories-all`
  - `MINIFIG-weapon_3`

## Step 7a: General image-text matching
- When there are no separators, a general image-text matching pass attempts to
  merge image-only and text-only clusters by distance.
- The match is only accepted if it improves the cluster score and reduces
  missing-text or image-order warnings.
- This pass runs before the whitelisted per-row pairing and uses that pairing as
  a fallback when it scores better.

## Step 8: Merge loose or grouped text-only labels (whitelisted)
- Some LBX files contain text-only groups that must be merged into nearby image-only
  groups in the same row.
- This merge is applied only for a whitelist:
  - `MINIFIG-accessories-all`
  - `MINIFIG-CATEGORY-clothing_hair`
  - `MINIFIG-hair-accessory`
  - `MINIFIG-weapon_3`
  - `NATURE-flower`
  - `OTHER-shooter_1`

## Step 8a: Merge adjacent text-only fragments (whitelisted)
- Some index labels split numeric suffixes into a separate text-only label.
- Adjacent text-only labels are merged when the left label ends in a digit and the
  right label starts with a digit.
- This merge is applied only for a whitelist:
  - `INDEX-slope`

## Step 9: Multi-image split exceptions (whitelisted)
- A few LBX files contain clusters with multiple images and texts in one row.
- These clusters are split using image-to-text pairing, then any text-only clusters
  are merged back into image-only clusters so the final result is `image + text`
  per label.
- Multi-image split whitelist:
  - `CLIP-clip_3`
  - `ANGLE-wedge_plate_63`

## Step 10: Recursive segmentation
- After initial clustering and pairing, oversized labels are split again using
  gap-based segmentation.
- A cluster is considered oversized when its height exceeds the median label
  height for that file multiplied by `BACKGROUND_SPAN_FACTOR`.
- The split is accepted only if it improves the label score.

## Determinism
- LBX file paths are sorted.
- Clusters are built from sorted object positions.
- No randomness is used.

## Current limitations
- If the LBX data has overlapping object boxes, overlaps can still occur.
- Labels that rely on non-polyline separators may need additional heuristics.
