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
- The best grid candidate is chosen by scoring how many labels contain both text and images.
- Image-only labels are merged into the nearest text label to keep images paired with text.

## Step 5: Geometry grouping when groups do not exist
- Only text and image objects are used for grouping.
- If horizontal separators exist, objects are split into row bins using those positions.
- If vertical separators exist, objects are split into column bins using those positions.
- Each row and column bin becomes a label.

## Step 6: Gap-based grouping when separators do not exist
- Rows are found by grouping objects on the Y axis.
- The Y gap threshold is computed from the distribution of Y gaps.
- Columns are found by grouping each row on the X axis.
- The X gap threshold is computed per row.
- The gap thresholds use a 2-group split of gap sizes when possible.
- A minimum threshold is enforced to avoid tiny noise splits.

## Determinism
- LBX file paths are sorted.
- Clusters are built from sorted object positions.
- No randomness is used.

## Current limitations
- If the LBX data has overlapping object boxes, overlaps can still occur.
- Labels that rely on non-polyline separators may need additional heuristics.
