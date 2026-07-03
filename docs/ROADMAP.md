# Roadmap

This roadmap tracks planned work after the segmentation refactor. The phased
refactor in [REFACTOR_PLAN.md](REFACTOR_PLAN.md) is complete (Phases 0 through 4
are all marked done), so future work focuses on reducing special cases and
supporting real-world printing.

## Near-term

- Reduce per-file special cases. Five whitelist sets still live in
  `brickarchitect_label_converter/config.py` (`ROW_STACK_WHITELIST`,
  `PAIRING_WHITELIST`, `TEXT_MERGE_WHITELIST`, `MULTI_IMAGE_SPLIT_WHITELIST`,
  `TEXT_ONLY_ADJACENT_MERGE_WHITELIST`). The goal is to replace these with
  general geometry-driven rules and shrink the whitelists over time.
- Remove manual split thresholds. `GROUP_SPLIT_TEXT_THRESHOLD` remains a
  hardcoded cutoff. REFACTOR_PLAN Phase 3 calls for retiring it once recursive
  segmentation proves stable across datasets.
- Document a printer calibration workflow. The `--calibration` page exists, but
  a step-by-step alignment guide for Avery 5167 sheets is still missing (see
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

## Longer-term

- Handle overlapping object boxes. Overlaps in LBX data can still produce
  overlapping labels (see [LABEL_BOUNDARIES.md](LABEL_BOUNDARIES.md) current
  limitations).
- Support non-polyline separators. Labels that rely on separators other than
  `draw:poly` lines may need additional heuristics.
- Define a release plan. `VERSION` tracks the current release string, but no
  tagged release or publishing workflow is defined yet. This is a standalone
  app, not a published package, so no `pyproject.toml` build backend is needed.

## Known gaps

- No dated release milestones are scheduled yet. Add target versions once a
  release plan exists.
