# Todo

Backlog scratchpad for small, near-term tasks. The plan of record is
[REFACTOR_PLAN.md](REFACTOR_PLAN.md); larger direction lives in
[ROADMAP.md](ROADMAP.md).

## Open items

- Shrink the special-case whitelists in
  `brickarchitect_label_converter/config.py` as general rules replace them.
- Retire `GROUP_SPLIT_TEXT_THRESHOLD` once recursive segmentation is stable
  (REFACTOR_PLAN Phase 3 follow-up).
- Write a printer calibration guide for aligning output to Avery 5167 sheets.
- Review low-label-count files logged to `output/label_counts.log` and
  multi-image files logged to `output/multi_image_labels.log` for missed splits.
- Investigate overlapping object boxes that can still produce overlapping labels.

## Known gaps

- No source code `TODO` or `FIXME` markers are present; this list is derived
  from planning docs and config whitelists.
