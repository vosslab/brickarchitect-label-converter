# Refactor plan

This plan describes the phased refactor to reduce special cases while keeping
the current v40 output stable. Each phase has exit criteria and a fallback path.

## Phase 0: Baseline harness (done)
- Capture v40 label counts and multi-image counts in fixtures.
- Add a regression test that compares current output to the fixtures.
- Provide a refresh script to rebuild fixtures when intentionally changing output.
- Add geometry and render smoke tests to catch merged labels and edge bleed.

Artifacts:
- `tests/fixtures/label_counts_v40.json`
- `tests/fixtures/multi_image_labels_v40.json`
- `tests/test_label_counts_v40.py`
- `refresh_v40_baseline.py`
- `tests/test_label_geometry_invariants.py`
- `tests/test_render_smoke_page.py`

Exit criteria:
- `pytest` passes with the current v40 dataset.

## Phase 1: Periodicity clustering (done)
- Add periodicity-based row and column inference.
- Score periodic clustering against the gap-based result and choose the better one.
- Keep the gap-based logic as the fallback.
- Skip periodicity for group splits to avoid oversplitting.

Exit criteria:
- v40 label-count baseline stays green.
- No new multi-image regressions against baseline.

## Phase 2: General image-text matching (done)
- Replace per-row pairing and most merge heuristics with a single matching pass.
- Use a distance-based pairing inside each provisional bin.
- Merge image-only bins into the closest text bin by score.
- Keep existing pairing as the fallback until the baseline is stable.

Exit criteria:
- v40 label-count baseline stays green.
- Multi-image baseline stays green.
- Image-after-text warnings drop or stay stable.

## Phase 3: Recursive segmentation (done)
- Replace `GROUP_SPLIT_TEXT_THRESHOLD` with recursive segmentation when a cluster
  spans multiple periods or has strong internal gaps.
- Remove manual split thresholds if the recursive method is stable.

Exit criteria:
- v40 baseline stays green.
- No new oversize clusters reported by validation.

## Phase 4: Modularization (done)
- Move CLI orchestration into a dedicated module.
- Move LBX parsing and normalization into a dedicated module.
- Move segmentation logic into a dedicated module.
- Move rendering and imposition into a dedicated module.

Exit criteria:
- All tests still pass.
- No change to v40 baselines unless explicitly refreshed.

## Test commands
- `pytest`
- `./refresh_v40_baseline.py` (only when the output change is intentional)
