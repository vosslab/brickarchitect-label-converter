import json
import os
import pathlib
import subprocess

import pytest

import lbx_to_avery_5167


_LABEL_DATA_CACHE = None


#============================================
def _get_repo_root() -> pathlib.Path:
	"""
	Get the repository root via git.

	Returns:
		Repository root path.
	"""
	result = subprocess.run(
		["git", "rev-parse", "--show-toplevel"],
		capture_output=True,
		text=True,
		check=False,
	)
	if result.returncode != 0:
		message = result.stderr.strip() or "git rev-parse failed"
		raise AssertionError(message)
	root = result.stdout.strip()
	if not root:
		raise AssertionError("git rev-parse returned empty output")
	return pathlib.Path(root)


#============================================
def _load_baseline(path: pathlib.Path) -> dict:
	"""
	Load a baseline JSON file.

	Args:
		path: JSON path.

	Returns:
		Loaded dict.
	"""
	text = path.read_text(encoding="utf-8")
	return json.loads(text)


#============================================
def _collect_label_data(repo_root: pathlib.Path) -> dict:
	"""
	Collect label counts and multi-image counts for the v40 dataset.

	Args:
		repo_root: Repository root path.

	Returns:
		Dict with counts and labels.
	"""
	global _LABEL_DATA_CACHE
	if _LABEL_DATA_CACHE is not None:
		return _LABEL_DATA_CACHE

	labels_dir = repo_root / "LEGO_BRICK_LABELS-v40" / "Labels"
	if not labels_dir.exists():
		pytest.skip("LEGO_BRICK_LABELS-v40/Labels not found.")

	paths = lbx_to_avery_5167.gather_lbx_paths([str(labels_dir)])
	# Collect labels and counts in one pass to avoid duplicate work.
	label_data = lbx_to_avery_5167.collect_labels(
		paths,
		None,
		True,
		None,
		verbose=False,
	)
	labels, counts_by_file, _hashes, _thresholds = label_data
	counts = {}
	for path, count in counts_by_file.items():
		rel_path = os.path.relpath(path, repo_root)
		counts[rel_path] = count

	multi_image_counts = {}
	for cluster in labels:
		image_count = sum(1 for obj in cluster.objects if obj.kind == "image")
		if image_count <= 1:
			continue
		source_name = os.path.basename(cluster.source_path)
		multi_image_counts[source_name] = multi_image_counts.get(source_name, 0) + 1

	_LABEL_DATA_CACHE = {
		"counts": counts,
		"multi_image": multi_image_counts,
	}
	return _LABEL_DATA_CACHE


#============================================
def _summarize_count_diff(expected: dict, actual: dict) -> str:
	"""
	Build a summary diff string for two dicts of counts.

	Args:
		expected: Expected counts.
		actual: Actual counts.

	Returns:
		Diff summary string.
	"""
	expected_keys = set(expected.keys())
	actual_keys = set(actual.keys())
	missing = sorted(expected_keys - actual_keys)
	extra = sorted(actual_keys - expected_keys)
	changed = []
	for key in sorted(expected_keys & actual_keys):
		expected_value = expected[key]
		actual_value = actual[key]
		if expected_value != actual_value:
			changed.append(f"{key}: expected {expected_value} got {actual_value}")
	lines = []
	if missing:
		lines.append("Missing entries:")
		for key in missing[:10]:
			lines.append(f"- {key}")
	if extra:
		lines.append("Extra entries:")
		for key in extra[:10]:
			lines.append(f"- {key}")
	if changed:
		lines.append("Changed counts:")
		for line in changed[:10]:
			lines.append(f"- {line}")
	return "\n".join(lines)


#============================================
def test_label_counts_v40() -> None:
	"""
	Compare current label counts to the v40 baseline.
	"""
	repo_root = _get_repo_root()
	baseline_path = repo_root / "tests" / "fixtures" / "label_counts_v40.json"
	if not baseline_path.exists():
		pytest.skip("Baseline label_counts_v40.json not found.")
	expected = _load_baseline(baseline_path)
	data = _collect_label_data(repo_root)
	actual = data["counts"]
	if expected != actual:
		diff = _summarize_count_diff(expected, actual)
		raise AssertionError(f"Label count mismatch against v40 baseline.\n{diff}")


#============================================
def test_multi_image_counts_v40() -> None:
	"""
	Compare multi-image label counts to the v40 baseline.
	"""
	repo_root = _get_repo_root()
	baseline_path = repo_root / "tests" / "fixtures" / "multi_image_labels_v40.json"
	if not baseline_path.exists():
		pytest.skip("Baseline multi_image_labels_v40.json not found.")
	expected = _load_baseline(baseline_path)
	data = _collect_label_data(repo_root)
	actual = data["multi_image"]
	diff = _summarize_count_diff(expected, actual)
	if diff:
		raise AssertionError(f"Multi-image label mismatch against v40 baseline.\n{diff}")
