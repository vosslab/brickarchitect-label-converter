#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refresh v40 label count baselines for tests.
"""

import json
import os
import pathlib
import subprocess

import lbx_to_avery_5167


#============================================
def get_repo_root() -> pathlib.Path:
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
def write_json(path: pathlib.Path, payload: dict) -> None:
	"""
	Write a JSON payload to disk.

	Args:
		path: Output path.
		payload: JSON payload.
	"""
	text = json.dumps(payload, indent=2, sort_keys=True)
	path.write_text(text, encoding="utf-8")


#============================================
def build_counts(repo_root: pathlib.Path) -> tuple[dict, dict]:
	"""
	Build label count and multi-image baselines from v40 labels.

	Args:
		repo_root: Repository root path.

	Returns:
		Tuple of (label_counts, multi_image_counts).
	"""
	labels_dir = repo_root / "LEGO_BRICK_LABELS-v40" / "Labels"
	if not labels_dir.exists():
		raise AssertionError("LEGO_BRICK_LABELS-v40/Labels not found.")

	paths = lbx_to_avery_5167.gather_lbx_paths([str(labels_dir)])
	labels, counts_by_file, _hashes, _thresholds = lbx_to_avery_5167.collect_labels(
		paths,
		None,
		True,
		None,
		verbose=False,
	)
	label_counts = {}
	for path, count in counts_by_file.items():
		rel_path = os.path.relpath(path, repo_root)
		label_counts[rel_path] = count

	multi_image_counts = {}
	for cluster in labels:
		image_count = sum(1 for obj in cluster.objects if obj.kind == "image")
		if image_count <= 1:
			continue
		source_name = os.path.basename(cluster.source_path)
		multi_image_counts[source_name] = multi_image_counts.get(source_name, 0) + 1

	return (label_counts, multi_image_counts)


#============================================
def main() -> None:
	"""
	Run the v40 baseline refresh.
	"""
	repo_root = get_repo_root()
	label_counts, multi_image_counts = build_counts(repo_root)
	fixtures_dir = repo_root / "tests" / "fixtures"
	fixtures_dir.mkdir(parents=True, exist_ok=True)
	write_json(fixtures_dir / "label_counts_v40.json", label_counts)
	write_json(fixtures_dir / "multi_image_labels_v40.json", multi_image_counts)
	print("Updated v40 baselines in tests/fixtures.")


if __name__ == "__main__":
	main()
