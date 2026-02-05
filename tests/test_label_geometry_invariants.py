import os
import pathlib

import lbx_to_avery_5167


EPSILON = 2.0


#============================================
def _is_category_label(cluster: lbx_to_avery_5167.LabelCluster) -> bool:
	"""
	Detect category labels which use a full black rectangle.

	Args:
		cluster: LabelCluster instance.

	Returns:
		True if cluster appears to be a category label.
	"""
	if (
		abs(cluster.width - lbx_to_avery_5167.DEFAULT_LABEL_WIDTH) > 0.1
		or abs(cluster.height - lbx_to_avery_5167.DEFAULT_LABEL_HEIGHT) > 0.1
	):
		return False
	for obj in cluster.objects:
		if obj.kind == "rect" and obj.fill_color == "#000000":
			return True
	return False


#============================================
def test_label_bboxes_within_background() -> None:
	"""
	Verify label bounding boxes fit within the background size.
	"""
	labels_dir = pathlib.Path("LEGO_BRICK_LABELS-v40/Labels")
	if not labels_dir.exists():
		return

	paths = lbx_to_avery_5167.gather_lbx_paths([str(labels_dir)])
	labels, _counts, _hashes, _thresholds = lbx_to_avery_5167.collect_labels(
		paths,
		None,
		True,
		None,
		verbose=False,
	)

	clusters_by_source: dict[str, list[lbx_to_avery_5167.LabelCluster]] = {}
	for cluster in labels:
		if _is_category_label(cluster):
			continue
		clusters_by_source.setdefault(cluster.source_path, []).append(cluster)

	failures = []
	for source_path, clusters in clusters_by_source.items():
		if len(clusters) < 3:
			continue
		heights = sorted(cluster.height for cluster in clusters)
		median_height = heights[len(heights) // 2]
		max_height = median_height * lbx_to_avery_5167.BACKGROUND_SPAN_FACTOR + EPSILON
		for cluster in clusters:
			if cluster.height > max_height:
				failures.append(
					f"{os.path.basename(source_path)} idx {cluster.index}: "
					f"height {cluster.height:.1f} > median {median_height:.1f}"
				)

	if failures:
		message = "Label bbox exceeds background bounds:\n"
		message += "\n".join(failures[:20])
		raise AssertionError(message)
