"""
Segmentation and label clustering logic.
"""

# Standard Library
import dataclasses
import hashlib
import pathlib
import statistics
import zipfile

# PIP3 modules
import reportlab.pdfbase.pdfmetrics

# local repo modules
import brickarchitect_label_converter as balc
import brickarchitect_label_converter.config
import brickarchitect_label_converter.lbx_lib


LabelObject = balc.lbx_lib.LabelObject
normalize_text = balc.lbx_lib.normalize_text
parse_label_xml_with_groups = balc.lbx_lib.parse_label_xml_with_groups
extract_background_bounds = balc.lbx_lib.extract_background_bounds

DEFAULT_LABEL_WIDTH = balc.config.DEFAULT_LABEL_WIDTH
DEFAULT_LABEL_HEIGHT = balc.config.DEFAULT_LABEL_HEIGHT
DEFAULT_GAP_THRESHOLD = balc.config.DEFAULT_GAP_THRESHOLD
DEFAULT_FONT_BOLD = balc.config.DEFAULT_FONT_BOLD
DEFAULT_TEXT_WEIGHT = balc.config.DEFAULT_TEXT_WEIGHT
CATEGORY_TEXT_SIZE = balc.config.CATEGORY_TEXT_SIZE
CATEGORY_TEXT_MARGIN = balc.config.CATEGORY_TEXT_MARGIN
SEPARATOR_THICKNESS = balc.config.SEPARATOR_THICKNESS
SEPARATOR_MIN_LENGTH = balc.config.SEPARATOR_MIN_LENGTH
BACKGROUND_SPAN_FACTOR = balc.config.BACKGROUND_SPAN_FACTOR
GROUP_SPLIT_TEXT_THRESHOLD = balc.config.GROUP_SPLIT_TEXT_THRESHOLD
PERIODICITY_BIN_SIZE = balc.config.PERIODICITY_BIN_SIZE
PERIODICITY_CONFIDENCE_MIN = balc.config.PERIODICITY_CONFIDENCE_MIN
PERIODICITY_MIN_DELTAS = balc.config.PERIODICITY_MIN_DELTAS
PERIODICITY_MIN_STEP = balc.config.PERIODICITY_MIN_STEP
RECURSIVE_SPLIT_MAX_DEPTH = balc.config.RECURSIVE_SPLIT_MAX_DEPTH
ROW_STACK_WHITELIST = balc.config.ROW_STACK_WHITELIST
PAIRING_WHITELIST = balc.config.PAIRING_WHITELIST
TEXT_MERGE_WHITELIST = balc.config.TEXT_MERGE_WHITELIST
MULTI_IMAGE_SPLIT_WHITELIST = balc.config.MULTI_IMAGE_SPLIT_WHITELIST
TEXT_ONLY_ADJACENT_MERGE_WHITELIST = balc.config.TEXT_ONLY_ADJACENT_MERGE_WHITELIST


@dataclasses.dataclass
class LabelCluster:
	source_path: str
	index: int
	objects: list[LabelObject]
	min_x: float
	min_y: float
	width: float
	height: float


#============================================
def merge_positions(values: list[float], tolerance: float) -> list[float]:
	"""
	Merge nearby positions into a reduced list.

	Args:
		values: Sorted positions.
		tolerance: Merge tolerance.

	Returns:
		Merged positions.
	"""
	if not values:
		return []
	values_sorted = sorted(values)
	merged = [values_sorted[0]]
	for value in values_sorted[1:]:
		if abs(value - merged[-1]) <= tolerance:
			merged[-1] = (merged[-1] + value) / 2.0
			continue
		merged.append(value)
	return merged


#============================================
def find_separators(
	objects: list[LabelObject],
) -> tuple[list[float], list[float]]:
	"""
	Find separators based on thin polyline objects.

	Args:
		objects: Label objects.

	Returns:
		Tuple of (vertical_positions, horizontal_positions).
	"""
	vertical: list[float] = []
	horizontal: list[float] = []
	for obj in objects:
		if obj.kind != "poly":
			continue
		if obj.width <= SEPARATOR_THICKNESS and obj.height >= SEPARATOR_MIN_LENGTH:
			vertical.append(obj.x + obj.width / 2.0)
			continue
		if obj.height <= SEPARATOR_THICKNESS and obj.width >= SEPARATOR_MIN_LENGTH:
			horizontal.append(obj.y + obj.height / 2.0)
	return (
		merge_positions(vertical, SEPARATOR_THICKNESS),
		merge_positions(horizontal, SEPARATOR_THICKNESS),
	)


#============================================
def update_cluster_bounds(cluster: "LabelCluster") -> None:
	"""
	Recompute the bounds for a LabelCluster.

	Args:
		cluster: LabelCluster to update.
	"""
	if not cluster.objects:
		return
	min_x = min(item.x for item in cluster.objects)
	min_y = min(item.y for item in cluster.objects)
	max_x = max(item.x + item.width for item in cluster.objects)
	max_y = max(item.y + item.height for item in cluster.objects)
	cluster.min_x = min_x
	cluster.min_y = min_y
	cluster.width = max_x - min_x
	cluster.height = max_y - min_y


#============================================
def cluster_objects_by_grid(
	objects: list["LabelObject"],
	source_path: str,
	origin_x: float,
	origin_y: float,
	col_step: float,
	row_step: float,
) -> list["LabelCluster"]:
	"""
	Cluster objects by snapping to a background grid.

	Args:
		objects: Label objects.
		source_path: Source path.
		origin_x: Grid origin X.
		origin_y: Grid origin Y.
		col_step: Column step in points.
		row_step: Row step in points.

	Returns:
		List of LabelCluster entries.
	"""
	if not objects or col_step <= 0.0 or row_step <= 0.0:
		return []
	grid: dict[tuple[int, int], list[LabelObject]] = {}
	for obj in objects:
		center_x = obj.x + obj.width / 2.0
		center_y = obj.y + obj.height / 2.0
		col = int(round((center_x - origin_x) / col_step))
		row = int(round((center_y - origin_y) / row_step))
		grid.setdefault((row, col), []).append(obj)
	clusters: list[LabelCluster] = []
	cluster_index = 1
	for key in sorted(grid.keys()):
		cluster = create_label_cluster(grid[key], source_path, cluster_index)
		if cluster is not None:
			clusters.append(cluster)
			cluster_index += 1
	return clusters


#============================================
def score_cluster_layout(
	clusters: list["LabelCluster"],
	expected_labels: int,
) -> float:
	"""
	Score cluster layout based on expected label count.

	Args:
		clusters: Label clusters.
		expected_labels: Expected label count.

	Returns:
		Score value.
	"""
	if expected_labels <= 0:
		return 0.0
	count = len(clusters)
	ratio = min(count, expected_labels) / max(count, expected_labels)
	return ratio


#============================================
def score_label_set(
	clusters: list["LabelCluster"],
	expected_labels: int,
) -> float:
	"""
	Score label set quality.

	Args:
		clusters: Label clusters.
		expected_labels: Expected label count.

	Returns:
		Score value.
	"""
	if not clusters or expected_labels <= 0:
		return 0.0
	score = score_cluster_layout(clusters, expected_labels)
	text_only = 0
	image_only = 0
	for cluster in clusters:
		has_text = any(
			obj.kind == "text" and (obj.text or "").strip()
			for obj in cluster.objects
		)
		has_image = any(obj.kind == "image" for obj in cluster.objects)
		if has_text and not has_image:
			text_only += 1
		if has_image and not has_text:
			image_only += 1
	if text_only > 0 or image_only > 0:
		score -= 0.2 * (text_only + image_only) / len(clusters)
	return score


#============================================
def summarize_label_warnings(clusters: list["LabelCluster"]) -> tuple[int, int]:
	"""
	Summarize missing text and image-after-text warnings.

	Args:
		clusters: Label clusters.

	Returns:
		Tuple of (missing_text, image_after_text).
	"""
	missing_text = 0
	image_after_text = 0
	for cluster in clusters:
		text_objects = [
			obj for obj in cluster.objects
			if obj.kind == "text" and (obj.text or "").strip()
		]
		image_objects = [obj for obj in cluster.objects if obj.kind == "image"]
		if not text_objects:
			missing_text += 1
		if text_objects and image_objects:
			min_text_x = min(obj.x for obj in text_objects)
			min_image_x = min(obj.x for obj in image_objects)
			if min_image_x > min_text_x:
				image_after_text += 1
	return (missing_text, image_after_text)


#============================================
def build_pairs_by_text(
	objects: list[LabelObject],
	source_path: str,
	row_step: float,
	origin_y: float,
) -> list[LabelCluster]:
	"""
	Pair images with text using row segmentation.

	Args:
		objects: Label objects.
		source_path: Source path.
		row_step: Row step height.
		origin_y: Origin Y offset.

	Returns:
		List of LabelCluster entries.
	"""
	if not objects:
		return []
	rows: dict[int, list[LabelObject]] = {}
	for obj in objects:
		center_y = obj.y + obj.height / 2.0
		row = int(round((center_y - origin_y) / row_step))
		rows.setdefault(row, []).append(obj)

	label_clusters: list[LabelCluster] = []
	cluster_index = 1
	for row_objects in rows.values():
		texts = [
			obj for obj in row_objects
			if obj.kind == "text" and (obj.text or "").strip()
		]
		images = [obj for obj in row_objects if obj.kind == "image"]
		texts_sorted = sorted(texts, key=lambda item: item.x)
		images_sorted = sorted(images, key=lambda item: item.x)

		image_index = 0
		for text_obj in texts_sorted:
			images_for_text: list[LabelObject] = []
			while image_index < len(images_sorted) and images_sorted[image_index].x <= text_obj.x:
				images_for_text.append(images_sorted[image_index])
				image_index += 1
			objects_for_label = images_for_text + [text_obj]
			label_cluster = create_label_cluster(objects_for_label, source_path, cluster_index)
			if label_cluster is not None:
				label_clusters.append(label_cluster)
				cluster_index += 1

		while image_index < len(images_sorted):
			objects_for_label = [images_sorted[image_index]]
			label_cluster = create_label_cluster(objects_for_label, source_path, cluster_index)
			if label_cluster is not None:
				label_clusters.append(label_cluster)
				cluster_index += 1
			image_index += 1

	return label_clusters


#============================================
def format_category_title(stem: str) -> str:
	"""
	Format a category label title from a file stem.

	Args:
		stem: LBX file stem.

	Returns:
		Formatted title.
	"""
	return stem.replace("_", " ").replace("-", " ").strip()


#============================================
def wrap_text_to_width(
	text: str,
	font_name: str,
	font_size: float,
	max_width: float,
) -> str:
	"""
	Wrap text to fit within a max width.

	Args:
		text: Input text.
		font_name: Font name for width calculation.
		font_size: Font size for width calculation.
		max_width: Maximum line width in points.

	Returns:
		Wrapped text with newlines.
	"""
	words = text.split()
	if not words:
		return text
	lines: list[str] = []
	current = ""
	for word in words:
		candidate = word if not current else f"{current} {word}"
		width = reportlab.pdfbase.pdfmetrics.stringWidth(candidate, font_name, font_size)
		if width <= max_width or not current:
			current = candidate
			continue
		lines.append(current)
		current = word
	if current:
		lines.append(current)
	return "\n".join(lines)


#============================================
def build_category_label(
	source_path: str,
	title: str,
	index: int,
) -> LabelCluster:
	"""
	Build a category label cluster with black background and white text.

	Args:
		source_path: LBX source path.
		title: Category title text.
		index: Cluster index.

	Returns:
		LabelCluster instance.
	"""
	rect = LabelObject(
		kind="rect",
		x=0.0,
		y=0.0,
		width=DEFAULT_LABEL_WIDTH,
		height=DEFAULT_LABEL_HEIGHT,
		fill_color="#000000",
		line_color="#000000",
	)
	wrapped_title = wrap_text_to_width(
		title,
		DEFAULT_FONT_BOLD,
		CATEGORY_TEXT_SIZE,
		DEFAULT_LABEL_WIDTH - CATEGORY_TEXT_MARGIN * 2.0,
	)
	text = LabelObject(
		kind="text",
		x=CATEGORY_TEXT_MARGIN,
		y=0.0,
		width=DEFAULT_LABEL_WIDTH - CATEGORY_TEXT_MARGIN * 2.0,
		height=DEFAULT_LABEL_HEIGHT,
		text=wrapped_title,
		font_size=CATEGORY_TEXT_SIZE,
		font_weight=DEFAULT_TEXT_WEIGHT,
		use_text_override=False,
		align_horizontal="LEFT",
		align_vertical="CENTER",
		text_color="#FFFFFF",
	)
	return LabelCluster(
		source_path=source_path,
		index=index,
		objects=[rect, text],
		min_x=0.0,
		min_y=0.0,
		width=DEFAULT_LABEL_WIDTH,
		height=DEFAULT_LABEL_HEIGHT,
	)


#============================================
def merge_loose_text_into_image_groups(
	group_clusters: list[list["LabelObject"]],
	loose_objects: list["LabelObject"],
	background: tuple[float, float, float, float] | None,
) -> tuple[list[list["LabelObject"]], list["LabelObject"]]:
	"""
	Merge loose text objects into image-only group clusters.

	Args:
		group_clusters: Existing group clusters.
		loose_objects: Loose objects outside of groups.
		background: Background bounds or None.

	Returns:
		Tuple of (updated group clusters, updated loose objects).
	"""
	if not group_clusters or not loose_objects or background is None:
		return (group_clusters, loose_objects)

	bg_x, bg_y, _bg_width, bg_height = background
	if bg_height <= 0.0:
		return (group_clusters, loose_objects)

	image_groups: list[tuple[list[LabelObject], float, float]] = []
	for group in group_clusters:
		has_image = any(obj.kind == "image" for obj in group)
		has_text = any(
			obj.kind == "text" and (obj.text or "").strip()
			for obj in group
		)
		if not has_image or has_text:
			continue
		cluster = create_label_cluster(group, "", 0)
		if cluster is None:
			continue
		center_x = cluster.min_x + cluster.width / 2.0
		center_y = cluster.min_y + cluster.height / 2.0
		image_groups.append((group, center_x, center_y))

	loose_texts = [
		obj for obj in loose_objects
		if obj.kind == "text" and (obj.text or "").strip()
	]
	if not image_groups or not loose_texts:
		return (group_clusters, loose_objects)

	rows: dict[int, list[tuple[list[LabelObject], float]]] = {}
	for group, center_x, center_y in image_groups:
		row = int((center_y - bg_y) // bg_height)
		rows.setdefault(row, []).append((group, center_x))

	text_rows: dict[int, list[LabelObject]] = {}
	for text_obj in loose_texts:
		center_y = text_obj.y + text_obj.height / 2.0
		row = int((center_y - bg_y) // bg_height)
		text_rows.setdefault(row, []).append(text_obj)

	assigned_ids: set[int] = set()
	for row, groups_in_row in rows.items():
		groups_sorted = sorted(groups_in_row, key=lambda item: item[1])
		texts = text_rows.get(row, [])
		if not texts:
			continue
		texts_sorted = sorted(texts, key=lambda item: item.x)
		text_index = 0
		for group, center_x in groups_sorted:
			while text_index < len(texts_sorted) and texts_sorted[text_index].x < center_x:
				text_index += 1
			if text_index >= len(texts_sorted):
				break
			text_obj = texts_sorted[text_index]
			group.append(text_obj)
			assigned_ids.add(id(text_obj))
			text_index += 1

	updated_loose = [
		obj for obj in loose_objects
		if not (obj.kind == "text" and id(obj) in assigned_ids)
	]
	return (group_clusters, updated_loose)


#============================================
def merge_text_only_groups_into_image_groups(
	group_clusters: list[list["LabelObject"]],
	background: tuple[float, float, float, float] | None,
) -> list[list["LabelObject"]]:
	"""
	Merge text-only group clusters into image-only group clusters.

	Args:
		group_clusters: Group clusters to merge.
		background: Background bounds or None.

	Returns:
		Updated group clusters.
	"""
	if not group_clusters or background is None:
		return group_clusters

	bg_x, bg_y, _bg_width, bg_height = background
	if bg_height <= 0.0:
		return group_clusters

	group_info: list[dict[str, object]] = []
	for index, group in enumerate(group_clusters):
		cluster = create_label_cluster(group, "", 0)
		if cluster is None:
			continue
		has_image = any(obj.kind == "image" for obj in group)
		has_text = any(
			obj.kind == "text" and (obj.text or "").strip()
			for obj in group
		)
		center_x = cluster.min_x + cluster.width / 2.0
		center_y = cluster.min_y + cluster.height / 2.0
		row = int((center_y - bg_y) // bg_height)
		group_info.append(
			{
				"index": index,
				"group": group,
				"has_image": has_image,
				"has_text": has_text,
				"center_x": center_x,
				"row": row,
			}
		)

	rows: dict[int, list[dict[str, object]]] = {}
	for info in group_info:
		rows.setdefault(info["row"], []).append(info)

	removed_indices: set[int] = set()
	max_dx = bg_height * 4.0

	for row, items in rows.items():
		image_groups = [
			item for item in items
			if item["has_image"] and not item["has_text"]
		]
		text_groups = [
			item for item in items
			if item["has_text"] and not item["has_image"]
		]
		if not image_groups or not text_groups:
			continue
		image_groups_sorted = sorted(image_groups, key=lambda item: item["center_x"])  # type: ignore[arg-type]
		for text_info in sorted(text_groups, key=lambda item: item["center_x"]):  # type: ignore[arg-type]
			text_center_x = float(text_info["center_x"])  # type: ignore[arg-type]
			best = None
			best_dx = None
			for image_info in image_groups_sorted:
				image_center_x = float(image_info["center_x"])  # type: ignore[arg-type]
				dx = abs(image_center_x - text_center_x)
				if dx > max_dx:
					continue
				if best_dx is None or dx < best_dx:
					best_dx = dx
					best = image_info
			if best is None:
				continue
			best_group = best["group"]
			if isinstance(best_group, list):
				best_group.extend(text_info["group"])  # type: ignore[arg-type]
				removed_indices.add(int(text_info["index"]))  # type: ignore[arg-type]

	if not removed_indices:
		return group_clusters

	return [
		group for idx, group in enumerate(group_clusters)
		if idx not in removed_indices
	]


#============================================
def merge_image_only_clusters(
	clusters: list["LabelCluster"],
	row_step: float,
	col_step: float,
) -> list["LabelCluster"]:
	"""
	Merge image-only clusters into the nearest text cluster.

	Args:
		clusters: Label clusters to merge.
		row_step: Grid row height.
		col_step: Grid column width.

	Returns:
		Merged list of LabelCluster entries.
	"""
	if not clusters or row_step <= 0.0 or col_step <= 0.0:
		return clusters

	image_only_indices = []
	text_only_indices = []
	text_indices = []
	for index, cluster in enumerate(clusters):
		has_text = any(obj.kind == "text" for obj in cluster.objects)
		has_image = any(obj.kind == "image" for obj in cluster.objects)
		if has_image and not has_text:
			image_only_indices.append(index)
		if has_text and not has_image:
			text_only_indices.append(index)
		if has_text:
			text_indices.append(index)

	if not image_only_indices or not text_indices:
		return clusters

	centers = []
	for cluster in clusters:
		center_x = cluster.min_x + cluster.width / 2.0
		center_y = cluster.min_y + cluster.height / 2.0
		centers.append((center_x, center_y))

	max_dx = col_step * 0.75
	max_dy = row_step * 1.5

	for index in image_only_indices:
		center_x, center_y = centers[index]
		candidates = text_only_indices or text_indices
		best_index = None
		best_score = None
		for cand in candidates:
			text_x, text_y = centers[cand]
			dx = text_x - center_x
			dy = abs(text_y - center_y)
			if abs(dx) > max_dx or dy > max_dy:
				continue
			score = abs(dx) + dy * 1.5
			if dx < 0:
				score += abs(dx) * 0.5
			if best_score is None or score < best_score:
				best_score = score
				best_index = cand
		if best_index is None:
			continue
		clusters[best_index].objects.extend(clusters[index].objects)
		update_cluster_bounds(clusters[best_index])
		clusters[index].objects = []

	return [cluster for cluster in clusters if cluster.objects]


#============================================
def merge_adjacent_text_only_clusters(
	clusters: list[LabelCluster],
	row_step: float,
) -> list[LabelCluster]:
	"""
	Merge adjacent text-only clusters within a row.

	Args:
		clusters: Label clusters to merge.
		row_step: Row height in points.

	Returns:
		Updated list of LabelCluster entries.
	"""
	if not clusters or row_step <= 0.0:
		return clusters

	row_map: dict[int, list[LabelCluster]] = {}
	for cluster in clusters:
		center_y = cluster.min_y + cluster.height / 2.0
		row = int(round(center_y / row_step))
		row_map.setdefault(row, []).append(cluster)

	removed_ids: set[int] = set()
	max_gap = row_step * 0.5

	for row_clusters in row_map.values():
		row_sorted = sorted(row_clusters, key=lambda item: item.min_x)
		index = 0
		while index < len(row_sorted) - 1:
			left = row_sorted[index]
			right = row_sorted[index + 1]
			if id(left) in removed_ids or id(right) in removed_ids:
				index += 1
				continue
			left_texts = [
				obj for obj in left.objects
				if obj.kind == "text" and (obj.text or "").strip()
			]
			right_texts = [
				obj for obj in right.objects
				if obj.kind == "text" and (obj.text or "").strip()
			]
			left_images = [obj for obj in left.objects if obj.kind == "image"]
			right_images = [obj for obj in right.objects if obj.kind == "image"]
			if (
				len(left_texts) != 1
				or len(right_texts) != 1
				or left_images
				or right_images
			):
				index += 1
				continue
			gap = right.min_x - (left.min_x + left.width)
			if gap > max_gap:
				index += 1
				continue
			left_text = left_texts[0]
			right_text = right_texts[0]
			left_value = (left_text.text or "").strip()
			right_value = (right_text.text or "").strip()
			if not left_value or not right_value:
				index += 1
				continue
			if not (left_value[-1].isdigit() and right_value[0].isdigit()):
				index += 1
				continue
			merged_text = f"{left_value}-{right_value}"
			new_obj = dataclasses.replace(
				left_text,
				text=merged_text,
				width=(right_text.x + right_text.width) - left_text.x,
				height=max(left_text.height, right_text.height),
			)
			left.objects = [
				obj for obj in left.objects
				if obj.kind != "text"
			]
			left.objects.append(new_obj)
			update_cluster_bounds(left)
			removed_ids.add(id(right))
			index += 2

	if not removed_ids:
		return clusters

	return [cluster for cluster in clusters if id(cluster) not in removed_ids]


#============================================
def clone_clusters(clusters: list["LabelCluster"]) -> list["LabelCluster"]:
	"""
	Clone clusters for safe comparison.

	Args:
		clusters: Label clusters.

	Returns:
		Cloned list of clusters.
	"""
	result: list[LabelCluster] = []
	for cluster in clusters:
		result.append(
			LabelCluster(
				source_path=cluster.source_path,
				index=cluster.index,
				objects=list(cluster.objects),
				min_x=cluster.min_x,
				min_y=cluster.min_y,
				width=cluster.width,
				height=cluster.height,
			)
		)
	return result


#============================================
def match_image_text_clusters(
	clusters: list[LabelCluster],
	expected_labels: int,
) -> list[LabelCluster] | None:
	"""
	Match images and text within clusters.

	Args:
		clusters: Label clusters.
		expected_labels: Expected label count.

	Returns:
		Updated clusters or None if not improved.
	"""
	if not clusters or expected_labels <= 0:
		return None

	original = clone_clusters(clusters)
	improved = False

	for cluster in clusters:
		text_objects = [
			obj for obj in cluster.objects
			if obj.kind == "text" and (obj.text or "").strip()
		]
		image_objects = [obj for obj in cluster.objects if obj.kind == "image"]
		if not text_objects or not image_objects:
			continue
		objects_sorted = sorted(
			text_objects + image_objects,
			key=lambda item: item.x,
		)
		paired = build_pairs_by_text(
			objects_sorted,
			cluster.source_path,
			cluster.height if cluster.height > 0.0 else DEFAULT_LABEL_HEIGHT,
			cluster.min_y,
		)
		if not paired or len(paired) <= 1:
			continue
		improved = True
		cluster.objects = []
		for pair in paired:
			cluster.objects.extend(pair.objects)
		update_cluster_bounds(cluster)

	if not improved:
		return None

	current_score = score_label_set(clusters, expected_labels)
	original_score = score_label_set(original, expected_labels)
	if current_score <= original_score:
		return None
	return clusters


#============================================
def recursive_split_clusters(
	clusters: list[LabelCluster],
	median_height: float,
	depth: int,
) -> list[LabelCluster]:
	"""
	Split overly tall clusters recursively.

	Args:
		clusters: Label clusters.
		median_height: Median label height.
		depth: Remaining split depth.

	Returns:
		Updated list of clusters.
	"""
	if depth <= 0:
		return clusters
	if median_height <= 0.0:
		return clusters
	updated: list[LabelCluster] = []
	updated_any = False
	for cluster in clusters:
		if cluster.height <= median_height * BACKGROUND_SPAN_FACTOR:
			updated.append(cluster)
			continue
		split = split_cluster_by_gaps(cluster.objects, cluster.source_path)
		if len(split) <= 1:
			updated.append(cluster)
			continue
		if score_cluster_layout(split, len(split)) <= score_cluster_layout([cluster], 1):
			updated.append(cluster)
			continue
		updated.extend(split)
		updated_any = True

	if not updated_any:
		return clusters
	return recursive_split_clusters(updated, median_height, depth - 1)


#============================================
def build_clusters_from_background(
	objects: list[LabelObject],
	source_path: str,
	background: tuple[float, float, float, float],
	allow_multi_row: bool = False,
) -> list[LabelCluster]:
	"""
	Build label clusters using background grid.

	Args:
		objects: Label objects.
		source_path: Source path.
		background: Background bounds.
		allow_multi_row: Allow stacked rows within a background row.

	Returns:
		List of LabelCluster entries.
	"""
	if not objects:
		return []
	bg_x, bg_y, bg_width, bg_height = background
	min_x = min(obj.x for obj in objects)
	min_y = min(obj.y for obj in objects)
	max_x = max(obj.x + obj.width for obj in objects)
	max_y = max(obj.y + obj.height for obj in objects)
	span_x = max_x - min_x
	span_y = max_y - min_y
	if span_x <= bg_width * BACKGROUND_SPAN_FACTOR and span_y <= bg_height * BACKGROUND_SPAN_FACTOR:
		return []

	text_count = sum(1 for obj in objects if obj.kind == "text")
	image_count = sum(1 for obj in objects if obj.kind == "image")
	expected_labels = max(text_count, image_count)

	row_multipliers = [1]
	if allow_multi_row:
		max_multiplier = int(span_y // bg_height) if bg_height > 0 else 1
		for multiplier in (2, 3):
			if multiplier <= max_multiplier:
				row_multipliers.append(multiplier)

	col_multipliers = [1]
	if span_x > bg_width * (BACKGROUND_SPAN_FACTOR + 0.1):
		col_multipliers.append(2)

	best_clusters: list[LabelCluster] = []
	best_score = None
	for row_multiplier in row_multipliers:
		row_step = bg_height * row_multiplier
		origin_y_values = [bg_y, bg_y + row_step / 2.0]
		for col_multiplier in col_multipliers:
			col_step = bg_width * col_multiplier
			origin_x_values = [bg_x, bg_x + col_step / 2.0]
			for origin_y in origin_y_values:
				for origin_x in origin_x_values:
					clusters = cluster_objects_by_grid(
						objects,
						source_path,
						origin_x,
						origin_y,
						col_step,
						row_step,
					)
					if not clusters:
						continue
					merged = merge_image_only_clusters(clusters, row_step, col_step)
					score = score_cluster_layout(merged, expected_labels)
					if best_score is None or score > best_score:
						best_score = score
						best_clusters = merged

	return best_clusters


#============================================
def split_objects_by_separators(
	objects: list[LabelObject],
	separators: list[float],
	axis: str = "x",
) -> list[list[LabelObject]]:
	"""
	Split objects into bins using separator positions.

	Args:
		objects: Label objects.
		separators: Separator positions.
		axis: Axis to split by ("x" or "y").

	Returns:
		List of object clusters.
	"""
	if not separators:
		return [objects] if objects else []
	use_x = axis != "y"
	centers = []
	for obj in objects:
		center = (obj.x + obj.width / 2.0) if use_x else (obj.y + obj.height / 2.0)
		centers.append(center)
	bins: list[list[LabelObject]] = [[] for _ in range(len(separators) + 1)]
	for obj, center in zip(objects, centers):
		index = 0
		while index < len(separators) and center >= separators[index]:
			index += 1
		bins[index].append(obj)
	return [bucket for bucket in bins if bucket]


#============================================
def compute_gap_threshold(
	objects: list[LabelObject],
	min_threshold: float,
	axis: str = "x",
) -> float:
	"""
	Compute a gap threshold from object spacing.

	Args:
		objects: List of LabelObject entries.
		min_threshold: Minimum gap threshold.
		axis: Axis to analyze ("x" or "y").

	Returns:
		Gap threshold in points.
	"""
	if len(objects) < 2:
		return min_threshold

	use_x = axis != "y"
	objects_sorted = sorted(objects, key=lambda item: item.x if use_x else item.y)
	gaps: list[float] = []
	for index in range(1, len(objects_sorted)):
		prev = objects_sorted[index - 1]
		cur = objects_sorted[index]
		prev_end = (prev.x + prev.width) if use_x else (prev.y + prev.height)
		cur_start = cur.x if use_x else cur.y
		gap = cur_start - prev_end
		if gap > 0:
			gaps.append(gap)
	if not gaps:
		return min_threshold

	median_gap = statistics.median(gaps)
	trim_limit = median_gap * 2.0
	trimmed_gaps = [gap for gap in gaps if gap <= trim_limit]
	gaps_for_cluster = gaps
	if len(trimmed_gaps) >= 2:
		gaps_for_cluster = trimmed_gaps

	if len(gaps_for_cluster) < 4:
		threshold = max(min_threshold, median_gap * 2.0)
		return threshold

	cluster_a = min(gaps_for_cluster)
	cluster_b = max(gaps_for_cluster)
	for _ in range(10):
		group_a: list[float] = []
		group_b: list[float] = []
		for gap in gaps_for_cluster:
			if abs(gap - cluster_a) <= abs(gap - cluster_b):
				group_a.append(gap)
			else:
				group_b.append(gap)
		if not group_a or not group_b:
			break
		cluster_a = sum(group_a) / len(group_a)
		cluster_b = sum(group_b) / len(group_b)

	low, high = sorted([cluster_a, cluster_b])
	if high <= low * 1.5:
		threshold = max(min_threshold, median_gap * 2.0)
		return threshold

	threshold = (low + high) / 2.0
	return max(min_threshold, threshold)


#============================================
def cluster_objects(
	objects: list[LabelObject],
	gap_threshold: float,
	axis: str = "x",
) -> list[list[LabelObject]]:
	"""
	Group objects into clusters based on axis gaps.

	Args:
		objects: List of LabelObject entries.
		gap_threshold: Gap threshold in points.
		axis: Axis to cluster by ("x" or "y").

	Returns:
		List of object clusters.
	"""
	if not objects:
		return []
	use_x = axis != "y"
	objects_sorted = sorted(objects, key=lambda item: item.x if use_x else item.y)
	clusters: list[list[LabelObject]] = []
	current: list[LabelObject] = []
	current_max_x = None

	for item in objects_sorted:
		if not current:
			current = [item]
			current_max_x = (item.x + item.width) if use_x else (item.y + item.height)
			continue
		position = item.x if use_x else item.y
		extent = (item.x + item.width) if use_x else (item.y + item.height)
		gap = position - current_max_x
		if gap > gap_threshold:
			clusters.append(current)
			current = [item]
			current_max_x = extent
			continue
		current.append(item)
		current_max_x = max(current_max_x, extent)

	if current:
		clusters.append(current)
	return clusters


#============================================
def compute_periodicity_step(values: list[float]) -> tuple[float | None, float]:
	"""
	Compute a dominant periodic step from a list of values.

	Args:
		values: Sorted center positions.

	Returns:
		Tuple of (step, confidence).
	"""
	if len(values) < 3:
		return (None, 0.0)
	ordered = sorted(values)
	deltas = []
	for index in range(1, len(ordered)):
		delta = ordered[index] - ordered[index - 1]
		if delta > 0:
			deltas.append(delta)
	if len(deltas) < PERIODICITY_MIN_DELTAS:
		return (None, 0.0)

	bins: dict[float, list[float]] = {}
	for delta in deltas:
		bucket = round(delta / PERIODICITY_BIN_SIZE) * PERIODICITY_BIN_SIZE
		bins.setdefault(bucket, []).append(delta)

	best_bucket = None
	best_count = 0
	for bucket, members in bins.items():
		if len(members) > best_count:
			best_bucket = bucket
			best_count = len(members)
	if best_bucket is None:
		return (None, 0.0)

	confidence = best_count / len(deltas)
	step = sum(bins[best_bucket]) / len(bins[best_bucket])
	if step < PERIODICITY_MIN_STEP:
		return (None, confidence)
	return (step, confidence)


#============================================
def compute_periodicity_offset(values: list[float], step: float) -> float:
	"""
	Compute a stable offset for periodic binning.

	Args:
		values: Center positions.
		step: Periodic step size.

	Returns:
		Offset in points.
	"""
	if step <= 0.0:
		return 0.0
	remainders = [value % step for value in values]
	if not remainders:
		return 0.0
	return statistics.median(sorted(remainders))


#============================================
def split_cluster_by_gaps(
	objects: list[LabelObject],
	source_path: str,
) -> list[LabelCluster]:
	"""
	Split a set of objects using gap-based clustering.

	Args:
		objects: Label objects.
		source_path: Source LBX path.

	Returns:
		List of label clusters.
	"""
	if not objects:
		return []
	row_threshold = compute_gap_threshold(objects, DEFAULT_GAP_THRESHOLD, axis="y")
	row_clusters = cluster_objects(objects, row_threshold, axis="y")
	label_clusters: list[LabelCluster] = []
	cluster_index = 1
	for row in row_clusters:
		col_threshold = compute_gap_threshold(row, DEFAULT_GAP_THRESHOLD, axis="x")
		col_clusters = cluster_objects(row, col_threshold, axis="x")
		for cluster in col_clusters:
			label_cluster = create_label_cluster(cluster, source_path, cluster_index)
			if label_cluster is not None:
				label_clusters.append(label_cluster)
				cluster_index += 1
	return label_clusters


#============================================
def cluster_objects_by_periodicity(
	objects: list[LabelObject],
	step: float,
	offset: float,
	axis: str = "x",
) -> list[list[LabelObject]]:
	"""
	Cluster objects by snapping centers to a periodic grid.

	Args:
		objects: Label objects.
		step: Periodic step size.
		offset: Periodic offset.
		axis: Axis to cluster by ("x" or "y").

	Returns:
		List of object clusters.
	"""
	if not objects or step <= 0.0:
		return []
	use_x = axis != "y"
	bins: dict[int, list[LabelObject]] = {}
	for obj in objects:
		center = (obj.x + obj.width / 2.0) if use_x else (obj.y + obj.height / 2.0)
		index = int(round((center - offset) / step))
		bins.setdefault(index, []).append(obj)
	return [bins[key] for key in sorted(bins.keys())]


#============================================
def create_label_cluster(
	objects: list[LabelObject],
	source_path: str,
	index: int,
) -> LabelCluster | None:
	"""
	Create a label cluster from objects.

	Args:
		objects: Label objects.
		source_path: Source LBX path.
		index: Cluster index.

	Returns:
		LabelCluster or None if objects are empty.
	"""
	if not objects:
		return None
	min_x = min(item.x for item in objects)
	min_y = min(item.y for item in objects)
	max_x = max(item.x + item.width for item in objects)
	max_y = max(item.y + item.height for item in objects)
	return LabelCluster(
		source_path=source_path,
		index=index,
		objects=objects,
		min_x=min_x,
		min_y=min_y,
		width=max_x - min_x,
		height=max_y - min_y,
	)


#============================================
def build_label_clusters(
	objects: list[LabelObject],
	source_path: str,
	gap_threshold: float,
	vertical_separators: list[float] | None = None,
	horizontal_separators: list[float] | None = None,
	use_periodicity: bool = True,
) -> list[LabelCluster]:
	"""
	Build label clusters from objects.

	Args:
		objects: List of LabelObject entries.
		source_path: Source LBX path.
		gap_threshold: Gap threshold in points.
		vertical_separators: Optional vertical separator positions.
		horizontal_separators: Optional horizontal separator positions.
		use_periodicity: Whether to attempt periodic clustering.

	Returns:
		List of LabelCluster entries.
	"""
	if not objects:
		return []

	row_clusters = split_objects_by_separators(objects, horizontal_separators or [], axis="y")
	if horizontal_separators or vertical_separators:
		if not horizontal_separators:
			row_threshold = compute_gap_threshold(objects, DEFAULT_GAP_THRESHOLD, axis="y")
			row_clusters = cluster_objects(objects, row_threshold, axis="y")
		label_clusters: list[LabelCluster] = []
		cluster_index = 1
		for row in row_clusters:
			col_clusters = split_objects_by_separators(row, vertical_separators or [], axis="x")
			if not vertical_separators:
				col_threshold = compute_gap_threshold(row, gap_threshold, axis="x")
				col_clusters = cluster_objects(row, col_threshold, axis="x")
			for cluster in col_clusters:
				label_cluster = create_label_cluster(cluster, source_path, cluster_index)
				if label_cluster is not None:
					label_clusters.append(label_cluster)
					cluster_index += 1
		return label_clusters

	text_count = sum(1 for obj in objects if obj.kind == "text")
	image_count = sum(1 for obj in objects if obj.kind == "image")
	expected_labels = max(text_count, image_count)

	row_threshold = compute_gap_threshold(objects, DEFAULT_GAP_THRESHOLD, axis="y")
	gap_row_clusters = cluster_objects(objects, row_threshold, axis="y")
	gap_clusters: list[LabelCluster] = []
	cluster_index = 1
	for row in gap_row_clusters:
		col_threshold = compute_gap_threshold(row, gap_threshold, axis="x")
		col_clusters = cluster_objects(row, col_threshold, axis="x")
		for cluster in col_clusters:
			label_cluster = create_label_cluster(cluster, source_path, cluster_index)
			if label_cluster is not None:
				gap_clusters.append(label_cluster)
				cluster_index += 1

	if use_periodicity:
		centers_x = [obj.x + obj.width / 2.0 for obj in objects]
		step_x, confidence_x = compute_periodicity_step(centers_x)
		if step_x is None or confidence_x < PERIODICITY_CONFIDENCE_MIN:
			step_x = None
		else:
			offset_x = compute_periodicity_offset(centers_x, step_x)

		centers_y = [obj.y + obj.height / 2.0 for obj in objects]
		step_y, confidence_y = compute_periodicity_step(centers_y)
		if step_y is None or confidence_y < PERIODICITY_CONFIDENCE_MIN:
			step_y = None
		else:
			offset_y = compute_periodicity_offset(centers_y, step_y)

		periodic_clusters: list[LabelCluster] | None = None
		if step_y is not None:
			row_clusters = cluster_objects_by_periodicity(objects, step_y, offset_y, axis="y")
			periodic_clusters = []
			cluster_index = 1
			for row in row_clusters:
				if step_x is not None:
					col_clusters = cluster_objects_by_periodicity(
						row,
						step_x,
						offset_x,
						axis="x",
					)
				else:
					col_threshold = compute_gap_threshold(row, gap_threshold, axis="x")
					col_clusters = cluster_objects(row, col_threshold, axis="x")
				for cluster in col_clusters:
					label_cluster = create_label_cluster(cluster, source_path, cluster_index)
					if label_cluster is not None:
						periodic_clusters.append(label_cluster)
						cluster_index += 1

		if periodic_clusters:
			periodic_score = score_cluster_layout(periodic_clusters, expected_labels)
			gap_score = score_cluster_layout(gap_clusters, expected_labels)
			if periodic_score > gap_score:
				return periodic_clusters

	return gap_clusters


#============================================
def compute_sha256(path: pathlib.Path) -> str:
	"""
	Compute SHA256 hash for a file.

	Args:
		path: File path.

	Returns:
		Hex digest.
	"""
	hasher = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


#============================================
def should_split_group(objects: list[LabelObject]) -> bool:
	"""
	Decide whether a group should be split into multiple label clusters.

	Args:
		objects: Group objects.

	Returns:
		True when the group likely contains multiple labels.
	"""
	text_count = sum(
		1 for obj in objects
		if obj.kind == "text" and (obj.text or "").strip()
	)
	return text_count >= GROUP_SPLIT_TEXT_THRESHOLD


#============================================
def split_cluster_by_text_pairs(cluster: LabelCluster) -> list[LabelCluster]:
	"""
	Split a cluster into multiple labels using image-to-text pairing.

	Args:
		cluster: Cluster to split.

	Returns:
		List of split clusters or the original cluster if no split applied.
	"""
	text_objects = [
		obj for obj in cluster.objects
		if obj.kind == "text" and (obj.text or "").strip()
	]
	image_objects = [obj for obj in cluster.objects if obj.kind == "image"]
	if not text_objects or len(image_objects) < 2:
		return [cluster]

	visual_objects = text_objects + image_objects
	row_step = cluster.height if cluster.height > 0.0 else DEFAULT_LABEL_HEIGHT
	paired = build_pairs_by_text(
		visual_objects,
		cluster.source_path,
		row_step,
		cluster.min_y,
	)
	if not paired or len(paired) <= 1:
		return [cluster]

	return paired


#============================================
def merge_text_only_clusters_into_image_clusters(
	clusters: list[LabelCluster],
	background: tuple[float, float, float, float],
) -> list[LabelCluster]:
	"""
	Merge text-only clusters into image-only clusters within the same row.

	Args:
		clusters: Label clusters for a single LBX file.
		background: Background bounds (x, y, width, height).

	Returns:
		Updated list of clusters.
	"""
	if not clusters:
		return clusters

	_bg_x, bg_y, _bg_width, bg_height = background
	if bg_height <= 0.0:
		return clusters

	def has_rect(cluster: LabelCluster) -> bool:
		return any(obj.kind == "rect" for obj in cluster.objects)

	cluster_info: list[dict[str, object]] = []
	for index, cluster in enumerate(clusters):
		has_image = any(obj.kind == "image" for obj in cluster.objects)
		has_text = any(
			obj.kind == "text" and (obj.text or "").strip()
			for obj in cluster.objects
		)
		center_x = cluster.min_x + cluster.width / 2.0
		center_y = cluster.min_y + cluster.height / 2.0
		row = int((center_y - bg_y) // bg_height)
		cluster_info.append(
			{
				"index": index,
				"cluster": cluster,
				"has_image": has_image,
				"has_text": has_text,
				"center_x": center_x,
				"row": row,
				"has_rect": has_rect(cluster),
			}
		)

	rows: dict[int, list[dict[str, object]]] = {}
	for info in cluster_info:
		rows.setdefault(info["row"], []).append(info)

	removed_indices: set[int] = set()
	max_dx = bg_height * 4.0

	for row, items in rows.items():
		image_clusters = [
			item for item in items
			if item["has_image"] and not item["has_text"]
		]
		text_clusters = [
			item for item in items
			if item["has_text"] and not item["has_image"] and not item["has_rect"]
		]
		if not image_clusters or not text_clusters:
			continue
		image_clusters_sorted = sorted(image_clusters, key=lambda item: item["center_x"])  # type: ignore[arg-type]
		for text_info in sorted(text_clusters, key=lambda item: item["center_x"]):  # type: ignore[arg-type]
			text_center_x = float(text_info["center_x"])  # type: ignore[arg-type]
			best = None
			best_dx = None
			for image_info in image_clusters_sorted:
				image_center_x = float(image_info["center_x"])  # type: ignore[arg-type]
				dx = abs(image_center_x - text_center_x)
				if dx > max_dx:
					continue
				if best_dx is None or dx < best_dx:
					best_dx = dx
					best = image_info
			if best is None:
				continue
			best_cluster = best["cluster"]
			if isinstance(best_cluster, LabelCluster):
				best_cluster.objects.extend(text_info["cluster"].objects)  # type: ignore[arg-type]
				update_cluster_bounds(best_cluster)
				removed_indices.add(int(text_info["index"]))  # type: ignore[arg-type]

	if not removed_indices:
		return clusters

	return [
		cluster for idx, cluster in enumerate(clusters)
		if idx not in removed_indices
	]


#============================================
def write_label_count_log(
	counts_by_file: dict[str, int],
	output_path: pathlib.Path,
) -> None:
	"""
	Write label count log for small label sets.

	Args:
		counts_by_file: Label count per LBX file.
		output_path: Output PDF path.
	"""
	lines = []
	for path, count in sorted(counts_by_file.items()):
		if count < 30:
			lines.append(f"{count}\t{path}")
	if not lines:
		return
	log_path = output_path.parent / "label_counts.log"
	with open(log_path, "w", encoding="utf-8") as handle:
		handle.write("count\tpath\n")
		handle.write("\n".join(lines))
		handle.write("\n")
	print(f"Label count log written: {log_path}")


#============================================
def gather_lbx_paths(inputs: list[str]) -> list[pathlib.Path]:
	"""
	Gather LBX paths from input paths.

	Args:
		inputs: Input paths.

	Returns:
		Sorted list of LBX file paths.
	"""
	paths: list[pathlib.Path] = []
	for entry in inputs:
		path = pathlib.Path(entry).expanduser().resolve()
		if path.is_dir():
			paths.extend(sorted(path.rglob("*.lbx")))
			continue
		if path.is_file() and path.suffix.lower() == ".lbx":
			paths.append(path)
	def sort_key(path: pathlib.Path) -> tuple[int, str, str]:
		category_num = 999
		category_name = ""
		for part in path.parts:
			if "." in part:
				prefix, _, suffix = part.partition(".")
				if prefix.isdigit():
					category_num = int(prefix)
					category_name = suffix.lower()
					break
		return (category_num, category_name, path.stem.lower())

	paths_sorted = sorted(paths, key=sort_key)
	return paths_sorted


#============================================
def collect_labels(
	paths: list[pathlib.Path],
	gap_threshold: float | None,
	apply_normalization: bool,
	max_labels: int | None = None,
	verbose: bool = False,
) -> tuple[list[LabelCluster], dict[str, int], dict[str, str], dict[str, float]]:
	"""
	Collect label clusters from LBX files.

	Args:
		paths: LBX file paths.
		gap_threshold: Optional gap threshold override.
		max_labels: Optional limit on total labels.

	Returns:
		Tuple of (labels, counts by file, hashes, thresholds by file).
	"""
	labels: list[LabelCluster] = []
	counts_by_file: dict[str, int] = {}
	hashes: dict[str, str] = {}
	thresholds: dict[str, float] = {}
	if verbose:
		print("Collecting labels")

	remaining_labels = max_labels
	total_paths = len(paths)
	for index, path in enumerate(paths, start=1):
		if remaining_labels is not None and remaining_labels <= 0:
			break
		with zipfile.ZipFile(path, "r") as archive:
			label_xml = archive.read("label.xml")
		group_clusters, loose_objects = parse_label_xml_with_groups(label_xml, apply_normalization)
		background_bounds = extract_background_bounds(label_xml)
		if path.stem in TEXT_MERGE_WHITELIST:
			group_clusters, loose_objects = merge_loose_text_into_image_groups(
				group_clusters,
				loose_objects,
				background_bounds,
			)
			group_clusters = merge_text_only_groups_into_image_groups(
				group_clusters,
				background_bounds,
			)
		label_clusters: list[LabelCluster] = []

		cluster_index = 1
		category_title = format_category_title(path.stem)
		if apply_normalization:
			category_title = normalize_text(category_title)
		label_clusters.append(
			build_category_label(str(path), category_title, cluster_index)
		)
		cluster_index += 1
		for group_objects in group_clusters:
			visual_group = [obj for obj in group_objects if obj.kind in ("text", "image")]
			if not visual_group:
				continue
			group_clusters_split: list[LabelCluster] | None = None
			if should_split_group(group_objects):
				group_threshold = compute_gap_threshold(
					visual_group,
					DEFAULT_GAP_THRESHOLD,
				)
				vertical_separators, horizontal_separators = find_separators(group_objects)
				group_clusters_split = build_label_clusters(
					visual_group,
					str(path),
					group_threshold,
					vertical_separators=vertical_separators,
					horizontal_separators=horizontal_separators,
					use_periodicity=False,
				)
			if group_clusters_split and len(group_clusters_split) > 1:
				for cluster in group_clusters_split:
					cluster.index = cluster_index
					cluster_index += 1
					label_clusters.append(cluster)
			else:
				label_cluster = create_label_cluster(group_objects, str(path), cluster_index)
				if label_cluster is not None:
					label_clusters.append(label_cluster)
					cluster_index += 1

		visual_objects = [obj for obj in loose_objects if obj.kind in ("text", "image")]
		vertical_separators, horizontal_separators = find_separators(loose_objects)
		threshold = gap_threshold
		if threshold is None:
			threshold = compute_gap_threshold(visual_objects, DEFAULT_GAP_THRESHOLD)
		thresholds[str(path)] = threshold

		loose_clusters = []
		if background_bounds is not None and not vertical_separators and not horizontal_separators:
			loose_clusters = build_clusters_from_background(
				visual_objects,
				str(path),
				background_bounds,
				allow_multi_row=path.stem in ROW_STACK_WHITELIST,
			)
		if not loose_clusters:
			loose_clusters = build_label_clusters(
				visual_objects,
				str(path),
				threshold,
				vertical_separators=vertical_separators,
				horizontal_separators=horizontal_separators,
			)
		if not vertical_separators and not horizontal_separators:
			text_count = sum(
				1 for obj in visual_objects
				if obj.kind == "text" and (obj.text or "").strip()
			)
			image_count = sum(1 for obj in visual_objects if obj.kind == "image")
			expected_labels = max(text_count, image_count)
			matched = match_image_text_clusters(loose_clusters, expected_labels)
			if matched is not None:
				loose_clusters = matched
		if (
			background_bounds is not None
			and not vertical_separators
			and not horizontal_separators
			and path.stem in PAIRING_WHITELIST
		):
			text_count = sum(
				1 for obj in visual_objects
				if obj.kind == "text" and (obj.text or "").strip()
			)
			image_count = sum(1 for obj in visual_objects if obj.kind == "image")
			if text_count == image_count and text_count > 0:
				_bg_x, bg_y, _bg_width, bg_height = background_bounds
				paired = build_pairs_by_text(
					visual_objects,
					str(path),
					bg_height,
					bg_y,
				)
				if paired:
					current_score = score_label_set(loose_clusters, text_count)
					paired_score = score_label_set(paired, text_count)
					if paired_score > current_score:
						loose_clusters = paired
		if (
			background_bounds is not None
			and not vertical_separators
			and not horizontal_separators
			and path.stem not in ROW_STACK_WHITELIST
		):
			text_count = sum(
				1 for obj in visual_objects
				if obj.kind == "text" and (obj.text or "").strip()
			)
			image_count = sum(1 for obj in visual_objects if obj.kind == "image")
			if text_count == image_count and text_count > 0:
				missing_text, image_after_text = summarize_label_warnings(loose_clusters)
				if missing_text > 0 or image_after_text > 0:
					_bg_x, bg_y, _bg_width, bg_height = background_bounds
					paired = build_pairs_by_text(
						visual_objects,
						str(path),
						bg_height,
						bg_y,
					)
					if paired:
						current_score = score_label_set(loose_clusters, text_count)
						paired_score = score_label_set(paired, text_count)
						if paired_score > current_score:
							loose_clusters = paired
		if (
			background_bounds is not None
			and not vertical_separators
			and not horizontal_separators
		):
			missing_text, _image_after = summarize_label_warnings(loose_clusters)
			if missing_text > 0:
				text_count = sum(
					1 for obj in visual_objects
					if obj.kind == "text" and (obj.text or "").strip()
				)
				image_count = sum(1 for obj in visual_objects if obj.kind == "image")
				expected_labels = max(text_count, image_count)
				row_step = background_bounds[3]
				col_step = background_bounds[2]
				merged = merge_image_only_clusters(
					clone_clusters(loose_clusters),
					row_step,
					col_step,
				)
				current_score = score_label_set(loose_clusters, expected_labels)
				merged_score = score_label_set(merged, expected_labels)
				if merged_score > current_score:
					loose_clusters = merged
		if loose_clusters:
			heights = sorted(cluster.height for cluster in loose_clusters)
			median_height = heights[len(heights) // 2]
			loose_clusters = recursive_split_clusters(
				loose_clusters,
				median_height,
				RECURSIVE_SPLIT_MAX_DEPTH,
			)
			if path.stem in TEXT_ONLY_ADJACENT_MERGE_WHITELIST:
				row_step = median_height
				if background_bounds is not None and background_bounds[3] > 0.0:
					row_step = background_bounds[3]
				loose_clusters = merge_adjacent_text_only_clusters(
					loose_clusters,
					row_step,
				)
		for cluster in loose_clusters:
			has_visual = any(obj.kind in ("text", "image") for obj in cluster.objects)
			if not has_visual:
				continue
			cluster.index = cluster_index
			cluster_index += 1
			label_clusters.append(cluster)

		if path.stem in MULTI_IMAGE_SPLIT_WHITELIST:
			updated_clusters: list[LabelCluster] = []
			cluster_index = 1
			for cluster in label_clusters:
				parts = split_cluster_by_text_pairs(cluster)
				if len(parts) > 1:
					for part in parts:
						part.index = cluster_index
						cluster_index += 1
						updated_clusters.append(part)
				else:
					cluster.index = cluster_index
					cluster_index += 1
					updated_clusters.append(cluster)
			label_clusters = updated_clusters
			if background_bounds is not None:
				label_clusters = merge_text_only_clusters_into_image_clusters(
					label_clusters,
					background_bounds,
				)

		if remaining_labels is not None and len(label_clusters) > remaining_labels:
			label_clusters = label_clusters[:remaining_labels]

		counts_by_file[str(path)] = len(label_clusters)
		labels.extend(label_clusters)
		hashes[str(path)] = compute_sha256(path)
		if remaining_labels is not None:
			remaining_labels -= len(label_clusters)
		if verbose and (index % 10 == 0 or index == total_paths):
			print(f"Processed {index} of {total_paths} LBX files")

	return (labels, counts_by_file, hashes, thresholds)
