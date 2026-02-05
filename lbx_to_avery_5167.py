#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Brother P-touch LBX labels into Avery 5167 label sheets.
"""

import argparse
import dataclasses
import hashlib
import io
import json
import pathlib
import statistics
import unicodedata
import zipfile
import defusedxml.ElementTree as ElementTree
import xml.etree.ElementTree as StdElementTree

import PIL.Image
import pypdf
import reportlab.lib.pagesizes
import reportlab.lib.utils
import reportlab.pdfbase.pdfmetrics
import reportlab.pdfgen.canvas


POINTS_PER_INCH = 72.0
LABELS_PER_PAGE = 80
COLUMNS = 4
ROWS = 20

DEFAULT_LABEL_WIDTH = 125.95
DEFAULT_LABEL_HEIGHT = 35.95

DEFAULT_LEFT_MARGIN = 21.6
DEFAULT_TOP_MARGIN = 36.1
DEFAULT_H_GAP = 21.65
DEFAULT_V_GAP = 0.15
DEFAULT_INSET = 1.44
DEFAULT_GAP_THRESHOLD = 6.0

DEFAULT_FONT_REGULAR = "Helvetica"
DEFAULT_FONT_BOLD = "Helvetica-Bold"
DEFAULT_FONT_ITALIC = "Helvetica-Oblique"
DEFAULT_FONT_BOLD_ITALIC = "Helvetica-BoldOblique"
DEFAULT_TEXT_SIZE = 7.5
DEFAULT_TEXT_WEIGHT = 700
DEFAULT_TEXT_MIN_SIZE = 5.0
TEXT_OVERLAP_PADDING = 0.5
PROGRESS_BAR_WIDTH = 20
PROGRESS_UPDATE_EVERY = 10
CATEGORY_TEXT_SIZE = 11.0
CATEGORY_TEXT_MARGIN = 2.0
IMAGE_SCALE = 0.95
MAX_IMAGE_UPSCALE = 2.0
SEPARATOR_THICKNESS = 1.0
SEPARATOR_MIN_LENGTH = 8.0
BACKGROUND_SPAN_FACTOR = 1.6
GROUP_SPLIT_TEXT_THRESHOLD = 6
ROW_STACK_WHITELIST = {
	"CLIP-flexible",
	"OTHER-chain_string",
	"TECHNIC-mechanical_1",
	"TECHNIC-mechanical_2",
}
PAIRING_WHITELIST = {
	"MINIFIG-accessories-all",
	"MINIFIG-weapon_3",
}
TEXT_MERGE_WHITELIST = {
	"MINIFIG-accessories-all",
	"MINIFIG-CATEGORY-clothing_hair",
	"MINIFIG-hair-accessory",
	"MINIFIG-weapon_3",
	"NATURE-flower",
	"OTHER-shooter_1",
}


@dataclasses.dataclass
class LabelObject:
	kind: str
	x: float
	y: float
	width: float
	height: float
	text: str = ""
	font_name: str = ""
	font_size: float = 10.0
	font_weight: int = 400
	font_italic: bool = False
	use_text_override: bool = True
	align_horizontal: str = "LEFT"
	align_vertical: str = "TOP"
	text_color: str = "#000000"
	image_name: str = ""
	line_width: float = 0.3
	line_color: str = "#000000"
	fill_color: str = ""
	poly_points: list[tuple[float, float]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class LabelCluster:
	source_path: str
	index: int
	objects: list[LabelObject]
	min_x: float
	min_y: float
	width: float
	height: float


@dataclasses.dataclass
class ImpositionConfig:
	label_width: float
	label_height: float
	columns: int
	rows: int
	left_margin: float
	top_margin: float
	h_gap: float
	v_gap: float
	inset: float
	x_scale: float
	y_scale: float
	include_partial: bool
	calibration: bool
	draw_outlines: bool
	max_pages: int | None
	max_labels: int | None
	cluster_align_horizontal: str
	cluster_align_vertical: str
	text_align_horizontal: str | None
	text_align_vertical: str | None
	text_font_size: float | None
	text_font_weight: int | None
	text_fit: bool


@dataclasses.dataclass
class TileConfig:
	label_width: float
	label_height: float
	inset: float
	cluster_align_horizontal: str
	cluster_align_vertical: str
	text_align_horizontal: str | None
	text_align_vertical: str | None
	text_font_size: float | None
	text_font_weight: int | None
	text_fit: bool


@dataclasses.dataclass
class ImpositionResult:
	total_labels: int
	printed_labels: int
	leftover_labels: int
	pages: int
	labels_per_page: int


#============================================
def inches_to_points(value: float) -> float:
	"""
	Convert inches to points.

	Args:
		value: Inches value.

	Returns:
		Points value.
	"""
	return value * POINTS_PER_INCH


#============================================
def parse_pt_value(value: str, default_value: float) -> float:
	"""
	Parse a point string into a float.

	Args:
		value: String value like "10pt".
		default_value: Fallback when parsing fails.

	Returns:
		Parsed float value.
	"""
	if value is None:
		return default_value
	value = value.strip()
	if not value:
		return default_value
	if value.endswith("pt"):
		return float(value[:-2])
	return float(value)


def compute_align_offset(available: float, scaled: float, align: str) -> float:
	"""
	Compute an offset based on alignment.

	Args:
		available: Available length.
		scaled: Content length.
		align: Alignment string.

	Returns:
		Offset in points.
	"""
	normalized = align.strip().upper()
	if normalized in ("LEFT", "BOTTOM"):
		return 0.0
	if normalized in ("RIGHT", "TOP"):
		return max(0.0, available - scaled)
	return max(0.0, (available - scaled) / 2.0)


#============================================
def compute_text_bbox(
	x: float,
	baseline_y: float,
	text: str,
	font_name: str,
	font_size: float,
	padding: float,
) -> tuple[float, float, float, float]:
	"""
	Compute a text bounding box from a baseline position.

	Args:
		x: Text x position.
		baseline_y: Text baseline y position.
		text: Text content.
		font_name: ReportLab font name.
		font_size: Font size in points.
		padding: Padding for overlap detection.

	Returns:
		Bounding box (x0, y0, x1, y1).
	"""
	width = reportlab.pdfbase.pdfmetrics.stringWidth(text, font_name, font_size)
	ascent = reportlab.pdfbase.pdfmetrics.getAscent(font_name) * font_size / 1000.0
	descent = reportlab.pdfbase.pdfmetrics.getDescent(font_name) * font_size / 1000.0
	x0 = x - padding
	x1 = x + width + padding
	y0 = baseline_y + descent - padding
	y1 = baseline_y + ascent + padding
	return (x0, y0, x1, y1)


#============================================
def compute_text_visual_bounds(
	obj: "LabelObject",
	text_align_horizontal: str | None = None,
	text_align_vertical: str | None = None,
	font_size_override: float | None = None,
	font_weight_override: int | None = None,
	text_fit: bool = False,
	min_font_size: float | None = None,
) -> tuple[float, float, float, float] | None:
	"""
	Compute the visual text bounds for a text object.

	Args:
		obj: LabelObject text entry.
		text_align_horizontal: Optional horizontal alignment override.
		text_align_vertical: Optional vertical alignment override.
		font_size_override: Optional font size override.
		font_weight_override: Optional font weight override.
		text_fit: Whether to shrink text to fit the box.
		min_font_size: Minimum font size when shrinking.

	Returns:
		Bounding box (x0, y0, x1, y1) or None if no text.
	"""
	text_value = obj.text or ""
	lines = text_value.splitlines()
	if not lines:
		return None

	font_weight = obj.font_weight
	if obj.use_text_override and font_weight_override is not None:
		font_weight = font_weight_override
	font_name = map_font_name(obj.font_name, font_weight, obj.font_italic)
	font_size = obj.font_size
	if obj.use_text_override and font_size_override is not None:
		font_size = font_size_override

	align_h = obj.align_horizontal.upper()
	align_v = obj.align_vertical.upper()
	if text_align_horizontal is not None:
		align_h = text_align_horizontal.upper()
	if text_align_vertical is not None:
		align_v = text_align_vertical.upper()

	def compute_leading(size: float) -> float:
		return size * 1.2

	leading = compute_leading(font_size)
	text_height = font_size + leading * (len(lines) - 1)
	max_width = max(
		reportlab.pdfbase.pdfmetrics.stringWidth(line, font_name, font_size)
		for line in lines
	)
	if text_fit and (max_width > obj.width or text_height > obj.height):
		scale_width = obj.width / max_width if max_width > 0 else 1.0
		scale_height = obj.height / text_height if text_height > 0 else 1.0
		scale = min(1.0, scale_width, scale_height)
		if scale < 1.0:
			target_size = font_size * scale
			if min_font_size is not None and target_size < min_font_size:
				font_size = min_font_size
			else:
				font_size = target_size
			leading = compute_leading(font_size)
			text_height = font_size + leading * (len(lines) - 1)

	if align_v == "CENTER":
		base_y = obj.y + (obj.height - text_height) / 2.0
	elif align_v == "BOTTOM":
		base_y = obj.y
	else:
		base_y = obj.y + obj.height - text_height

	bounds: tuple[float, float, float, float] | None = None
	for index, line in enumerate(lines):
		line_width = reportlab.pdfbase.pdfmetrics.stringWidth(line, font_name, font_size)
		if align_h == "CENTER":
			text_x = obj.x + (obj.width - line_width) / 2.0
		elif align_h == "RIGHT":
			text_x = obj.x + obj.width - line_width
		else:
			text_x = obj.x
		text_y = base_y + index * leading
		bbox = compute_text_bbox(text_x, text_y, line, font_name, font_size, 0.0)
		if bounds is None:
			bounds = bbox
		else:
			bounds = (
				min(bounds[0], bbox[0]),
				min(bounds[1], bbox[1]),
				max(bounds[2], bbox[2]),
				max(bounds[3], bbox[3]),
			)
	return bounds


#============================================
def compute_visual_bounds(
	objects: list["LabelObject"],
	text_align_horizontal: str | None = None,
	text_align_vertical: str | None = None,
	font_size_override: float | None = None,
	font_weight_override: int | None = None,
	text_fit: bool = False,
	min_font_size: float | None = None,
) -> tuple[float, float, float, float] | None:
	"""
	Compute visual bounds for a list of objects.

	Args:
		objects: Label objects.
		text_align_horizontal: Optional horizontal alignment override.
		text_align_vertical: Optional vertical alignment override.
		font_size_override: Optional font size override.
		font_weight_override: Optional font weight override.
		text_fit: Whether to shrink text to fit the box.
		min_font_size: Minimum font size when shrinking.

	Returns:
		Bounding box (x0, y0, x1, y1) or None if no visual objects.
	"""
	bounds: tuple[float, float, float, float] | None = None
	for obj in objects:
		if obj.kind == "text":
			text_bounds = compute_text_visual_bounds(
				obj,
				text_align_horizontal=text_align_horizontal,
				text_align_vertical=text_align_vertical,
				font_size_override=font_size_override,
				font_weight_override=font_weight_override,
				text_fit=text_fit,
				min_font_size=min_font_size,
			)
			if text_bounds is None:
				continue
			x0, y0, x1, y1 = text_bounds
		elif obj.kind == "poly":
			if len(obj.poly_points) < 2:
				continue
			x_values = [point[0] for point in obj.poly_points]
			y_values = [point[1] for point in obj.poly_points]
			x0, y0, x1, y1 = min(x_values), min(y_values), max(x_values), max(y_values)
		else:
			x0 = obj.x
			y0 = obj.y
			x1 = obj.x + obj.width
			y1 = obj.y + obj.height
		if bounds is None:
			bounds = (x0, y0, x1, y1)
		else:
			bounds = (
				min(bounds[0], x0),
				min(bounds[1], y0),
				max(bounds[2], x1),
				max(bounds[3], y1),
			)
	return bounds


#============================================
def boxes_intersect(
	first: tuple[float, float, float, float],
	second: tuple[float, float, float, float],
) -> bool:
	"""
	Check whether two bounding boxes intersect.

	Args:
		first: First bounding box.
		second: Second bounding box.

	Returns:
		True if boxes intersect.
	"""
	if first[2] <= second[0]:
		return False
	if second[2] <= first[0]:
		return False
	if first[3] <= second[1]:
		return False
	if second[3] <= first[1]:
		return False
	return True


#============================================
def merge_positions(values: list[float], tolerance: float) -> list[float]:
	"""
	Merge nearby separator positions.

	Args:
		values: Raw separator positions.
		tolerance: Merge tolerance in points.

	Returns:
		Merged separator positions.
	"""
	if not values:
		return []
	positions = sorted(values)
	merged = [positions[0]]
	for value in positions[1:]:
		if abs(value - merged[-1]) <= tolerance:
			merged[-1] = (merged[-1] + value) / 2.0
			continue
		merged.append(value)
	return merged


#============================================
def print_progress(prefix: str, current: int, total: int) -> None:
	"""
	Print a single-line progress bar.

	Args:
		prefix: Progress prefix label.
		current: Current count.
		total: Total count.
	"""
	if total <= 0:
		return
	ratio = min(1.0, max(0.0, current / total))
	filled = int(ratio * PROGRESS_BAR_WIDTH)
	bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
	percent = int(ratio * 100)
	print(f"\r{prefix} [{bar}] {current}/{total} ({percent}%)", end="", flush=True)


#============================================
def find_separators(
	objects: list[LabelObject],
) -> tuple[list[float], list[float]]:
	"""
	Detect separator lines from poly objects.

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
def extract_background_bounds(
	label_xml: bytes,
) -> tuple[float, float, float, float] | None:
	"""
	Extract background bounds from label XML.

	Args:
		label_xml: XML bytes.

	Returns:
		Tuple of (x, y, width, height) or None.
	"""
	root = ElementTree.fromstring(label_xml)
	background = root.find(".//{*}backGround")
	if background is None:
		return None
	x = parse_pt_value(background.attrib.get("x"), 0.0)
	y = parse_pt_value(background.attrib.get("y"), 0.0)
	width = parse_pt_value(background.attrib.get("width"), 0.0)
	height = parse_pt_value(background.attrib.get("height"), 0.0)
	if width <= 0.0 or height <= 0.0:
		return None
	return (x, y, width, height)


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
	Cluster objects into grid cells using a fixed origin and step sizes.

	Args:
		objects: Label objects to cluster.
		source_path: Source LBX path.
		origin_x: Grid origin x.
		origin_y: Grid origin y.
		col_step: Column step width.
		row_step: Row step height.

	Returns:
		List of LabelCluster entries.
	"""
	if not objects or col_step <= 0.0 or row_step <= 0.0:
		return []
	bins: dict[tuple[int, int], list[LabelObject]] = {}
	for obj in objects:
		center_x = obj.x + obj.width / 2.0
		center_y = obj.y + obj.height / 2.0
		col = int((center_x - origin_x) // col_step)
		row = int((center_y - origin_y) // row_step)
		bins.setdefault((row, col), []).append(obj)

	label_clusters: list[LabelCluster] = []
	cluster_index = 1
	for key in sorted(bins.keys()):
		cluster = bins[key]
		label_cluster = create_label_cluster(cluster, source_path, cluster_index)
		if label_cluster is not None:
			label_clusters.append(label_cluster)
			cluster_index += 1
	return label_clusters


#============================================
def score_cluster_layout(
	clusters: list["LabelCluster"],
	expected_labels: int,
) -> float:
	"""
	Score a clustering layout to select the best grid settings.

	Args:
		clusters: Candidate clusters.
		expected_labels: Expected label count.

	Returns:
		Score value (higher is better).
	"""
	both_count = 0
	text_only = 0
	image_only = 0
	oversize = 0
	for cluster in clusters:
		has_text = any(obj.kind == "text" for obj in cluster.objects)
		has_image = any(obj.kind == "image" for obj in cluster.objects)
		if has_text and has_image:
			both_count += 1
		elif has_text:
			text_only += 1
		elif has_image:
			image_only += 1
		if len(cluster.objects) > 3:
			oversize += 1

	cluster_count = len(clusters)
	missing = max(0, expected_labels - cluster_count)
	extra = max(0, cluster_count - expected_labels)

	score = 0.0
	score += both_count * 5.0
	score -= image_only * 4.0
	score -= text_only * 1.0
	score -= oversize * 2.0
	score -= missing * 5.0
	score -= extra * 1.0
	return score


#============================================
def summarize_label_warnings(
	clusters: list["LabelCluster"],
) -> tuple[int, int]:
	"""
	Count missing-text and image-after-text warnings for labels.

	Args:
		clusters: Label clusters to inspect.

	Returns:
		Tuple of (missing_text_count, image_after_text_count).
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
			continue
		if text_objects and image_objects:
			min_text_x = min(obj.x for obj in text_objects)
			min_image_x = min(obj.x for obj in image_objects)
			if min_image_x > min_text_x:
				image_after_text += 1
	return (missing_text, image_after_text)


#============================================
def build_pairs_by_text(
	objects: list["LabelObject"],
	source_path: str,
	row_step: float,
	origin_y: float,
) -> list["LabelCluster"]:
	"""
	Pair images to text within a single row height.

	Args:
		objects: Label objects to pair.
		source_path: Source LBX path.
		row_step: Row height for grouping.
		origin_y: Row origin y.

	Returns:
		List of LabelCluster entries.
	"""
	if not objects or row_step <= 0.0:
		return []

	rows: dict[int, list[LabelObject]] = {}
	for obj in objects:
		center_y = obj.y + obj.height / 2.0
		row = int((center_y - origin_y) // row_step)
		rows.setdefault(row, []).append(obj)

	label_clusters: list[LabelCluster] = []
	cluster_index = 1
	for row in sorted(rows.keys()):
		row_objects = rows[row]
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
	Merge text-only groups into image-only groups within the same row.

	Args:
		group_clusters: Existing group clusters.
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
def build_clusters_from_background(
	objects: list[LabelObject],
	source_path: str,
	background: tuple[float, float, float, float],
	allow_multi_row: bool = True,
) -> list[LabelCluster]:
	"""
	Build label clusters based on background grid sizing.

	Args:
		objects: List of LabelObject entries.
		source_path: Source LBX path.
		background: Background bounds (x, y, width, height).

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
	separator_positions: list[float],
	axis: str,
) -> list[list[LabelObject]]:
	"""
	Split objects into bins based on separator positions.

	Args:
		objects: Label objects.
		separator_positions: Separator positions.
		axis: Axis to split by ("x" or "y").

	Returns:
		List of object bins.
	"""
	if not objects:
		return []
	if not separator_positions:
		return [objects]
	separators = sorted(separator_positions)
	bins: list[list[LabelObject]] = [[] for _ in range(len(separators) + 1)]
	use_x = axis != "y"
	for obj in objects:
		center = (obj.x + obj.width / 2.0) if use_x else (obj.y + obj.height / 2.0)
		index = 0
		while index < len(separators) and center >= separators[index]:
			index += 1
		bins[index].append(obj)
	return [bucket for bucket in bins if bucket]


#============================================
def parse_hex_color(value: str) -> tuple[float, float, float]:
	"""
	Parse a hex color string into RGB floats.

	Args:
		value: Color string like "#AABBCC".

	Returns:
		Tuple of (r, g, b) in 0.0-1.0 range.
	"""
	if not value or not value.startswith("#") or len(value) != 7:
		return (0.0, 0.0, 0.0)
	red = int(value[1:3], 16) / 255.0
	green = int(value[3:5], 16) / 255.0
	blue = int(value[5:7], 16) / 255.0
	return (red, green, blue)


#============================================
def find_first_child(
	element: StdElementTree.Element,
	suffix: str,
) -> StdElementTree.Element | None:
	"""
	Find the first child element that ends with the given suffix.

	Args:
		element: XML element to search.
		suffix: Tag suffix to match.

	Returns:
		Matching child element or None.
	"""
	for child in list(element):
		if child.tag.endswith(suffix):
			return child
	return None


#============================================
def parse_text_element(element: StdElementTree.Element) -> LabelObject | None:
	"""
	Parse a text element into a LabelObject.

	Args:
		element: XML element.

	Returns:
		LabelObject or None.
	"""
	style = find_first_child(element, "objectStyle")
	if style is None:
		return None
	x = parse_pt_value(style.attrib.get("x"), 0.0)
	y = parse_pt_value(style.attrib.get("y"), 0.0)
	width = parse_pt_value(style.attrib.get("width"), 0.0)
	height = parse_pt_value(style.attrib.get("height"), 0.0)

	font_info = find_first_child(element, "ptFontInfo")
	log_font = None
	font_ext = None
	if font_info is not None:
		log_font = find_first_child(font_info, "logFont")
		font_ext = find_first_child(font_info, "fontExt")

	font_name = "Arial"
	font_weight = 400
	font_italic = False
	font_size = 10.0
	text_color = "#000000"

	if log_font is not None:
		font_name = log_font.attrib.get("name", font_name)
		font_weight = int(log_font.attrib.get("weight", str(font_weight)))
		font_italic = log_font.attrib.get("italic", "false").lower() == "true"

	if font_ext is not None:
		font_size = parse_pt_value(font_ext.attrib.get("size"), font_size)
		text_color = font_ext.attrib.get("textColor", text_color)

	text_data = ""
	data_elem = find_first_child(element, "data")
	if data_elem is not None and data_elem.text is not None:
		text_data = data_elem.text

	align_horizontal = "LEFT"
	align_vertical = "TOP"
	align_elem = find_first_child(element, "textAlign")
	if align_elem is not None:
		align_horizontal = align_elem.attrib.get("horizontalAlignment", align_horizontal)
		align_vertical = align_elem.attrib.get("verticalAlignment", align_vertical)

	label_object = LabelObject(
		kind="text",
		x=x,
		y=y,
		width=width,
		height=height,
		text=text_data,
		font_name=font_name,
		font_size=font_size,
		font_weight=font_weight,
		font_italic=font_italic,
		align_horizontal=align_horizontal,
		align_vertical=align_vertical,
		text_color=text_color,
	)
	return label_object


#============================================
def parse_image_element(element: StdElementTree.Element) -> LabelObject | None:
	"""
	Parse an image element into a LabelObject.

	Args:
		element: XML element.

	Returns:
		LabelObject or None.
	"""
	style = find_first_child(element, "objectStyle")
	if style is None:
		return None
	x = parse_pt_value(style.attrib.get("x"), 0.0)
	y = parse_pt_value(style.attrib.get("y"), 0.0)
	width = parse_pt_value(style.attrib.get("width"), 0.0)
	height = parse_pt_value(style.attrib.get("height"), 0.0)

	image_style = find_first_child(element, "imageStyle")
	image_name = ""
	if image_style is not None:
		image_name = image_style.attrib.get("fileName", "")

	if not image_name:
		return None

	label_object = LabelObject(
		kind="image",
		x=x,
		y=y,
		width=width,
		height=height,
		image_name=image_name,
	)
	return label_object


#============================================
def parse_poly_element(element: StdElementTree.Element) -> LabelObject | None:
	"""
	Parse a poly element into a LabelObject.

	Args:
		element: XML element.

	Returns:
		LabelObject or None.
	"""
	style = find_first_child(element, "objectStyle")
	if style is None:
		return None
	x = parse_pt_value(style.attrib.get("x"), 0.0)
	y = parse_pt_value(style.attrib.get("y"), 0.0)
	width = parse_pt_value(style.attrib.get("width"), 0.0)
	height = parse_pt_value(style.attrib.get("height"), 0.0)

	line_width = 0.3
	line_color = "#000000"
	pen = find_first_child(style, "pen")
	if pen is not None:
		line_width = parse_pt_value(pen.attrib.get("widthX"), line_width)
		line_color = pen.attrib.get("color", line_color)

	poly_style = find_first_child(element, "polyStyle")
	if poly_style is None:
		return None
	points_elem = find_first_child(poly_style, "polyLinePoints")
	if points_elem is None:
		return None
	points_value = points_elem.attrib.get("points", "")
	if not points_value:
		return None

	points = []
	for token in points_value.split():
		coords = token.split(",")
		if len(coords) != 2:
			continue
		points.append(
			(
				parse_pt_value(coords[0], 0.0),
				parse_pt_value(coords[1], 0.0),
			)
		)

	if not points:
		return None

	label_object = LabelObject(
		kind="poly",
		x=x,
		y=y,
		width=width,
		height=height,
		line_width=line_width,
		line_color=line_color,
		poly_points=points,
	)
	return label_object


#============================================
def parse_label_xml(label_xml: bytes, apply_normalization: bool) -> list[LabelObject]:
	"""
	Parse label XML content into label objects.

	Args:
		label_xml: XML bytes.

	Returns:
		List of LabelObject entries.
	"""
	root = ElementTree.fromstring(label_xml)
	objects: list[LabelObject] = []
	for element in root.iter():
		if element.tag.endswith("text"):
			text_object = parse_text_element(element)
			if text_object is not None:
				if apply_normalization:
					text_object.text = normalize_text(text_object.text)
				objects.append(text_object)
			continue
		if element.tag.endswith("image"):
			image_object = parse_image_element(element)
			if image_object is not None:
				objects.append(image_object)
			continue
		if element.tag.endswith("poly"):
			poly_object = parse_poly_element(element)
			if poly_object is not None:
				objects.append(poly_object)
	return objects


#============================================
def parse_objects_in_container(
	element: StdElementTree.Element,
	apply_normalization: bool,
) -> list[LabelObject]:
	"""
	Parse label objects within a container element.

	Args:
		element: Container XML element.

	Returns:
		List of LabelObject entries.
	"""
	objects: list[LabelObject] = []
	for child in element.iter():
		if child is element:
			continue
		if child.tag.endswith("text"):
			text_object = parse_text_element(child)
			if text_object is not None:
				if apply_normalization:
					text_object.text = normalize_text(text_object.text)
				objects.append(text_object)
			continue
		if child.tag.endswith("image"):
			image_object = parse_image_element(child)
			if image_object is not None:
				objects.append(image_object)
			continue
		if child.tag.endswith("poly"):
			poly_object = parse_poly_element(child)
			if poly_object is not None:
				objects.append(poly_object)
	return objects


#============================================
def parse_label_xml_with_groups(
	label_xml: bytes,
	apply_normalization: bool,
) -> tuple[list[list[LabelObject]], list[LabelObject]]:
	"""
	Parse label XML into group clusters and loose objects.

	Args:
		label_xml: XML bytes.

	Returns:
		Tuple of (group_clusters, loose_objects).
	"""
	root = ElementTree.fromstring(label_xml)
	objects_root = None
	for element in root.iter():
		if element.tag.endswith("objects"):
			objects_root = element
			break
	if objects_root is None:
		return ([], parse_label_xml(label_xml, apply_normalization))

	group_clusters: list[list[LabelObject]] = []
	loose_objects: list[LabelObject] = []
	for child in list(objects_root):
		if child.tag.endswith("group"):
			group_objects = parse_objects_in_container(child, apply_normalization)
			if group_objects:
				group_clusters.append(group_objects)
			continue
		if child.tag.endswith("text"):
			text_object = parse_text_element(child)
			if text_object is not None:
				if apply_normalization:
					text_object.text = normalize_text(text_object.text)
				loose_objects.append(text_object)
			continue
		if child.tag.endswith("image"):
			image_object = parse_image_element(child)
			if image_object is not None:
				loose_objects.append(image_object)
			continue
		if child.tag.endswith("poly"):
			poly_object = parse_poly_element(child)
			if poly_object is not None:
				loose_objects.append(poly_object)
	return (group_clusters, loose_objects)


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
) -> list[LabelCluster]:
	"""
	Build label clusters from objects.

	Args:
		objects: List of LabelObject entries.
		source_path: Source LBX path.
		gap_threshold: Gap threshold in points.
		vertical_separators: Optional vertical separator positions.
		horizontal_separators: Optional horizontal separator positions.

	Returns:
		List of LabelCluster entries.
	"""
	row_clusters = split_objects_by_separators(objects, horizontal_separators or [], axis="y")
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
def sanitize_token(value: str) -> str:
	"""
	Sanitize a string for filenames.

	Args:
		value: Input string.

	Returns:
		Sanitized string.
	"""
	result: list[str] = []
	for char in value:
		if char.isalnum():
			result.append(char)
		else:
			result.append("_")
	sanitized = "".join(result).strip("_")
	if not sanitized:
		return "label"
	return sanitized


#============================================
def normalize_text(value: str) -> str:
	"""
	Normalize label text to ASCII for PDF fonts.

	Args:
		value: Input text.

	Returns:
		Normalized text.
	"""
	if not value:
		return value
	replacements = {
		"\u00d7": "x",
		"\u00f7": "/",
		"\u00d8": "O",
		"\u00f8": "o",
		"\u00bd": "1/2",
		"\u00bc": "1/4",
		"\u00be": "3/4",
		"\u00b0": "deg",
		"\u2122": "TM",
		"\u00ae": "R",
		"\u00a0": " ",
	}
	for old, new in replacements.items():
		value = value.replace(old, new)
	value = unicodedata.normalize("NFKD", value)
	value = value.encode("ascii", "ignore").decode("ascii")
	return value


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
						paired_missing, paired_image_after = summarize_label_warnings(paired)
						if paired_missing + paired_image_after < missing_text + image_after_text:
							loose_clusters = paired
		for cluster in loose_clusters:
			has_visual = any(obj.kind in ("text", "image") for obj in cluster.objects)
			if not has_visual:
				continue
			cluster.index = cluster_index
			cluster_index += 1
			label_clusters.append(cluster)

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


#============================================
def map_font_name(font_name: str, weight: int, italic: bool) -> str:
	"""
	Map LBX font metadata to a PDF font name.

	Args:
		font_name: LBX font name.
		weight: Font weight.
		italic: Italic flag.

	Returns:
		ReportLab font name.
	"""
	is_bold = weight >= 700
	if italic and is_bold:
		return DEFAULT_FONT_BOLD_ITALIC
	if italic:
		return DEFAULT_FONT_ITALIC
	if is_bold:
		return DEFAULT_FONT_BOLD
	return DEFAULT_FONT_REGULAR


#============================================
def draw_text_object(
	pdf: reportlab.pdfgen.canvas.Canvas,
	obj: LabelObject,
	text_align_horizontal: str | None = None,
	text_align_vertical: str | None = None,
	font_size_override: float | None = None,
	font_weight_override: int | None = None,
	text_fit: bool = False,
	min_font_size: float | None = None,
	clamp_counter: list[int] | None = None,
) -> list[tuple[float, float, float, float]]:
	"""
	Draw a text object onto the PDF canvas.

	Args:
		pdf: ReportLab canvas.
		obj: LabelObject to draw.

	Returns:
		List of text bounding boxes for overlap checks.
	"""
	font_weight = obj.font_weight
	if obj.use_text_override and font_weight_override is not None:
		font_weight = font_weight_override
	font_name = map_font_name(obj.font_name, font_weight, obj.font_italic)
	font_size = obj.font_size
	if obj.use_text_override and font_size_override is not None:
		font_size = font_size_override
	pdf.setFont(font_name, font_size)

	color = parse_hex_color(obj.text_color)
	pdf.setFillColorRGB(color[0], color[1], color[2])

	text_value = obj.text or ""
	lines = text_value.splitlines()
	if not lines:
		return []

	align_h = obj.align_horizontal.upper()
	align_v = obj.align_vertical.upper()
	if text_align_horizontal is not None:
		align_h = text_align_horizontal.upper()
	if text_align_vertical is not None:
		align_v = text_align_vertical.upper()

	def compute_leading(size: float) -> float:
		return size * 1.2

	leading = compute_leading(font_size)
	text_height = font_size + leading * (len(lines) - 1)
	max_width = max(pdf.stringWidth(line, font_name, font_size) for line in lines)
	if text_fit and (max_width > obj.width or text_height > obj.height):
		scale_width = obj.width / max_width if max_width > 0 else 1.0
		scale_height = obj.height / text_height if text_height > 0 else 1.0
		scale = min(1.0, scale_width, scale_height)
		if scale < 1.0:
			target_size = font_size * scale
			clamped = False
			if min_font_size is not None and target_size < min_font_size:
				font_size = min_font_size
				clamped = True
			else:
				font_size = target_size
			pdf.setFont(font_name, font_size)
			leading = compute_leading(font_size)
			text_height = font_size + leading * (len(lines) - 1)
			if clamped and clamp_counter is not None:
				clamp_counter[0] += 1

	bboxes: list[tuple[float, float, float, float]] = []

	if align_v == "CENTER":
		base_y = obj.y + (obj.height - text_height) / 2.0
	elif align_v == "BOTTOM":
		base_y = obj.y
	else:
		base_y = obj.y + obj.height - text_height

	for index, line in enumerate(lines):
		line_width = pdf.stringWidth(line, font_name, font_size)
		if align_h == "CENTER":
			text_x = obj.x + (obj.width - line_width) / 2.0
		elif align_h == "RIGHT":
			text_x = obj.x + obj.width - line_width
		else:
			text_x = obj.x
		text_y = base_y + index * leading
		pdf.drawString(text_x, text_y, line)
		bboxes.append(
			compute_text_bbox(
				text_x,
				text_y,
				line,
				font_name,
				font_size,
				TEXT_OVERLAP_PADDING,
			)
		)
	return bboxes


#============================================
def draw_image_object(
	pdf: reportlab.pdfgen.canvas.Canvas,
	obj: LabelObject,
	image_reader: reportlab.lib.utils.ImageReader,
) -> None:
	"""
	Draw an image object onto the PDF canvas.

	Args:
		pdf: ReportLab canvas.
		obj: LabelObject to draw.
		image_reader: ImageReader instance.
	"""
	scale = IMAGE_SCALE
	if scale <= 0.0:
		scale = 1.0
	if scale >= 1.0:
		image_x = obj.x
		image_y = obj.y
		image_width = obj.width
		image_height = obj.height
	else:
		image_width = obj.width * scale
		image_height = obj.height * scale
		offset_x = (obj.width - image_width) / 2.0
		offset_y = (obj.height - image_height) / 2.0
		image_x = obj.x + offset_x
		image_y = obj.y + offset_y
	pdf.drawImage(
		image_reader,
		image_x,
		image_y,
		width=image_width,
		height=image_height,
		mask=None,
		preserveAspectRatio=False,
		anchor="sw",
	)


#============================================
def draw_poly_object(
	pdf: reportlab.pdfgen.canvas.Canvas,
	obj: LabelObject,
) -> None:
	"""
	Draw a polyline object onto the PDF canvas.

	Args:
		pdf: ReportLab canvas.
		obj: LabelObject to draw.
	"""
	if len(obj.poly_points) < 2:
		return
	color = parse_hex_color(obj.line_color)
	pdf.setStrokeColorRGB(color[0], color[1], color[2])
	pdf.setLineWidth(obj.line_width)
	for index in range(1, len(obj.poly_points)):
		start = obj.poly_points[index - 1]
		end = obj.poly_points[index]
		pdf.line(start[0], start[1], end[0], end[1])


#============================================
def draw_rect_object(
	pdf: reportlab.pdfgen.canvas.Canvas,
	obj: LabelObject,
) -> None:
	"""
	Draw a filled rectangle object onto the PDF canvas.

	Args:
		pdf: ReportLab canvas.
		obj: LabelObject to draw.
	"""
	fill = obj.fill_color or obj.line_color
	color = parse_hex_color(fill)
	pdf.setFillColorRGB(color[0], color[1], color[2])
	pdf.rect(obj.x, obj.y, obj.width, obj.height, stroke=0, fill=1)


#============================================
def build_image_cache(paths: list[pathlib.Path]) -> dict[tuple[str, str], reportlab.lib.utils.ImageReader]:
	"""
	Build a cache of images for LBX files.

	Args:
		paths: LBX file paths.

	Returns:
		Cache of ImageReader instances keyed by (lbx_path, image_name).
	"""
	image_cache: dict[tuple[str, str], reportlab.lib.utils.ImageReader] = {}
	for path in paths:
		with zipfile.ZipFile(path, "r") as archive:
			for name in archive.namelist():
				if not name.lower().endswith(".bmp"):
					continue
				data = archive.read(name)
				image = PIL.Image.open(io.BytesIO(data))
				image.load()
				image_reader = reportlab.lib.utils.ImageReader(image)
				image_cache[(str(path), name)] = image_reader
	return image_cache


#============================================
def draw_label_cluster(
	pdf: reportlab.pdfgen.canvas.Canvas,
	cluster: LabelCluster,
	cell_x: float,
	cell_y: float,
	config: ImpositionConfig,
	image_cache: dict[tuple[str, str], reportlab.lib.utils.ImageReader],
) -> None:
	"""
	Draw a label cluster within the label cell.

	Args:
		pdf: ReportLab canvas.
		cluster: LabelCluster to draw.
		cell_x: Cell x origin.
		cell_y: Cell y origin.
		config: Imposition configuration.
		image_cache: Image cache.
	"""
	available_width = config.label_width - 2.0 * config.inset
	available_height = config.label_height - 2.0 * config.inset
	if cluster.width <= 0 or cluster.height <= 0:
		return

	scale = min(available_width / cluster.width, available_height / cluster.height)
	scaled_height = cluster.height * scale
	visual_bounds = compute_visual_bounds(
		cluster.objects,
		text_align_horizontal=config.text_align_horizontal,
		text_align_vertical=config.text_align_vertical,
		font_size_override=config.text_font_size,
		font_weight_override=config.text_font_weight,
		text_fit=config.text_fit,
		min_font_size=DEFAULT_TEXT_MIN_SIZE,
	)
	if visual_bounds is not None:
		visual_min_x = visual_bounds[0]
		visual_width = max(0.0, visual_bounds[2] - visual_bounds[0])
	else:
		visual_min_x = cluster.min_x
		visual_width = cluster.width
	offset_x = compute_align_offset(
		available_width,
		visual_width * scale,
		config.cluster_align_horizontal,
	)
	offset_y = compute_align_offset(available_height, scaled_height, config.cluster_align_vertical)

	base_x = cell_x + config.inset + offset_x - (visual_min_x - cluster.min_x) * scale
	base_y = cell_y + config.inset + offset_y

	def transform_x(value: float) -> float:
		return base_x + (value - cluster.min_x) * scale

	def transform_y(value: float) -> float:
		return base_y + (cluster.height - (value - cluster.min_y)) * scale

	for obj in cluster.objects:
		if obj.kind == "rect":
			new_obj = dataclasses.replace(
				obj,
				x=transform_x(obj.x),
				y=transform_y(obj.y + obj.height),
				width=obj.width * scale,
				height=obj.height * scale,
			)
			draw_rect_object(pdf, new_obj)
			continue
		if obj.kind == "text":
			new_obj = dataclasses.replace(
				obj,
				x=transform_x(obj.x),
				y=transform_y(obj.y + obj.height),
				width=obj.width * scale,
				height=obj.height * scale,
			)
			draw_text_object(
				pdf,
				new_obj,
				text_align_horizontal=config.text_align_horizontal,
				text_align_vertical=config.text_align_vertical,
				font_size_override=config.text_font_size,
				font_weight_override=config.text_font_weight,
				text_fit=config.text_fit,
				min_font_size=DEFAULT_TEXT_MIN_SIZE,
			)
			continue
		if obj.kind == "image":
			key = (cluster.source_path, obj.image_name)
			image_reader = image_cache.get(key)
			if image_reader is not None:
				image_x = transform_x(obj.x)
				image_y = transform_y(obj.y + obj.height)
				target_width = obj.width * scale
				target_height = obj.height * scale
				image_scale = min(scale, MAX_IMAGE_UPSCALE)
				if image_scale < scale:
					image_width = obj.width * image_scale
					image_height = obj.height * image_scale
					image_x += (target_width - image_width) / 2.0
					image_y += (target_height - image_height) / 2.0
				else:
					image_width = target_width
					image_height = target_height
				new_obj = dataclasses.replace(
					obj,
					x=image_x,
					y=image_y,
					width=image_width,
					height=image_height,
				)
				draw_image_object(pdf, new_obj, image_reader)
			continue
		if obj.kind == "poly":
			points: list[tuple[float, float]] = []
			for point in obj.poly_points:
				points.append((transform_x(point[0]), transform_y(point[1])))
			new_obj = dataclasses.replace(
				obj,
				poly_points=points,
				line_width=obj.line_width * scale,
			)
			draw_poly_object(pdf, new_obj)
			continue


#============================================
def draw_cluster_to_tile(
	pdf: reportlab.pdfgen.canvas.Canvas,
	cluster: LabelCluster,
	config: TileConfig,
	image_cache: dict[tuple[str, str], reportlab.lib.utils.ImageReader],
) -> tuple[int, int]:
	"""
	Draw a label cluster onto a tile PDF.

	Args:
		pdf: ReportLab canvas.
		cluster: LabelCluster to draw.
		config: Tile configuration.
		image_cache: Image cache.

	Returns:
		Tuple of (overlap_count, min_font_clamp_count).
	"""
	available_width = config.label_width - 2.0 * config.inset
	available_height = config.label_height - 2.0 * config.inset
	if cluster.width <= 0 or cluster.height <= 0:
		return (0, 0)

	scale = min(available_width / cluster.width, available_height / cluster.height)
	scaled_height = cluster.height * scale
	visual_bounds = compute_visual_bounds(
		cluster.objects,
		text_align_horizontal=config.text_align_horizontal,
		text_align_vertical=config.text_align_vertical,
		font_size_override=config.text_font_size,
		font_weight_override=config.text_font_weight,
		text_fit=config.text_fit,
		min_font_size=DEFAULT_TEXT_MIN_SIZE,
	)
	if visual_bounds is not None:
		visual_min_x = visual_bounds[0]
		visual_width = max(0.0, visual_bounds[2] - visual_bounds[0])
	else:
		visual_min_x = cluster.min_x
		visual_width = cluster.width
	offset_x = compute_align_offset(
		available_width,
		visual_width * scale,
		config.cluster_align_horizontal,
	)
	offset_y = compute_align_offset(available_height, scaled_height, config.cluster_align_vertical)
	base_x = config.inset + offset_x - (visual_min_x - cluster.min_x) * scale
	base_y = config.inset + offset_y

	def transform_x(value: float) -> float:
		return base_x + (value - cluster.min_x) * scale

	def transform_y(value: float) -> float:
		return base_y + (cluster.height - (value - cluster.min_y)) * scale

	text_boxes: list[tuple[float, float, float, float]] = []
	overlap_count = 0
	min_font_clamps = [0]

	for obj in cluster.objects:
		if obj.kind == "rect":
			new_obj = dataclasses.replace(
				obj,
				x=transform_x(obj.x),
				y=transform_y(obj.y + obj.height),
				width=obj.width * scale,
				height=obj.height * scale,
			)
			draw_rect_object(pdf, new_obj)
			continue
		if obj.kind == "text":
			new_obj = dataclasses.replace(
				obj,
				x=transform_x(obj.x),
				y=transform_y(obj.y + obj.height),
				width=obj.width * scale,
				height=obj.height * scale,
			)
			new_boxes = draw_text_object(
				pdf,
				new_obj,
				text_align_horizontal=config.text_align_horizontal,
				text_align_vertical=config.text_align_vertical,
				font_size_override=config.text_font_size,
				font_weight_override=config.text_font_weight,
				text_fit=config.text_fit,
				min_font_size=DEFAULT_TEXT_MIN_SIZE,
				clamp_counter=min_font_clamps,
			)
			for new_box in new_boxes:
				for prior in text_boxes:
					if boxes_intersect(new_box, prior):
						overlap_count += 1
						break
			text_boxes.extend(new_boxes)
			continue
		if obj.kind == "image":
			key = (cluster.source_path, obj.image_name)
			image_reader = image_cache.get(key)
			if image_reader is not None:
				image_x = transform_x(obj.x)
				image_y = transform_y(obj.y + obj.height)
				target_width = obj.width * scale
				target_height = obj.height * scale
				image_scale = min(scale, MAX_IMAGE_UPSCALE)
				if image_scale < scale:
					image_width = obj.width * image_scale
					image_height = obj.height * image_scale
					image_x += (target_width - image_width) / 2.0
					image_y += (target_height - image_height) / 2.0
				else:
					image_width = target_width
					image_height = target_height
				new_obj = dataclasses.replace(
					obj,
					x=image_x,
					y=image_y,
					width=image_width,
					height=image_height,
				)
				draw_image_object(pdf, new_obj, image_reader)
			continue
		if obj.kind == "poly":
			points: list[tuple[float, float]] = []
			for point in obj.poly_points:
				points.append((transform_x(point[0]), transform_y(point[1])))
			new_obj = dataclasses.replace(
				obj,
				poly_points=points,
				line_width=obj.line_width * scale,
			)
			draw_poly_object(pdf, new_obj)
			continue
	return (overlap_count, min_font_clamps[0])


#============================================
def render_tile_pdf(
	cluster: LabelCluster,
	output_path: pathlib.Path,
	config: TileConfig,
	image_cache: dict[tuple[str, str], reportlab.lib.utils.ImageReader],
) -> tuple[int, int]:
	"""
	Render a single label tile PDF.

	Args:
		cluster: LabelCluster to render.
		output_path: Output file path.
		config: Tile configuration.
		image_cache: Image cache.

	Returns:
		Tuple of (overlap_count, min_font_clamp_count).
	"""
	pdf = reportlab.pdfgen.canvas.Canvas(
		str(output_path),
		pagesize=(config.label_width, config.label_height),
	)
	overlap_count, clamp_count = draw_cluster_to_tile(pdf, cluster, config, image_cache)
	pdf.save()
	return (overlap_count, clamp_count)


#============================================
def draw_calibration_page(pdf: reportlab.pdfgen.canvas.Canvas, config: ImpositionConfig) -> None:
	"""
	Draw calibration boxes and a 1 inch ruler mark.

	Args:
		pdf: ReportLab canvas.
		config: Imposition configuration.
	"""
	page_width, page_height = reportlab.lib.pagesizes.letter
	pdf.setLineWidth(0.3)
	pdf.setStrokeColorRGB(0.6, 0.6, 0.6)

	scaled_label_width = config.label_width * config.x_scale
	scaled_label_height = config.label_height * config.y_scale
	scaled_h_gap = config.h_gap * config.x_scale
	scaled_v_gap = config.v_gap * config.y_scale

	for row in range(config.rows):
		for col in range(config.columns):
			cell_x = config.left_margin + col * (scaled_label_width + scaled_h_gap)
			cell_y = page_height - config.top_margin - scaled_label_height - row * (
				scaled_label_height + scaled_v_gap
			)
			pdf.rect(cell_x, cell_y, scaled_label_width, scaled_label_height, stroke=1, fill=0)

	ruler_x = config.left_margin
	ruler_y = page_height - config.top_margin + 10.0
	pdf.setStrokeColorRGB(0.0, 0.0, 0.0)
	pdf.setLineWidth(0.6)
	pdf.line(ruler_x, ruler_y, ruler_x + POINTS_PER_INCH, ruler_y)
	pdf.setFont(DEFAULT_FONT_REGULAR, 8)
	pdf.drawString(ruler_x, ruler_y + 4.0, "1 in")


def draw_label_outlines(pdf: reportlab.pdfgen.canvas.Canvas, config: ImpositionConfig) -> None:
	"""
	Draw label outlines on the current page.

	Args:
		pdf: ReportLab canvas.
		config: Imposition configuration.
	"""
	page_width, page_height = reportlab.lib.pagesizes.letter
	pdf.setLineWidth(0.3)
	pdf.setStrokeColorRGB(0.7, 0.7, 0.7)

	scaled_label_width = config.label_width * config.x_scale
	scaled_label_height = config.label_height * config.y_scale
	scaled_h_gap = config.h_gap * config.x_scale
	scaled_v_gap = config.v_gap * config.y_scale

	for row in range(config.rows):
		for col in range(config.columns):
			cell_x = config.left_margin + col * (scaled_label_width + scaled_h_gap)
			cell_y = page_height - config.top_margin - scaled_label_height - row * (
				scaled_label_height + scaled_v_gap
			)
			pdf.rect(cell_x, cell_y, scaled_label_width, scaled_label_height, stroke=1, fill=0)


#============================================
def render_labels_to_pdf(
	labels: list[LabelCluster],
	paths: list[pathlib.Path],
	output_path: pathlib.Path,
	config: ImpositionConfig,
) -> ImpositionResult:
	"""
	Render labels to a PDF.

	Args:
		labels: List of LabelCluster entries.
		paths: LBX file paths.
		output_path: Output PDF path.
		config: Imposition configuration.

	Returns:
		ImpositionResult.
	"""
	page_width, page_height = reportlab.lib.pagesizes.letter
	pdf = reportlab.pdfgen.canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

	if config.calibration:
		draw_calibration_page(pdf, config)
		pdf.showPage()

	labels_per_page = config.columns * config.rows
	total_labels = len(labels)
	if not config.include_partial:
		labels_to_print = (total_labels // labels_per_page) * labels_per_page
	else:
		labels_to_print = total_labels

	image_cache = build_image_cache(paths)
	for index in range(labels_to_print):
		cluster = labels[index]
		if index > 0 and index % labels_per_page == 0:
			pdf.showPage()
		if config.draw_outlines and index % labels_per_page == 0:
			draw_label_outlines(pdf, config)
		slot = index % labels_per_page
		row = slot % config.rows
		col = slot // config.rows
		cell_x = config.left_margin + col * (config.label_width + config.h_gap)
		cell_y = page_height - config.top_margin - config.label_height - row * (
			config.label_height + config.v_gap
		)
		draw_label_cluster(pdf, cluster, cell_x, cell_y, config, image_cache)

	pdf.save()

	printed_labels = labels_to_print
	leftover_labels = total_labels - labels_to_print
	pages = 0
	if labels_to_print > 0:
		pages = (labels_to_print + labels_per_page - 1) // labels_per_page
	if config.calibration:
		pages += 1

	result = ImpositionResult(
		total_labels=total_labels,
		printed_labels=printed_labels,
		leftover_labels=leftover_labels,
		pages=pages,
		labels_per_page=labels_per_page,
	)
	return result


#============================================
def render_tiles(
	labels: list[LabelCluster],
	output_dir: pathlib.Path,
	tile_config: TileConfig,
	image_cache: dict[tuple[str, str], reportlab.lib.utils.ImageReader],
	hashes: dict[str, str],
) -> list[dict[str, str]]:
	"""
	Render label clusters into tile PDFs.

	Args:
		labels: Label clusters.
		output_dir: Output directory.
		tile_config: Tile configuration.
		image_cache: Image cache.
		hashes: SHA256 hashes by LBX path.

	Returns:
		List of tile metadata dictionaries.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	tiles: list[dict[str, str]] = []
	overlap_labels = 0
	overlap_total = 0
	min_font_labels = 0
	min_font_total = 0
	overlap_messages: list[str] = []
	missing_text_labels = 0
	image_after_text_labels = 0
	validation_messages: list[str] = []
	total = len(labels)
	if total > 0:
		print_progress("Tiles", 0, total)
	for index, cluster in enumerate(labels, start=1):
		source_path = pathlib.Path(cluster.source_path)
		source_hash = hashes.get(cluster.source_path, "")[:8]
		stem = sanitize_token(source_path.stem)
		tile_name = f"{stem}_{cluster.index:03d}_{source_hash}.pdf"
		tile_path = output_dir / tile_name
		text_objects = [
			obj for obj in cluster.objects
			if obj.kind == "text" and (obj.text or "").strip()
		]
		image_objects = [obj for obj in cluster.objects if obj.kind == "image"]
		if not text_objects:
			missing_text_labels += 1
			validation_messages.append(f"No text in {tile_name}")
		if text_objects and image_objects:
			min_text_x = min(obj.x for obj in text_objects)
			min_image_x = min(obj.x for obj in image_objects)
			if min_image_x > min_text_x:
				image_after_text_labels += 1
				validation_messages.append(
					f"Image after text in {tile_name} (text x={min_text_x:.1f}, image x={min_image_x:.1f})"
				)
		overlap_count, clamp_count = render_tile_pdf(
			cluster,
			tile_path,
			tile_config,
			image_cache,
		)
		if overlap_count > 0:
			overlap_labels += 1
			overlap_total += overlap_count
			overlap_messages.append(f"Text overlaps in {tile_name}: {overlap_count}")
		if clamp_count > 0:
			min_font_labels += 1
			min_font_total += clamp_count
		tiles.append(
			{
				"id": tile_name.replace(".pdf", ""),
				"path": str(tile_path),
				"source": cluster.source_path,
				"index": str(cluster.index),
			}
		)
		if total > 0 and (index % PROGRESS_UPDATE_EVERY == 0 or index == total):
			print_progress("Tiles", index, total)
	if total > 0:
		print()
	if overlap_messages:
		for message in overlap_messages:
			print(message)
	if validation_messages:
		for message in validation_messages:
			print(message)
	if overlap_total > 0:
		print(f"Text overlap summary: {overlap_labels} labels, {overlap_total} overlaps")
	if min_font_total > 0:
		print(f"Min font size clamps: {min_font_labels} labels, {min_font_total} text runs")
	if missing_text_labels > 0:
		print(f"Missing text summary: {missing_text_labels} labels")
	if image_after_text_labels > 0:
		print(f"Image-after-text summary: {image_after_text_labels} labels")
	return tiles


def build_outline_overlay(config: ImpositionConfig) -> pypdf.PageObject:
	"""
	Build a PDF overlay page with label outlines.

	Args:
		config: Imposition configuration.

	Returns:
		PDF page object.
	"""
	buffer = io.BytesIO()
	page_width, page_height = reportlab.lib.pagesizes.letter
	pdf = reportlab.pdfgen.canvas.Canvas(buffer, pagesize=(page_width, page_height))
	draw_label_outlines(pdf, config)
	pdf.save()
	buffer.seek(0)
	reader = pypdf.PdfReader(buffer)
	return reader.pages[0]


#============================================
def build_calibration_page(config: ImpositionConfig) -> pypdf.PageObject:
	"""
	Build a calibration page PDF.

	Args:
		config: Imposition configuration.

	Returns:
		PDF page object.
	"""
	buffer = io.BytesIO()
	page_width, page_height = reportlab.lib.pagesizes.letter
	pdf = reportlab.pdfgen.canvas.Canvas(buffer, pagesize=(page_width, page_height))

	draw_calibration_page(pdf, config)

	pdf.setLineWidth(0.6)
	pdf.setStrokeColorRGB(0.0, 0.0, 0.0)
	pdf.setFont(DEFAULT_FONT_REGULAR, 8)

	row_samples = [0, config.rows - 1]
	col_samples = [0, config.columns - 1]
	scaled_label_width = config.label_width * config.x_scale
	scaled_label_height = config.label_height * config.y_scale
	scaled_h_gap = config.h_gap * config.x_scale
	scaled_v_gap = config.v_gap * config.y_scale
	for row in row_samples:
		for col in col_samples:
			cell_x = config.left_margin + col * (scaled_label_width + scaled_h_gap)
			cell_y = page_height - config.top_margin - scaled_label_height - row * (
				scaled_label_height + scaled_v_gap
			)
			center_x = cell_x + scaled_label_width / 2.0
			center_y = cell_y + scaled_label_height / 2.0
			size = 6.0
			pdf.line(center_x - size, center_y, center_x + size, center_y)
			pdf.line(center_x, center_y - size, center_x, center_y + size)

	pdf.save()
	buffer.seek(0)
	reader = pypdf.PdfReader(buffer)
	return reader.pages[0]


#============================================
def impose_tiles(
	tile_paths: list[pathlib.Path],
	output_path: pathlib.Path,
	config: ImpositionConfig,
) -> ImpositionResult:
	"""
	Impose tile PDFs onto Avery 5167 sheets.

	Args:
		tile_paths: Tile PDF paths.
		output_path: Output PDF path.
		config: Imposition configuration.

	Returns:
		ImpositionResult.
	"""
	writer = pypdf.PdfWriter()
	page_width, page_height = reportlab.lib.pagesizes.letter
	labels_per_page = config.columns * config.rows

	tiles_to_print = len(tile_paths)
	if config.max_labels is not None:
		tiles_to_print = min(tiles_to_print, config.max_labels)
	if config.max_pages is not None:
		tiles_to_print = min(tiles_to_print, config.max_pages * labels_per_page)
	if not config.include_partial:
		tiles_to_print = (tiles_to_print // labels_per_page) * labels_per_page

	outline_page = None
	if config.draw_outlines:
		outline_page = build_outline_overlay(config)

	if config.calibration:
		calibration_page = build_calibration_page(config)
		writer.add_page(calibration_page)

	tile_cache: dict[str, pypdf.PageObject] = {}
	for index in range(tiles_to_print):
		if index % labels_per_page == 0:
			page = pypdf.PageObject.create_blank_page(
				width=page_width,
				height=page_height,
			)
			if outline_page is not None:
				page.merge_page(outline_page)
			writer.add_page(page)

		page = writer.pages[-1]
		slot = index % labels_per_page
		row = slot % config.rows
		col = slot // config.rows

		scaled_label_width = config.label_width * config.x_scale
		scaled_label_height = config.label_height * config.y_scale
		scaled_h_gap = config.h_gap * config.x_scale
		scaled_v_gap = config.v_gap * config.y_scale

		cell_x = config.left_margin + col * (scaled_label_width + scaled_h_gap)
		cell_y = page_height - config.top_margin - scaled_label_height - row * (
			scaled_label_height + scaled_v_gap
		)

		tile_path = str(tile_paths[index])
		if tile_path not in tile_cache:
			reader = pypdf.PdfReader(tile_path)
			tile_cache[tile_path] = reader.pages[0]
		tile_page = tile_cache[tile_path]

		transform = pypdf.Transformation().scale(config.x_scale, config.y_scale).translate(
			cell_x,
			cell_y,
		)
		page.merge_transformed_page(tile_page, transform)

	writer.write(str(output_path))

	printed_labels = tiles_to_print
	leftover_labels = len(tile_paths) - tiles_to_print
	pages = 0
	if tiles_to_print > 0:
		pages = (tiles_to_print + labels_per_page - 1) // labels_per_page
	if config.calibration:
		pages += 1

	return ImpositionResult(
		total_labels=len(tile_paths),
		printed_labels=printed_labels,
		leftover_labels=leftover_labels,
		pages=pages,
		labels_per_page=labels_per_page,
	)
#============================================
def write_manifest(
	manifest_path: pathlib.Path,
	inputs: list[pathlib.Path],
	counts_by_file: dict[str, int],
	hashes: dict[str, str],
	thresholds: dict[str, float],
	result: ImpositionResult,
	config: ImpositionConfig,
) -> None:
	"""
	Write a manifest JSON file.

	Args:
		manifest_path: Output path.
		inputs: Input LBX files.
		counts_by_file: Label counts by file.
		hashes: SHA256 hashes by file.
		thresholds: Gap thresholds by file.
		result: Imposition result.
		config: Imposition configuration.
	"""
	data = {
		"inputs": [str(path) for path in inputs],
		"label_counts": counts_by_file,
		"lbx_hashes": hashes,
		"gap_thresholds": thresholds,
		"labels_per_page": result.labels_per_page,
		"total_labels": result.total_labels,
		"printed_labels": result.printed_labels,
		"leftover_labels": result.leftover_labels,
		"pages": result.pages,
		"layout": {
			"label_width": config.label_width,
			"label_height": config.label_height,
			"columns": config.columns,
			"rows": config.rows,
			"left_margin": config.left_margin,
			"top_margin": config.top_margin,
			"h_gap": config.h_gap,
			"v_gap": config.v_gap,
			"inset": config.inset,
			"x_scale": config.x_scale,
			"y_scale": config.y_scale,
			"draw_outlines": config.draw_outlines,
			"max_pages": config.max_pages,
			"max_labels": config.max_labels,
		},
		"fonts": {
			"regular": DEFAULT_FONT_REGULAR,
			"bold": DEFAULT_FONT_BOLD,
			"italic": DEFAULT_FONT_ITALIC,
			"bold_italic": DEFAULT_FONT_BOLD_ITALIC,
		},
	}
	with manifest_path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2, sort_keys=True)


#============================================
#============================================
def build_config(args: argparse.Namespace) -> ImpositionConfig:
	"""
	Build imposition config from CLI args.

	Args:
		args: Parsed argparse namespace.

	Returns:
		ImpositionConfig.
	"""
	config = ImpositionConfig(
		label_width=DEFAULT_LABEL_WIDTH,
		label_height=DEFAULT_LABEL_HEIGHT,
		columns=COLUMNS,
		rows=ROWS,
		left_margin=DEFAULT_LEFT_MARGIN,
		top_margin=DEFAULT_TOP_MARGIN,
		h_gap=DEFAULT_H_GAP,
		v_gap=DEFAULT_V_GAP,
		inset=DEFAULT_INSET,
		x_scale=1.0,
		y_scale=1.0,
		include_partial=args.include_partial,
		calibration=args.calibration,
		draw_outlines=args.draw_outlines,
		max_pages=args.max_pages,
		max_labels=args.max_labels,
		cluster_align_horizontal="CENTER",
		cluster_align_vertical="CENTER",
		text_align_horizontal=None,
		text_align_vertical=None,
		text_font_size=None,
		text_font_weight=None,
		text_fit=True,
	)
	return config


#============================================
def build_tile_config(args: argparse.Namespace) -> TileConfig:
	"""
	Build tile config from CLI args.

	Args:
		args: Parsed argparse namespace.

	Returns:
		TileConfig.
	"""
	return TileConfig(
		label_width=DEFAULT_LABEL_WIDTH,
		label_height=DEFAULT_LABEL_HEIGHT,
		inset=DEFAULT_INSET,
		cluster_align_horizontal="CENTER",
		cluster_align_vertical="CENTER",
		text_align_horizontal=None,
		text_align_vertical=None,
		text_font_size=DEFAULT_TEXT_SIZE,
		text_font_weight=DEFAULT_TEXT_WEIGHT,
		text_fit=True,
	)


#============================================
def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
		Parsed argparse namespace.
	"""
	parser = argparse.ArgumentParser(description="Convert LBX files to Avery 5167 PDF sheets.")
	parser.add_argument("inputs", nargs="+", help="LBX files or directories.")

	output_group = parser.add_argument_group("Output")
	output_group.add_argument("-o", "--output", dest="output_path", required=True, help="Output PDF path.")
	output_group.add_argument("-m", "--manifest", dest="manifest_path", default=None, help="Output manifest JSON path.")

	behavior_group = parser.add_argument_group("Behavior")
	behavior_group.add_argument("-d", "--draw-outlines", dest="draw_outlines", action="store_true", help="Draw sticker outlines.")
	behavior_group.add_argument("-D", "--no-draw-outlines", dest="draw_outlines", action="store_false", help="Disable sticker outlines.")
	behavior_group.add_argument("-c", "--calibration", dest="calibration", action="store_true", help="Add a calibration page.")
	behavior_group.add_argument("-C", "--no-calibration", dest="calibration", action="store_false", help="Disable calibration page.")
	behavior_group.add_argument("-p", "--include-partial", dest="include_partial", action="store_true", help="Include a partial final page.")
	behavior_group.add_argument("-P", "--no-include-partial", dest="include_partial", action="store_false", help="Exclude partial final page.")
	behavior_group.add_argument("-n", "--normalize-text", dest="normalize_text", action="store_true", help="Normalize text to ASCII.")
	behavior_group.add_argument("-N", "--no-normalize-text", dest="normalize_text", action="store_false", help="Preserve original text.")
	behavior_group.add_argument(
		"--stop-before-rendering",
		dest="stop_before_rendering",
		action="store_true",
		help="Stop after collecting labels (skip rendering and imposition).",
	)

	limit_group = parser.add_argument_group("Limits")
	limit_group.add_argument("-g", "--max-pages", dest="max_pages", type=int, default=None, help="Limit number of label pages.")
	limit_group.add_argument("-l", "--max-labels", dest="max_labels", type=int, default=None, help="Limit number of labels.")

	parser.set_defaults(
		draw_outlines=False,
		calibration=False,
		include_partial=False,
		normalize_text=True,
		stop_before_rendering=False,
	)

	args = parser.parse_args()
	return args


#============================================
def run_pipeline(args: argparse.Namespace) -> None:
	"""
	Run the full pipeline from LBX input to Avery output.

	Args:
		args: Parsed argparse namespace.
	"""
	print("LBX to Avery 5167 pipeline")
	print(f"Output PDF: {args.output_path}")
	if args.manifest_path:
		print(f"Manifest: {args.manifest_path}")
	print(f"Normalize text: {args.normalize_text}")
	print(f"Draw outlines: {args.draw_outlines}")
	print(f"Calibration: {args.calibration}")
	print(f"Include partial: {args.include_partial}")
	if args.max_labels is not None:
		print(f"Max labels: {args.max_labels}")
	if args.max_pages is not None:
		print(f"Max pages: {args.max_pages}")
	if args.stop_before_rendering:
		print("Stop before rendering: True")

	paths = gather_lbx_paths(args.inputs)
	print(f"LBX files found: {len(paths)}")

	labels, counts_by_file, hashes, thresholds = collect_labels(
		paths,
		None,
		args.normalize_text,
		args.max_labels,
		verbose=True,
	)
	print(f"Labels collected: {len(labels)}")

	output_path = pathlib.Path(args.output_path)
	write_label_count_log(counts_by_file, output_path)
	if args.stop_before_rendering:
		print("Stopping before rendering tiles.")
		return

	image_cache = build_image_cache(paths)
	tile_config = build_tile_config(args)
	tiles_dir = output_path.parent / "tiles"
	print(f"Tiles directory: {tiles_dir}")
	print("Rendering tiles")
	tiles = render_tiles(labels, tiles_dir, tile_config, image_cache, hashes)
	print(f"Tiles rendered: {len(tiles)}")

	tile_paths = [pathlib.Path(tile["path"]) for tile in tiles]
	config = build_config(args)
	print("Imposing tiles")
	result = impose_tiles(tile_paths, output_path, config)
	print(f"Pages written: {result.pages}")
	print(f"Labels printed: {result.printed_labels}")
	print(f"Labels leftover: {result.leftover_labels}")

	manifest_path = args.manifest_path
	if manifest_path is None:
		manifest_path = f"{output_path}.json"
	write_manifest(
		pathlib.Path(manifest_path),
		paths,
		counts_by_file,
		hashes,
		thresholds,
		result,
		config,
	)
	print(f"Manifest written: {manifest_path}")


#============================================
def main() -> None:
	"""
	Main entry point.
	"""
	args = parse_args()
	run_pipeline(args)


if __name__ == "__main__":
	main()
