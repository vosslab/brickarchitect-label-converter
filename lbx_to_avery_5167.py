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
import zipfile
import xml.etree.ElementTree as ElementTree

import PIL.Image
import pypdf
import reportlab.lib.pagesizes
import reportlab.lib.utils
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
	align_horizontal: str = "LEFT"
	align_vertical: str = "TOP"
	text_color: str = "#000000"
	image_name: str = ""
	line_width: float = 0.3
	line_color: str = "#000000"
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


@dataclasses.dataclass
class TileConfig:
	label_width: float
	label_height: float
	inset: float


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
def find_first_child(element: ElementTree.Element, suffix: str) -> ElementTree.Element | None:
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
def parse_text_element(element: ElementTree.Element) -> LabelObject | None:
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
def parse_image_element(element: ElementTree.Element) -> LabelObject | None:
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
def parse_poly_element(element: ElementTree.Element) -> LabelObject | None:
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
def parse_label_xml(label_xml: bytes) -> list[LabelObject]:
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
def compute_gap_threshold(objects: list[LabelObject], min_threshold: float) -> float:
	"""
	Compute a gap threshold from object spacing.

	Args:
		objects: List of LabelObject entries.
		min_threshold: Minimum gap threshold.

	Returns:
		Gap threshold in points.
	"""
	if len(objects) < 2:
		return min_threshold
	objects_sorted = sorted(objects, key=lambda item: item.x)
	gaps: list[float] = []
	for index in range(1, len(objects_sorted)):
		prev = objects_sorted[index - 1]
		cur = objects_sorted[index]
		prev_end = prev.x + prev.width
		gap = cur.x - prev_end
		if gap > 0:
			gaps.append(gap)
	if not gaps:
		return min_threshold
	median_gap = statistics.median(gaps)
	threshold = max(min_threshold, median_gap * 2.0)
	return threshold


#============================================
def cluster_objects(
	objects: list[LabelObject],
	gap_threshold: float,
) -> list[list[LabelObject]]:
	"""
	Group objects into clusters based on horizontal gaps.

	Args:
		objects: List of LabelObject entries.
		gap_threshold: Gap threshold in points.

	Returns:
		List of object clusters.
	"""
	if not objects:
		return []
	objects_sorted = sorted(objects, key=lambda item: item.x)
	clusters: list[list[LabelObject]] = []
	current: list[LabelObject] = []
	current_max_x = None

	for item in objects_sorted:
		if not current:
			current = [item]
			current_max_x = item.x + item.width
			continue
		gap = item.x - current_max_x
		if gap > gap_threshold:
			clusters.append(current)
			current = [item]
			current_max_x = item.x + item.width
			continue
		current.append(item)
		current_max_x = max(current_max_x, item.x + item.width)

	if current:
		clusters.append(current)
	return clusters


#============================================
def build_label_clusters(
	objects: list[LabelObject],
	source_path: str,
	gap_threshold: float,
) -> list[LabelCluster]:
	"""
	Build label clusters from objects.

	Args:
		objects: List of LabelObject entries.
		source_path: Source LBX path.
		gap_threshold: Gap threshold in points.

	Returns:
		List of LabelCluster entries.
	"""
	clusters = cluster_objects(objects, gap_threshold)
	label_clusters: list[LabelCluster] = []
	for index, cluster in enumerate(clusters, 1):
		min_x = min(item.x for item in cluster)
		min_y = min(item.y for item in cluster)
		max_x = max(item.x + item.width for item in cluster)
		max_y = max(item.y + item.height for item in cluster)
		label_clusters.append(
			LabelCluster(
				source_path=source_path,
				index=index,
				objects=cluster,
				min_x=min_x,
				min_y=min_y,
				width=max_x - min_x,
				height=max_y - min_y,
			)
		)
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
	paths_sorted = sorted(paths)
	return paths_sorted


#============================================
def collect_labels(
	paths: list[pathlib.Path],
	gap_threshold: float | None,
) -> tuple[list[LabelCluster], dict[str, int], dict[str, str], dict[str, float]]:
	"""
	Collect label clusters from LBX files.

	Args:
		paths: LBX file paths.
		gap_threshold: Optional gap threshold override.

	Returns:
		Tuple of (labels, counts by file, hashes, thresholds by file).
	"""
	labels: list[LabelCluster] = []
	counts_by_file: dict[str, int] = {}
	hashes: dict[str, str] = {}
	thresholds: dict[str, float] = {}

	for path in paths:
		with zipfile.ZipFile(path, "r") as archive:
			label_xml = archive.read("label.xml")
		objects = parse_label_xml(label_xml)
		threshold = gap_threshold
		if threshold is None:
			threshold = compute_gap_threshold(objects, DEFAULT_GAP_THRESHOLD)
		thresholds[str(path)] = threshold
		clusters = build_label_clusters(objects, str(path), threshold)
		counts_by_file[str(path)] = len(clusters)
		labels.extend(clusters)
		hashes[str(path)] = compute_sha256(path)

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
) -> None:
	"""
	Draw a text object onto the PDF canvas.

	Args:
		pdf: ReportLab canvas.
		obj: LabelObject to draw.
	"""
	font_name = map_font_name(obj.font_name, obj.font_weight, obj.font_italic)
	font_size = obj.font_size
	pdf.setFont(font_name, font_size)

	color = parse_hex_color(obj.text_color)
	pdf.setFillColorRGB(color[0], color[1], color[2])

	text_value = obj.text or ""
	lines = text_value.splitlines()
	if not lines:
		return

	leading = font_size * 1.2
	text_height = leading * len(lines)

	align_h = obj.align_horizontal.upper()
	align_v = obj.align_vertical.upper()

	if align_v == "CENTER":
		base_y = obj.y + (obj.height - text_height) / 2.0 + font_size
	elif align_v == "BOTTOM":
		base_y = obj.y + (obj.height - text_height) + font_size
	else:
		base_y = obj.y + font_size

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
	pdf.drawImage(
		image_reader,
		obj.x,
		obj.y,
		width=obj.width,
		height=obj.height,
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
	scaled_width = cluster.width * scale
	scaled_height = cluster.height * scale
	offset_x = (available_width - scaled_width) / 2.0
	offset_y = (available_height - scaled_height) / 2.0

	base_x = cell_x + config.inset + offset_x
	base_y = cell_y + config.inset + offset_y

	def transform_x(value: float) -> float:
		return base_x + (value - cluster.min_x) * scale

	def transform_y(value: float) -> float:
		return base_y + (cluster.height - (value - cluster.min_y)) * scale

	for obj in cluster.objects:
		if obj.kind == "text":
			new_obj = dataclasses.replace(
				obj,
				x=transform_x(obj.x),
				y=transform_y(obj.y + obj.height),
				width=obj.width * scale,
				height=obj.height * scale,
			)
			draw_text_object(pdf, new_obj)
			continue
		if obj.kind == "image":
			key = (cluster.source_path, obj.image_name)
			image_reader = image_cache.get(key)
			if image_reader is not None:
				new_obj = dataclasses.replace(
					obj,
					x=transform_x(obj.x),
					y=transform_y(obj.y + obj.height),
					width=obj.width * scale,
					height=obj.height * scale,
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
) -> None:
	"""
	Draw a label cluster onto a tile PDF.

	Args:
		pdf: ReportLab canvas.
		cluster: LabelCluster to draw.
		config: Tile configuration.
		image_cache: Image cache.
	"""
	available_width = config.label_width - 2.0 * config.inset
	available_height = config.label_height - 2.0 * config.inset
	if cluster.width <= 0 or cluster.height <= 0:
		return

	scale = min(available_width / cluster.width, available_height / cluster.height)
	scaled_width = cluster.width * scale
	scaled_height = cluster.height * scale
	base_x = config.inset + (available_width - scaled_width) / 2.0
	base_y = config.inset + (available_height - scaled_height) / 2.0

	def transform_x(value: float) -> float:
		return base_x + (value - cluster.min_x) * scale

	def transform_y(value: float) -> float:
		return base_y + (cluster.height - (value - cluster.min_y)) * scale

	for obj in cluster.objects:
		if obj.kind == "text":
			new_obj = dataclasses.replace(
				obj,
				x=transform_x(obj.x),
				y=transform_y(obj.y + obj.height),
				width=obj.width * scale,
				height=obj.height * scale,
			)
			draw_text_object(pdf, new_obj)
			continue
		if obj.kind == "image":
			key = (cluster.source_path, obj.image_name)
			image_reader = image_cache.get(key)
			if image_reader is not None:
				new_obj = dataclasses.replace(
					obj,
					x=transform_x(obj.x),
					y=transform_y(obj.y + obj.height),
					width=obj.width * scale,
					height=obj.height * scale,
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
def render_tile_pdf(
	cluster: LabelCluster,
	output_path: pathlib.Path,
	config: TileConfig,
	image_cache: dict[tuple[str, str], reportlab.lib.utils.ImageReader],
) -> None:
	"""
	Render a single label tile PDF.

	Args:
		cluster: LabelCluster to render.
		output_path: Output file path.
		config: Tile configuration.
		image_cache: Image cache.
	"""
	pdf = reportlab.pdfgen.canvas.Canvas(
		str(output_path),
		pagesize=(config.label_width, config.label_height),
	)
	draw_cluster_to_tile(pdf, cluster, config, image_cache)
	pdf.save()


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
		row = slot // config.columns
		col = slot % config.columns
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
	for cluster in labels:
		source_path = pathlib.Path(cluster.source_path)
		source_hash = hashes.get(cluster.source_path, "")[:8]
		stem = sanitize_token(source_path.stem)
		tile_name = f"{stem}_{cluster.index:03d}_{source_hash}.pdf"
		tile_path = output_dir / tile_name
		render_tile_pdf(cluster, tile_path, tile_config, image_cache)
		tiles.append(
			{
				"id": tile_name.replace(".pdf", ""),
				"path": str(tile_path),
				"source": cluster.source_path,
				"index": str(cluster.index),
			}
		)
	return tiles


#============================================
def write_tiles_manifest(
	manifest_path: pathlib.Path,
	tiles: list[dict[str, str]],
	counts_by_file: dict[str, int],
	hashes: dict[str, str],
	thresholds: dict[str, float],
	tile_config: TileConfig,
) -> None:
	"""
	Write a tile manifest JSON file.

	Args:
		manifest_path: Output path.
		tiles: Tile metadata.
		counts_by_file: Label counts by file.
		hashes: SHA256 hashes by file.
		thresholds: Gap thresholds by file.
		tile_config: Tile configuration.
	"""
	data = {
		"tiles": tiles,
		"label_counts": counts_by_file,
		"lbx_hashes": hashes,
		"gap_thresholds": thresholds,
		"tile_config": {
			"label_width": tile_config.label_width,
			"label_height": tile_config.label_height,
			"inset": tile_config.inset,
		},
	}
	with manifest_path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2, sort_keys=True)


#============================================
def gather_tile_paths(inputs: list[str]) -> list[pathlib.Path]:
	"""
	Gather tile PDF paths from inputs.

	Args:
		inputs: Input paths.

	Returns:
		Sorted list of PDF file paths.
	"""
	paths: list[pathlib.Path] = []
	for entry in inputs:
		path = pathlib.Path(entry).expanduser().resolve()
		if path.is_dir():
			paths.extend(sorted(path.rglob("*.pdf")))
			continue
		if path.is_file() and path.suffix.lower() == ".pdf":
			paths.append(path)
	return sorted(paths)


#============================================
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
		row = slot // config.columns
		col = slot % config.columns

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
	label_width = inches_to_points(args.label_width)
	label_height = inches_to_points(args.label_height)
	left_margin = inches_to_points(args.left_margin)
	top_margin = inches_to_points(args.top_margin)
	h_gap = inches_to_points(args.h_gap)
	v_gap = inches_to_points(args.v_gap)
	inset_value = getattr(args, "inset", DEFAULT_INSET / POINTS_PER_INCH)
	inset = inches_to_points(inset_value)

	config = ImpositionConfig(
		label_width=label_width,
		label_height=label_height,
		columns=COLUMNS,
		rows=ROWS,
		left_margin=left_margin,
		top_margin=top_margin,
		h_gap=h_gap,
		v_gap=v_gap,
		inset=inset,
		x_scale=args.x_scale,
		y_scale=args.y_scale,
		include_partial=args.include_partial,
		calibration=args.calibration,
		draw_outlines=args.draw_outlines,
	)
	return config


#============================================
def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
		Parsed argparse namespace.
	"""
	parser = argparse.ArgumentParser(description="Convert LBX files to Avery 5167 PDF sheets.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	tiles = subparsers.add_parser("tiles", help="Render LBX labels into tile PDFs.")
	tiles.add_argument("inputs", nargs="+", help="LBX files or directories.")
	tiles.add_argument("--tiles-dir", required=True, help="Output directory for tile PDFs.")
	tiles.add_argument("--manifest", required=True, help="Tile manifest JSON path.")
	tiles.add_argument("--gap-threshold", type=float, default=None, help="Gap threshold in points.")
	tiles.add_argument("--label-width", type=float, default=1.7493, help="Label width in inches.")
	tiles.add_argument("--label-height", type=float, default=0.4993, help="Label height in inches.")
	tiles.add_argument("--inset", type=float, default=0.02, help="Inset in inches.")

	impose = subparsers.add_parser("impose", help="Impose tile PDFs onto Avery 5167 sheets.")
	impose.add_argument("inputs", nargs="+", help="Tile PDF files or directories.")
	impose.add_argument("--tiles-manifest", help="Tile manifest JSON path for ordering.")
	impose.add_argument("--output", required=True, help="Output PDF path.")
	impose.add_argument("--manifest", help="Output manifest JSON path.")
	impose.add_argument("--include-partial", action="store_true", help="Include partial final page.")
	impose.add_argument("--calibration", action="store_true", help="Add calibration page.")
	impose.add_argument("--draw-outlines", action="store_true", help="Draw sticker outlines.")
	impose.add_argument("--label-width", type=float, default=1.7493, help="Label width in inches.")
	impose.add_argument("--label-height", type=float, default=0.4993, help="Label height in inches.")
	impose.add_argument("--left-margin", type=float, default=0.3, help="Left margin in inches.")
	impose.add_argument("--top-margin", type=float, default=0.5014, help="Top margin in inches.")
	impose.add_argument("--h-gap", type=float, default=0.3007, help="Horizontal gap in inches.")
	impose.add_argument("--v-gap", type=float, default=0.0021, help="Vertical gap in inches.")
	impose.add_argument("--x-scale", type=float, default=1.0, help="Horizontal scale for placement.")
	impose.add_argument("--y-scale", type=float, default=1.0, help="Vertical scale for placement.")

	return parser.parse_args()


#============================================
def run_tiles(args: argparse.Namespace) -> None:
	"""
	Run tiles command.

	Args:
		args: Parsed argparse namespace.
	"""
	paths = gather_lbx_paths(args.inputs)
	labels, counts_by_file, hashes, thresholds = collect_labels(paths, args.gap_threshold)
	image_cache = build_image_cache(paths)
	tile_config = TileConfig(
		label_width=inches_to_points(args.label_width),
		label_height=inches_to_points(args.label_height),
		inset=inches_to_points(args.inset),
	)
	tiles = render_tiles(
		labels,
		pathlib.Path(args.tiles_dir),
		tile_config,
		image_cache,
		hashes,
	)
	write_tiles_manifest(
		pathlib.Path(args.manifest),
		tiles,
		counts_by_file,
		hashes,
		thresholds,
		tile_config,
	)


#============================================
def run_impose(args: argparse.Namespace) -> None:
	"""
	Run impose command.

	Args:
		args: Parsed argparse namespace.
	"""
	if args.tiles_manifest:
		manifest_path = pathlib.Path(args.tiles_manifest)
		with manifest_path.open("r", encoding="utf-8") as handle:
			data = json.load(handle)
		tile_paths = [pathlib.Path(tile["path"]) for tile in data.get("tiles", [])]
	else:
		tile_paths = gather_tile_paths(args.inputs)

	config = build_config(args)
	output_path = pathlib.Path(args.output)
	result = impose_tiles(tile_paths, output_path, config)

	manifest_path = args.manifest
	if manifest_path is None:
		manifest_path = f"{output_path}.json"
	write_manifest(
		pathlib.Path(manifest_path),
		tile_paths,
		{},
		{},
		{},
		result,
		config,
	)


#============================================
def main() -> None:
	"""
	Main entry point.
	"""
	args = parse_args()
	if args.command == "tiles":
		run_tiles(args)
		return
	if args.command == "impose":
		run_impose(args)
		return
	raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
	main()
