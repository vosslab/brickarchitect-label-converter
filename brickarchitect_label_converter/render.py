"""
Rendering and imposition logic.
"""

# Standard Library
import dataclasses
import io
import json
import pathlib
import zipfile

# PIP3 modules
import PIL.Image
import pypdf
import reportlab.lib.pagesizes
import reportlab.lib.utils
import reportlab.pdfbase.pdfmetrics
import reportlab.pdfgen.canvas

# local repo modules
import brickarchitect_label_converter as balc
import brickarchitect_label_converter.config
import brickarchitect_label_converter.lbx_lib
import brickarchitect_label_converter.segment


LabelObject = balc.lbx_lib.LabelObject
LabelCluster = balc.segment.LabelCluster
ImpositionConfig = balc.config.ImpositionConfig
TileConfig = balc.config.TileConfig
ImpositionResult = balc.config.ImpositionResult

POINTS_PER_INCH = balc.config.POINTS_PER_INCH
DEFAULT_FONT_REGULAR = balc.config.DEFAULT_FONT_REGULAR
DEFAULT_FONT_BOLD = balc.config.DEFAULT_FONT_BOLD
DEFAULT_FONT_ITALIC = balc.config.DEFAULT_FONT_ITALIC
DEFAULT_FONT_BOLD_ITALIC = balc.config.DEFAULT_FONT_BOLD_ITALIC
DEFAULT_TEXT_SIZE = balc.config.DEFAULT_TEXT_SIZE
DEFAULT_TEXT_WEIGHT = balc.config.DEFAULT_TEXT_WEIGHT
DEFAULT_TEXT_MIN_SIZE = balc.config.DEFAULT_TEXT_MIN_SIZE
TEXT_OVERLAP_PADDING = balc.config.TEXT_OVERLAP_PADDING
PROGRESS_BAR_WIDTH = balc.config.PROGRESS_BAR_WIDTH
PROGRESS_UPDATE_EVERY = balc.config.PROGRESS_UPDATE_EVERY
IMAGE_SCALE = balc.config.IMAGE_SCALE
MAX_IMAGE_UPSCALE = balc.config.MAX_IMAGE_UPSCALE


#============================================
def compute_align_offset(available: float, scaled: float, align: str) -> float:
	"""
	Compute an alignment offset.

	Args:
		available: Available dimension.
		scaled: Scaled dimension.
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
	scale_for_images: float | None = None,
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
		elif obj.kind == "image":
			image_scale = IMAGE_SCALE
			if scale_for_images is not None and scale_for_images > 0.0:
				image_scale *= min(1.0, MAX_IMAGE_UPSCALE / scale_for_images)
			center_x = obj.x + obj.width / 2.0
			center_y = obj.y + obj.height / 2.0
			eff_width = obj.width * image_scale
			eff_height = obj.height * image_scale
			x0 = center_x - eff_width / 2.0
			x1 = center_x + eff_width / 2.0
			y0 = center_y - eff_height / 2.0
			y1 = center_y + eff_height / 2.0
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
	box_a: tuple[float, float, float, float],
	box_b: tuple[float, float, float, float],
) -> bool:
	"""
	Check whether two boxes overlap.

	Args:
		box_a: First bounding box.
		box_b: Second bounding box.

	Returns:
		True if boxes overlap.
	"""
	left = max(box_a[0], box_b[0])
	right = min(box_a[2], box_b[2])
	bottom = max(box_a[1], box_b[1])
	top = min(box_a[3], box_b[3])
	return right > left and top > bottom


#============================================
def print_progress(prefix: str, current: int, total: int) -> None:
	"""
	Print a simple progress bar.

	Args:
		prefix: Label text.
		current: Current count.
		total: Total count.
	"""
	if total <= 0:
		return
	percent = int(round((current / total) * 100.0))
	filled = int(round(PROGRESS_BAR_WIDTH * percent / 100.0))
	bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
	print(f"{prefix} [{bar}] {current}/{total} ({percent}%)", end="\r")


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
		scale_for_images=scale,
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
		scale_for_images=scale,
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
	multi_image_messages: list[str] = []
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
		if len(image_objects) > 1:
			multi_image_messages.append(
				f"{tile_name} (images={len(image_objects)}, source={source_path.name})"
			)
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
	if multi_image_messages:
		log_path = output_dir.parent / "multi_image_labels.log"
		with log_path.open("w", encoding="utf-8") as handle:
			handle.write("\n".join(multi_image_messages) + "\n")
		print(f"Multi-image label log written: {log_path}")
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
