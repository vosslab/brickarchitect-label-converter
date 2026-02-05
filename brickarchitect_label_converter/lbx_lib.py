"""
LBX parsing and normalization.
"""

# Standard Library
import dataclasses
import unicodedata
import xml.etree.ElementTree as StdElementTree

# PIP3 modules
import defusedxml.ElementTree as ElementTree


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
