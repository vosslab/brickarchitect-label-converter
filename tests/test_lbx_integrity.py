import math
import pathlib
import zipfile
import defusedxml.ElementTree

import lbx_to_avery_5167


#============================================
def fixture_paths() -> list[pathlib.Path]:
	"""
	Return a small, deterministic set of LBX fixtures.
	"""
	paths = [
		pathlib.Path("LEGO_BRICK_LABELS-v40/Labels/1.BASIC/BASIC-brick-1x.lbx"),
		pathlib.Path("LEGO_BRICK_LABELS-v40/Labels/5.HINGE/HINGE-hinge.lbx"),
		pathlib.Path("LEGO_BRICK_LABELS-v40/Labels/12.TECHNIC/TECHNIC-rack.lbx"),
	]
	for path in paths:
		assert path.exists()
	return paths


#============================================
def extract_namespace(tag: str) -> str:
	"""
	Extract the namespace URI from an XML tag.

	Args:
		tag: XML tag string.

	Returns:
		Namespace URI or empty string.
	"""
	if tag.startswith("{") and "}" in tag:
		return tag.split("}", 1)[0][1:]
	return ""


#============================================
def load_label_xml(path: pathlib.Path) -> bytes:
	"""
	Read label.xml from an LBX file.

	Args:
		path: LBX file path.

	Returns:
		XML bytes.
	"""
	with zipfile.ZipFile(path, "r") as archive:
		return archive.read("label.xml")


#============================================
def collect_objects(label_xml: bytes) -> list[lbx_to_avery_5167.LabelObject]:
	"""
	Collect label objects from XML.

	Args:
		label_xml: XML bytes.

	Returns:
		List of LabelObject entries.
	"""
	group_clusters, loose_objects = lbx_to_avery_5167.parse_label_xml_with_groups(
		label_xml,
		apply_normalization=True,
	)
	objects: list[lbx_to_avery_5167.LabelObject] = []
	for group in group_clusters:
		objects.extend(group)
	objects.extend(loose_objects)
	return objects


#============================================
def test_lbx_zip_and_xml_sanity() -> None:
	"""
	Verify each fixture LBX is a zip and has expected XML structure.
	"""
	allowed_namespaces = {
		"http://schemas.brother.info/ptouch/2007/lbx/main",
		"http://schemas.brother.info/ptouch/2007/lbx/style",
		"http://schemas.brother.info/ptouch/2007/lbx/text",
		"http://schemas.brother.info/ptouch/2007/lbx/draw",
		"http://schemas.brother.info/ptouch/2007/lbx/image",
		"http://schemas.brother.info/ptouch/2007/lbx/barcode",
		"http://schemas.brother.info/ptouch/2007/lbx/database",
		"http://schemas.brother.info/ptouch/2007/lbx/table",
		"http://schemas.brother.info/ptouch/2007/lbx/cable",
	}
	for path in fixture_paths():
		assert zipfile.is_zipfile(path)
		with zipfile.ZipFile(path, "r") as archive:
			assert "label.xml" in archive.namelist()
		label_xml = load_label_xml(path)
		root = defusedxml.ElementTree.fromstring(label_xml)
		assert root.tag.endswith("document")
		paper = root.find(".//{*}paper")
		objects = root.find(".//{*}objects")
		background = root.find(".//{*}backGround")
		assert paper is not None
		assert objects is not None
		assert background is not None

		for element in root.iter():
			namespace = extract_namespace(element.tag)
			if namespace:
				assert namespace in allowed_namespaces


#============================================
def test_lbx_image_references_exist() -> None:
	"""
	Verify all image objects reference assets in the LBX archive.
	"""
	for path in fixture_paths():
		label_xml = load_label_xml(path)
		objects = collect_objects(label_xml)
		with zipfile.ZipFile(path, "r") as archive:
			names = set(archive.namelist())
		for obj in objects:
			if obj.kind != "image":
				continue
			assert obj.image_name
			assert obj.image_name in names


#============================================
def test_lbx_font_and_bounds_sanity() -> None:
	"""
	Validate text font mapping and object bounds.
	"""
	max_coord = 5000.0
	min_size = 0.1
	for path in fixture_paths():
		label_xml = load_label_xml(path)
		objects = collect_objects(label_xml)
		for obj in objects:
			assert math.isfinite(obj.x)
			assert math.isfinite(obj.y)
			assert math.isfinite(obj.width)
			assert math.isfinite(obj.height)
			assert abs(obj.x) <= max_coord
			assert abs(obj.y) <= max_coord
			assert abs(obj.width) <= max_coord
			assert abs(obj.height) <= max_coord
			if obj.kind in ("text", "image"):
				assert obj.width > min_size
				assert obj.height > min_size
			if obj.kind == "poly":
				for point in obj.poly_points:
					assert math.isfinite(point[0])
					assert math.isfinite(point[1])
					assert abs(point[0]) <= max_coord
					assert abs(point[1]) <= max_coord
			if obj.kind == "text":
				mapped = lbx_to_avery_5167.map_font_name(
					obj.font_name,
					obj.font_weight,
					obj.font_italic,
				)
				assert mapped in {
					lbx_to_avery_5167.DEFAULT_FONT_REGULAR,
					lbx_to_avery_5167.DEFAULT_FONT_BOLD,
					lbx_to_avery_5167.DEFAULT_FONT_ITALIC,
					lbx_to_avery_5167.DEFAULT_FONT_BOLD_ITALIC,
				}


#============================================
def test_lbx_canvas_bounds_consistent() -> None:
	"""
	Ensure objects fall within a reasonable canvas range.
	"""
	padding = 20.0
	for path in fixture_paths():
		label_xml = load_label_xml(path)
		root = defusedxml.ElementTree.fromstring(label_xml)
		background = root.find(".//{*}backGround")
		assert background is not None

		bg_width = lbx_to_avery_5167.parse_pt_value(background.attrib.get("width"), 0.0)
		bg_height = lbx_to_avery_5167.parse_pt_value(background.attrib.get("height"), 0.0)
		assert bg_width > 0.0
		assert bg_height > 0.0

		objects = collect_objects(label_xml)
		for obj in objects:
			max_x = obj.x + obj.width
			max_y = obj.y + obj.height
			assert max_x <= bg_width + padding
			assert max_y <= bg_height + padding


#============================================
def test_label_clusters_do_not_span_rows() -> None:
	"""
	Ensure label clusters do not mix objects from multiple rows.
	"""
	for path in fixture_paths():
		label_xml = load_label_xml(path)
		objects = collect_objects(label_xml)
		visual_objects = [obj for obj in objects if obj.kind in ("text", "image")]
		if not visual_objects:
			continue
		row_threshold = lbx_to_avery_5167.compute_gap_threshold(
			visual_objects,
			lbx_to_avery_5167.DEFAULT_GAP_THRESHOLD,
			axis="y",
		)
		row_clusters = lbx_to_avery_5167.cluster_objects(
			visual_objects,
			row_threshold,
			axis="y",
		)
		row_index: dict[int, int] = {}
		for index, row in enumerate(row_clusters):
			for obj in row:
				row_index[id(obj)] = index

		x_threshold = lbx_to_avery_5167.compute_gap_threshold(
			visual_objects,
			lbx_to_avery_5167.DEFAULT_GAP_THRESHOLD,
		)
		label_clusters = lbx_to_avery_5167.build_label_clusters(
			visual_objects,
			str(path),
			x_threshold,
		)
		for cluster in label_clusters:
			rows = {row_index[id(obj)] for obj in cluster.objects if id(obj) in row_index}
			assert len(rows) <= 1
