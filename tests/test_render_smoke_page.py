import argparse
import pathlib

import fitz
import PIL.Image

import lbx_to_avery_5167


DPI = 300
INK_THRESHOLD = 240
EDGE_RATIO_LIMIT = 0.01


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
def _render_pdf_first_page(path: pathlib.Path) -> PIL.Image.Image:
	"""
	Render the first page of a PDF to an image.

	Args:
		path: PDF path.

	Returns:
		PIL image.
	"""
	document = fitz.open(path)
	page = document[0]
	scale = DPI / 72.0
	matrix = fitz.Matrix(scale, scale)
	pixmap = page.get_pixmap(matrix=matrix, alpha=False)
	image = PIL.Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
	document.close()
	return image


#============================================
def _count_ink_ratio(gray: PIL.Image.Image, threshold: int) -> float:
	"""
	Compute the ink ratio for a grayscale strip.

	Args:
		gray: Grayscale image region.
		threshold: Pixel intensity threshold.

	Returns:
		Ink ratio.
	"""
	pixels = gray.getdata()
	if not pixels:
		return 0.0
	ink = sum(1 for value in pixels if value < threshold)
	return ink / len(pixels)


#============================================
def test_rendered_page_edge_strips(tmp_path: pathlib.Path) -> None:
	"""
	Smoke test the first page for edge-strip ink bleed.
	"""
	source = pathlib.Path("LEGO_BRICK_LABELS-v40/Labels/1.BASIC/BASIC-brick-1x.lbx")
	if not source.exists():
		return

	labels, _counts, hashes, _thresholds = lbx_to_avery_5167.collect_labels(
		[source],
		None,
		True,
		None,
		verbose=False,
	)
	labels = [cluster for cluster in labels if not _is_category_label(cluster)]

	image_cache = lbx_to_avery_5167.build_image_cache([source])
	args = argparse.Namespace(
		include_partial=True,
		calibration=False,
		draw_outlines=False,
		max_pages=1,
		max_labels=None,
	)
	tile_config = lbx_to_avery_5167.build_tile_config(args)
	tiles_dir = tmp_path / "tiles"
	tiles = lbx_to_avery_5167.render_tiles(labels, tiles_dir, tile_config, image_cache, hashes)

	output_pdf = tmp_path / "smoke.pdf"
	config = lbx_to_avery_5167.build_config(args)
	lbx_to_avery_5167.impose_tiles(
		[pathlib.Path(tile["path"]) for tile in tiles],
		output_pdf,
		config,
	)

	image = _render_pdf_first_page(output_pdf)
	gray = image.convert("L")
	scale = DPI / 72.0
	strip = max(2, int(round(scale * 0.01)))

	page_width, page_height = lbx_to_avery_5167.reportlab.lib.pagesizes.letter
	violations = []
	for row in range(config.rows):
		for col in range(config.columns):
			cell_x = config.left_margin + col * (config.label_width + config.h_gap)
			cell_y = page_height - config.top_margin - config.label_height - row * (
				config.label_height + config.v_gap
			)
			x0 = int(round(cell_x * scale))
			x1 = int(round((cell_x + config.label_width) * scale))
			y0 = int(round((page_height - (cell_y + config.label_height)) * scale))
			y1 = int(round((page_height - cell_y) * scale))
			if x1 <= x0 or y1 <= y0:
				continue
			left = gray.crop((x0, y0, x0 + strip, y1))
			right = gray.crop((x1 - strip, y0, x1, y1))
			top = gray.crop((x0, y0, x1, y0 + strip))
			bottom = gray.crop((x0, y1 - strip, x1, y1))
			for edge_name, edge in (
				("left", left),
				("right", right),
				("top", top),
				("bottom", bottom),
			):
				ratio = _count_ink_ratio(edge, INK_THRESHOLD)
				if ratio > EDGE_RATIO_LIMIT:
					violations.append(
						f"row {row} col {col} edge {edge_name} ratio {ratio:.3f}"
					)

	if violations:
		message = "Edge strip ink detected in rendered page:\n"
		message += "\n".join(violations[:10])
		raise AssertionError(message)
