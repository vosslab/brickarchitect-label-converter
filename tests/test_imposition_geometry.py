import lbx_to_avery_5167
import reportlab.lib.pagesizes


#============================================
def build_default_config() -> lbx_to_avery_5167.ImpositionConfig:
	"""
	Build a default ImpositionConfig for tests.
	"""
	return lbx_to_avery_5167.ImpositionConfig(
		label_width=lbx_to_avery_5167.DEFAULT_LABEL_WIDTH,
		label_height=lbx_to_avery_5167.DEFAULT_LABEL_HEIGHT,
		columns=lbx_to_avery_5167.COLUMNS,
		rows=lbx_to_avery_5167.ROWS,
		left_margin=lbx_to_avery_5167.DEFAULT_LEFT_MARGIN,
		top_margin=lbx_to_avery_5167.DEFAULT_TOP_MARGIN,
		h_gap=lbx_to_avery_5167.DEFAULT_H_GAP,
		v_gap=lbx_to_avery_5167.DEFAULT_V_GAP,
		inset=lbx_to_avery_5167.DEFAULT_INSET,
		x_scale=1.0,
		y_scale=1.0,
		include_partial=True,
		calibration=False,
		draw_outlines=False,
		max_pages=None,
		max_labels=None,
		cluster_align_horizontal="CENTER",
		cluster_align_vertical="CENTER",
		text_align_horizontal=None,
		text_align_vertical=None,
		text_font_size=None,
		text_font_weight=None,
		text_fit=True,
	)


#============================================
def compute_cell_box(
	config: lbx_to_avery_5167.ImpositionConfig,
	row: int,
	col: int,
) -> tuple[float, float, float, float]:
	"""
	Compute the bounding box for a label slot.

	Args:
		config: Imposition configuration.
		row: Row index.
		col: Column index.

	Returns:
		Tuple of (x0, y0, x1, y1).
	"""
	page_width, page_height = reportlab.lib.pagesizes.letter
	scaled_label_width = config.label_width * config.x_scale
	scaled_label_height = config.label_height * config.y_scale
	scaled_h_gap = config.h_gap * config.x_scale
	scaled_v_gap = config.v_gap * config.y_scale

	cell_x = config.left_margin + col * (scaled_label_width + scaled_h_gap)
	cell_y = page_height - config.top_margin - scaled_label_height - row * (
		scaled_label_height + scaled_v_gap
	)

	return (cell_x, cell_y, cell_x + scaled_label_width, cell_y + scaled_label_height)


#============================================
def test_grid_boxes_within_page() -> None:
	"""
	Ensure all label slots are on-page.
	"""
	config = build_default_config()
	page_width, page_height = reportlab.lib.pagesizes.letter
	for row in range(config.rows):
		for col in range(config.columns):
			x0, y0, x1, y1 = compute_cell_box(config, row, col)
			assert 0.0 <= x0 < x1 <= page_width
			assert 0.0 <= y0 < y1 <= page_height


#============================================
def test_grid_boxes_non_overlapping() -> None:
	"""
	Ensure adjacent slots do not overlap.
	"""
	config = build_default_config()
	epsilon = 0.001
	for col in range(config.columns - 1):
		left_box = compute_cell_box(config, 0, col)
		right_box = compute_cell_box(config, 0, col + 1)
		assert right_box[0] >= left_box[2] - epsilon

	for row in range(config.rows - 1):
		upper_box = compute_cell_box(config, row, 0)
		lower_box = compute_cell_box(config, row + 1, 0)
		assert lower_box[3] <= upper_box[1] + epsilon


#============================================
def test_scale_to_fit_math() -> None:
	"""
	Ensure scale-to-fit math keeps content within the inset box.
	"""
	config = build_default_config()
	available_width = config.label_width - 2.0 * config.inset
	available_height = config.label_height - 2.0 * config.inset
	assert available_width > 0.0
	assert available_height > 0.0

	cluster_width = available_width * 1.2
	cluster_height = available_height * 0.8
	scale = min(available_width / cluster_width, available_height / cluster_height)
	scaled_width = cluster_width * scale
	scaled_height = cluster_height * scale
	assert scale <= 1.0
	assert scaled_width <= available_width + 0.001
	assert scaled_height <= available_height + 0.001
