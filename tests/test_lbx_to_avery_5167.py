import pathlib

import lbx_to_avery_5167


#============================================
def test_basic_brick_label_count_and_pdf(tmp_path: pathlib.Path) -> None:
	"""
	Verify label clustering and PDF output for a known LBX file.
	"""
	path = pathlib.Path("LEGO_BRICK_LABELS-v40/Labels/1.BASIC/BASIC-brick-1x.lbx")
	labels, counts_by_file, hashes, thresholds = lbx_to_avery_5167.collect_labels([path], None)
	assert len(labels) == 10
	assert counts_by_file[str(path)] == 10
	assert str(path) in hashes
	assert str(path) in thresholds

	output_path = tmp_path / "labels.pdf"
	config = lbx_to_avery_5167.ImpositionConfig(
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
	)
	result = lbx_to_avery_5167.render_labels_to_pdf(labels, [path], output_path, config)
	assert result.printed_labels == 10
	assert output_path.exists()
	assert output_path.stat().st_size > 0
