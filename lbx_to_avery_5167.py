#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Brother P-touch LBX labels into Avery 5167 label sheets.
"""

# PIP3 modules
import reportlab
import reportlab.lib.pagesizes as reportlab_pagesizes

# local repo modules
import brickarchitect_label_converter as balc
import brickarchitect_label_converter.cli
import brickarchitect_label_converter.config
import brickarchitect_label_converter.lbx_lib
import brickarchitect_label_converter.render
import brickarchitect_label_converter.segment


REPORTLAB = reportlab
REPORTLAB_PAGESIZES = reportlab_pagesizes
POINTS_PER_INCH = balc.config.POINTS_PER_INCH
LABELS_PER_PAGE = balc.config.LABELS_PER_PAGE
COLUMNS = balc.config.COLUMNS
ROWS = balc.config.ROWS
DEFAULT_LABEL_WIDTH = balc.config.DEFAULT_LABEL_WIDTH
DEFAULT_LABEL_HEIGHT = balc.config.DEFAULT_LABEL_HEIGHT
DEFAULT_LEFT_MARGIN = balc.config.DEFAULT_LEFT_MARGIN
DEFAULT_TOP_MARGIN = balc.config.DEFAULT_TOP_MARGIN
DEFAULT_H_GAP = balc.config.DEFAULT_H_GAP
DEFAULT_V_GAP = balc.config.DEFAULT_V_GAP
DEFAULT_INSET = balc.config.DEFAULT_INSET
DEFAULT_GAP_THRESHOLD = balc.config.DEFAULT_GAP_THRESHOLD
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
CATEGORY_TEXT_SIZE = balc.config.CATEGORY_TEXT_SIZE
CATEGORY_TEXT_MARGIN = balc.config.CATEGORY_TEXT_MARGIN
IMAGE_SCALE = balc.config.IMAGE_SCALE
MAX_IMAGE_UPSCALE = balc.config.MAX_IMAGE_UPSCALE
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

LabelObject = balc.lbx_lib.LabelObject
LabelCluster = balc.segment.LabelCluster
ImpositionConfig = balc.config.ImpositionConfig
TileConfig = balc.config.TileConfig
ImpositionResult = balc.config.ImpositionResult

parse_pt_value = balc.lbx_lib.parse_pt_value
normalize_text = balc.lbx_lib.normalize_text
parse_label_xml = balc.lbx_lib.parse_label_xml
parse_label_xml_with_groups = balc.lbx_lib.parse_label_xml_with_groups
extract_background_bounds = balc.lbx_lib.extract_background_bounds

merge_positions = balc.segment.merge_positions
find_separators = balc.segment.find_separators
update_cluster_bounds = balc.segment.update_cluster_bounds
cluster_objects_by_grid = balc.segment.cluster_objects_by_grid
score_cluster_layout = balc.segment.score_cluster_layout
score_label_set = balc.segment.score_label_set
summarize_label_warnings = balc.segment.summarize_label_warnings
build_pairs_by_text = balc.segment.build_pairs_by_text
format_category_title = balc.segment.format_category_title
wrap_text_to_width = balc.segment.wrap_text_to_width
build_category_label = balc.segment.build_category_label
merge_loose_text_into_image_groups = balc.segment.merge_loose_text_into_image_groups
merge_text_only_groups_into_image_groups = balc.segment.merge_text_only_groups_into_image_groups
merge_image_only_clusters = balc.segment.merge_image_only_clusters
clone_clusters = balc.segment.clone_clusters
match_image_text_clusters = balc.segment.match_image_text_clusters
recursive_split_clusters = balc.segment.recursive_split_clusters
build_clusters_from_background = balc.segment.build_clusters_from_background
split_objects_by_separators = balc.segment.split_objects_by_separators
compute_gap_threshold = balc.segment.compute_gap_threshold
cluster_objects = balc.segment.cluster_objects
compute_periodicity_step = balc.segment.compute_periodicity_step
compute_periodicity_offset = balc.segment.compute_periodicity_offset
split_cluster_by_gaps = balc.segment.split_cluster_by_gaps
cluster_objects_by_periodicity = balc.segment.cluster_objects_by_periodicity
create_label_cluster = balc.segment.create_label_cluster
build_label_clusters = balc.segment.build_label_clusters
compute_sha256 = balc.segment.compute_sha256
should_split_group = balc.segment.should_split_group
split_cluster_by_text_pairs = balc.segment.split_cluster_by_text_pairs
merge_text_only_clusters_into_image_clusters = balc.segment.merge_text_only_clusters_into_image_clusters
write_label_count_log = balc.segment.write_label_count_log
gather_lbx_paths = balc.segment.gather_lbx_paths
collect_labels = balc.segment.collect_labels

compute_align_offset = balc.render.compute_align_offset
compute_text_bbox = balc.render.compute_text_bbox
compute_text_visual_bounds = balc.render.compute_text_visual_bounds
compute_visual_bounds = balc.render.compute_visual_bounds
boxes_intersect = balc.render.boxes_intersect
print_progress = balc.render.print_progress
parse_hex_color = balc.render.parse_hex_color
sanitize_token = balc.render.sanitize_token
map_font_name = balc.render.map_font_name
draw_text_object = balc.render.draw_text_object
draw_image_object = balc.render.draw_image_object
draw_poly_object = balc.render.draw_poly_object
draw_rect_object = balc.render.draw_rect_object
build_image_cache = balc.render.build_image_cache
draw_label_cluster = balc.render.draw_label_cluster
draw_cluster_to_tile = balc.render.draw_cluster_to_tile
render_tile_pdf = balc.render.render_tile_pdf
draw_calibration_page = balc.render.draw_calibration_page
draw_label_outlines = balc.render.draw_label_outlines
render_labels_to_pdf = balc.render.render_labels_to_pdf
render_tiles = balc.render.render_tiles
build_outline_overlay = balc.render.build_outline_overlay
build_calibration_page = balc.render.build_calibration_page
impose_tiles = balc.render.impose_tiles
write_manifest = balc.render.write_manifest

build_config = balc.cli.build_config
build_tile_config = balc.cli.build_tile_config
parse_args = balc.cli.parse_args
run_pipeline = balc.cli.run_pipeline
main = balc.cli.main


if __name__ == "__main__":
	main()
