"""
CLI entry points for LBX to Avery conversion.
"""

# Standard Library
import argparse
import pathlib
import time

# local repo modules
import brickarchitect_label_converter as balc
import brickarchitect_label_converter.config
import brickarchitect_label_converter.render
import brickarchitect_label_converter.segment


ImpositionConfig = balc.config.ImpositionConfig
TileConfig = balc.config.TileConfig

DEFAULT_LABEL_WIDTH = balc.config.DEFAULT_LABEL_WIDTH
DEFAULT_LABEL_HEIGHT = balc.config.DEFAULT_LABEL_HEIGHT
COLUMNS = balc.config.COLUMNS
ROWS = balc.config.ROWS
DEFAULT_LEFT_MARGIN = balc.config.DEFAULT_LEFT_MARGIN
DEFAULT_TOP_MARGIN = balc.config.DEFAULT_TOP_MARGIN
DEFAULT_H_GAP = balc.config.DEFAULT_H_GAP
DEFAULT_V_GAP = balc.config.DEFAULT_V_GAP
DEFAULT_INSET = balc.config.DEFAULT_INSET
DEFAULT_TEXT_SIZE = balc.config.DEFAULT_TEXT_SIZE
DEFAULT_TEXT_WEIGHT = balc.config.DEFAULT_TEXT_WEIGHT


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

	paths = balc.segment.gather_lbx_paths(args.inputs)
	print(f"LBX files found: {len(paths)}")

	start_time = time.perf_counter()
	collect_start = time.perf_counter()
	labels, counts_by_file, hashes, thresholds = balc.segment.collect_labels(
		paths,
		None,
		args.normalize_text,
		args.max_labels,
		verbose=True,
	)
	collect_end = time.perf_counter()
	print(f"Labels collected: {len(labels)}")

	output_path = pathlib.Path(args.output_path)
	balc.segment.write_label_count_log(counts_by_file, output_path)
	if args.stop_before_rendering:
		print("Stopping before rendering tiles.")
		total_time = time.perf_counter() - start_time
		print(
			"Timing: collect={:.2f}s total={:.2f}s".format(
				collect_end - collect_start,
				total_time,
			)
		)
		return

	render_start = time.perf_counter()
	image_cache = balc.render.build_image_cache(paths)
	tile_config = build_tile_config(args)
	tiles_dir = output_path.parent / "tiles"
	print(f"Tiles directory: {tiles_dir}")
	print("Rendering tiles")
	tiles = balc.render.render_tiles(labels, tiles_dir, tile_config, image_cache, hashes)
	render_end = time.perf_counter()
	print(f"Tiles rendered: {len(tiles)}")

	tile_paths = [pathlib.Path(tile["path"]) for tile in tiles]
	config = build_config(args)
	print("Imposing tiles")
	impose_start = time.perf_counter()
	result = balc.render.impose_tiles(tile_paths, output_path, config)
	impose_end = time.perf_counter()
	print(f"Pages written: {result.pages}")
	print(f"Labels printed: {result.printed_labels}")
	print(f"Labels leftover: {result.leftover_labels}")

	manifest_path = args.manifest_path
	if manifest_path is None:
		manifest_path = f"{output_path}.json"
	balc.render.write_manifest(
		pathlib.Path(manifest_path),
		paths,
		counts_by_file,
		hashes,
		thresholds,
		result,
		config,
	)

	total_time = time.perf_counter() - start_time
	print(
		"Timing: collect={:.2f}s render={:.2f}s impose={:.2f}s total={:.2f}s".format(
			collect_end - collect_start,
			render_end - render_start,
			impose_end - impose_start,
			total_time,
		)
	)
	print(f"Manifest written: {manifest_path}")


#============================================
def main() -> None:
	"""
	Main entry point.
	"""
	args = parse_args()
	run_pipeline(args)
