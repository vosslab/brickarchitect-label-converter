"""
Shared configuration and constants.
"""

import dataclasses


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
PERIODICITY_BIN_SIZE = 1.0
PERIODICITY_CONFIDENCE_MIN = 0.45
PERIODICITY_MIN_DELTAS = 3
PERIODICITY_MIN_STEP = 6.0
RECURSIVE_SPLIT_MAX_DEPTH = 2
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
MULTI_IMAGE_SPLIT_WHITELIST = {
	"CLIP-clip_3",
	"ANGLE-wedge_plate_63",
}
TEXT_ONLY_ADJACENT_MERGE_WHITELIST = {
	"INDEX-slope",
}


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
