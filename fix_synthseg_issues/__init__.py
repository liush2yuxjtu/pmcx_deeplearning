# fix_synthseg_issues package initialization

from .core import (
    detect_segmentation_tool,
    run_stage1,
    run_stage2,
    show_src_vol,
    view_result,
    view_result_mask,
    from_csv_get_subject_coord
)

from .core import tissue_mappings, mcx_tissue_mapping

__version__ = "1.0.0"
__author__ = "Trae AI"
__email__ = "trae.ai@example.com"
__description__ = "MCX pipeline with SynthSeg support"

__all__ = [
    "detect_segmentation_tool",
    "run_stage1",
    "run_stage2",
    "show_src_vol",
    "view_result",
    "view_result_mask",
    "from_csv_get_subject_coord",
    "tissue_mappings",
    "mcx_tissue_mapping"
]