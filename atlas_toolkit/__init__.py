"""ATLAS toolkit — Python port of the ATLAS molecular simulation library."""
from .io.bgf import read_bgf, write_bgf, parse_struct_file
from .io.bgf_parmed import (
    bgf_to_parmed, parmed_to_bgf,
    load_bgf_as_parmed, save_parmed_as_bgf,
)
from .io.ff_detect import detect_ff, suggest_ff_files, fftype_family
from .core.headers import create_headers, insert_header_remark, add_box_to_header
from .core.manip_atoms import get_mols, select_atoms, build_selection, add_mols_to_selection
from .core.general import trim, file_tester, has_cell

__all__ = [
    "read_bgf", "write_bgf", "parse_struct_file",
    "bgf_to_parmed", "parmed_to_bgf", "load_bgf_as_parmed", "save_parmed_as_bgf",
    "detect_ff", "suggest_ff_files", "fftype_family",
    "create_headers", "insert_header_remark", "add_box_to_header",
    "get_mols", "select_atoms", "build_selection", "add_mols_to_selection",
    "trim", "file_tester", "has_cell",
]
