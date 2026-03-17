"""
atlas-add-box-to-bgf — Add (or replace) a CRYSTX periodic-box record in a BGF.

If the BGF already has a CRYSTX line, the box dimensions are taken from the
atom coordinates (bounding box) and any existing CRYSTX / PERIOD / AXES /
SGNAME / CELLS header lines are replaced.

Usage
-----
atlas-add-box-to-bgf input.bgf [output.bgf]

If output is omitted, the input file is overwritten.

Port of addBoxToBGF.pl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_BOX_LINES_BEFORE_CRYSTX = (
    "PERIOD 111",
    "AXES   ZYX",
    "SGNAME P 1                  1    1",
)
_BOX_LINES_AFTER_CRYSTX = (
    "CELLS    -1    1   -1    1   -1    1",
)
_HEADER_BOX_TAGS = {"PERIOD", "AXES", "SGNAME", "CRYSTX", "CELLS"}


def add_box_to_bgf(
    atoms: dict,
    bonds: dict,
    headers: list,
    padding: float = 0.0,
) -> list:
    """Recompute the bounding box and return an updated headers list.

    Any existing PERIOD / AXES / SGNAME / CRYSTX / CELLS lines are removed
    and fresh ones are appended.  All other header lines are kept.

    Parameters
    ----------
    atoms   : atom dict from read_bgf
    bonds   : bonds dict from read_bgf  (unused, kept for API consistency)
    headers : header list from read_bgf
    padding : extra padding added to each side of the bounding box (Å)

    Returns
    -------
    Updated headers list (new object; originals are not mutated).
    """
    from atlas_toolkit.core.box import get_box

    # Strip existing box headers
    clean = [h for h in headers
             if not any(h.strip().startswith(tag) for tag in _HEADER_BOX_TAGS)]

    # Compute box from atom coordinates (ignores any CRYSTX in headers)
    box = get_box(atoms, headers=None, padding=padding)

    lx = box["X"]["len"]
    ly = box["Y"]["len"]
    lz = box["Z"]["len"]
    ax = box["X"]["angle"]
    ay = box["Y"]["angle"]
    az = box["Z"]["angle"]

    crystx = f"CRYSTX {lx:11.5f}{ly:11.5f}{lz:11.5f}{ax:11.5f}{ay:11.5f}{az:11.5f}"

    new_headers = list(clean)
    for line in _BOX_LINES_BEFORE_CRYSTX:
        new_headers.append(line)
    new_headers.append(crystx)
    for line in _BOX_LINES_AFTER_CRYSTX:
        new_headers.append(line)

    return new_headers


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="atlas-add-box-to-bgf",
        description="Add or replace the CRYSTX box record in a BGF file.",
    )
    p.add_argument("bgf",  help="Input BGF file")
    p.add_argument("save", nargs="?", default=None,
                   help="Output BGF file (default: overwrite input)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    bgf_path = Path(args.bgf)
    if not bgf_path.exists():
        sys.exit(f"Error: file not found: {bgf_path}")

    save_path = Path(args.save) if args.save else bgf_path

    from atlas_toolkit.io.bgf import read_bgf, write_bgf

    atoms, bonds, headers = read_bgf(bgf_path)
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    write_bgf(atoms, bonds, save_path, new_headers)
    print(f"Written: {save_path}")


if __name__ == "__main__":
    main()
