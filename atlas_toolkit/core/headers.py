"""
BGF header manipulation — port of relevant functions from FileFormats.pm.
"""
import os
import time
from typing import Optional

from ..types import HeadersList

__all__ = ["create_headers", "insert_header_remark", "add_box_to_header"]

# Records that appear in the periodic-box block (removed/replaced by add_box_to_header)
_BOX_KEYS = ("PERIOD", "AXES", "SGNAME", "CRYSTX", "CELLS")


def create_headers(bbox: dict | None = None, title: str = "default.bgf") -> HeadersList:
    """Create a minimal BGF header list (Perl createHeaders).

    Args:
        bbox:  optional box dict {X: {hi, lo, angle}, Y: ..., Z: ...}
        title: DESCRP title string
    """
    muser = os.environ.get("USER", "unknown")
    mdate = time.strftime("%a %b %d %H:%M:%S %Y")

    headers: HeadersList = [
        "BIOGRF 200",
        f"DESCRP {title}",
        f"REMARK Created by {muser} on {mdate}",
        "FORCEFIELD AMBER",
    ]

    if bbox:
        x_len = bbox["X"]["hi"] - bbox["X"]["lo"]
        y_len = bbox["Y"]["hi"] - bbox["Y"]["lo"]
        z_len = bbox["Z"]["hi"] - bbox["Z"]["lo"]
        x_ang = bbox["X"].get("angle", 90.0)
        y_ang = bbox["Y"].get("angle", 90.0)
        z_ang = bbox["Z"].get("angle", 90.0)
        crystx = (f"CRYSTX {x_len:11.5f}{y_len:11.5f}{z_len:11.5f}"
                  f"{x_ang:11.5f}{y_ang:11.5f}{z_ang:11.5f}")
        headers += [
            "PERIOD 111",
            "AXES   ZYX",
            "SGNAME P 1  1 1",
            crystx,
            "CELLS  -1    1   -1    1   -1    1",
        ]

    return headers


def insert_header_remark(headers: HeadersList, remark: str) -> HeadersList:
    """Insert a REMARK line after the last existing REMARK (Perl insertHeaderRemark).

    Inserts after the block of REMARK/DESCRP lines near the top.
    Modifies headers in place and returns it.
    """
    if not headers:
        headers = create_headers(title="default.bgf")

    start = end = 0
    for i, h in enumerate(headers):
        if h is None:
            continue
        if h.startswith("DESCRP"):
            start = i
        elif h.startswith("REMARK"):
            if not start:
                start = i - 1
            end = i

    if not start and not end:
        return headers

    # Insert after 'end' (last REMARK)
    headers.insert(end + 1, remark)

    # Remove any None entries that may have been left
    headers[:] = [h for h in headers if h is not None]
    return headers


def add_box_to_header(headers: HeadersList, box: dict) -> None:
    """Update or insert the periodic-box block in headers (Perl addBoxToHeader).

    Removes any existing PERIOD/AXES/SGNAME/CRYSTX/CELLS lines then appends
    a fresh block.  Modifies headers in place.
    """
    x_len = box["X"]["hi"] - box["X"]["lo"]
    y_len = box["Y"]["hi"] - box["Y"]["lo"]
    z_len = box["Z"]["hi"] - box["Z"]["lo"]

    angles = ""
    for dim in ("X", "Y", "Z"):
        ang = box[dim].get("angle", 90.0)
        angles += f"{ang:11.5f}"

    rec = (
        f"PERIOD 111\nAXES   ZYX\nSGNAME P 1  1 1\n"
        f"CRYSTX {x_len:11.5f}{y_len:11.5f}{z_len:11.5f}{angles}\n"
        f"CELLS -1 1   -1 1   -1 1"
    )

    # Remove existing box records
    i = 0
    while i < len(headers):
        if headers[i] and any(headers[i].startswith(k) for k in _BOX_KEYS):
            headers.pop(i)
        else:
            i += 1

    headers.append(rec)
