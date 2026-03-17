"""Tests for scripts/add_box_to_bgf.py."""
from pathlib import Path

import pytest

from atlas_toolkit.io.bgf import read_bgf, write_bgf
from atlas_toolkit.core.box import get_box
from atlas_toolkit.scripts.add_box_to_bgf import add_box_to_bgf

FIXTURES = Path(__file__).parent / "fixtures"
TIP3_BGF = FIXTURES / "tip3.bgf"    # no CRYSTX
SPC_BOX  = FIXTURES / "spc_box.bgf" # has CRYSTX


# ── add_box_to_bgf: BGF without CRYSTX ────────────────────────────────────────

@pytest.fixture(scope="module")
def tip3_no_box():
    atoms, bonds, headers = read_bgf(TIP3_BGF)
    return atoms, bonds, headers


def test_add_box_adds_crystx(tip3_no_box):
    atoms, bonds, headers = tip3_no_box
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    crystx_lines = [h for h in new_headers if h.startswith("CRYSTX")]
    assert len(crystx_lines) == 1


def test_add_box_has_period_axes_sgname_cells(tip3_no_box):
    atoms, bonds, headers = tip3_no_box
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    tags = {h.strip().split()[0] for h in new_headers if h.strip()}
    for tag in ("PERIOD", "AXES", "SGNAME", "CRYSTX", "CELLS"):
        assert tag in tags, f"{tag} missing from headers"


def test_add_box_dimensions_cover_span(tip3_no_box):
    """Box lengths must be ≥ the coordinate span of atoms (including VDW padding)."""
    atoms, bonds, headers = tip3_no_box
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    box = get_box(atoms, new_headers)
    xs = [float(a["XCOORD"]) for a in atoms.values()]
    ys = [float(a["YCOORD"]) for a in atoms.values()]
    zs = [float(a["ZCOORD"]) for a in atoms.values()]
    assert box["X"]["len"] >= max(xs) - min(xs)
    assert box["Y"]["len"] >= max(ys) - min(ys)
    assert box["Z"]["len"] >= max(zs) - min(zs)


def test_add_box_positive_lengths(tip3_no_box):
    atoms, bonds, headers = tip3_no_box
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    box = get_box(atoms, new_headers)
    for dim in ("X", "Y", "Z"):
        assert box[dim]["len"] > 0


def test_add_box_does_not_mutate_original_headers(tip3_no_box):
    atoms, bonds, headers = tip3_no_box
    original_len = len(headers)
    add_box_to_bgf(atoms, bonds, headers)
    assert len(headers) == original_len  # original unchanged


# ── add_box_to_bgf: BGF that already has CRYSTX ───────────────────────────────

@pytest.fixture(scope="module")
def spc_with_box():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    return atoms, bonds, headers


def test_replace_existing_crystx(spc_with_box):
    """Existing CRYSTX is removed and a new one added — exactly one in result."""
    atoms, bonds, headers = spc_with_box
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    crystx_lines = [h for h in new_headers if h.startswith("CRYSTX")]
    assert len(crystx_lines) == 1


def test_no_duplicate_box_tags(spc_with_box):
    atoms, bonds, headers = spc_with_box
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    for tag in ("PERIOD", "AXES", "SGNAME", "CELLS"):
        matches = [h for h in new_headers if h.strip().startswith(tag)]
        assert len(matches) == 1, f"Expected 1 {tag} line, got {len(matches)}"


# ── round-trip: write then re-read ────────────────────────────────────────────

def test_round_trip_crystx(tmp_path, tip3_no_box):
    atoms, bonds, headers = tip3_no_box
    new_headers = add_box_to_bgf(atoms, bonds, headers)
    out = tmp_path / "tip3_with_box.bgf"
    write_bgf(atoms, bonds, out, new_headers)
    # Re-read: should now have a box
    atoms2, bonds2, headers2 = read_bgf(out)
    box = get_box(atoms2, headers2)
    assert box["X"]["len"] > 0
    assert box["Y"]["len"] > 0
    assert box["Z"]["len"] > 0
