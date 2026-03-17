"""Tests for core/box.py."""
import math
from pathlib import Path

import pytest

from atlas_toolkit.core.box import (
    cart2frac,
    center_atoms,
    frac2cart,
    get_box,
    get_box_displacement_tensor,
    get_box_vol,
    init_box,
    map2unit_cell,
)
from atlas_toolkit.io.bgf import read_bgf

FIXTURES = Path(__file__).parent / "fixtures"
SPC_BOX  = FIXTURES / "spc_box.bgf"
TIP3     = FIXTURES / "tip3.bgf"


# ── get_box from CRYSTX header ───────────────────────────────────────────────

def test_get_box_from_crystx():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)
    # CRYSTX line should give a/b/c > 0
    assert box["X"]["len"] > 0
    assert box["Y"]["len"] > 0
    assert box["Z"]["len"] > 0


def test_get_box_crystx_lo_is_zero():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)
    # CRYSTX convention: lo = 0
    assert box["X"]["lo"] == pytest.approx(0.0)
    assert box["Y"]["lo"] == pytest.approx(0.0)
    assert box["Z"]["lo"] == pytest.approx(0.0)


def test_get_box_crystx_angles_default_90():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)
    assert box["X"]["angle"] == pytest.approx(90.0)
    assert box["Y"]["angle"] == pytest.approx(90.0)
    assert box["Z"]["angle"] == pytest.approx(90.0)


def test_get_box_no_headers_uses_atoms():
    atoms, bonds, _ = read_bgf(TIP3)
    box = get_box(atoms, headers=None)
    # Bounding box should enclose all atoms
    for a in atoms.values():
        assert a["XCOORD"] >= box["X"]["lo"]
        assert a["XCOORD"] <= box["X"]["hi"]
        assert a["YCOORD"] >= box["Y"]["lo"]
        assert a["YCOORD"] <= box["Y"]["hi"]


# ── H / F matrices (orthogonal box) ─────────────────────────────────────────

def test_h_matrix_orthogonal():
    """For a 10×10×10 orthogonal box H should be close to 10*I."""
    atoms = {1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    H = box["H"]
    # Diagonal entries ~ 10; off-diagonal ~ 0
    assert abs(H[0][0]) == pytest.approx(10.0, abs=1e-6)
    assert abs(H[1][1]) == pytest.approx(10.0, abs=1e-6)
    assert abs(H[2][2]) == pytest.approx(10.0, abs=1e-6)
    assert abs(H[0][1]) == pytest.approx(0.0, abs=1e-6)
    assert abs(H[1][0]) == pytest.approx(0.0, abs=1e-6)


def test_f_matrix_orthogonal():
    """F should be ~ (1/10)*I for a 10×10×10 orthogonal box."""
    atoms = {1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    F = box["F"]
    assert F[0][0] == pytest.approx(1/10, abs=1e-9)
    assert F[1][1] == pytest.approx(1/10, abs=1e-9)
    assert F[2][2] == pytest.approx(1/10, abs=1e-9)


# ── cart2frac / frac2cart round-trip ─────────────────────────────────────────

def test_cart2frac_roundtrip():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    import copy
    atoms_orig = copy.deepcopy(atoms)
    box = get_box(atoms, headers)
    # cart2frac already called by get_box/init_box; now convert back
    frac2cart(atoms, box)
    for idx in atoms:
        assert atoms[idx]["XCOORD"] == pytest.approx(atoms_orig[idx]["XCOORD"], abs=1e-6)
        assert atoms[idx]["YCOORD"] == pytest.approx(atoms_orig[idx]["YCOORD"], abs=1e-6)
        assert atoms[idx]["ZCOORD"] == pytest.approx(atoms_orig[idx]["ZCOORD"], abs=1e-6)


def test_cart2frac_orthogonal_values():
    """For a 10×10×10 box, atom at (5,5,5) → FA=FB=FC=0.5."""
    atoms = {1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    assert atoms[1]["FA"] == pytest.approx(0.5, abs=1e-6)
    assert atoms[1]["FB"] == pytest.approx(0.5, abs=1e-6)
    assert atoms[1]["FC"] == pytest.approx(0.5, abs=1e-6)


# ── displacement tensor ──────────────────────────────────────────────────────

def test_displacement_tensor_orthogonal():
    atoms = {1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    get_box_displacement_tensor(box)
    assert box["X"]["DISP_V"]["X"] == pytest.approx(10.0, abs=1e-6)
    assert box["Y"]["DISP_V"]["X"] == pytest.approx(0.0, abs=1e-6)
    assert box["Y"]["DISP_V"]["Y"] == pytest.approx(10.0, abs=1e-6)
    assert box["Z"]["DISP_V"]["Z"] == pytest.approx(10.0, abs=1e-6)


def test_displacement_tensor_cached():
    """Calling get_box_displacement_tensor twice is idempotent."""
    atoms = {1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    get_box_displacement_tensor(box)
    val1 = box["X"]["DISP_V"]["X"]
    get_box_displacement_tensor(box)
    assert box["X"]["DISP_V"]["X"] == val1


# ── map2unit_cell ─────────────────────────────────────────────────────────────

def test_map2unit_cell_inside():
    """Atom already inside cell should not move."""
    atoms = {1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    map2unit_cell(atoms[1], box)
    assert atoms[1]["XCOORD"] == pytest.approx(5.0, abs=1e-6)
    assert atoms[1]["YCOORD"] == pytest.approx(5.0, abs=1e-6)
    assert atoms[1]["ZCOORD"] == pytest.approx(5.0, abs=1e-6)


def test_map2unit_cell_outside_positive():
    """Atom at 12.0 in a 10.0 box should image to 2.0."""
    atoms = {1: {"XCOORD": 12.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    map2unit_cell(atoms[1], box)
    assert atoms[1]["XCOORD"] == pytest.approx(2.0, abs=1e-4)


def test_map2unit_cell_outside_negative():
    """Atom at -2.0 in a 10.0 box should image to 8.0."""
    atoms = {1: {"XCOORD": -2.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  10.0 10.0 10.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    map2unit_cell(atoms[1], box)
    assert atoms[1]["XCOORD"] == pytest.approx(8.0, abs=1e-4)


# ── center_atoms ──────────────────────────────────────────────────────────────

def test_center_atoms_moves_lo_to_zero():
    atoms = {
        1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
            "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
            "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0},
    }
    box = {
        "X": {"hi": 10.0, "lo": 5.0, "len": 5.0, "angle": 90.0},
        "Y": {"hi": 10.0, "lo": 5.0, "len": 5.0, "angle": 90.0},
        "Z": {"hi": 10.0, "lo": 5.0, "len": 5.0, "angle": 90.0},
    }
    center_atoms(atoms, box)
    assert box["X"]["lo"] == pytest.approx(0.0)
    assert atoms[1]["XCOORD"] == pytest.approx(0.0)


# ── box volume ────────────────────────────────────────────────────────────────

def test_box_vol_orthogonal():
    atoms = {1: {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = ["CRYSTX  3.0 4.0 5.0 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    assert get_box_vol(box) == pytest.approx(60.0, abs=1e-4)
