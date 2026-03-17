"""Tests for scripts/trim_cell.py."""
from pathlib import Path

import pytest

from atlas_toolkit.core.box import get_box, init_box
from atlas_toolkit.io.bgf import read_bgf
from atlas_toolkit.scripts.trim_cell import (
    _parse_cell,
    center_sys,
    make_atoms_mols,
    trim_cell,
)

FIXTURES = Path(__file__).parent / "fixtures"
SPC_BOX = FIXTURES / "spc_box.bgf"


# ── _parse_cell ────────────────────────────────────────────────────────────────

def test_parse_cell_orthogonal():
    b = _parse_cell("10 20 30")
    assert b["X"]["len"] == pytest.approx(10.0)
    assert b["Y"]["len"] == pytest.approx(20.0)
    assert b["Z"]["len"] == pytest.approx(30.0)


def test_parse_cell_default_angles():
    b = _parse_cell("10 10 10")
    assert b["X"]["angle"] == pytest.approx(90.0)
    assert b["Y"]["angle"] == pytest.approx(90.0)
    assert b["Z"]["angle"] == pytest.approx(90.0)


def test_parse_cell_custom_angles():
    b = _parse_cell("10 10 10 90 90 120")
    assert b["Z"]["angle"] == pytest.approx(120.0)


def test_parse_cell_too_few_values():
    with pytest.raises(ValueError):
        _parse_cell("10 20")


# ── make_atoms_mols ────────────────────────────────────────────────────────────

def test_make_atoms_mols_one_per_atom():
    atoms = {1: {}, 2: {}, 5: {}}
    mols = make_atoms_mols(atoms)
    assert set(mols.keys()) == {1, 2, 5}
    for k, mol in mols.items():
        assert mol["MEMBERS"] == {k: 1}
        assert mol["SIZE"] == 1


# ── center_sys ─────────────────────────────────────────────────────────────────

def _make_atom(x, y, z):
    return {"XCOORD": x, "YCOORD": y, "ZCOORD": z,
            "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
            "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}


def test_center_sys_box_center_shifts_to_origin():
    """origin=0: atoms shift by -len/2 in each dim."""
    atoms = {1: _make_atom(5.0, 5.0, 5.0)}
    box = {"X": {"len": 10.0, "lo": 0.0, "hi": 10.0, "angle": 90.0},
           "Y": {"len": 10.0, "lo": 0.0, "hi": 10.0, "angle": 90.0},
           "Z": {"len": 10.0, "lo": 0.0, "hi": 10.0, "angle": 90.0}}
    center_sys(atoms, box, start_origin=0)
    assert atoms[1]["XCOORD"] == pytest.approx(0.0)
    assert atoms[1]["YCOORD"] == pytest.approx(0.0)
    assert atoms[1]["ZCOORD"] == pytest.approx(0.0)


def test_center_sys_origin_no_shift():
    """origin=1: no shift applied."""
    atoms = {1: _make_atom(3.0, 4.0, 5.0)}
    box = {"X": {"len": 10.0, "lo": 0.0, "hi": 10.0, "angle": 90.0},
           "Y": {"len": 10.0, "lo": 0.0, "hi": 10.0, "angle": 90.0},
           "Z": {"len": 10.0, "lo": 0.0, "hi": 10.0, "angle": 90.0}}
    center_sys(atoms, box, start_origin=1)
    assert atoms[1]["XCOORD"] == pytest.approx(3.0)
    assert atoms[1]["YCOORD"] == pytest.approx(4.0)
    assert atoms[1]["ZCOORD"] == pytest.approx(5.0)


# ── trim_cell ──────────────────────────────────────────────────────────────────

def test_trim_cell_removes_outside_atoms():
    """Atoms clearly outside [0, new_box) should be removed."""
    atoms = {
        1: _make_atom(2.0, 2.0, 2.0),  # inside 5x5x5
        2: _make_atom(8.0, 2.0, 2.0),  # outside
    }
    bonds = {1: [], 2: []}
    mols = make_atoms_mols(atoms)

    new_box = _parse_cell("5 5 5")
    init_box(new_box, atoms)

    trim_cell(atoms, bonds, mols, new_box)
    assert 1 in atoms
    assert 2 not in atoms


def test_trim_cell_keeps_inside_atoms():
    atoms = {
        1: _make_atom(1.0, 1.0, 1.0),
        2: _make_atom(2.0, 2.0, 2.0),
    }
    bonds = {1: [], 2: []}
    mols = make_atoms_mols(atoms)
    new_box = _parse_cell("5 5 5")
    init_box(new_box, atoms)
    trim_cell(atoms, bonds, mols, new_box)
    assert 1 in atoms
    assert 2 in atoms


def test_trim_cell_spc_box_reduces_atom_count():
    """Trimming SPC box to a smaller cell should reduce atom count."""
    atoms, bonds, headers = read_bgf(SPC_BOX)
    n_orig = len(atoms)
    box = get_box(atoms, headers)
    orig_len = box["X"]["len"]

    new_box = _parse_cell(f"{orig_len * 0.5} {orig_len * 0.5} {orig_len * 0.5}")
    init_box(new_box, atoms)

    from atlas_toolkit.core.manip_atoms import get_mols
    mols = get_mols(atoms, bonds)
    trim_cell(atoms, bonds, mols, new_box)

    assert len(atoms) < n_orig
    assert len(atoms) > 0
