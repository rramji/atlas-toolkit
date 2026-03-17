"""Tests for scripts/embed_molecule.py."""
import copy
from pathlib import Path

import pytest

from atlas_toolkit.core.box import get_box, init_box
from atlas_toolkit.io.bgf import read_bgf
from atlas_toolkit.scripts.embed_molecule import (
    OVERLAP_CUTOFF,
    _center_mols,
    _min_image_dist,
    embed_molecule,
)

FIXTURES = Path(__file__).parent / "fixtures"
SPC_BOX = FIXTURES / "spc_box.bgf"
TIP3    = FIXTURES / "tip3.bgf"


def _ortho_box(lx=30.0, ly=30.0, lz=30.0):
    return {
        "X": {"lo": 0.0, "hi": lx, "len": lx, "angle": 90.0},
        "Y": {"lo": 0.0, "hi": ly, "len": ly, "angle": 90.0},
        "Z": {"lo": 0.0, "hi": lz, "len": lz, "angle": 90.0},
    }


def _atom(x, y, z, idx=1, resname="SOL", resnum=1, mass=1.0):
    return {
        "INDEX": idx, "LABEL": "HETATM",
        "XCOORD": float(x), "YCOORD": float(y), "ZCOORD": float(z),
        "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
        "RESNAME": resname, "RESNUM": resnum,
        "NUMBONDS": 0, "LONEPAIRS": 0, "MASS": mass,
    }


# ── _min_image_dist ────────────────────────────────────────────────────────────

def test_min_image_dist_same_point():
    a = {"XCOORD": 5.0, "YCOORD": 5.0, "ZCOORD": 5.0}
    box = _ortho_box(10.0, 10.0, 10.0)
    assert _min_image_dist(a, a, box) == pytest.approx(0.0)


def test_min_image_dist_across_boundary():
    """Distance between 0.5 and 9.5 in a 10 Å box should be 1.0, not 9.0."""
    a = {"XCOORD": 0.5, "YCOORD": 0.0, "ZCOORD": 0.0}
    b = {"XCOORD": 9.5, "YCOORD": 0.0, "ZCOORD": 0.0}
    box = _ortho_box(10.0, 10.0, 10.0)
    assert _min_image_dist(a, b, box) == pytest.approx(1.0)


def test_min_image_dist_3d():
    a = {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0}
    b = {"XCOORD": 3.0, "YCOORD": 4.0, "ZCOORD": 0.0}
    box = _ortho_box(100.0, 100.0, 100.0)
    assert _min_image_dist(a, b, box) == pytest.approx(5.0)


# ── _center_mols ───────────────────────────────────────────────────────────────

def test_center_mols_aligns_coms():
    solu = {1: _atom(0.0, 0.0, 0.0)}
    solv = {1: _atom(10.0, 10.0, 10.0)}
    _center_mols(solu, solv)
    # solu CoM should now be at solv CoM (10, 10, 10)
    assert solu[1]["XCOORD"] == pytest.approx(10.0)
    assert solu[1]["YCOORD"] == pytest.approx(10.0)
    assert solu[1]["ZCOORD"] == pytest.approx(10.0)


def test_center_mols_does_not_move_solvent():
    solu = {1: _atom(0.0, 0.0, 0.0)}
    solv = {1: _atom(5.0, 5.0, 5.0)}
    _center_mols(solu, solv)
    assert solv[1]["XCOORD"] == pytest.approx(5.0)


# ── embed_molecule integration ─────────────────────────────────────────────────

def test_embed_molecule_total_atom_count():
    """After embedding, atom count should be ≤ n_solu + n_solv (overlaps removed)."""
    solu_atoms, solu_bonds, solu_headers = read_bgf(TIP3)
    solv_atoms, solv_bonds, solv_headers = read_bgf(SPC_BOX)

    n_solu = len(solu_atoms)
    n_solv = len(solv_atoms)

    box = get_box(solv_atoms, solv_headers)
    atoms, bonds = embed_molecule(
        copy.deepcopy(solu_atoms), copy.deepcopy(solu_bonds),
        copy.deepcopy(solv_atoms), copy.deepcopy(solv_bonds),
        copy.deepcopy(box),
        center=True,
        check_overlap=True,
    )
    assert len(atoms) <= n_solu + n_solv
    assert len(atoms) > 0


def test_embed_molecule_no_overlap_check_gives_full_count():
    """Without overlap removal, all atoms from both systems should be present."""
    solu_atoms, solu_bonds, _ = read_bgf(TIP3)
    solv_atoms, solv_bonds, solv_headers = read_bgf(SPC_BOX)
    n_total = len(solu_atoms) + len(solv_atoms)
    box = get_box(solv_atoms, solv_headers)

    atoms, bonds = embed_molecule(
        copy.deepcopy(solu_atoms), copy.deepcopy(solu_bonds),
        copy.deepcopy(solv_atoms), copy.deepcopy(solv_bonds),
        copy.deepcopy(box),
        center=False,
        check_overlap=False,
    )
    assert len(atoms) == n_total


def test_embed_molecule_indices_sequential():
    """Atom indices in the result should be sequential starting from 1."""
    solu_atoms, solu_bonds, _ = read_bgf(TIP3)
    solv_atoms, solv_bonds, solv_headers = read_bgf(SPC_BOX)
    box = get_box(solv_atoms, solv_headers)

    atoms, bonds = embed_molecule(
        copy.deepcopy(solu_atoms), copy.deepcopy(solu_bonds),
        copy.deepcopy(solv_atoms), copy.deepcopy(solv_bonds),
        copy.deepcopy(box),
        center=True,
        check_overlap=True,
    )
    keys = sorted(atoms.keys())
    assert keys == list(range(1, len(atoms) + 1))


def test_embed_molecule_overlap_removes_close_solvent():
    """A solvent atom placed < OVERLAP_CUTOFF from the solute should be removed."""
    # Place solute at origin
    solu_atoms = {1: _atom(0.0, 0.0, 0.0, idx=1, resname="SOL")}
    solu_bonds = {1: []}
    # Place one solvent atom at 1 Å (inside cutoff) and one at 10 Å (outside)
    solv_atoms = {
        1: _atom(1.0, 0.0, 0.0, idx=1, resname="WAT", resnum=1),  # too close
        2: _atom(10.0, 0.0, 0.0, idx=2, resname="WAT", resnum=2),  # OK
    }
    solv_bonds = {1: [], 2: []}

    box = _ortho_box(20.0, 20.0, 20.0)
    init_box(box, {**solu_atoms, **solv_atoms})

    atoms, bonds = embed_molecule(
        copy.deepcopy(solu_atoms), copy.deepcopy(solu_bonds),
        copy.deepcopy(solv_atoms), copy.deepcopy(solv_bonds),
        box,
        center=False,
        check_overlap=True,
    )
    # Should have the solute atom + the far solvent atom (the close one removed)
    assert len(atoms) == 2
