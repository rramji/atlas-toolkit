"""Tests for scripts/remove_mols.py."""
import copy
from pathlib import Path

import pytest

from atlas_toolkit.core.manip_atoms import get_mols
from atlas_toolkit.io.bgf import read_bgf
from atlas_toolkit.scripts.remove_mols import remove_mols

FIXTURES = Path(__file__).parent / "fixtures"
SPC_BOX = FIXTURES / "spc_box.bgf"


def _make_atom(idx, resname="WAT", resnum=1):
    return {
        "INDEX": idx, "LABEL": "HETATM",
        "XCOORD": float(idx), "YCOORD": 0.0, "ZCOORD": 0.0,
        "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
        "RESNAME": resname, "RESNUM": resnum, "NUMBONDS": 0, "LONEPAIRS": 0,
    }


def _three_water_system():
    """Three isolated single-atom 'molecules' (no bonds)."""
    atoms = {
        1: _make_atom(1, "WAT", 1),
        2: _make_atom(2, "WAT", 2),
        3: _make_atom(3, "WAT", 3),
    }
    bonds = {1: [], 2: [], 3: []}
    return atoms, bonds


# ── remove by atom count ───────────────────────────────────────────────────────

def test_remove_mols_by_atom_count():
    atoms, bonds = _three_water_system()
    selection = {1: 1, 2: 1, 3: 1}
    n_atoms, n_mols = remove_mols(atoms, bonds, selection, max_atoms=1)
    assert n_atoms == 1
    assert n_mols == 1
    assert len(atoms) == 2


def test_remove_mols_by_mol_count():
    atoms, bonds = _three_water_system()
    selection = {1: 1, 2: 1, 3: 1}
    n_atoms, n_mols = remove_mols(atoms, bonds, selection, max_mols=2)
    assert n_mols == 2
    assert len(atoms) == 1


def test_remove_mols_zero_removes_nothing():
    atoms, bonds = _three_water_system()
    selection = {1: 1, 2: 1, 3: 1}
    n_atoms, n_mols = remove_mols(atoms, bonds, selection, max_mols=0)
    assert n_mols == 0
    assert len(atoms) == 3


def test_remove_mols_max_exceeds_available():
    """Asking to remove more molecules than are in the selection removes all selected ones.
    A non-selected atom keeps the structure non-empty."""
    atoms, bonds = _three_water_system()
    # Add a 4th atom not in selection so we don't empty the structure
    atoms[4] = _make_atom(4, "SOL", 4)
    bonds[4] = []
    selection = {1: 1, 2: 1, 3: 1}
    n_atoms, n_mols = remove_mols(atoms, bonds, selection, max_mols=99)
    assert n_mols == 3
    assert len(atoms) == 1  # only atom 4 remains


# ── selection restriction ──────────────────────────────────────────────────────

def test_remove_mols_selection_limits_scope():
    """Only atoms in selection should be candidates for removal."""
    atoms, bonds = _three_water_system()
    # Only atom 1 in selection
    selection = {1: 1}
    n_atoms, n_mols = remove_mols(atoms, bonds, selection, max_mols=99)
    assert 2 in atoms
    assert 3 in atoms
    assert 1 not in atoms


# ── randomize ─────────────────────────────────────────────────────────────────

def test_remove_mols_randomize_does_not_crash():
    atoms, bonds = _three_water_system()
    selection = {1: 1, 2: 1, 3: 1}
    remove_mols(atoms, bonds, selection, max_mols=1, randomize=True)
    assert len(atoms) == 2


# ── integration with SPC box ──────────────────────────────────────────────────

def test_remove_mols_from_spc_box():
    from atlas_toolkit.core.manip_atoms import get_mols
    atoms, bonds, headers = read_bgf(SPC_BOX)
    get_mols(atoms, bonds)  # populate MOLECULE field so remove_mols sees whole molecules
    n_orig = len(atoms)
    # Select all atoms (WAT resname)
    selection = {i: 1 for i in atoms}
    n_atoms, n_mols = remove_mols(atoms, bonds, selection, max_mols=10)
    assert n_mols == 10
    assert len(atoms) == n_orig - 10 * 3  # 3 atoms per SPC water
