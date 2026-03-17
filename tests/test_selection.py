"""Tests for atom selection DSL."""
import pytest
from atlas_toolkit.core.manip_atoms import build_selection, select_atoms, get_mols

# Minimal synthetic atoms dict for selection testing
ATOMS = {
    1: {"INDEX": 1, "ATMNAME": "O1",  "RESNAME": "WAT", "FFTYPE": "OW",
        "CHARGE": -0.834, "RESNUM": 1, "XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0,
        "NUMBONDS": 2, "LONEPAIRS": 0},
    2: {"INDEX": 2, "ATMNAME": "H2",  "RESNAME": "WAT", "FFTYPE": "HW",
        "CHARGE":  0.417, "RESNUM": 1, "XCOORD": 1.0, "YCOORD": 0.0, "ZCOORD": 0.0,
        "NUMBONDS": 1, "LONEPAIRS": 0},
    3: {"INDEX": 3, "ATMNAME": "H3",  "RESNAME": "WAT", "FFTYPE": "HW",
        "CHARGE":  0.417, "RESNUM": 1, "XCOORD": 2.0, "YCOORD": 0.0, "ZCOORD": 0.0,
        "NUMBONDS": 1, "LONEPAIRS": 0},
    4: {"INDEX": 4, "ATMNAME": "CL1", "RESNAME": "RES", "FFTYPE": "Cl-",
        "CHARGE": -1.000, "RESNUM": 2, "XCOORD": 5.0, "YCOORD": 0.0, "ZCOORD": 0.0,
        "NUMBONDS": 0, "LONEPAIRS": 0},
}

BONDS = {1: [2, 3], 2: [1], 3: [1], 4: []}


def test_select_all_star():
    pred = build_selection("*")
    result = {idx for idx, a in ATOMS.items() if pred(a)}
    assert result == {1, 2, 3, 4}

def test_select_fftype_eq():
    result = select_atoms("fftype eq OW", ATOMS)
    assert set(result) == {1}

def test_select_fftype_eq_caseinsensitive_field():
    # field name is case-insensitive
    result = select_atoms("FFTYPE eq OW", ATOMS)
    assert set(result) == {1}

def test_select_fftype_eq_cl():
    result = select_atoms("fftype eq Cl-", ATOMS)
    assert set(result) == {4}

def test_select_resname_eq():
    result = select_atoms("resname eq WAT", ATOMS)
    assert set(result) == {1, 2, 3}

def test_select_charge_lt():
    result = select_atoms("charge < 0", ATOMS)
    assert set(result) == {1, 4}

def test_select_charge_gt():
    result = select_atoms("charge > 0", ATOMS)
    assert set(result) == {2, 3}

def test_select_index_gt():
    result = select_atoms("index > 2", ATOMS)
    assert set(result) == {3, 4}

def test_select_and():
    result = select_atoms("resname eq WAT and charge < 0", ATOMS)
    assert set(result) == {1}

def test_select_or():
    result = select_atoms("fftype eq OW or fftype eq Cl-", ATOMS)
    assert set(result) == {1, 4}

def test_select_ne():
    result = select_atoms("resname ne RES", ATOMS)
    assert set(result) == {1, 2, 3}

def test_select_regex():
    result = select_atoms("fftype =~ HW", ATOMS)
    assert set(result) == {2, 3}

def test_select_no_match_raises():
    with pytest.raises(ValueError, match="No atoms matched"):
        select_atoms("fftype eq NONEXISTENT", ATOMS)

def test_select_invalid_raises():
    with pytest.raises(ValueError):
        select_atoms("this is not valid ???", ATOMS)

def test_get_mols_single_atom():
    atoms = {4: dict(ATOMS[4])}
    bonds = {4: []}
    mols = get_mols(atoms, bonds)
    assert len(mols) == 1
    assert atoms[4]["MOLECULEID"] == 1
    assert atoms[4]["MOLSIZE"] == 1

def test_get_mols_water():
    import copy
    atoms_copy = copy.deepcopy(ATOMS)
    mols = get_mols(atoms_copy, BONDS)
    # WAT (1,2,3) should be one mol; Cl- (4) another
    assert len(mols) == 2
    # All WAT atoms share the same MOLECULEID
    wids = {atoms_copy[i]["MOLECULEID"] for i in (1, 2, 3)}
    assert len(wids) == 1
    # Cl- has a different mol id
    assert atoms_copy[4]["MOLECULEID"] not in wids

def test_get_mols_molsize():
    import copy
    atoms_copy = copy.deepcopy(ATOMS)
    get_mols(atoms_copy, BONDS)
    assert atoms_copy[1]["MOLSIZE"] == 3
    assert atoms_copy[4]["MOLSIZE"] == 1
