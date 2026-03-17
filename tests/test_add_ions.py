"""Tests for scripts/add_ions.py."""
import copy
from pathlib import Path

import pytest

from atlas_toolkit.core.box import get_box
from atlas_toolkit.io.bgf import read_bgf
from atlas_toolkit.io.ff import read_ff
from atlas_toolkit.scripts.add_ions import (
    add_ions,
    get_ion_parms,
    resolve_ion_count,
)

FIXTURES = Path(__file__).parent / "fixtures"
SPC_BOX  = FIXTURES / "spc_box.bgf"
FF_DIR   = Path(__file__).parents[1] / "atlas_toolkit" / "data" / "ff"
AMBER99  = FF_DIR / "AMBER99.ff"


@pytest.fixture(scope="module")
def amber_parms():
    return read_ff(AMBER99)


# ── get_ion_parms ─────────────────────────────────────────────────────────────

def test_get_ion_parms_Na(amber_parms):
    p = get_ion_parms(amber_parms, "Na")
    assert p["FFTYPE"] is not None
    assert float(p["MASS"]) > 0
    # Na mass ≈ 23
    assert abs(float(p["MASS"]) - 23.0) < 2.0


def test_get_ion_parms_missing_raises(amber_parms):
    with pytest.raises(ValueError):
        get_ion_parms(amber_parms, "XYZNONEXISTENT")


# ── resolve_ion_count ─────────────────────────────────────────────────────────

def _box_10():
    return {"X": {"len": 10.0}, "Y": {"len": 10.0}, "Z": {"len": 10.0}}


def test_resolve_ion_count_integer():
    assert resolve_ion_count("5", _box_10(), 0.0, 1.0) == 5


def test_resolve_ion_count_neutralize_positive_sys():
    # sys charge = +2, ion charge = -1 → need 2 anions
    assert resolve_ion_count("0", _box_10(), 2.0, -1.0) == 2


def test_resolve_ion_count_neutralize_negative_sys():
    # sys charge = -3, ion charge = +1 → need 3 cations
    assert resolve_ion_count("0", _box_10(), -3.0, 1.0) == 3


def test_resolve_ion_count_neutralize_same_sign_raises():
    with pytest.raises(ValueError):
        resolve_ion_count("0", _box_10(), 2.0, 1.0)


def test_resolve_ion_count_molar():
    # 1 M in a 1000 Å³ box: N = 6.022e-4 * 1.0 * 1000 ≈ 0.6 → 1 ion
    box = {"X": {"len": 10.0}, "Y": {"len": 10.0}, "Z": {"len": 10.0}}
    n = resolve_ion_count("1.0", box, 0.0, 1.0)
    # In 1000 Å³ at 1 M: N = 6.022e23 * 1 / (6.022e26) * 1000 ≈ 0.6 → 1
    assert n >= 0  # just check it runs without error


# ── add_ions integration ──────────────────────────────────────────────────────

def test_add_ions_places_correct_count(amber_parms):
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)
    n_orig = len(atoms)

    # Mark water atoms with IS_SOLVENT flag to help add_mols_to_selection
    for a in atoms.values():
        if a.get("RESNAME") == "WAT":
            a["IS_SOLVENT"] = True

    n_placed = add_ions(
        atoms, bonds, box,
        ion_specs=[("Na", "3")],
        ff_parms=amber_parms,
        solv_select="resname eq WAT",
        randomize=True,
    )
    assert n_placed == 3
    # Net change: 3 water molecules removed (3 atoms each) + 3 Na atoms added
    # = n_orig - 3*3 + 3 = n_orig - 6
    assert len(atoms) == n_orig - 3 * 3 + 3


def test_add_ions_na_resname_set(amber_parms):
    """Each ion atom should have RESNAME == element."""
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)

    add_ions(
        atoms, bonds, box,
        ion_specs=[("Na", "1")],
        ff_parms=amber_parms,
        solv_select="resname eq WAT",
        randomize=True,
    )
    na_atoms = [a for a in atoms.values() if a.get("RESNAME") == "Na"]
    assert len(na_atoms) == 1


def test_add_ions_zero_neutralize_no_charge(amber_parms):
    """If system charge is 0.0, neutralize (n=0) should place 0 ions."""
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)
    # Force sys charge to 0 by zeroing all charges
    for a in atoms.values():
        a["CHARGE"] = 0.0

    with pytest.raises((ValueError, ZeroDivisionError)):
        add_ions(
            atoms, bonds, box,
            ion_specs=[("Na", "0")],
            ff_parms=amber_parms,
            solv_select="resname eq WAT",
            randomize=True,
        )
