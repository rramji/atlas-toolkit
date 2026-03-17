"""Tests for io/ff.py — Cerius2 FF parser."""
from pathlib import Path

import pytest

from atlas_toolkit.io.ff import find_ff, get_vdw_radius, read_ff

FIXTURES = Path(__file__).parent / "fixtures"
FF_DIR   = Path(__file__).parents[1] / "atlas_toolkit" / "data" / "ff"
TIP3P_FF = FF_DIR / "tip3p.ff"
AMBER99  = FF_DIR / "AMBER99.ff"


# ── find_ff ───────────────────────────────────────────────────────────────────

def test_find_ff_by_stem():
    p = find_ff("tip3p")
    assert p is not None
    assert p.exists()
    assert p.suffix == ".ff"


def test_find_ff_full_name():
    p = find_ff("tip3p.ff")
    assert p is not None
    assert p.exists()


def test_find_ff_missing():
    assert find_ff("nonexistent_ff_xyzzy") is None


# ── read_ff — ATOMTYPES ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tip3p_parms():
    return read_ff(TIP3P_FF)


def test_read_ff_has_atomtypes(tip3p_parms):
    assert "ATOMTYPES" in tip3p_parms
    assert len(tip3p_parms["ATOMTYPES"]) > 0


def test_read_ff_atomtype_has_mass(tip3p_parms):
    for atype, data in tip3p_parms["ATOMTYPES"].items():
        assert "MASS" in data, f"ATOMTYPES[{atype}] missing MASS"
        assert float(data["MASS"]) >= 0


def test_read_ff_has_vdw(tip3p_parms):
    assert "VDW" in tip3p_parms
    assert len(tip3p_parms["VDW"]) > 0


# ── read_ff — VDW / alter ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def amber99_parms():
    return read_ff(AMBER99)


def test_amber99_has_Na_atomtype(amber99_parms):
    atypes = amber99_parms["ATOMTYPES"]
    assert any(k.strip() == "Na" for k in atypes), "Na not found in ATOMTYPES"


def test_amber99_Na_vdw_has_epsilon_sigma(amber99_parms):
    vdw = amber99_parms["VDW"]
    na_vdw = vdw.get("Na", {}).get("Na", {}).get(1, {})
    vals = na_vdw.get("VALS", [])
    assert len(vals) >= 2, "Na VDW should have at least 2 values (epsilon, sigma)"
    # epsilon should be positive; sigma > 0
    assert float(vals[0]) > 0, "Na epsilon should be > 0"
    assert float(vals[1]) > 0, "Na sigma should be > 0"


def test_lj_alter_sigma_smaller_than_rmin(amber99_parms):
    """After alter, sigma_LJ = r_min / 2^(1/6) < r_min."""
    import math
    vdw = amber99_parms["VDW"]
    # Read the raw file to get original r_min for comparison
    raw = AMBER99.read_text()
    # Na line: LJ_6_12   3.7360   0.0028
    for line in raw.splitlines():
        if "Na" in line and "LJ_6_12" in line:
            parts = line.split()
            rmin = float(parts[2])
            sigma_lj = float(vdw["Na"]["Na"][1]["VALS"][1])
            # sigma_LJ = rmin / 2^(1/6)
            assert abs(sigma_lj - rmin / 2 ** (1 / 6)) < 1e-6
            break


# ── get_vdw_radius ────────────────────────────────────────────────────────────

def test_get_vdw_radius_returns_float(amber99_parms):
    r = get_vdw_radius("Na", amber99_parms)
    assert r is not None
    assert r > 0


def test_get_vdw_radius_missing_returns_none(amber99_parms):
    assert get_vdw_radius("NONEXISTENT_XYZ", amber99_parms) is None
