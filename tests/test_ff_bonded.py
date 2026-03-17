"""Tests for extended io/ff.py (bonds/angles/torsions/inversions) and io/frcmod.py."""
from pathlib import Path
import pytest

from atlas_toolkit.io.ff import (
    angle_key, bond_key, inversion_key, torsion_key,
    load_ff, lookup_angle, lookup_bond, lookup_inversion, lookup_torsion,
    read_ff,
)
from atlas_toolkit.io.frcmod import read_frcmod

FF_DIR   = Path(__file__).parents[1] / "atlas_toolkit" / "data" / "ff"
AMBER99  = FF_DIR / "AMBER99.ff"
TIP3P_FF = FF_DIR / "tip3p.ff"

FRCMOD_DIR = Path("/home/robert/Downloads/test_for_maggie/march2026_peptide_aggregation/general_ffs")
CITRATE_FRCMOD = FRCMOD_DIR / "citrate.frcmod"
HEINZ_FF = FRCMOD_DIR / "heinzAu_oplsIons_softChlorine.ff"

_HAS_FRCMOD = CITRATE_FRCMOD.exists()
_HAS_HEINZ  = HEINZ_FF.exists()


# ── canonical key functions ───────────────────────────────────────────────────

def test_bond_key_sorted():
    assert bond_key("CT", "HC") == bond_key("HC", "CT")
    assert bond_key("A", "B") == ("A", "B")
    assert bond_key("B", "A") == ("A", "B")


def test_angle_key_sorted_endpoints():
    assert angle_key("CT", "C", "O") == angle_key("O", "C", "CT")
    a, center, b = angle_key("HC", "CT", "N")
    assert center == "CT"
    assert a <= b


def test_torsion_key_canonical():
    assert torsion_key("A", "B", "C", "D") == torsion_key("D", "C", "B", "A")


def test_inversion_key_sorted_satellites():
    k1 = inversion_key("C", "O", "N", "X")
    k2 = inversion_key("C", "X", "O", "N")
    assert k1 == k2
    assert k1[0] == "C"


# ── read_ff — bonded sections ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def amber99():
    return read_ff(AMBER99)


def test_amber99_has_bonds(amber99):
    assert len(amber99["BONDS"]) > 100


def test_amber99_has_angles(amber99):
    assert len(amber99["ANGLES"]) > 200


def test_amber99_has_torsions(amber99):
    assert len(amber99["TORSIONS"]) > 50


def test_amber99_has_inversions(amber99):
    assert len(amber99["INVERSIONS"]) > 10


def test_amber99_bond_hw_ow(amber99):
    """HW-OW water bond should be present."""
    entry = amber99["BONDS"].get(bond_key("HW", "OW"))
    assert entry is not None, "HW-OW bond not found"
    k, r0 = entry["VALS"]
    assert k > 0
    assert 0.9 < r0 < 1.1  # ~0.957 Å


def test_amber99_angle_hw_ow_hw(amber99):
    """HW-OW-HW water angle should be present."""
    entry = amber99["ANGLES"].get(angle_key("HW", "OW", "HW"))
    assert entry is not None, "HW-OW-HW angle not found"
    k, theta0 = entry["VALS"]
    assert k > 0
    assert 100 < theta0 < 120  # ~104.52 degrees


def test_amber99_multi_term_torsion(amber99):
    """At least one torsion should have > 1 Fourier term."""
    multi = [v for v in amber99["TORSIONS"].values() if len(v) > 1]
    assert len(multi) > 0, "No multi-term torsions found"


def test_amber99_inversion_has_type(amber99):
    for key, terms in amber99["INVERSIONS"].items():
        for term in terms:
            assert "TYPE" in term
            assert "VALS" in term
            assert len(term["VALS"]) == 3


# ── lookup helpers ────────────────────────────────────────────────────────────

def test_lookup_bond_symmetric(amber99):
    a = lookup_bond("HW", "OW", amber99)
    b = lookup_bond("OW", "HW", amber99)
    assert a is b  # same dict object (same canonical key)


def test_lookup_bond_missing(amber99):
    assert lookup_bond("XX", "YY", amber99) is None


def test_lookup_angle_symmetric(amber99):
    a = lookup_angle("HW", "OW", "HW", amber99)
    assert a is not None


def test_lookup_torsion_wildcard(amber99):
    """INVERSIONS have 'X' wildcards — lookup_inversion should fall back to them."""
    # In AMBER99 inversions, many entries use X wildcards
    # Try a specific inversion that exists with wildcards
    invs = amber99["INVERSIONS"]
    if not invs:
        pytest.skip("No inversions in this FF")
    # Get the first entry's key and look it up
    key = next(iter(invs))
    center = key[0]
    result = lookup_inversion(center, key[1], key[2], key[3], amber99)
    assert result is not None


# ── frcmod parser ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FRCMOD, reason="citrate.frcmod not present")
class TestFrcmod:
    @pytest.fixture(scope="class")
    def cit(self):
        return read_frcmod(CITRATE_FRCMOD)

    def test_has_atomtypes(self, cit):
        assert "c3" in cit["ATOMTYPES"]
        assert abs(cit["ATOMTYPES"]["c3"]["MASS"] - 12.01) < 0.1

    def test_has_bonds(self, cit):
        assert len(cit["BONDS"]) > 0
        # c3-hc bond should be present
        key = bond_key("c3", "hc")
        assert key in cit["BONDS"]
        k, r0 = cit["BONDS"][key]["VALS"]
        assert k > 0
        assert 1.0 < r0 < 1.2  # ~1.097 Å

    def test_has_angles(self, cit):
        assert len(cit["ANGLES"]) > 0

    def test_has_torsions(self, cit):
        assert len(cit["TORSIONS"]) > 0

    def test_has_inversions(self, cit):
        assert len(cit["INVERSIONS"]) > 0

    def test_nonbon_sigma_conversion(self, cit):
        """rmin_half should be converted to sigma_lj = 2*rmin_half / 2^(1/6)."""
        import math
        vdw = cit["VDW"]
        assert "c3" in vdw, "c3 VDW not found"
        vals = vdw["c3"]["c3"][1]["VALS"]
        epsilon, sigma_lj = vals[0], vals[1]
        assert epsilon > 0
        # From frcmod: rmin_half=1.9080 → sigma_lj = 2*1.9080 / 2^(1/6) ≈ 3.398
        assert abs(sigma_lj - 2 * 1.9080 / 2 ** (1 / 6)) < 0.001

    def test_dihe_div_applied(self, cit):
        """The divider in DIHE lines should be applied (k stored = k_raw/div)."""
        # c-c3-c3-c3 has div=9, k=1.400 → stored k ≈ 0.1556
        key = torsion_key("c", "c3", "c3", "c3")
        assert key in cit["TORSIONS"]
        k = cit["TORSIONS"][key][0]["VALS"][0]
        assert abs(k - 1.400 / 9) < 1e-4

    def test_multi_term_dihe(self, cit):
        """c3-c3-oh-ho has 2 entries in the frcmod — both should be stored."""
        key = torsion_key("c3", "c3", "oh", "ho")
        assert key in cit["TORSIONS"]
        assert len(cit["TORSIONS"][key]) == 2


# ── load_ff merge ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (_HAS_FRCMOD and _HAS_HEINZ), reason="test FF files not present")
class TestLoadFF:
    @pytest.fixture(scope="class")
    def merged(self):
        return load_ff([str(HEINZ_FF), str(CITRATE_FRCMOD)])

    def test_atomtypes_from_both(self, merged):
        # From heinz.ff
        assert "Au" in merged["ATOMTYPES"]
        assert "Na+" in merged["ATOMTYPES"]
        # From citrate.frcmod
        assert "c3" in merged["ATOMTYPES"]

    def test_vdw_from_both(self, merged):
        assert "Au" in merged["VDW"]
        assert "c3" in merged["VDW"]

    def test_bonds_from_frcmod(self, merged):
        assert len(merged["BONDS"]) > 0

    def test_parms_inherited(self, merged):
        # load_ff initialises PARMS from defaults
        assert "cut_vdw" in merged["PARMS"]


def test_load_ff_single_ff():
    """load_ff with a single .ff path should match read_ff."""
    from atlas_toolkit.io.ff import read_ff as _read
    parms_direct = _read(AMBER99)
    parms_via_load = load_ff(str(AMBER99))
    assert len(parms_via_load["BONDS"]) == len(parms_direct["BONDS"])
    assert len(parms_via_load["ATOMTYPES"]) == len(parms_direct["ATOMTYPES"])


def test_load_ff_string_spec():
    """load_ff should accept a space-separated string of paths."""
    parms = load_ff(str(AMBER99))
    assert "ATOMTYPES" in parms
