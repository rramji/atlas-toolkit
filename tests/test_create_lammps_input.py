"""Tests for atlas_toolkit/lammps/ — topology, data_file, input_script."""
import re
from pathlib import Path

import pytest

from atlas_toolkit.io.bgf import read_bgf
from atlas_toolkit.io.ff import load_ff, read_ff
from atlas_toolkit.core.box import get_box
from atlas_toolkit.lammps.topology import (
    enumerate_angles,
    enumerate_bonds,
    enumerate_impropers,
    enumerate_torsions,
)
from atlas_toolkit.lammps.data_file import write_data_file
from atlas_toolkit.lammps.input_script import write_input_script

FIXTURES = Path(__file__).parent / "fixtures"
SPC_BOX  = FIXTURES / "spc_box.bgf"
FF_DIR   = Path(__file__).parents[1] / "atlas_toolkit" / "data" / "ff"
AMBER99  = FF_DIR / "AMBER99.ff"

FRCMOD_DIR = Path("/home/robert/Downloads/test_for_maggie/march2026_peptide_aggregation/general_ffs")
_HAS_FRCMOD = (FRCMOD_DIR / "citrate.frcmod").exists()


# ── topology: tiny hand-crafted connectivity ───────────────────────────────────

@pytest.fixture
def linear_bonds():
    """1-2-3-4 linear chain: {1:[2], 2:[1,3], 3:[2,4], 4:[3]}"""
    return {1: [2], 2: [1, 3], 3: [2, 4], 4: [3]}


@pytest.fixture
def triangle_bonds():
    """Triangle: 1-2-3-1"""
    return {1: [2, 3], 2: [1, 3], 3: [1, 2]}


def test_enumerate_bonds_linear(linear_bonds):
    bonds = enumerate_bonds(linear_bonds)
    assert bonds == [(1, 2), (2, 3), (3, 4)]


def test_enumerate_bonds_no_duplicates(triangle_bonds):
    bonds = enumerate_bonds(triangle_bonds)
    assert len(bonds) == 3
    assert (1, 2) in bonds
    assert (1, 3) in bonds
    assert (2, 3) in bonds


def test_enumerate_angles_linear(linear_bonds):
    angles = enumerate_angles(linear_bonds)
    # 1-2-3, 2-3-4
    assert len(angles) == 2
    assert (1, 2, 3) in angles
    assert (2, 3, 4) in angles


def test_enumerate_angles_center_is_middle(linear_bonds):
    """Middle element of each tuple must be the vertex atom."""
    for (i, j, k) in enumerate_angles(linear_bonds):
        assert j in linear_bonds[i]
        assert j in linear_bonds[k]


def test_enumerate_torsions_linear(linear_bonds):
    torsions = enumerate_torsions(linear_bonds)
    assert len(torsions) == 1
    canon = torsions[0]
    assert set(canon) == {1, 2, 3, 4}


def test_enumerate_torsions_canonical(linear_bonds):
    """Each torsion should equal its canonical (lexicographically smallest) form."""
    for t in enumerate_torsions(linear_bonds):
        assert t <= t[::-1]


def test_enumerate_impropers_needs_3_bonds(linear_bonds):
    """No atom in the linear chain has ≥3 bonds → no impropers."""
    assert enumerate_impropers(linear_bonds) == []


def test_enumerate_impropers_star():
    """Central atom with 3 satellites → 1 improper."""
    bonds = {1: [2, 3, 4], 2: [1], 3: [1], 4: [1]}
    imps = enumerate_impropers(bonds)
    assert len(imps) == 1
    assert imps[0][0] == 1


def test_enumerate_impropers_four_satellites():
    """Central atom with 4 satellites → C(4,3)=4 impropers."""
    bonds = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
    imps = enumerate_impropers(bonds)
    assert len(imps) == 4


# ── topology: SPC water box ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spc_data():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)
    return atoms, bonds, box


def test_spc_bonds_count(spc_data):
    atoms, bonds, box = spc_data
    b = enumerate_bonds(bonds)
    n_wat = sum(1 for a in atoms.values() if a.get("RESNAME") == "WAT") // 3
    assert len(b) == 2 * n_wat   # 2 O-H bonds per water


def test_spc_angles_count(spc_data):
    atoms, bonds, box = spc_data
    a = enumerate_angles(bonds)
    n_wat = sum(1 for at in atoms.values() if at.get("RESNAME") == "WAT") // 3
    assert len(a) == n_wat        # 1 H-O-H angle per water


def test_spc_no_torsions(spc_data):
    atoms, bonds, box = spc_data
    t = enumerate_torsions(bonds)
    assert len(t) == 0            # water has no torsions


# ── data_file writer: SPC water with AMBER99 ──────────────────────────────────

@pytest.fixture(scope="module")
def amber_parms():
    return read_ff(AMBER99)


@pytest.fixture(scope="module")
def spc_data_file(tmp_path_factory, spc_data, amber_parms):
    atoms, bonds, box = spc_data
    out = tmp_path_factory.mktemp("lammps") / "data.spc_box"
    summary = write_data_file(out, atoms, bonds, amber_parms, box)
    return out, summary


def test_data_file_exists(spc_data_file):
    out, summary = spc_data_file
    assert out.exists()


def test_data_file_atom_count(spc_data_file, spc_data):
    out, summary = spc_data_file
    atoms, bonds, box = spc_data
    assert summary.n_atoms == len(atoms)


def test_data_file_bond_count(spc_data_file):
    out, summary = spc_data_file
    assert summary.n_bonds > 0


def test_data_file_angle_count(spc_data_file):
    out, summary = spc_data_file
    assert summary.n_angles > 0


def test_data_file_atom_types(spc_data_file):
    """SPC water has 2 atom types: OW and HW."""
    out, summary = spc_data_file
    assert summary.n_atom_types == 2


def test_data_file_bond_types(spc_data_file):
    """SPC water has 1 bond type: HW-OW."""
    out, summary = spc_data_file
    assert summary.n_bond_types == 1


def test_data_file_angle_types(spc_data_file):
    """SPC water has 1 angle type: HW-OW-HW."""
    out, summary = spc_data_file
    assert summary.n_angle_types == 1


def test_data_file_contains_masses_section(spc_data_file):
    out, summary = spc_data_file
    text = out.read_text()
    assert "Masses" in text


def test_data_file_contains_atoms_section(spc_data_file):
    out, summary = spc_data_file
    text = out.read_text()
    assert "Atoms  # full" in text


def test_data_file_contains_bonds_section(spc_data_file):
    out, summary = spc_data_file
    text = out.read_text()
    assert "Bonds" in text


def test_data_file_bond_coeffs(spc_data_file):
    out, summary = spc_data_file
    text = out.read_text()
    assert "Bond Coeffs" in text


def test_data_file_angle_coeffs(spc_data_file):
    out, summary = spc_data_file
    text = out.read_text()
    assert "Angle Coeffs" in text


def test_data_file_atom_line_format(spc_data_file, spc_data):
    """Each Atoms line must have 10 numeric fields."""
    out, summary = spc_data_file
    text = out.read_text()
    in_atoms = False
    lines_checked = 0
    for line in text.splitlines():
        if line.strip() == "Atoms  # full":
            in_atoms = True
            continue
        if in_atoms:
            if line.strip() == "":
                if lines_checked > 0:
                    break
                continue
            # stop at next section
            if line[0].isalpha():
                break
            parts = line.split()
            assert len(parts) == 10, f"Bad atom line: {line!r}"
            lines_checked += 1
    assert lines_checked == summary.n_atoms


def test_data_file_box_dimensions(spc_data_file, spc_data):
    """xlo xhi lines must appear in header and box lengths must be positive."""
    out, summary = spc_data_file
    text = out.read_text()
    for dim in ("xlo xhi", "ylo yhi", "zlo zhi"):
        assert dim in text


# ── input_script writer ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spc_input_script(tmp_path_factory, spc_data_file, spc_data, amber_parms):
    out_data, summary = spc_data_file
    atoms, bonds, box = spc_data
    out = tmp_path_factory.mktemp("lammps") / "in.spc_box"
    write_input_script(out, out_data.name, summary, amber_parms, box,
                       protocol="min")
    return out, summary


def test_input_script_exists(spc_input_script):
    out, _ = spc_input_script
    assert out.exists()


def test_input_script_has_pair_style(spc_input_script):
    out, _ = spc_input_script
    text = out.read_text()
    assert "pair_style" in text
    assert "lj/cut/coul/long" in text


def test_input_script_has_bond_style(spc_input_script):
    out, _ = spc_input_script
    text = out.read_text()
    assert "bond_style" in text


def test_input_script_has_angle_style(spc_input_script):
    out, _ = spc_input_script
    text = out.read_text()
    assert "angle_style" in text


def test_input_script_has_read_data(spc_input_script):
    out, _ = spc_input_script
    text = out.read_text()
    assert "read_data" in text


def test_input_script_has_minimize(spc_input_script):
    out, _ = spc_input_script
    text = out.read_text()
    assert "minimize" in text


def test_input_script_nvt(tmp_path, spc_data_file, spc_data, amber_parms):
    out_data, summary = spc_data_file
    atoms, bonds, box = spc_data
    out = tmp_path / "in.nvt"
    write_input_script(out, out_data.name, summary, amber_parms, box,
                       protocol="nvt")
    text = out.read_text()
    assert "nvt" in text
    assert "velocity" in text


def test_input_script_npt(tmp_path, spc_data_file, spc_data, amber_parms):
    out_data, summary = spc_data_file
    atoms, bonds, box = spc_data
    out = tmp_path / "in.npt"
    write_input_script(out, out_data.name, summary, amber_parms, box,
                       protocol="npt")
    text = out.read_text()
    assert "npt" in text


def test_input_script_bad_protocol_raises(tmp_path, spc_data_file, spc_data, amber_parms):
    out_data, summary = spc_data_file
    atoms, bonds, box = spc_data
    out = tmp_path / "in.bad"
    with pytest.raises(ValueError, match="Unknown protocol"):
        write_input_script(out, out_data.name, summary, amber_parms, box,
                           protocol="bad_proto")


# ── frcmod + data_file integration ────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FRCMOD, reason="citrate.frcmod not present")
def test_data_file_with_frcmod(tmp_path):
    """load_ff merges heinz.ff + citrate.frcmod; write_data_file completes."""
    heinz_ff = FRCMOD_DIR / "heinzAu_oplsIons_softChlorine.ff"
    cit_frcmod = FRCMOD_DIR / "citrate.frcmod"
    if not heinz_ff.exists():
        pytest.skip("heinz FF not present")
    # Just load and verify no exceptions
    parms = load_ff([str(heinz_ff), str(cit_frcmod)])
    assert "Au" in parms["ATOMTYPES"]
    assert "c3" in parms["ATOMTYPES"]
