"""Tests for core/replicate.py and scripts/replicate.py."""
import copy
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from atlas_toolkit.core.box import get_box, get_box_displacement_tensor
from atlas_toolkit.core.replicate import (
    combine_mols,
    replicate_cell,
    rotate,
    set_pbc_bonds,
    trans_atom,
    trans_cell_atoms,
)
from atlas_toolkit.io.bgf import read_bgf

FIXTURES   = Path(__file__).parent / "fixtures"
SPC_BOX    = FIXTURES / "spc_box.bgf"
TIP3       = FIXTURES / "tip3.bgf"
PERL_SCRIPT = Path(__file__).parents[2] / "ATLAS-toolkit/scripts/replicate.pl"


# ── trans_atom / trans_cell_atoms ────────────────────────────────────────────

def _ortho_box(a=10.0, b=10.0, c=10.0):
    atoms = {1: {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    headers = [f"CRYSTX  {a} {b} {c} 90.0 90.0 90.0"]
    box = get_box(atoms, headers)
    return box


def test_trans_atom_zero_vec():
    box = _ortho_box()
    atom = {"XCOORD": 1.0, "YCOORD": 2.0, "ZCOORD": 3.0}
    result = trans_atom(atom, box, [0, 0, 0])
    assert result["XCOORD"] == pytest.approx(1.0)
    assert result["YCOORD"] == pytest.approx(2.0)
    assert result["ZCOORD"] == pytest.approx(3.0)


def test_trans_atom_x_shift():
    box = _ortho_box(a=10.0)
    atom = {"XCOORD": 1.0, "YCOORD": 2.0, "ZCOORD": 3.0}
    result = trans_atom(atom, box, [1, 0, 0])
    assert result["XCOORD"] == pytest.approx(11.0)
    assert result["YCOORD"] == pytest.approx(2.0)
    assert result["ZCOORD"] == pytest.approx(3.0)


def test_trans_cell_atoms_is_deep_copy():
    atoms = {1: {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    box = _ortho_box()
    result = trans_cell_atoms(atoms, box, [1, 0, 0])
    result[1]["XCOORD"] = 999.0
    assert atoms[1]["XCOORD"] == pytest.approx(0.0)  # original unmodified


def test_trans_cell_atoms_y_shift():
    atoms = {1: {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0,
                 "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
                 "RESNAME": "RES", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0}}
    box = _ortho_box(b=5.0)
    result = trans_cell_atoms(atoms, box, [0, 2, 0])
    assert result[1]["YCOORD"] == pytest.approx(10.0)


# ── combine_mols ─────────────────────────────────────────────────────────────

def test_combine_mols_atom_count():
    a1 = {1: {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0,
              "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
              "RESNAME": "R", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0, "INDEX": 1}}
    b1 = {1: []}
    a2 = {1: {"XCOORD": 5.0, "YCOORD": 0.0, "ZCOORD": 0.0,
              "CHARGE": 0.0, "FFTYPE": "Y", "ATMNAME": "Y",
              "RESNAME": "R", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0, "INDEX": 1}}
    b2 = {1: []}
    merged_atoms, merged_bonds = combine_mols(a1, b1, a2, b2)
    assert len(merged_atoms) == 2
    assert 1 in merged_atoms and 2 in merged_atoms


def test_combine_mols_index_offset():
    a1 = {1: {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0,
              "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
              "RESNAME": "R", "RESNUM": 1, "NUMBONDS": 1, "LONEPAIRS": 0, "INDEX": 1},
          2: {"XCOORD": 1.0, "YCOORD": 0.0, "ZCOORD": 0.0,
              "CHARGE": 0.0, "FFTYPE": "Y", "ATMNAME": "Y",
              "RESNAME": "R", "RESNUM": 1, "NUMBONDS": 1, "LONEPAIRS": 0, "INDEX": 2}}
    b1 = {1: [2], 2: [1]}
    a2 = copy.deepcopy(a1)
    b2 = copy.deepcopy(b1)
    merged_atoms, merged_bonds = combine_mols(a1, b1, a2, b2)
    # Atoms from a2 get indices 3,4
    assert 3 in merged_atoms and 4 in merged_atoms
    # Bond 3→4 should exist
    assert 4 in merged_bonds[3]


def test_combine_mols_bonds_renumbered():
    a1 = {1: {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0,
              "CHARGE": 0.0, "FFTYPE": "X", "ATMNAME": "X",
              "RESNAME": "R", "RESNUM": 1, "NUMBONDS": 0, "LONEPAIRS": 0, "INDEX": 1}}
    b1 = {1: []}
    a2 = {1: {"XCOORD": 5.0, "YCOORD": 0.0, "ZCOORD": 0.0,
              "CHARGE": 0.0, "FFTYPE": "Y", "ATMNAME": "Y",
              "RESNAME": "R", "RESNUM": 1, "NUMBONDS": 1, "LONEPAIRS": 0, "INDEX": 1},
          2: {"XCOORD": 6.0, "YCOORD": 0.0, "ZCOORD": 0.0,
              "CHARGE": 0.0, "FFTYPE": "Z", "ATMNAME": "Z",
              "RESNAME": "R", "RESNUM": 1, "NUMBONDS": 1, "LONEPAIRS": 0, "INDEX": 2}}
    b2 = {1: [2], 2: [1]}
    _, merged_bonds = combine_mols(a1, b1, a2, b2)
    # a2's atom 1 → index 2, a2's atom 2 → index 3
    assert 3 in merged_bonds[2]


# ── replicate_cell ────────────────────────────────────────────────────────────

def test_replicate_2x1x1_atom_count():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    n_orig = len(atoms)
    box = get_box(atoms, headers)
    new_atoms, new_bonds, new_box = replicate_cell(
        atoms, bonds, box, {"X": 2, "Y": 1, "Z": 1}, pbc=False
    )
    assert len(new_atoms) == 2 * n_orig


def test_replicate_2x2x1_atom_count():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    n_orig = len(atoms)
    box = get_box(atoms, headers)
    new_atoms, new_bonds, new_box = replicate_cell(
        atoms, bonds, box, {"X": 2, "Y": 2, "Z": 1}, pbc=False
    )
    assert len(new_atoms) == 4 * n_orig


def test_replicate_box_dims_updated():
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)
    orig_len_x = box["X"]["len"]
    _, _, new_box = replicate_cell(
        atoms, bonds, box, {"X": 3, "Y": 1, "Z": 1}, pbc=False
    )
    assert new_box["X"]["len"] == pytest.approx(3 * orig_len_x, rel=1e-5)


def test_replicate_x_coordinates_shifted():
    """Second replica atoms should be shifted by one unit cell in X."""
    atoms, bonds, headers = read_bgf(SPC_BOX)
    n_orig = len(atoms)
    box = get_box(atoms, headers)
    cell_len_x = box["X"]["len"]
    new_atoms, _, _ = replicate_cell(
        atoms, bonds, box, {"X": 2, "Y": 1, "Z": 1}, pbc=False
    )
    # The replica atoms (indices n_orig+1 .. 2*n_orig) should be shifted
    for idx in range(n_orig + 1, 2 * n_orig + 1):
        orig_x = atoms[idx - n_orig]["XCOORD"]
        assert new_atoms[idx]["XCOORD"] == pytest.approx(
            float(orig_x) + cell_len_x, abs=1e-4
        )


# ── rotate ────────────────────────────────────────────────────────────────────

def test_rotate_identity():
    """Zero angles → no change."""
    atoms = {1: {"XCOORD": 1.0, "YCOORD": 2.0, "ZCOORD": 3.0}}
    rotate(atoms, [0.0, 0.0, 0.0], coord=3)
    assert atoms[1]["XCOORD"] == pytest.approx(1.0, abs=1e-9)
    assert atoms[1]["YCOORD"] == pytest.approx(2.0, abs=1e-9)
    assert atoms[1]["ZCOORD"] == pytest.approx(3.0, abs=1e-9)


def test_rotate_rz_90():
    """90° around Z: (1,0,0) → (0,1,0) (up to sign convention)."""
    atoms = {1: {"XCOORD": 1.0, "YCOORD": 0.0, "ZCOORD": 0.0}}
    rotate(atoms, [0.0, 0.0, math.pi / 2], coord=2)
    assert abs(atoms[1]["XCOORD"]) == pytest.approx(0.0, abs=1e-6)
    assert abs(atoms[1]["YCOORD"]) == pytest.approx(1.0, abs=1e-6)


def test_rotate_preserves_norm():
    """Rotation should preserve the vector length."""
    atoms = {1: {"XCOORD": 1.0, "YCOORD": 2.0, "ZCOORD": 3.0}}
    orig_norm = math.sqrt(1**2 + 2**2 + 3**2)
    rotate(atoms, [0.5, 1.0, 1.5], coord=3)
    new_norm = math.sqrt(
        atoms[1]["XCOORD"]**2 + atoms[1]["YCOORD"]**2 + atoms[1]["ZCOORD"]**2
    )
    assert new_norm == pytest.approx(orig_norm, abs=1e-9)


# ── oracle: compare Python vs Perl replicate ─────────────────────────────────

@pytest.mark.skipif(
    not shutil.which("perl") or not PERL_SCRIPT.exists(),
    reason="Perl not available or replicate.pl not found",
)
def test_oracle_replicate_2x1x1():
    """Python and Perl replicate.pl produce the same atom coordinates for 2x1x1."""
    import os
    atoms, bonds, headers = read_bgf(SPC_BOX)
    box = get_box(atoms, headers)

    with tempfile.TemporaryDirectory() as tmpdir:
        py_out   = os.path.join(tmpdir, "py_out.bgf")
        perl_out = os.path.join(tmpdir, "perl_out.bgf")

        # Perl replicate
        ff_path = Path(__file__).parents[2] / "ATLAS-toolkit/ff/tip3p.ff"
        subprocess.run(
            ["perl", str(PERL_SCRIPT),
             "-b", str(SPC_BOX), "-d", "2 1 1",
             "-f", str(ff_path),
             "-s", perl_out],
            check=True, capture_output=True,
        )

        # Python replicate
        from atlas_toolkit.core.replicate import replicate_cell
        from atlas_toolkit.core.headers import add_box_to_header, insert_header_remark
        from atlas_toolkit.io.bgf import write_bgf
        import copy
        a2, b2, headers2 = read_bgf(SPC_BOX)
        bx2 = get_box(a2, headers2)
        na, nb, nbx = replicate_cell(copy.deepcopy(a2), copy.deepcopy(b2), bx2, {"X":2,"Y":1,"Z":1}, pbc=True)
        insert_header_remark(headers2, f"REMARK {SPC_BOX} replicated 2x1x1")
        add_box_to_header(headers2, nbx)
        write_bgf(na, nb, py_out, headers2)

        def norm_lines(path):
            return [
                l.rstrip()
                for l in Path(path).read_text().splitlines()
                if l.strip() and not l.startswith("REMARK")
            ]

        pl = norm_lines(perl_out)
        py = norm_lines(py_out)
        # Compare HETATM/ATOM lines only (coord lines)
        pl_coords = [l for l in pl if l.startswith(("HETATM", "ATOM"))]
        py_coords = [l for l in py if l.startswith(("HETATM", "ATOM"))]
        assert len(py_coords) == len(pl_coords), "Atom count mismatch"
        # Check coordinate fields match within floating-point tolerance
        for pl_line, py_line in zip(pl_coords, py_coords):
            # Fields: x = cols 30-39, y = 40-49, z = 50-59
            for start in (30, 40, 50):
                pv = float(pl_line[start:start+10])
                yv = float(py_line[start:start+10])
                assert abs(pv - yv) < 0.001, f"Coord mismatch:\nPerl: {pl_line}\nPy:   {py_line}"
