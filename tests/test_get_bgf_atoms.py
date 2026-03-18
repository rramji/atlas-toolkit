"""Tests for scripts/get_bgf_atoms.py and scripts/combine_bgf.py."""
import pytest
from atlas_toolkit.io.bgf import parse_struct_file, get_bgf_atoms
from atlas_toolkit.core.manip_atoms import get_bounds, get_mols, select_atoms
from atlas_toolkit.scripts.combine_bgf import combine_bgf

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"
SPC_BOX = FIXTURES / "spc_box.bgf"
TIP3 = FIXTURES / "tip3.bgf"


# ── get_bgf_atoms ─────────────────────────────────────────────────────────────

class TestGetBgfAtoms:
    def test_select_subset_atoms(self):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        sel = {idx: 1 for idx in list(atoms)[:6]}  # first 6 atoms
        sub_atoms, sub_bonds = get_bgf_atoms(sel, atoms, bonds)
        assert len(sub_atoms) == 6

    def test_reindexed_sequential(self):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        sel = {idx: 1 for idx in list(atoms)[10:20]}  # non-contiguous slice
        sub_atoms, sub_bonds = get_bgf_atoms(sel, atoms, bonds)
        assert list(sub_atoms.keys()) == list(range(1, len(sub_atoms) + 1))

    def test_bonds_remapped(self):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        # Select one complete water molecule (atoms 1-3 are bonded in spc_box)
        sel = {1: 1, 2: 1, 3: 1}
        sub_atoms, sub_bonds = get_bgf_atoms(sel, atoms, bonds)
        # All bond targets should be within the new index range
        for src, targets in sub_bonds.items():
            for t in targets:
                assert t in sub_atoms

    def test_cross_molecule_bonds_dropped(self):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        # Select only oxygen atoms — no bonds should remain (O is bonded to H only)
        get_mols(atoms, bonds)
        sel = select_atoms("fftype eq OW", atoms)
        sub_atoms, sub_bonds = get_bgf_atoms(sel, atoms, bonds)
        # All bond lists should be empty (H atoms excluded)
        assert all(len(v) == 0 for v in sub_bonds.values())

    def test_all_atoms_selection(self):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        sel = {idx: 1 for idx in atoms}
        sub_atoms, sub_bonds = get_bgf_atoms(sel, atoms, bonds)
        assert len(sub_atoms) == len(atoms)


# ── combine_bgf ───────────────────────────────────────────────────────────────

class TestCombineBgf:
    def test_atom_count_additive(self):
        atoms1, _, _ = parse_struct_file(str(TIP3))
        atoms2, _, _ = parse_struct_file(str(SPC_BOX))
        merged, _, _ = combine_bgf([str(TIP3), str(SPC_BOX)])
        assert len(merged) == len(atoms1) + len(atoms2)

    def test_sequential_indices(self):
        merged, _, _ = combine_bgf([str(TIP3), str(SPC_BOX)])
        assert list(merged.keys()) == list(range(1, len(merged) + 1))

    def test_no_bond_collisions(self):
        merged, bonds, _ = combine_bgf([str(TIP3), str(SPC_BOX)])
        for src, targets in bonds.items():
            for t in targets:
                assert t in merged, f"bond target {t} not in merged atoms"

    def test_headers_from_last_file(self):
        # Headers should come from the last file (SPC_BOX has CRYSTX, TIP3 does not)
        _, _, headers = combine_bgf([str(TIP3), str(SPC_BOX)])
        assert any("CRYSTX" in h for h in headers)

    def test_three_files(self, tmp_path):
        a = str(TIP3); b = str(SPC_BOX)
        atoms_a, _, _ = parse_struct_file(a)
        atoms_b, _, _ = parse_struct_file(b)
        merged, _, _ = combine_bgf([a, b, a])
        assert len(merged) == len(atoms_a) * 2 + len(atoms_b)


# ── get_bounds ────────────────────────────────────────────────────────────────

class TestGetBounds:
    def test_all_atoms_bounds(self):
        atoms, _, _ = parse_struct_file(str(SPC_BOX))
        bounds = get_bounds(atoms)
        for dim in ("X", "Y", "Z"):
            assert bounds[dim]["min"] < bounds[dim]["max"]

    def test_selection_subset(self):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        sel = select_atoms("fftype eq OW", atoms)
        bounds_all = get_bounds(atoms)
        bounds_ow = get_bounds(atoms, sel)
        # OW bounds should be within total bounds
        for dim in ("X", "Y", "Z"):
            assert bounds_ow[dim]["min"] >= bounds_all[dim]["min"] - 1e-6
            assert bounds_ow[dim]["max"] <= bounds_all[dim]["max"] + 1e-6

    def test_single_atom(self):
        atoms, _, _ = parse_struct_file(str(TIP3))
        sel = {1: 1}
        bounds = get_bounds(atoms, sel)
        for dim in ("X", "Y", "Z"):
            assert bounds[dim]["min"] == bounds[dim]["max"]

    def test_empty_selection_returns_zeros(self):
        atoms, _, _ = parse_struct_file(str(SPC_BOX))
        bounds = get_bounds(atoms, {})
        for dim in ("X", "Y", "Z"):
            assert bounds[dim]["min"] == 0.0
            assert bounds[dim]["max"] == 0.0
