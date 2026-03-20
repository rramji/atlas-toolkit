"""Tests for io/mol2.py (write_mol2, read_mol2) and scripts/bgf_to_mol2.py."""
import pytest
from pathlib import Path

from atlas_toolkit.io.bgf import parse_struct_file
from atlas_toolkit.io.mol2 import read_mol2, write_mol2
from atlas_toolkit.core.manip_atoms import get_mols

FIXTURES = Path(__file__).parent / "fixtures"
SPC_BOX = FIXTURES / "spc_box.bgf"
TIP3 = FIXTURES / "tip3.bgf"


# ── write_mol2 ────────────────────────────────────────────────────────────────

class TestWriteMol2:
    def test_sections_present(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        out = tmp_path / "out.mol2"
        write_mol2(atoms, bonds, out)
        text = out.read_text()
        for sec in ("@<TRIPOS>MOLECULE", "@<TRIPOS>ATOM",
                    "@<TRIPOS>BOND", "@<TRIPOS>SUBSTRUCTURE"):
            assert sec in text

    def test_atom_count_in_molecule_header(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        out = tmp_path / "out.mol2"
        write_mol2(atoms, bonds, out)
        lines = out.read_text().splitlines()
        # Line after @<TRIPOS>MOLECULE and title is the count line
        mol_idx = lines.index("@<TRIPOS>MOLECULE")
        counts = lines[mol_idx + 2].split()
        assert int(counts[0]) == len(atoms)

    def test_bond_count_in_molecule_header(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        out = tmp_path / "out.mol2"
        write_mol2(atoms, bonds, out)
        lines = out.read_text().splitlines()
        mol_idx = lines.index("@<TRIPOS>MOLECULE")
        counts = lines[mol_idx + 2].split()
        expected_bonds = sum(len(v) for v in bonds.values()) // 2
        assert int(counts[1]) == expected_bonds

    def test_charges_written(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        out = tmp_path / "out.mol2"
        write_mol2(atoms, bonds, out)
        atom_lines = [l for l in out.read_text().splitlines()
                      if l and not l.startswith("@") and len(l.split()) >= 9]
        # First charge should be OW charge ~ -0.8476
        charge = float(atom_lines[0].split()[-1])
        assert abs(charge - (-0.8476)) < 1e-3

    def test_custom_title(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        out = tmp_path / "out.mol2"
        write_mol2(atoms, bonds, out, title="MySPC")
        lines = out.read_text().splitlines()
        mol_idx = lines.index("@<TRIPOS>MOLECULE")
        assert lines[mol_idx + 1] == "MySPC"

    def test_no_bonds_in_tip3(self, tmp_path):
        """tip3.bgf has explicit bonds — verify they appear."""
        atoms, bonds, _ = parse_struct_file(str(TIP3))
        out = tmp_path / "out.mol2"
        write_mol2(atoms, bonds, out)
        bond_lines = []
        in_bond = False
        for line in out.read_text().splitlines():
            if line.startswith("@<TRIPOS>BOND"):
                in_bond = True
                continue
            if line.startswith("@<TRIPOS>") and in_bond:
                break
            if in_bond and line.strip():
                bond_lines.append(line)
        expected = sum(len(v) for v in bonds.values()) // 2
        assert len(bond_lines) == expected


# ── read_mol2 round-trip ──────────────────────────────────────────────────────

class TestReadMol2:
    def test_roundtrip_atom_count(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        mol2 = tmp_path / "rt.mol2"
        write_mol2(atoms, bonds, mol2)
        rt_atoms, rt_bonds, _ = read_mol2(mol2)
        assert len(rt_atoms) == len(atoms)

    def test_roundtrip_bond_count(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        mol2 = tmp_path / "rt.mol2"
        write_mol2(atoms, bonds, mol2)
        rt_atoms, rt_bonds, _ = read_mol2(mol2)
        orig_bonds = sum(len(v) for v in bonds.values()) // 2
        rt_bonds_count = sum(len(v) for v in rt_bonds.values()) // 2
        assert rt_bonds_count == orig_bonds

    def test_roundtrip_coords(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(TIP3))
        mol2 = tmp_path / "rt.mol2"
        write_mol2(atoms, bonds, mol2)
        rt_atoms, _, _ = read_mol2(mol2)
        for idx in atoms:
            for coord in ("XCOORD", "YCOORD", "ZCOORD"):
                assert abs(float(atoms[idx][coord]) - float(rt_atoms[idx][coord])) < 1e-3

    def test_roundtrip_charges(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        mol2 = tmp_path / "rt.mol2"
        write_mol2(atoms, bonds, mol2)
        rt_atoms, _, _ = read_mol2(mol2)
        for idx in atoms:
            assert abs(float(atoms[idx].get("CHARGE", 0)) -
                       float(rt_atoms[idx].get("CHARGE", 0))) < 1e-4

    def test_roundtrip_fftype(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(SPC_BOX))
        mol2 = tmp_path / "rt.mol2"
        write_mol2(atoms, bonds, mol2)
        rt_atoms, _, _ = read_mol2(mol2)
        for idx in atoms:
            assert atoms[idx].get("FFTYPE") == rt_atoms[idx].get("FFTYPE")

    def test_headers_empty_list(self, tmp_path):
        atoms, bonds, _ = parse_struct_file(str(TIP3))
        mol2 = tmp_path / "rt.mol2"
        write_mol2(atoms, bonds, mol2)
        _, _, headers = read_mol2(mol2)
        assert headers == []


# ── CLI bgf_to_mol2 ───────────────────────────────────────────────────────────

class TestBgfToMol2Script:
    def test_default_output_name(self, tmp_path):
        import shutil
        from atlas_toolkit.scripts.bgf_to_mol2 import bgf_to_mol2
        src = shutil.copy(str(SPC_BOX), tmp_path / "spc_box.bgf")
        bgf_to_mol2(str(src))
        assert (tmp_path / "spc_box.mol2").exists()

    def test_explicit_output(self, tmp_path):
        from atlas_toolkit.scripts.bgf_to_mol2 import bgf_to_mol2
        out = str(tmp_path / "explicit.mol2")
        bgf_to_mol2(str(SPC_BOX), save_path=out)
        assert Path(out).exists()

    def test_selection_reduces_atoms(self, tmp_path):
        from atlas_toolkit.scripts.bgf_to_mol2 import bgf_to_mol2
        out = str(tmp_path / "ow.mol2")
        bgf_to_mol2(str(SPC_BOX), save_path=out, selection="fftype eq OW")
        rt_atoms, _, _ = read_mol2(out)
        orig_atoms, _, _ = parse_struct_file(str(SPC_BOX))
        assert len(rt_atoms) < len(orig_atoms)
