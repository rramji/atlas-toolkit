"""Tests for BGF reader and writer."""
import os
import tempfile
from pathlib import Path

import pytest

from atlas_toolkit.io.bgf import read_bgf, write_bgf

FIXTURES = Path(__file__).parent / "fixtures"
TIP3 = FIXTURES / "tip3.bgf"
SPC_BOX = FIXTURES / "spc_box.bgf"
CL = FIXTURES / "Cl-Chlorine.bgf"


# ── reader tests ───────────────────────────────────────────────────────────

def test_read_tip3_atom_count():
    atoms, bonds, headers = read_bgf(TIP3)
    assert len(atoms) == 3

def test_read_tip3_indices():
    atoms, bonds, _ = read_bgf(TIP3)
    assert set(atoms) == {1, 2, 3}

def test_read_tip3_hetatm_label():
    atoms, _, _ = read_bgf(TIP3)
    assert atoms[1]["LABEL"] == "HETATM"

def test_read_tip3_coords():
    atoms, _, _ = read_bgf(TIP3)
    assert abs(atoms[1]["XCOORD"] - (-0.23900)) < 1e-9
    assert abs(atoms[1]["YCOORD"] - (-0.30900)) < 1e-9
    assert abs(atoms[1]["ZCOORD"] -   0.00000)  < 1e-9

def test_read_tip3_fftype():
    atoms, _, _ = read_bgf(TIP3)
    assert atoms[1]["FFTYPE"] == "OW"
    assert atoms[2]["FFTYPE"] == "HW"

def test_read_tip3_charge():
    atoms, _, _ = read_bgf(TIP3)
    assert abs(atoms[1]["CHARGE"] - (-0.83400)) < 1e-9
    assert abs(atoms[2]["CHARGE"] -   0.41700)  < 1e-9

def test_read_tip3_resname():
    atoms, _, _ = read_bgf(TIP3)
    assert atoms[1]["RESNAME"] == "WAT"

def test_read_tip3_no_bonds():
    _, bonds, _ = read_bgf(TIP3)
    # tip3.bgf has empty CONECT lines
    assert all(len(v) == 0 for v in bonds.values())

def test_read_tip3_headers():
    _, _, headers = read_bgf(TIP3)
    assert any("BIOGRF" in h for h in headers)
    assert any("DESCRP" in h for h in headers)

def test_read_tip3_no_headers():
    _, _, headers = read_bgf(TIP3, save_headers=False)
    assert headers == []

def test_read_cl_single_atom():
    atoms, bonds, headers = read_bgf(CL)
    assert len(atoms) == 1
    assert 1 in atoms
    assert atoms[1]["FFTYPE"] == "Cl-"
    assert abs(atoms[1]["CHARGE"] - (-1.0)) < 1e-6

def test_read_spc_box_has_crystx():
    _, _, headers = read_bgf(SPC_BOX)
    assert any(h.startswith("CRYSTX") for h in headers)


# ── writer tests ───────────────────────────────────────────────────────────

def test_write_produces_file():
    atoms, bonds, headers = read_bgf(TIP3)
    with tempfile.NamedTemporaryFile(suffix=".bgf", delete=False) as f:
        tmp = f.name
    try:
        write_bgf(atoms, bonds, tmp, headers)
        assert os.path.getsize(tmp) > 0
    finally:
        os.unlink(tmp)

def test_roundtrip_tip3():
    """Read → write → re-read should give same atoms."""
    atoms, bonds, headers = read_bgf(TIP3)
    with tempfile.NamedTemporaryFile(suffix=".bgf", delete=False, mode="w") as f:
        tmp = f.name
    try:
        write_bgf(atoms, bonds, tmp, headers)
        atoms2, bonds2, headers2 = read_bgf(tmp)

        assert set(atoms2) == set(atoms)
        for idx in atoms:
            assert abs(atoms2[idx]["XCOORD"] - atoms[idx]["XCOORD"]) < 1e-9
            assert abs(atoms2[idx]["YCOORD"] - atoms[idx]["YCOORD"]) < 1e-9
            assert abs(atoms2[idx]["ZCOORD"] - atoms[idx]["ZCOORD"]) < 1e-9
            assert abs(atoms2[idx]["CHARGE"] - atoms[idx]["CHARGE"]) < 1e-9
            assert atoms2[idx]["FFTYPE"] == atoms[idx]["FFTYPE"]
            assert atoms2[idx]["LABEL"]  == atoms[idx]["LABEL"]
    finally:
        os.unlink(tmp)

def test_roundtrip_spc_box():
    """Round-trip a periodic-box file — CRYSTX header must survive."""
    atoms, bonds, headers = read_bgf(SPC_BOX)
    with tempfile.NamedTemporaryFile(suffix=".bgf", delete=False, mode="w") as f:
        tmp = f.name
    try:
        write_bgf(atoms, bonds, tmp, headers)
        _, _, headers2 = read_bgf(tmp)
        assert any(h.startswith("CRYSTX") for h in headers2)
        # CRYSTX values preserved
        crystx_in  = next(h for h in headers  if h.startswith("CRYSTX"))
        crystx_out = next(h for h in headers2 if h.startswith("CRYSTX"))
        assert crystx_in == crystx_out
    finally:
        os.unlink(tmp)

def test_write_format_atom_line():
    """Verify the ATOM line matches the expected Perl printf format."""
    atoms, bonds, headers = read_bgf(CL)
    with tempfile.NamedTemporaryFile(suffix=".bgf", delete=False, mode="w") as f:
        tmp = f.name
    try:
        write_bgf(atoms, bonds, tmp, headers)
        lines = Path(tmp).read_text().splitlines()
        atom_line = next(l for l in lines if l.startswith("HETATM"))
        # Perl: "%-6s %5d %-5s %3s %1s %5d%10.5f%10.5f%10.5f %-5s %2d %1d %9.6f"
        # Check coordinate fields are 10.5f
        assert "   0.00000" in atom_line
        assert "   0.82600" in atom_line
        assert "-1.000000" in atom_line
    finally:
        os.unlink(tmp)

def test_write_end_line():
    atoms, bonds, headers = read_bgf(TIP3)
    with tempfile.NamedTemporaryFile(suffix=".bgf", delete=False, mode="w") as f:
        tmp = f.name
    try:
        write_bgf(atoms, bonds, tmp)
        content = Path(tmp).read_text()
        assert content.strip().endswith("END")
    finally:
        os.unlink(tmp)
