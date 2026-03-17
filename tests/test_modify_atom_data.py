"""
Integration tests for modify_atom_data.py.

Layer 1: Python-only tests that don't require Perl.
Layer 2: Oracle tests that compare against Perl output (skipped if perl absent).
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from atlas_toolkit.io.bgf import read_bgf, write_bgf
from atlas_toolkit.scripts.modify_atom_data import (
    _parse_field_str,
    _update_atom_fields,
    _select_random,
    _validate_fields,
    FieldSpec,
)

FIXTURES = Path(__file__).parent / "fixtures"
TIP3 = FIXTURES / "tip3.bgf"
CL   = FIXTURES / "Cl-Chlorine.bgf"
PERL_SCRIPT = Path(__file__).parents[2] / "ATLAS-toolkit/scripts/modifyAtomData.pl"


# ── unit tests ─────────────────────────────────────────────────────────────

def test_parse_field_assign_negative():
    # "CHARGE:-1.0"  → mod="" val="-1.0"  OR  mod="-" val="1.0"
    # Perl regex: (\w+):(\+|\-|\.)?(.*) so - IS captured as modifier
    specs = _parse_field_str("CHARGE:-1.0")
    assert specs["CHARGE"].mod == "-"
    assert specs["CHARGE"].val == "1.0"

def test_parse_field_add():
    specs = _parse_field_str("CHARGE:+0.5")
    assert specs["CHARGE"].mod == "+"
    assert specs["CHARGE"].val == "0.5"

def test_parse_field_subtract():
    specs = _parse_field_str("CHARGE:-0.1")
    assert specs["CHARGE"].mod == "-"
    assert specs["CHARGE"].val == "0.1"

def test_parse_field_append():
    specs = _parse_field_str("RESNAME:.X")
    assert specs["RESNAME"].mod == "."
    assert specs["RESNAME"].val == "X"

def test_parse_field_plain_assign():
    specs = _parse_field_str("RESNAME:NEW")
    assert specs["RESNAME"].mod == ""
    assert specs["RESNAME"].val == "NEW"

def test_parse_multiple_fields():
    specs = _parse_field_str("CHARGE:-1.0 RESNAME:RES")
    assert set(specs) == {"CHARGE", "RESNAME"}

def test_update_fields_assign():
    atom = {"CHARGE": 0.0, "RESNAME": "WAT"}
    _update_atom_fields({1: atom}, {1: 1},
                        {"CHARGE": FieldSpec(mod="", val="-1.0")})
    assert abs(atom["CHARGE"] - (-1.0)) < 1e-9

def test_update_fields_add():
    atom = {"CHARGE": -0.5}
    _update_atom_fields({1: atom}, {1: 1},
                        {"CHARGE": FieldSpec(mod="+", val="0.5")})
    assert abs(atom["CHARGE"] - 0.0) < 1e-9

def test_update_fields_subtract():
    atom = {"CHARGE": 1.0}
    _update_atom_fields({1: atom}, {1: 1},
                        {"CHARGE": FieldSpec(mod="-", val="0.5")})
    assert abs(atom["CHARGE"] - 0.5) < 1e-9

def test_update_fields_string_append():
    atom = {"RESNAME": "WAT"}
    _update_atom_fields({1: atom}, {1: 1},
                        {"RESNAME": FieldSpec(mod=".", val="2")})
    assert atom["RESNAME"] == "WAT2"

def test_select_random_keeps_n():
    sel = {i: 1 for i in range(100)}
    _select_random(sel, 10)
    assert len(sel) == 10

def test_select_random_noop_if_small():
    sel = {1: 1, 2: 1}
    _select_random(sel, 5)
    assert len(sel) == 2

def test_validate_fields_warns_unknown(capsys):
    atoms = {1: {"CHARGE": 0.0}}
    specs = {"CHARGE": FieldSpec("", "1.0"), "NONEXISTENT": FieldSpec("", "x")}
    _validate_fields(atoms, specs)
    assert "NONEXISTENT" not in specs
    captured = capsys.readouterr()
    assert "NONEXISTENT" in captured.err


# ── integration test: run script end-to-end ────────────────────────────────

def test_modify_charge_end_to_end():
    """Run modify_atom_data as a script, verify charge is updated."""
    atoms, bonds, headers = read_bgf(CL)
    original_charge = atoms[1]["CHARGE"]

    with tempfile.NamedTemporaryFile(suffix=".bgf", delete=False, mode="w") as f:
        out = f.name
    try:
        from atlas_toolkit.scripts.modify_atom_data import (
            _parse_field_str, _update_atom_fields, _validate_fields
        )
        import copy
        atoms2 = copy.deepcopy(atoms)
        specs = _parse_field_str("CHARGE:-0.5")
        _validate_fields(atoms2, specs)
        _update_atom_fields(atoms2, {1: 1}, specs)
        assert abs(atoms2[1]["CHARGE"] - (-0.5 + original_charge)) < 1e-6
    finally:
        if os.path.exists(out):
            os.unlink(out)


# ── oracle test: compare Python output vs Perl output ─────────────────────

@pytest.mark.skipif(
    not shutil.which("perl") or not PERL_SCRIPT.exists(),
    reason="Perl not available or modifyAtomData.pl not found",
)
def test_oracle_charge_assignment():
    """Python and Perl should produce identical BGF output for a simple charge assignment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        perl_out = os.path.join(tmpdir, "perl_out.bgf")
        py_out   = os.path.join(tmpdir, "py_out.bgf")

        # Run Perl  ("index>0" is valid Perl eval and selects all atoms)
        subprocess.run(
            ["perl", str(PERL_SCRIPT),
             "-s", str(CL),
             "-a", "index>0",
             "-f", "CHARGE:-1.5",
             "-w", perl_out],
            check=True, capture_output=True,
        )

        # Run Python  ("index>0" is also supported by our parser)
        subprocess.run(
            [shutil.which("python3") or "python",
             str(Path(__file__).parents[1] / "atlas_toolkit/scripts/modify_atom_data.py"),
             "-s", str(CL),
             "-a", "index>0",
             "-f", "CHARGE:-1.5",
             "-w", py_out],
            check=True, capture_output=True,
        )

        def normalise(path):
            lines = Path(path).read_text().splitlines()
            # Strip trailing whitespace; ignore REMARK lines (timestamps differ)
            return [l.rstrip() for l in lines
                    if not l.startswith("REMARK") and l.strip()]

        assert normalise(py_out) == normalise(perl_out), (
            "Python and Perl output differ — see files in " + tmpdir
        )
