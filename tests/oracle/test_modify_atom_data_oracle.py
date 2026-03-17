"""Oracle: compare atlas-modify-atom-data output to modifyAtomData.pl."""

import pytest

from .conftest import run_perl, run_python, FIXTURES
from .bgf_compare import assert_bgf_equal

pytestmark = pytest.mark.perl_oracle

SPC_BOX = FIXTURES / "spc_box.bgf"
PY_MOD  = "atlas_toolkit.scripts.modify_atom_data"

# Note: Perl's eval-based selection requires quoted string values: eq 'SPC'
# Python's predicate parser accepts both quoted and bare: eq SPC


class TestModifyAtomDataOracle:
    def test_set_charge(self, tmp_path):
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("modifyAtomData.pl", ["-s", str(SPC_BOX),
                                           "-a", "resname eq 'WAT'",
                                           "-f", "CHARGE:0.0",
                                           "-w", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-s", str(SPC_BOX),
                                  "-a", "resname eq WAT",
                                  "-f", "CHARGE:0.0",
                                  "-w", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        assert_bgf_equal(py_out, perl_out, msg="set CHARGE:0.0", check_coords=False)

    def test_add_to_charge(self, tmp_path):
        # Perl's eval-based parser rejects bare "*"; use "index > 0" which
        # selects all atoms in both implementations.
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("modifyAtomData.pl", ["-s", str(SPC_BOX),
                                           "-a", "index > 0",
                                           "-f", "CHARGE:+0.01",
                                           "-w", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-s", str(SPC_BOX),
                                  "-a", "index > 0",
                                  "-f", "CHARGE:+0.01",
                                  "-w", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        assert_bgf_equal(py_out, perl_out, msg="add +0.01 to all charges",
                         check_coords=False)

    def test_set_resname(self, tmp_path):
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("modifyAtomData.pl", ["-s", str(SPC_BOX),
                                           "-a", "fftype eq 'OW'",
                                           "-f", "RESNAME:WAT",
                                           "-w", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-s", str(SPC_BOX),
                                  "-a", "fftype eq OW",
                                  "-f", "RESNAME:WAT",
                                  "-w", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        assert_bgf_equal(py_out, perl_out, msg="rename OW to WAT",
                         check_coords=False, fields=["RESNAME", "FFTYPE"])
