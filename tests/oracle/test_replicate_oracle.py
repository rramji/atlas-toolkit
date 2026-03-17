"""Oracle: compare atlas-replicate output to replicate.pl for the same input."""

import pytest

from .conftest import run_perl, run_python, FIXTURES
from .bgf_compare import assert_bgf_equal

pytestmark = pytest.mark.perl_oracle

SPC_BOX = FIXTURES / "spc_box.bgf"
PY_MOD  = "atlas_toolkit.scripts.replicate"


class TestReplicateOracle:
    def test_replicate_2x2x1(self, tmp_path):
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("replicate.pl", ["-b", str(SPC_BOX), "-f", "AMBER99",
                                      "-d", "2 2 1", "-s", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-b", str(SPC_BOX), "-d", "2 2 1", "-s", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        assert_bgf_equal(py_out, perl_out, msg="replicate 2x2x1", coord_tol=0.001)

    def test_replicate_3x1x1(self, tmp_path):
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("replicate.pl", ["-b", str(SPC_BOX), "-f", "AMBER99",
                                      "-d", "3 1 1", "-s", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-b", str(SPC_BOX), "-d", "3 1 1", "-s", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        assert_bgf_equal(py_out, perl_out, msg="replicate 3x1x1", coord_tol=0.001)
