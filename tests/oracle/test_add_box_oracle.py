"""Oracle: compare atlas-add-box-to-bgf output to addBoxToBGF.pl."""

import pytest

from .conftest import run_perl, run_python, FIXTURES
from .bgf_compare import compare_bgf

pytestmark = pytest.mark.perl_oracle

TIP3    = FIXTURES / "tip3.bgf"
SPC_BOX = FIXTURES / "spc_box.bgf"
PY_MOD  = "atlas_toolkit.scripts.add_box_to_bgf"


class TestAddBoxOracle:
    def _run_both(self, tmp_path, input_bgf):
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("addBoxToBGF.pl", [str(input_bgf), str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, [str(input_bgf), str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        return perl_out, py_out

    def test_box_added_to_tip3(self, tmp_path):
        perl_out, py_out = self._run_both(tmp_path, TIP3)
        diff = compare_bgf(py_out, perl_out,
                           check_coords=True, coord_tol=0.001,
                           check_bonds=True, check_box=True)
        assert diff.ok, f"Differences:\n{diff.summary()}"

    def test_box_updated_in_spc_atoms_unchanged(self, tmp_path):
        # Deliberate behavioral difference: Perl's addBoxToBGF.pl respects an existing
        # CRYSTX (GetBox reads from headers first), while our implementation always
        # recomputes the bounding box from atom coordinates.  Atoms and bonds are identical.
        perl_out, py_out = self._run_both(tmp_path, SPC_BOX)
        diff = compare_bgf(py_out, perl_out,
                           check_coords=True, coord_tol=0.001,
                           check_bonds=True, check_box=False)
        assert diff.ok, f"Atom/bond differences:\n{diff.summary()}"
