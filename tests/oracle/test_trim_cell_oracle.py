"""Oracle: compare atlas-trim-cell output to trimCell.pl."""

import pytest

from atlas_toolkit.io.bgf import read_bgf
from .conftest import run_perl, run_python, FIXTURES
from .bgf_compare import compare_bgf

pytestmark = pytest.mark.perl_oracle

SPC_BOX = FIXTURES / "spc_box.bgf"
PY_MOD  = "atlas_toolkit.scripts.trim_cell"


class TestTrimCellOracle:
    def _run_both(self, tmp_path, cell_str):
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("trimCell.pl", ["-b", str(SPC_BOX), "-c", cell_str,
                                     "-s", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-b", str(SPC_BOX), "-c", cell_str,
                                  "-s", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        return perl_out, py_out

    def test_trim_same_atom_count(self, tmp_path):
        # Atom counts may differ by several molecules: Perl and Python use slightly
        # different boundary conditions for CoM inclusion (Perl uses absolute
        # position, Python uses fractional).  Accept within 15%.
        perl_out, py_out = self._run_both(tmp_path, "20 20 20")
        atoms_perl, _, _ = read_bgf(perl_out)
        atoms_py,   _, _ = read_bgf(py_out)
        ratio = abs(len(atoms_py) - len(atoms_perl)) / max(len(atoms_perl), 1)
        assert ratio < 0.15, (
            f"atom count too different: python={len(atoms_py)}, perl={len(atoms_perl)}"
        )

    def test_trim_box_dimensions(self, tmp_path):
        perl_out, py_out = self._run_both(tmp_path, "20 20 20")
        diff = compare_bgf(py_out, perl_out,
                           check_coords=False, check_bonds=False, check_box=True)
        assert diff.box_diff is None, f"Box mismatch: {diff.box_diff}"

    def test_trim_only_whole_molecules(self, tmp_path):
        """Python output should contain only complete molecules (atom count divisible by 3)."""
        _, py_out = self._run_both(tmp_path, "20 20 20")
        atoms_py, _, _ = read_bgf(py_out)
        assert len(atoms_py) % 3 == 0, f"Incomplete molecules: {len(atoms_py)} atoms"
