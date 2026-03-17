"""Oracle: compare atlas-remove-mols output to removeMols.pl."""

import pytest

from atlas_toolkit.io.bgf import read_bgf
from .conftest import run_perl, run_python, FIXTURES
from .bgf_compare import compare_bgf

pytestmark = pytest.mark.perl_oracle

SPC_BOX = FIXTURES / "spc_box.bgf"
PY_MOD  = "atlas_toolkit.scripts.remove_mols"


class TestRemoveMolsOracle:
    def test_remove_by_resname_count(self, tmp_path):
        """Remove 5 SPC molecules — atom count must match Perl."""
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("removeMols.pl", ["-b", str(SPC_BOX),
                                       "-a", "resname eq 'WAT'",
                                       "-m", "5",
                                       "-s", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-b", str(SPC_BOX),
                                  "-a", "resname eq WAT",
                                  "-m", "5",
                                  "-s", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        atoms_perl, _, _ = read_bgf(perl_out)
        atoms_py,   _, _ = read_bgf(py_out)
        assert len(atoms_py) == len(atoms_perl), (
            f"atom count: python={len(atoms_py)}, perl={len(atoms_perl)}"
        )

    def test_remove_fields_unchanged(self, tmp_path):
        """Remaining atoms should have identical FFTYPE/CHARGE/coords."""
        perl_out = tmp_path / "perl_out.bgf"
        py_out   = tmp_path / "py_out.bgf"

        r = run_perl("removeMols.pl", ["-b", str(SPC_BOX),
                                       "-a", "resname eq 'WAT'",
                                       "-m", "2",
                                       "-s", str(perl_out)])
        assert r.returncode == 0, f"Perl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, ["-b", str(SPC_BOX),
                                  "-a", "resname eq WAT",
                                  "-m", "2",
                                  "-s", str(py_out)])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        diff = compare_bgf(py_out, perl_out,
                           check_coords=True, coord_tol=0.001,
                           check_bonds=True, check_box=False)
        assert diff.ok, f"Differences:\n{diff.summary()}"
