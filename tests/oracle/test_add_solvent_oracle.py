"""Oracle: compare atlas-add-solvent output to addSolvent.pl.

add_solvent involves random placement, so we compare structure (atom count,
residue composition, FF types) rather than exact coordinates.
"""

import pytest

from atlas_toolkit.io.bgf import read_bgf
from .conftest import run_perl, run_python, FIXTURES

pytestmark = pytest.mark.perl_oracle

TIP3   = FIXTURES / "tip3.bgf"
PY_MOD = "atlas_toolkit.scripts.add_solvent"


class TestAddSolventOracle:
    def _run_both(self, tmp_path, n_spec, solvent="tip3"):
        perl_dir = tmp_path / "perl"
        py_dir   = tmp_path / "py"
        perl_dir.mkdir()
        py_dir.mkdir()
        perl_out = perl_dir / "perl_out.bgf"
        py_out   = py_dir   / "py_out.bgf"

        r = run_perl("addSolvent.pl", [
            "-i", str(TIP3),
            "-f", "AMBER99",
            "-n", n_spec,
            "-w", solvent,
            "-s", str(perl_out),
        ])
        assert r.returncode == 0, f"addSolvent.pl failed:\n{r.stderr}"

        r2 = run_python(PY_MOD, [
            "-i", str(TIP3),
            "-n", n_spec,
            "-w", solvent,
            "-s", str(py_out),
        ])
        assert r2.returncode == 0, f"Python failed:\n{r2.stderr}"

        return perl_out, py_out

    def test_total_count(self, tmp_path):
        """Both should produce approximately total: 10 solvent molecules (±3 tolerance).

        Perl's removal has an off-by-a-few quirk when the solute shares RESNAME
        with the solvent; both aim for 10 but may differ by 1-3 molecules.
        """
        perl_out, py_out = self._run_both(tmp_path, "total: 10")
        atoms_perl, _, _ = read_bgf(perl_out)
        atoms_py,   _, _ = read_bgf(py_out)
        # Each molecule is 3 atoms; accept ±3 molecule difference
        diff_mols = abs(len(atoms_py) - len(atoms_perl)) // 3
        assert diff_mols <= 3, (
            f"molecule count too different: python={len(atoms_py)//3}, perl={len(atoms_perl)//3}"
        )

    def test_resname_composition(self, tmp_path):
        """Residue name keys must match; per-name counts may differ by ±3 molecules."""
        perl_out, py_out = self._run_both(tmp_path, "total: 10")

        def resname_counts(path):
            atoms, _, _ = read_bgf(path)
            counts = {}
            for a in atoms.values():
                rn = a.get("RESNAME", "")
                counts[rn] = counts.get(rn, 0) + 1
            return counts

        py_c = resname_counts(py_out)
        pl_c = resname_counts(perl_out)
        assert set(py_c.keys()) == set(pl_c.keys()), (
            f"residue name mismatch: python={set(py_c.keys())}, perl={set(pl_c.keys())}"
        )
        for rn in py_c:
            diff = abs(py_c[rn] - pl_c[rn])
            assert diff <= 9, (  # ≤3 molecules × 3 atoms each
                f"RESNAME {rn!r}: python={py_c[rn]}, perl={pl_c[rn]}"
            )

    def test_fftype_set(self, tmp_path):
        """The set of FF types in the output should match."""
        perl_out, py_out = self._run_both(tmp_path, "total: 10")

        def fftype_set(path):
            atoms, _, _ = read_bgf(path)
            return {a.get("FFTYPE", "") for a in atoms.values()}

        assert fftype_set(py_out) == fftype_set(perl_out)
