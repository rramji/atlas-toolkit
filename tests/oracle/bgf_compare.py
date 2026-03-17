"""BGF comparison utilities for oracle tests.

These functions compare two BGF structures and return structured diffs
rather than raising immediately, so tests can report exactly what differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from atlas_toolkit.io.bgf import read_bgf


@dataclass
class BgfDiff:
    """Accumulated differences between two BGF structures."""
    atom_count: tuple[int, int] | None = None       # (got, expected) if mismatch
    missing_atoms: list[int] = field(default_factory=list)
    extra_atoms: list[int] = field(default_factory=list)
    field_diffs: list[str] = field(default_factory=list)
    coord_diffs: list[str] = field(default_factory=list)
    bond_diffs: list[str] = field(default_factory=list)
    box_diff: str | None = None

    @property
    def ok(self) -> bool:
        return not any([
            self.atom_count, self.missing_atoms, self.extra_atoms,
            self.field_diffs, self.coord_diffs, self.bond_diffs, self.box_diff,
        ])

    def summary(self) -> str:
        lines = []
        if self.atom_count:
            lines.append(f"  atom count: got {self.atom_count[0]}, expected {self.atom_count[1]}")
        if self.missing_atoms:
            lines.append(f"  missing atom indices: {self.missing_atoms[:10]}")
        if self.extra_atoms:
            lines.append(f"  extra atom indices: {self.extra_atoms[:10]}")
        for d in self.field_diffs[:5]:
            lines.append(f"  field: {d}")
        for d in self.coord_diffs[:5]:
            lines.append(f"  coord: {d}")
        for d in self.bond_diffs[:5]:
            lines.append(f"  bond: {d}")
        if self.box_diff:
            lines.append(f"  box: {self.box_diff}")
        return "\n".join(lines) or "  (no differences)"


def compare_bgf(
    path_a: str | Path,
    path_b: str | Path,
    coord_tol: float = 0.001,
    check_coords: bool = True,
    check_bonds: bool = True,
    check_box: bool = True,
    fields: list[str] | None = None,
) -> BgfDiff:
    """Compare two BGF files and return a BgfDiff.

    Parameters
    ----------
    path_a, path_b : BGF files to compare (a=test output, b=reference/Perl)
    coord_tol      : absolute tolerance for coordinate comparison (Å)
    check_coords   : whether to compare XCOORD/YCOORD/ZCOORD
    check_bonds    : whether to compare bond topology
    check_box      : whether to compare CRYSTX box dimensions
    fields         : atom fields to compare (default: FFTYPE CHARGE RESNAME ATMNAME)
    """
    atoms_a, bonds_a, headers_a = read_bgf(path_a)
    atoms_b, bonds_b, headers_b = read_bgf(path_b)

    diff = BgfDiff()
    fields = fields or ["FFTYPE", "CHARGE", "RESNAME", "ATMNAME"]

    # --- Atom count ---
    if len(atoms_a) != len(atoms_b):
        diff.atom_count = (len(atoms_a), len(atoms_b))
        return diff  # further comparisons are meaningless

    # --- Per-atom fields + coords ---
    keys_a, keys_b = set(atoms_a), set(atoms_b)
    diff.missing_atoms = sorted(keys_b - keys_a)
    diff.extra_atoms   = sorted(keys_a - keys_b)

    for idx in sorted(keys_a & keys_b):
        a, b = atoms_a[idx], atoms_b[idx]

        for f in fields:
            va = _normalise(a.get(f))
            vb = _normalise(b.get(f))
            if f == "CHARGE":
                try:
                    if abs(float(va) - float(vb)) > 1e-5:
                        diff.field_diffs.append(
                            f"atom {idx} CHARGE: {va} vs {vb}"
                        )
                    continue
                except (TypeError, ValueError):
                    pass
            if va != vb:
                diff.field_diffs.append(f"atom {idx} {f}: {va!r} vs {vb!r}")

        if check_coords:
            for c in ("XCOORD", "YCOORD", "ZCOORD"):
                try:
                    if abs(float(a.get(c, 0)) - float(b.get(c, 0))) > coord_tol:
                        diff.coord_diffs.append(
                            f"atom {idx} {c}: {a.get(c)} vs {b.get(c)}"
                        )
                except (TypeError, ValueError):
                    pass

    # --- Bond topology ---
    if check_bonds:
        for idx in sorted(keys_a & keys_b):
            ba = sorted(bonds_a.get(idx, []))
            bb = sorted(bonds_b.get(idx, []))
            if ba != bb:
                diff.bond_diffs.append(f"atom {idx}: {ba} vs {bb}")

    # --- Box ---
    if check_box:
        crystx_a = _get_crystx(headers_a)
        crystx_b = _get_crystx(headers_b)
        if crystx_a is not None and crystx_b is not None:
            for i, (va, vb) in enumerate(zip(crystx_a, crystx_b)):
                if abs(va - vb) > 0.01:
                    diff.box_diff = f"CRYSTX[{i}]: {va:.4f} vs {vb:.4f}"
                    break
        elif crystx_a != crystx_b:
            diff.box_diff = f"CRYSTX present: {crystx_a is not None} vs {crystx_b is not None}"

    return diff


def assert_bgf_equal(path_a, path_b, msg="", **kwargs):
    """Assert two BGF files are equivalent; raise AssertionError with diff on failure."""
    diff = compare_bgf(path_a, path_b, **kwargs)
    if not diff.ok:
        prefix = f"{msg}\n" if msg else ""
        raise AssertionError(f"{prefix}BGF files differ:\n{diff.summary()}")


def compare_bgf_structural(path_a: str | Path, path_b: str | Path) -> BgfDiff:
    """Structural comparison only: atom count, bond topology, FF types.

    Ignores coordinates — suitable for stochastic scripts (add_solvent, add_ions)
    where placement differs but topology should match.
    """
    return compare_bgf(
        path_a, path_b,
        check_coords=False,
        check_bonds=True,
        check_box=False,
        fields=["FFTYPE", "RESNAME"],
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalise(val):
    if val is None:
        return None
    return str(val).strip()


def _get_crystx(headers: list) -> list[float] | None:
    for h in headers:
        if h.startswith("CRYSTX"):
            try:
                return [float(x) for x in h.split()[1:7]]
            except (ValueError, IndexError):
                pass
    return None
