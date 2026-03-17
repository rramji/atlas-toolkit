"""
General utility functions — port of General.pm (subset needed for modifyAtomData).
"""
import os
import random
from pathlib import Path
from typing import Any

from ..types import AtomsDict

__all__ = ["trim", "file_tester", "has_cell", "shuffle_array", "com"]


def trim(s: str) -> str:
    """Strip leading/trailing whitespace (Perl Trim)."""
    return s.strip()


def file_tester(path: str | Path) -> None:
    """Die if path does not exist or is not readable (Perl FileTester).

    Raises:
        FileNotFoundError: if the file does not exist.
        PermissionError:   if the file is not readable.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ERROR: File {path} does not exist!")
    if not os.access(p, os.R_OK):
        raise PermissionError(f"ERROR: File {path} is not readable!")


def has_cell(headers: list[str]) -> bool:
    """Return True if the header list contains a CRYSTX record (Perl HasCell)."""
    return any(h.startswith("CRYSTX") for h in headers if h)


def shuffle_array(lst: list) -> None:
    """Shuffle a list in place (Perl ShuffleArray)."""
    random.shuffle(lst)


def com(atoms: AtomsDict, box: dict | None = None) -> dict[str, float]:
    """Compute centre of mass (Perl CoM).

    For a single-atom dict this returns a *copy* of the coordinate values,
    not a reference to the atom.  This is an intentional fix over the Perl
    version which returned the atom hash directly and caused aliasing bugs
    (see rotateMols / single-ion placement issue).

    Args:
        atoms: dict of atom dicts, each with XCOORD/YCOORD/ZCOORD and
               optionally MASS.
        box:   optional periodic-box dict (not used for simple COM).

    Returns:
        {'XCOORD': cx, 'YCOORD': cy, 'ZCOORD': cz}
    """
    keys = list(atoms)
    if len(keys) == 1:
        a = atoms[keys[0]]
        return {"XCOORD": float(a["XCOORD"]),
                "YCOORD": float(a["YCOORD"]),
                "ZCOORD": float(a["ZCOORD"])}

    total_mass = 0.0
    cx = cy = cz = 0.0
    for a in atoms.values():
        mass = float(a.get("MASS", 1.0))
        cx += mass * float(a["XCOORD"])
        cy += mass * float(a["YCOORD"])
        cz += mass * float(a["ZCOORD"])
        total_mass += mass

    if total_mass == 0:
        total_mass = 1.0
    return {"XCOORD": cx / total_mass,
            "YCOORD": cy / total_mass,
            "ZCOORD": cz / total_mass}
