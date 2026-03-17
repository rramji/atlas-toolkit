"""
Type aliases and TypedDict definitions for the ATLAS toolkit data model.

The core data model mirrors the Perl hash-of-hashes:
  atoms  : dict[int, dict]  — atom index (1-based) → atom fields
  bonds  : dict[int, list]  — atom index → list of bonded atom indices
  headers: list[str]        — BGF/structure file header lines
"""
from typing import Any
from typing import TypedDict

__all__ = ["AtomRecord", "AtomsDict", "BondsDict", "HeadersList", "MolRecord", "MolsDict"]


class AtomRecord(TypedDict, total=False):
    """All possible fields on an atom dict.  total=False because most fields are optional
    depending on which operations have been run."""
    INDEX: int
    ATMNAME: str
    RESNAME: str
    CHAIN: str
    RESNUM: int
    XCOORD: float
    YCOORD: float
    ZCOORD: float
    FFTYPE: str
    FFORIG: str
    NUMBONDS: int
    LONEPAIRS: int
    CHARGE: float
    LABEL: str          # "ATOM" or "HETATM"
    MOLECULEID: int
    MOLSIZE: int
    MOLECULE: dict      # {INDEX, MEMBERS, MOLSIZE}
    RADII: float
    OCCUPANCY: int
    RESONANCE: int
    FA: float           # fractional coord a
    FB: float           # fractional coord b
    FC: float           # fractional coord c
    DNB: int            # do-not-bond flag
    ORDER: list         # bond orders
    DISPX: list         # displacement X per bond
    DISPY: list
    DISPZ: list
    OFFSET: dict        # coordinate offset {XCOORD, YCOORD, ZCOORD}


class MolRecord(TypedDict, total=False):
    INDEX: int
    MEMBERS: dict       # {atom_idx: 1, ...}
    MOLSIZE: int


# Convenient aliases used throughout the codebase
AtomsDict = dict[int, dict[str, Any]]
BondsDict = dict[int, list[int]]
HeadersList = list[str]
MolsDict = dict[int, MolRecord]
