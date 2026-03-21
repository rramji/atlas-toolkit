"""
BGF ↔ ParmEd Structure bridge.

Converts between atlas-toolkit's native BGF representation
(atoms dict, bonds dict, headers list) and a ParmEd Structure.

The bridge is intentionally *topology-only*: it carries coordinates,
connectivity, charges, and FF type labels.  Force-field parameters
(LJ ε/σ, bond/angle/torsion constants) are NOT injected here — that
is the job of the downstream parameterisation step.  This keeps the
bridge fast, predictable, and format-agnostic.

Usage
-----
    from atlas_toolkit.io.bgf import read_bgf
    from atlas_toolkit.io.bgf_parmed import bgf_to_parmed, parmed_to_bgf

    atoms, bonds, headers = read_bgf("system.bgf")
    struct = bgf_to_parmed(atoms, bonds)

    # ... modify struct with parmed ...

    atoms2, bonds2 = parmed_to_bgf(struct)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import parmed as pmd
from parmed.periodic_table import AtomicNum, Element, Mass

from .bgf import read_bgf, write_bgf
from ..types import AtomsDict, BondsDict, HeadersList

__all__ = [
    "bgf_to_parmed",
    "parmed_to_bgf",
    "load_bgf_as_parmed",
    "save_parmed_as_bgf",
]

# ── element inference ──────────────────────────────────────────────────────

# Common BGF/OPLS/GAFF/AMBER/Heinz FF type prefixes → element symbol.
# The lookup is tried in order; the first regex match wins.
# Falls back to stripping digits then looking in the periodic table.
_FF_TYPE_ELEMENT_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Au",  re.I), "Au"),
    (re.compile(r"^Ag",  re.I), "Ag"),
    (re.compile(r"^Na",  re.I), "Na"),
    (re.compile(r"^Cl",  re.I), "Cl"),
    (re.compile(r"^Ca",  re.I), "Ca"),
    (re.compile(r"^Mg",  re.I), "Mg"),
    (re.compile(r"^Fe",  re.I), "Fe"),
    (re.compile(r"^Zn",  re.I), "Zn"),
    (re.compile(r"^Cu",  re.I), "Cu"),
    (re.compile(r"^Br",  re.I), "Br"),
    (re.compile(r"^Si",  re.I), "Si"),
    (re.compile(r"^LP",  re.I), "LP"),  # lone pair — no atomic number
    (re.compile(r"^C",   re.I), "C"),
    (re.compile(r"^N",   re.I), "N"),
    (re.compile(r"^O",   re.I), "O"),
    (re.compile(r"^S",   re.I), "S"),
    (re.compile(r"^P",   re.I), "P"),
    (re.compile(r"^F",   re.I), "F"),
    (re.compile(r"^H",   re.I), "H"),
]

# Symbols that parmed's periodic table knows (lowercase key → symbol)
_ELEMENT_SET: set[str] = {Element[i].lower() for i in range(1, 119)
                           if Element[i]}


def _element_from_fftype(fftype: str) -> str:
    """Guess element symbol from a FF type string.

    Returns a capitalised symbol ('C', 'Au', …) or '' if unknown.
    """
    for pattern, sym in _FF_TYPE_ELEMENT_MAP:
        if pattern.match(fftype):
            return sym
    # strip trailing digits and punctuation, try periodic table
    stripped = re.sub(r"[\d_\-\+\*]+$", "", fftype).capitalize()
    if stripped.lower() in _ELEMENT_SET:
        return stripped
    return ""


def _atomic_num(sym: str) -> int:
    return AtomicNum.get(sym, 0)


def _mass_from_sym(sym: str) -> float:
    return Mass.get(sym, 0.0)


# ── BGF → ParmEd ──────────────────────────────────────────────────────────

def bgf_to_parmed(
    atoms: AtomsDict,
    bonds: BondsDict,
    *,
    default_resname: str = "MOL",
) -> pmd.Structure:
    """Convert a BGF atoms/bonds dict to a ParmEd Structure.

    Parameters
    ----------
    atoms          : dict from read_bgf (1-based index → atom fields)
    bonds          : dict from read_bgf (atom index → list of bonded indices)
    default_resname: residue name used when RESNAME is missing/blank

    Returns
    -------
    pmd.Structure  : topology + coordinates; no FF parameters loaded
    """
    struct = pmd.Structure()

    # ── pass 1: create atoms, grouped by residue ──────────────────────────
    # We need to build Residue objects first because ParmEd requires
    # them to exist before atoms are added.
    residues: dict[tuple[int, str], pmd.Residue] = {}
    pmd_atoms: dict[int, pmd.Atom] = {}   # bgf index → pmd.Atom

    sorted_indices = sorted(atoms.keys())

    for idx in sorted_indices:
        a = atoms[idx]

        resnum  = a.get("RESNUM",  1)
        resname = (a.get("RESNAME") or default_resname).strip() or default_resname
        chain   = a.get("CHAIN", "A") or "A"
        res_key = (resnum, resname, chain)

        if res_key not in residues:
            res = pmd.Residue(resname, number=resnum, chain=chain)
            residues[res_key] = res
            struct.residues.append(res)
        else:
            res = residues[res_key]

        fftype  = a.get("FFTYPE", "")
        element = _element_from_fftype(fftype)
        anum    = _atomic_num(element) if element else 0
        mass    = _mass_from_sym(element) if element else 0.0

        atom = pmd.Atom(
            name    = (a.get("ATMNAME") or f"A{idx}").strip(),
            type    = fftype,
            charge  = a.get("CHARGE", 0.0),
            mass    = mass,
            atomic_number = anum,
        )
        atom.xx = a.get("XCOORD", 0.0)
        atom.xy = a.get("YCOORD", 0.0)
        atom.xz = a.get("ZCOORD", 0.0)

        # Store original BGF index as an extra attribute for round-trip fidelity
        atom.bgf_index = idx

        res.add_atom(atom)
        struct.atoms.append(atom)
        pmd_atoms[idx] = atom

    # ── pass 2: bonds (undirected, deduplicated) ──────────────────────────
    seen: set[frozenset] = set()
    for src_idx, neighbours in bonds.items():
        if src_idx not in pmd_atoms:
            continue
        for tgt_idx in neighbours:
            if tgt_idx not in pmd_atoms:
                continue
            key = frozenset((src_idx, tgt_idx))
            if key in seen:
                continue
            seen.add(key)
            struct.bonds.append(
                pmd.Bond(pmd_atoms[src_idx], pmd_atoms[tgt_idx])
            )

    return struct


# ── ParmEd → BGF ──────────────────────────────────────────────────────────

def parmed_to_bgf(
    struct: pmd.Structure,
    *,
    default_resname: str = "MOL",
    default_label:   str = "HETATM",
) -> tuple[AtomsDict, BondsDict]:
    """Convert a ParmEd Structure back to BGF atoms/bonds dicts.

    Parameters
    ----------
    struct         : ParmEd Structure (must have coordinates set)
    default_resname: fallback residue name
    default_label  : ATOM or HETATM for records without residue info

    Returns
    -------
    (atoms, bonds) : same format as read_bgf() — ready for write_bgf()
    """
    atoms: AtomsDict = {}
    bonds: BondsDict = {}

    for i, atom in enumerate(struct.atoms, start=1):
        res     = atom.residue
        resname = (res.name if res else default_resname) or default_resname
        resnum  = res.number if res else 1
        chain   = (res.chain if res and res.chain else "A") or "A"

        atoms[i] = {
            "INDEX":     i,
            "ATMNAME":   atom.name or f"A{i}",
            "RESNAME":   resname,
            "RESNUM":    resnum,
            "XCOORD":    getattr(atom, "xx", 0.0),
            "YCOORD":    getattr(atom, "xy", 0.0),
            "ZCOORD":    getattr(atom, "xz", 0.0),
            "FFTYPE":    atom.type or atom.name or "DUM",
            "NUMBONDS":  len(atom.bond_partners),
            "LONEPAIRS": 0,
            "CHARGE":    atom.charge or 0.0,
            "MOLECULEID": 1,
            "CHAIN":     chain,
            "LABEL":     default_label,
        }
        bonds[i] = []

    # Build a parmed-atom → new-1-based-index map
    atom_map: dict[int, int] = {id(a): i for i, a in enumerate(struct.atoms, 1)}

    for bond in struct.bonds:
        src = atom_map.get(id(bond.atom1))
        tgt = atom_map.get(id(bond.atom2))
        if src and tgt:
            if tgt not in bonds[src]:
                bonds[src].append(tgt)
            if src not in bonds[tgt]:
                bonds[tgt].append(src)

    # Update NUMBONDS from the live bond lists
    for idx in atoms:
        atoms[idx]["NUMBONDS"] = len(bonds[idx])

    return atoms, bonds


# ── convenience I/O ───────────────────────────────────────────────────────

def load_bgf_as_parmed(
    path: str | Path,
    **kwargs,
) -> tuple[pmd.Structure, HeadersList]:
    """Read a BGF file and return a (ParmEd Structure, headers) tuple.

    The headers list lets you reconstruct the BGF preamble (BIOGRF,
    CRYSTX, FORCEFIELD …) when writing back to BGF.
    """
    atoms, bonds, headers = read_bgf(path)
    struct = bgf_to_parmed(atoms, bonds, **kwargs)
    return struct, headers


def save_parmed_as_bgf(
    struct: pmd.Structure,
    path: str | Path,
    headers: Optional[HeadersList] = None,
    **kwargs,
) -> None:
    """Write a ParmEd Structure to a BGF file.

    Parameters
    ----------
    struct  : ParmEd Structure
    path    : output file path
    headers : optional header lines from the original BGF (BIOGRF, CRYSTX …)
    """
    atoms, bonds = parmed_to_bgf(struct, **kwargs)
    write_bgf(atoms, bonds, path, headers=headers)
