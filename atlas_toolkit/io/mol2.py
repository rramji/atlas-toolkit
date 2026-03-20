"""
MOL2 (Tripos SYBYL) reader and writer.

write_mol2(atoms, bonds, path, ...)  — BGF atom/bond dicts → .mol2
read_mol2(path)                      — .mol2 → BGF atom/bond dicts
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from atlas_toolkit.types import AtomsDict, BondsDict

# Map integer bond order → Tripos bond type string
_ORDER_TO_TRIPOS: dict[int, str] = {
    1: "1",
    2: "2",
    3: "3",
    4: "ar",   # aromatic (used when order stored as 4)
}

# Default Sybyl type when FFTYPE has no mapping
_DEFAULT_SYBYL = "X"


def _sybyl_type(atom: dict) -> str:
    """Best-effort FFTYPE → Sybyl atom type.

    For most MD force fields the FFTYPE is not a Sybyl type, so we just
    return it as-is; parmed/openbabel will re-assign types if needed.
    """
    return str(atom.get("FFTYPE") or atom.get("ATMNAME") or _DEFAULT_SYBYL)


def write_mol2(
    atoms: AtomsDict,
    bonds: BondsDict,
    path: str | Path,
    title: str = "",
    mol_type: str = "SMALL",
    charge_type: str = "USER_CHARGES",
) -> None:
    """Write a Tripos MOL2 file from BGF atom/bond dicts.

    Bond orders are taken from atoms[idx]["ORDER"] (aligned with
    sorted(bonds[idx])), defaulting to 1 for any missing entry.

    Parameters
    ----------
    atoms       : BGF atoms dict
    bonds       : BGF bonds dict (adjacency list)
    path        : output file path
    title       : molecule name written to @<TRIPOS>MOLECULE
    mol_type    : SMALL | BIOPOLYMER | PROTEIN | NUCLEIC_ACID | SACCHARIDE
    charge_type : charge type string (default USER_CHARGES)
    """
    path = Path(path)
    title = title or path.stem

    # Collect unique (residue_num, residue_name) pairs in order
    substruct_map: dict[int, str] = {}
    for a in atoms.values():
        rn = int(a.get("RESNUM", 1))
        substruct_map.setdefault(rn, str(a.get("RESNAME", "RES")))

    # Build bond list: (bond_id, atom1, atom2, type_str)
    bond_records: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    bid = 1
    for src in sorted(atoms):
        neighbors = sorted(bonds.get(src, []))
        orders = atoms[src].get("ORDER", [])
        for i, dst in enumerate(neighbors):
            pair = (min(src, dst), max(src, dst))
            if pair in seen:
                continue
            seen.add(pair)
            order = int(orders[i]) if i < len(orders) else 1
            btype = _ORDER_TO_TRIPOS.get(order, str(order))
            bond_records.append((bid, src, dst, btype))
            bid += 1

    n_atoms = len(atoms)
    n_bonds = len(bond_records)
    n_subst = len(substruct_map)

    with open(path, "w") as fh:
        # ── MOLECULE ──────────────────────────────────────────────────────
        fh.write("@<TRIPOS>MOLECULE\n")
        fh.write(f"{title}\n")
        fh.write(f"{n_atoms:5d} {n_bonds:5d} {n_subst:5d}     0     0\n")
        fh.write(f"{mol_type}\n")
        fh.write(f"{charge_type}\n")
        fh.write("\n")

        # ── ATOM ──────────────────────────────────────────────────────────
        fh.write("@<TRIPOS>ATOM\n")
        for idx in sorted(atoms):
            a = atoms[idx]
            name  = str(a.get("ATMNAME", f"X{idx}"))
            x     = float(a.get("XCOORD", 0.0))
            y     = float(a.get("YCOORD", 0.0))
            z     = float(a.get("ZCOORD", 0.0))
            stype = _sybyl_type(a)
            resnum = int(a.get("RESNUM", 1))
            resname = str(a.get("RESNAME", "RES"))
            charge = float(a.get("CHARGE", 0.0))
            fh.write(
                f"{idx:7d} {name:<8s} {x:10.4f} {y:10.4f} {z:10.4f}"
                f" {stype:<8s} {resnum:4d}  {resname:<8s} {charge:10.6f}\n"
            )

        # ── BOND ──────────────────────────────────────────────────────────
        fh.write("@<TRIPOS>BOND\n")
        for bid, a1, a2, btype in bond_records:
            fh.write(f"{bid:6d} {a1:5d} {a2:5d} {btype}\n")

        # ── SUBSTRUCTURE ──────────────────────────────────────────────────
        fh.write("@<TRIPOS>SUBSTRUCTURE\n")
        # Find first atom index for each residue (root atom)
        res_root: dict[int, int] = {}
        for idx in sorted(atoms):
            rn = int(atoms[idx].get("RESNUM", 1))
            res_root.setdefault(rn, idx)

        for rn in sorted(substruct_map):
            rname = substruct_map[rn]
            root  = res_root.get(rn, 1)
            chain = str(atoms[root].get("CHAIN", "A"))
            fh.write(
                f"{rn:6d}  {rname:<8s} {root:6d} TEMP              "
                f"0 {chain}     1 ROOT\n"
            )


def read_mol2(path: str | Path) -> tuple[AtomsDict, BondsDict, list]:
    """Parse a Tripos MOL2 file into BGF atom/bond dicts.

    Returns (atoms, bonds, headers) where headers is an empty list
    (MOL2 has no BGF-style periodic box headers).
    """
    path = Path(path)
    atoms: AtomsDict = {}
    bonds: BondsDict = {}
    section = ""

    # Temporary bond list; we'll build adjacency after
    raw_bonds: list[tuple[int, int, int]] = []  # (a1, a2, order_int)

    _TRIPOS_ORDER = {"1": 1, "2": 2, "3": 3, "ar": 4, "am": 1, "du": 1,
                     "un": 1, "nc": 0}

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("@<TRIPOS>"):
                section = line.strip()
                continue
            if not line.strip() or line.startswith("#"):
                continue

            if section == "@<TRIPOS>ATOM":
                parts = line.split()
                if len(parts) < 6:
                    continue
                idx      = int(parts[0])
                name     = parts[1]
                x, y, z  = float(parts[2]), float(parts[3]), float(parts[4])
                stype    = parts[5]
                resnum   = int(parts[6]) if len(parts) > 6 else 1
                resname  = parts[7] if len(parts) > 7 else "RES"
                charge   = float(parts[8]) if len(parts) > 8 else 0.0
                atoms[idx] = {
                    "INDEX":   idx,
                    "ATMNAME": name,
                    "FFTYPE":  stype,
                    "XCOORD":  x,
                    "YCOORD":  y,
                    "ZCOORD":  z,
                    "RESNUM":  resnum,
                    "RESNAME": resname,
                    "CHARGE":  charge,
                    "CHAIN":   "A",
                    "LONEPAIRS": 0,
                    "NUMBONDS":  0,
                }

            elif section == "@<TRIPOS>BOND":
                parts = line.split()
                if len(parts) < 4:
                    continue
                a1 = int(parts[1])
                a2 = int(parts[2])
                order_str = parts[3].lower()
                order = _TRIPOS_ORDER.get(order_str, 1)
                raw_bonds.append((a1, a2, order))

    # Build adjacency and store ORDER on each atom
    for a1, a2, order in raw_bonds:
        bonds.setdefault(a1, [])
        bonds.setdefault(a2, [])
        bonds[a1].append(a2)
        bonds[a2].append(a1)

    # Populate ORDER lists (aligned with sorted(bonds[idx]))
    # Build per-atom ordered list of (neighbor, order)
    bond_order_map: dict[tuple[int, int], int] = {}
    for a1, a2, order in raw_bonds:
        bond_order_map[(min(a1, a2), max(a1, a2))] = order

    for idx in bonds:
        nbrs = sorted(bonds[idx])
        bonds[idx] = nbrs
        orders = [bond_order_map.get((min(idx, n), max(idx, n)), 1) for n in nbrs]
        if idx in atoms:
            atoms[idx]["ORDER"] = orders
            atoms[idx]["NUMBONDS"] = len(nbrs)

    return atoms, bonds, []
