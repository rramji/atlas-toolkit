"""
BGF file reader and writer — port of FileFormats.pm (GetBGFFileInfo + createBGF).

Format reference (FORMAT ATOM line):
  (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5,f10.5)

Actual printf used in createBGF:
  "%-6s %5d %-5s %3s %1s %5d%10.5f%10.5f%10.5f %-5s %2d %1d %9.6f"

Key fidelity notes:
  - CONECT lines are parsed in strict 6-char chunks (not split on whitespace).
  - HETATM label is stored in atom['LABEL'] and restored on write.
  - NUMBONDS on write is always taken from the live bonds list, capped at 9.
  - The HEADER key is never stuffed into the atoms dict (unlike Perl).
"""
import re
from pathlib import Path
from typing import Optional

from ..types import AtomsDict, BondsDict, HeadersList

__all__ = ["read_bgf", "write_bgf", "parse_struct_file"]

# ── compiled patterns ──────────────────────────────────────────────────────

# Header record types saved to the headers list
_HEADER_RE = re.compile(
    r"^(XTLGRF|BIOGRF|DESCRP|REMARK|FORCEFIELD|PERIOD|AXES|SGNAME|CELLS|CRYSTX)"
)

# Atom record regex (matches after HETATM→"ATOM  " substitution)
_ATM_RE = re.compile(
    r"^ATOM\s*(\d+)\s+"         # 1: index
    r"(\S+)\S?\s+"              # 2: atom name  (extra \S? eats trailing non-ws)
    r"(\S+)\s\w?\s*"            # 3: res name
    r"(\d+)\s*"                 # 4: res num
    r"(-?\d+\.\d{5})\s*"        # 5: x
    r"(-?\d+\.\d{5})\s*"        # 6: y
    r"(-?\d+\.\d{5})\s*"        # 7: z
    r"(\S+)\s+"                 # 8: ff type
    r"(\d+)\s+"                 # 9: numbonds
    r"(\d+)\s+"                 # 10: lone pairs
    r"(-?\d+\.\d+)"             # 11: charge
    r"(?:\s+(\d+\s+\d+\s+\d+\.\d+))?"  # 12: optional occ/res/radii
)

_CHAIN_RE = re.compile(r"^ATOM\s+\d+\s+\S+\s+\S+\s+(\w)\s+")
_CONECT_RE = re.compile(r"^CONECT(.+)$")
_ORDER_DISP_RE = re.compile(r"^(ORDER|DISP\S)\s+(.+)$")


# ── reader ─────────────────────────────────────────────────────────────────

def read_bgf(
    path: str | Path,
    save_headers: bool = True,
) -> tuple[AtomsDict, BondsDict, HeadersList]:
    """Parse a BGF structure file.

    Returns:
        atoms   : dict[int, dict]  — 1-based atom index → atom fields
        bonds   : dict[int, list]  — atom index → list of bonded atom indices
        headers : list[str]        — header lines (empty if save_headers=False)
    """
    atoms: AtomsDict = {}
    bonds: BondsDict = {}
    headers: HeadersList = []
    molid = 1
    num_hetatm = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            # Normalise HETATM → "ATOM  " for parsing, keep original for label
            is_hetatm = line.startswith("HETATM")
            norm = "ATOM  " + line[6:] if is_hetatm else line

            # ── header records ───────────────────────────────────────────
            if _HEADER_RE.match(norm):
                if save_headers:
                    headers.append(norm)
                continue

            # ── atom record ──────────────────────────────────────────────
            m = _ATM_RE.match(norm)
            if m:
                idx = int(m.group(1).strip())
                fftype = m.group(8)
                atom: dict = {
                    "INDEX":      idx,
                    "ATMNAME":    m.group(2),
                    "RESNAME":    m.group(3),
                    "RESNUM":     int(m.group(4)),
                    "XCOORD":     float(m.group(5)),
                    "YCOORD":     float(m.group(6)),
                    "ZCOORD":     float(m.group(7)),
                    "FFTYPE":     fftype,
                    "NUMBONDS":   int(m.group(9)),
                    "LONEPAIRS":  int(m.group(10)),
                    "CHARGE":     float(m.group(11)),
                    "MOLECULEID": molid,
                    "LABEL":      "HETATM" if is_hetatm else "ATOM",
                }

                # Optional occupancy / resonance / radii triplet
                if m.group(12):
                    parts = m.group(12).split()
                    if len(parts) >= 3:
                        atom["OCCUPANCY"] = int(parts[0])
                        atom["RESONANCE"] = int(parts[1])
                        atom["RADII"]     = float(parts[2])

                # Chain: look for it in the normalised line
                cm = _CHAIN_RE.match(norm)
                atom["CHAIN"] = cm.group(1) if cm else "A"

                if is_hetatm:
                    num_hetatm += 1

                atoms[idx] = atom
                bonds[idx] = []
                continue

            # ── CONECT record — strict 6-char chunks ─────────────────────
            mc = _CONECT_RE.match(norm)
            if mc:
                raw_con = mc.group(1)
                parts = [raw_con[i:i+6].strip()
                         for i in range(0, len(raw_con), 6)]
                parts = [p for p in parts if p]
                if not parts:
                    continue
                try:
                    src = int(parts[0])
                except ValueError:
                    continue
                if src not in atoms:
                    continue
                for tgt_s in parts[1:]:
                    try:
                        tgt = int(tgt_s)
                    except ValueError:
                        continue
                    if tgt in atoms:
                        bonds[src].append(tgt)
                        atoms[src]["DNB"] = 1
                        atoms[tgt]["DNB"] = 1
                atoms[src]["NUMBONDS"] = len(parts) - 1
                continue

            # ── ORDER / DISP records ──────────────────────────────────────
            mo = _ORDER_DISP_RE.match(norm)
            if mo:
                rec_type = mo.group(1)
                vals = mo.group(2).split()
                if not vals:
                    continue
                try:
                    atom_id = int(vals[0])
                except ValueError:
                    continue
                data_strs = vals[1:]
                if atom_id in atoms:
                    parsed = []
                    nb = 0
                    for v in data_strs:
                        try:
                            iv = int(v)
                            parsed.append(iv)
                            nb += iv
                        except ValueError:
                            parsed.append(float(v))
                    atoms[atom_id][rec_type] = parsed
                    if rec_type == "ORDER":
                        atoms[atom_id]["NUMBONDS"] = nb
                continue

            # ── molecule separator ────────────────────────────────────────
            if line.startswith(("TER", "ENDMDL")):
                molid += 1

    if not atoms:
        raise ValueError(f"{path} contains no ATOM/HETATM records")

    return atoms, bonds, headers


# ── writer ─────────────────────────────────────────────────────────────────

def write_bgf(
    atoms: AtomsDict,
    bonds: BondsDict,
    path: str | Path,
    headers: Optional[HeadersList] = None,
    save_radii: bool = False,
) -> None:
    """Write a BGF structure file (Perl createBGF).

    The output format matches the Perl printf exactly:
      "%-6s %5d %-5s %3s %1s %5d%10.5f%10.5f%10.5f %-5s %2d %1d %9.6f"
    """
    with open(path, "w", encoding="utf-8") as fh:
        # Write header lines
        if headers:
            for h in headers:
                fh.write(h + "\n")

        # FORMAT ATOM descriptor
        fh.write(
            "FORMAT ATOM   "
            "(a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5,f10.5)\n"
        )

        # Atom records in index order
        for idx in sorted(atoms):
            atom = atoms[idx]
            label = atom.get("LABEL", "ATOM")

            # Use FFORIG if present, strip any :: prefix qualifier
            fftype = atom.get("FFORIG") or atom.get("FFTYPE", "")
            if "::" in str(fftype):
                fftype = str(fftype).split("::")[-1]

            # NUMBONDS from live bond list (overrides stored field), capped at 9
            numbonds = len(bonds.get(idx, []))
            numbonds = min(numbonds, 9)

            lonepairs = atom.get("LONEPAIRS", 0)
            lonepairs = min(int(lonepairs), 9)

            # Apply coordinate offset if present
            offset = atom.get("OFFSET", {})
            x = float(atom["XCOORD"]) - float(offset.get("XCOORD", 0))
            y = float(atom["YCOORD"]) - float(offset.get("YCOORD", 0))
            z = float(atom["ZCOORD"]) - float(offset.get("ZCOORD", 0))

            chain = atom.get("CHAIN", "X")
            resnum = int(atom.get("RESNUM", 0))

            # Perl: "%-6s %5d %-5s %3s %1s %5d%10.5f%10.5f%10.5f %-5s %2d %1d %9.6f"
            line = (
                f"{label:<6s} {idx:5d} {atom['ATMNAME']:<5s} {atom['RESNAME']:>3s} "
                f"{chain:1s} {resnum:5d}"
                f"{x:10.5f}{y:10.5f}{z:10.5f} "
                f"{fftype:<5s} {numbonds:2d} {lonepairs:1d} {atom['CHARGE']:9.6f}"
            )

            if save_radii:
                resonance = int(atom.get("RESONANCE", 0))
                occupancy = int(atom.get("OCCUPANCY", 0))
                radii = float(atom.get("RADII", 0.0))
                line += f"{resonance:2d}{occupancy:4d}{radii:8.3f}"

            fh.write(line + "\n")

        # FORMAT CONECT descriptor
        fh.write("FORMAT CONECT (a6,12i6)\n")

        # CONECT records
        for atom_idx in sorted(atoms):
            con = bonds.get(atom_idx) or []
            atom = atoms[atom_idx]

            if not con:
                fh.write(f"{'CONECT':<6s}{atom_idx:6d}\n")
                continue

            sorted_bonds = sorted(con)

            # CONECT line: "%-6s%6d" for label+atom, then "%6d" per bond
            fh.write(f"{'CONECT':<6s}{atom_idx:6d}")
            for b in sorted_bonds:
                fh.write(f"{b:6d}")
            fh.write("\n")

            # Optional ORDER / DISP records
            for rec_type in ("ORDER", "DISPX", "DISPY", "DISPZ"):
                if rec_type not in atom:
                    continue
                vals = atom[rec_type]
                fh.write(f"{rec_type:<6s}{atom_idx:6d}")
                for i, _ in enumerate(sorted_bonds):
                    v = int(vals[i]) if i < len(vals) and vals[i] is not None else 0
                    fh.write(f"{v:6d}")
                fh.write("\n")

        fh.write("END\n")


# ── atom renumbering ────────────────────────────────────────────────────────

def get_bgf_atoms(
    atm_list: dict,
    atoms: AtomsDict,
    bonds: BondsDict,
) -> tuple[AtomsDict, BondsDict]:
    """Re-index a subset of atoms to sequential 1-based integers.

    Port of FileFormats.pm::GetBGFAtoms.

    Parameters
    ----------
    atm_list : dict whose keys are the original atom indices to include
    atoms    : full atom dict (may contain more atoms than atm_list)
    bonds    : full bond dict

    Returns
    -------
    (new_atoms, new_bonds) with sequential 1-based indices and fixed bonds.
    """
    new_atoms: AtomsDict = {}
    new_idx = 1
    old_to_new: dict[int, int] = {}

    for old_idx in sorted(atm_list):
        if old_idx not in atoms:
            continue
        new_atoms[new_idx] = dict(atoms[old_idx])
        new_atoms[new_idx]["INDEX"] = new_idx
        old_to_new[old_idx] = new_idx
        new_idx += 1

    new_bonds: BondsDict = {}
    for new_i, old_i in zip(sorted(new_atoms), sorted(old_to_new)):
        old_i = list(old_to_new.keys())[list(old_to_new.values()).index(new_i)]
        new_bonds[new_i] = [
            old_to_new[j] for j in bonds.get(old_i, []) if j in old_to_new
        ]

    return new_atoms, new_bonds


def make_seq_atom_index(
    atoms: AtomsDict,
    bonds: BondsDict,
) -> tuple[AtomsDict, BondsDict]:
    """Renumber atoms to sequential 1-based integers, preserving relative order.

    Port of General.pm::MakeSeqAtomIndex.  Handles sparse dicts (atoms with
    holes after deletions).
    """
    sorted_old = sorted(atoms)
    old_to_new = {old: new for new, old in enumerate(sorted_old, start=1)}

    new_atoms: AtomsDict = {}
    new_bonds: BondsDict = {}
    for old_idx in sorted_old:
        new_idx = old_to_new[old_idx]
        new_atoms[new_idx] = dict(atoms[old_idx])
        new_atoms[new_idx]["INDEX"] = new_idx
        new_bonds[new_idx] = [
            old_to_new[j] for j in bonds.get(old_idx, []) if j in old_to_new
        ]

    return new_atoms, new_bonds


# ── dispatcher ─────────────────────────────────────────────────────────────

def parse_struct_file(
    path: str | Path,
    save_headers: bool = True,
) -> tuple[AtomsDict, BondsDict, HeadersList]:
    """Dispatch to the correct reader based on file extension (Perl ParseStructFile, BGF only).

    Additional formats (PDB, MOL2, AMBER, CHARMM …) will be added in later milestones.
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")

    if ext == "bgf":
        return read_bgf(p, save_headers=save_headers)

    raise NotImplementedError(
        f"File type '{ext}' is not yet supported. "
        "Currently only BGF files are handled."
    )
