"""
Force field file reader — Cerius2 .ff format.

Port of CERIUS2.pm::parseCerius2FF (ATOMTYPES, VDW, BONDS, ANGLES, TORSIONS,
INVERSIONS sections).

Public API
----------
read_ff(path, alter=True) -> dict
    Parse a Cerius2 .ff file and return a parms dict with keys:
        ATOMTYPES  : {fftype -> {MASS, TYPEID, NUMBONDS, LONEPAIRS, CHARGE, LABEL}}
        VDW        : {t1 -> {t2 -> {1 -> {TYPE, VALS}}}}
                     For LJ_6_12 with alter=True:
                         VALS[0] = epsilon (kcal/mol)
                         VALS[1] = sigma_LJ (Å)  [= r_min / 2^(1/6)]
        BONDS      : {(t1, t2) -> {TYPE, VALS: [k, r0]}}
        ANGLES     : {(t1, t2, t3) -> {TYPE, VALS: [k, theta0]}}
        TORSIONS   : {(t1, t2, t3, t4) -> [{TYPE, VALS: [k, n, phi0]}, ...]}
        INVERSIONS : {(t1, t2, t3, t4) -> [{TYPE, VALS: [k, phi0, n]}, ...]}
        PARMS      : {cut_vdw, cut_coul, mix_rule, ...}

load_ff(specs, alter=True) -> dict
    Load and merge one or more .ff / .frcmod files (space-separated string or list).

find_ff(name) -> Path | None
    Locate a bundled .ff file by bare name (e.g. "AMBER99", "tip3p").

get_vdw_radius(fftype, parms) -> float | None
    Return the VDW sigma/2 for a given atom type, or None if not found.

Canonical key helpers (importable)
    bond_key(t1, t2)             -> (str, str)
    angle_key(t1, t2, t3)        -> (str, str, str)
    torsion_key(t1, t2, t3, t4)  -> (str, str, str, str)
    inversion_key(t1, t2, t3, t4)-> (str, str, str, str)

Lookup helpers (handle wildcard 'X')
    lookup_bond(t1, t2, parms)              -> dict | None
    lookup_angle(t1, t2, t3, parms)         -> dict | None
    lookup_torsion(t1, t2, t3, t4, parms)   -> list | None
    lookup_inversion(t1, t2, t3, t4, parms) -> list | None
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Bundled FF directory
_FF_DIR = Path(__file__).parent.parent / "data" / "ff"

_TWO_16 = 2.0 ** (1.0 / 6.0)


# ── canonical key helpers ─────────────────────────────────────────────────────

def bond_key(t1: str, t2: str) -> tuple[str, str]:
    """Sorted canonical bond key."""
    a, b = t1.strip(), t2.strip()
    return (a, b) if a <= b else (b, a)


def angle_key(t1: str, t2: str, t3: str) -> tuple[str, str, str]:
    """Canonical angle key: sorted endpoints, fixed center."""
    a, c = t1.strip(), t3.strip()
    return (min(a, c), t2.strip(), max(a, c))


def torsion_key(t1: str, t2: str, t3: str, t4: str) -> tuple[str, str, str, str]:
    """Canonical torsion key: lexicographically smallest direction."""
    fwd = (t1.strip(), t2.strip(), t3.strip(), t4.strip())
    rev = (t4.strip(), t3.strip(), t2.strip(), t1.strip())
    return fwd if fwd <= rev else rev


def inversion_key(t1: str, t2: str, t3: str, t4: str) -> tuple[str, str, str, str]:
    """Canonical inversion key: (central_atom, *sorted_satellites)."""
    center = t1.strip()
    sats = sorted([t2.strip(), t3.strip(), t4.strip()])
    return (center, sats[0], sats[1], sats[2])


# ── wildcard lookup helpers ───────────────────────────────────────────────────

def lookup_bond(t1: str, t2: str, parms: dict) -> Optional[dict]:
    """Return bond parms for (t1, t2), or None."""
    bonds = parms.get("BONDS", {})
    return bonds.get(bond_key(t1, t2))


def lookup_angle(t1: str, t2: str, t3: str, parms: dict) -> Optional[dict]:
    """Return angle parms for (t1, t2, t3), or None."""
    angles = parms.get("ANGLES", {})
    return angles.get(angle_key(t1, t2, t3))


def lookup_torsion(t1: str, t2: str, t3: str, t4: str, parms: dict) -> Optional[list]:
    """Return torsion term list for (t1,t2,t3,t4), with wildcard 'X' fallback."""
    tors = parms.get("TORSIONS", {})
    key = torsion_key(t1, t2, t3, t4)
    if key in tors:
        return tors[key]
    # Try wildcards: X at one or both ends (keeping inner types fixed)
    for k1, k4 in [(t1, "X"), ("X", t4), ("X", "X")]:
        wkey = torsion_key(k1, t2, t3, k4)
        if wkey in tors:
            return tors[wkey]
    return None


def lookup_inversion(t1: str, t2: str, t3: str, t4: str, parms: dict) -> Optional[list]:
    """Return inversion term list, with wildcard 'X' fallback.

    t1 is the central atom (IT_JIKL convention).
    """
    invs = parms.get("INVERSIONS", {})
    key = inversion_key(t1, t2, t3, t4)
    if key in invs:
        return invs[key]
    # Try replacing satellites with X progressively
    satellites = [t2.strip(), t3.strip(), t4.strip()]
    for n_wildcards in range(1, 4):
        import itertools
        for combo in itertools.combinations(range(3), n_wildcards):
            trial = list(satellites)
            for idx in combo:
                trial[idx] = "X"
            wkey = inversion_key(t1, trial[0], trial[1], trial[2])
            if wkey in invs:
                return invs[wkey]
    return None


# ── public API ────────────────────────────────────────────────────────────────

def find_ff(name: str) -> Optional[Path]:
    """Return Path to a bundled FF file matching *name* (case-insensitive).

    Searches the bundled ff/ tree for .ff and .frcmod files.
    Returns None if not found.
    """
    candidate = Path(name)
    if candidate.exists():
        return candidate

    name_lower = name.lower()
    # strip trailing extension for stem comparison
    name_stem = name_lower
    for ext in (".ff", ".frcmod"):
        if name_stem.endswith(ext):
            name_stem = name_stem[: -len(ext)]

    for p in _FF_DIR.rglob("*"):
        if not p.is_file():
            continue
        # Accept .ff, .frcmod, files whose name starts with "frcmod.", or no extension
        pname_lower = p.name.lower()
        if not (p.suffix.lower() in (".ff", ".frcmod")
                or pname_lower.startswith("frcmod.")
                or "." not in p.name):
            continue
        if p.stem.lower() == name_stem or pname_lower == name_lower:
            return p
    return None


def read_ff(path: str | Path, alter: bool = True) -> dict:
    """Parse a Cerius2 .ff file.

    Parameters
    ----------
    path  : path to the .ff file
    alter : if True, apply Perl's alter transformations (swap epsilon/sigma
            and convert sigma from r_min to LJ convention for LJ_6_12)

    Returns
    -------
    dict with keys ATOMTYPES, VDW, BONDS, ANGLES, TORSIONS, INVERSIONS, PARMS
    """
    path = Path(path)
    parms: dict = {
        "PARMS": {
            "cut_vdw": 14.0,
            "cut_coul": 15.0,
            "coul_accuracy": 0.00001,
            "mix_rule": "geometric",
        },
        "ATOMTYPES": {},
        "VDW": {},
        "BONDS": {},
        "ANGLES": {},
        "TORSIONS": {},
        "INVERSIONS": {},
    }

    # section codes
    # 0=ignore, 1=ATOMTYPES, 3=DIAG_VDW, 8=OFF_DIAG_VDW
    # 4=BOND_STRETCH, 5=ANGLE_BEND, 6=TORSIONS, 7=INVERSIONS, 99=skip
    section = 0
    type_counter = 0
    _current_torsion_key: Optional[tuple] = None  # for continuation lines

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            # Strip comments for most sections
            data = line
            if section not in (3, 8):
                data = re.sub(r"#.*$", "", line)
            stripped = data.strip()

            if not stripped:
                continue

            # ── Section transitions ──────────────────────────────────────────
            if stripped.startswith("END"):
                section = 0
                _current_torsion_key = None
                continue
            if stripped.startswith("ATOMTYPES"):
                section = 1; continue
            if stripped.startswith("DIAGONAL_VDW"):
                section = 3; continue
            if stripped.startswith("OFF_DIAGONAL_VDW"):
                section = 8; continue
            if stripped.startswith("BOND_STRETCH"):
                section = 4; continue
            if stripped.startswith("ANGLE_BEND"):
                section = 5; continue
            if stripped.startswith("TORSIONS"):
                section = 6; continue
            if stripped.startswith("INVERSIONS"):
                section = 7; continue
            if re.match(r"^(PREFERENCES|HYDROGEN_BONDS|ATOM_TYPING_RULES|"
                        r"EQUIVALENCE|GENERATOR|QEq|COULOMBIC|CROSS_TERMS|"
                        r"STRETCH_STRETCH|STRETCH_BEND|BEND_BEND|"
                        r"TORSION_STRETCH|UREY_BRADLEY)", stripped):
                section = 99; continue

            # ── PREFERENCES inline keys ──────────────────────────────────────
            if section in (0, 99):
                m = re.match(r"^\s*VDW_SPLINE_OFF\s+([\d.]+)", stripped)
                if m:
                    parms["PARMS"]["cut_vdw"] = float(m.group(1)); continue
                m = re.match(r"^\s*COU_SPLINE_OFF\s+([\d.]+)", stripped)
                if m:
                    parms["PARMS"]["cut_coul"] = float(m.group(1)); continue
                m = re.match(r"^\s*VDW_COMBINATION_RULE\s+(\w+)", stripped)
                if m:
                    parms["PARMS"]["mix_rule"] = m.group(1).lower(); continue
                m = re.match(r"^\s*COU_1-4_SCALE_FACTOR\s+([\d.]+)", stripped)
                if m:
                    parms["PARMS"]["coul_14_scale"] = float(m.group(1)); continue
                m = re.match(r"^\s*VDW_1-4_SCALE_FACTOR\s+([\d.]+)", stripped)
                if m:
                    parms["PARMS"]["vdw_14_scale"] = float(m.group(1)); continue
                m = re.match(r"^\s*EWALD_SUM_COU_ACCURACY\s+([\d.eE+\-]+)", stripped)
                if m:
                    parms["PARMS"]["coul_accuracy"] = float(m.group(1)); continue

            # ── ATOMTYPES ────────────────────────────────────────────────────
            if section == 1:
                m = re.match(
                    r"^\s*(\S+)\s+(\w+)\s+([\d.eE+\-]+)\s+([\d.\-]+)\s+(\d+)\s+(\d+)\s+(\d+)",
                    stripped,
                )
                if m:
                    type_counter += 1
                    label = m.group(1)
                    parms["ATOMTYPES"][label] = {
                        "TYPEID":    type_counter,
                        "ATOM":      m.group(2),
                        "MASS":      float(m.group(3)),
                        "CHARGE":    float(m.group(4)),
                        "NUMBONDS":  int(m.group(5)),
                        "OTHER":     int(m.group(6)),
                        "LONEPAIRS": int(m.group(7)),
                        "LABEL":     label,
                        "USED":      0,
                    }

            # ── DIAGONAL_VDW / OFF_DIAGONAL_VDW ─────────────────────────────
            elif section in (3, 8):
                data_clean = re.sub(r"#.*$", "", stripped).strip()
                if not data_clean:
                    continue
                parts = data_clean.split()
                if len(parts) < 3:
                    continue

                if section == 3:
                    atom1, parm_type, vals_raw = parts[0], parts[1], parts[2:]
                    atom2 = atom1
                else:
                    if len(parts) < 4:
                        continue
                    atom1, atom2, parm_type, vals_raw = parts[0], parts[1], parts[2], parts[3:]

                if (atom1 not in parms["ATOMTYPES"] and "_shell" not in atom1 and
                        atom2 not in parms["ATOMTYPES"] and "_shell" not in atom2):
                    continue
                if "IGNORE" in " ".join(vals_raw).upper():
                    continue

                try:
                    vals = [abs(float(v)) for v in vals_raw if _is_float(v)]
                except ValueError:
                    continue
                if not vals:
                    continue

                if alter:
                    vals = _apply_alter(parm_type, vals)

                k1, k2 = (atom1, atom2) if atom1 >= atom2 else (atom2, atom1)
                vdw_entry = parms["VDW"].setdefault(k1, {}).setdefault(k2, {})
                vdw_entry[1] = {
                    "TYPE": parm_type,
                    "VALS": vals,
                    "ATOM": f"{k1} {k2}",
                    "USED": 0,
                }

            # ── BOND_STRETCH ─────────────────────────────────────────────────
            # format: type1  type2  PARM_TYPE  k  r0
            elif section == 4:
                parts = stripped.split()
                if len(parts) < 5:
                    continue
                try:
                    t1, t2, ptype = parts[0], parts[1], parts[2]
                    k, r0 = float(parts[3]), float(parts[4])
                except (ValueError, IndexError):
                    continue
                parms["BONDS"][bond_key(t1, t2)] = {"TYPE": ptype, "VALS": [k, r0]}

            # ── ANGLE_BEND ────────────────────────────────────────────────────
            # format: type1  type2  type3  PARM_TYPE  k  theta0
            elif section == 5:
                parts = stripped.split()
                if len(parts) < 6:
                    continue
                try:
                    t1, t2, t3, ptype = parts[0], parts[1], parts[2], parts[3]
                    k, theta0 = float(parts[4]), float(parts[5])
                except (ValueError, IndexError):
                    continue
                parms["ANGLES"][angle_key(t1, t2, t3)] = {"TYPE": ptype, "VALS": [k, theta0]}

            # ── TORSIONS ──────────────────────────────────────────────────────
            # Main line:  type1  type2  type3  type4  PARM_TYPE  k  n  phi0
            # Continuation:                            k  n  phi0  (3 numbers only)
            elif section == 6:
                parts = stripped.split()
                # Detect continuation: starts with a float
                if parts and _is_float(parts[0]):
                    # Continuation line — append to current torsion
                    if _current_torsion_key is not None and len(parts) >= 3:
                        try:
                            k, n, phi0 = float(parts[0]), float(parts[1]), float(parts[2])
                            parms["TORSIONS"][_current_torsion_key].append(
                                {"TYPE": "SHFT_DIHDR", "VALS": [k, n, phi0]}
                            )
                        except (ValueError, IndexError):
                            pass
                    continue
                # Main line
                if len(parts) < 8:
                    continue
                try:
                    t1, t2, t3, t4, ptype = parts[0], parts[1], parts[2], parts[3], parts[4]
                    k, n, phi0 = float(parts[5]), float(parts[6]), float(parts[7])
                except (ValueError, IndexError):
                    continue
                key = torsion_key(t1, t2, t3, t4)
                _current_torsion_key = key
                parms["TORSIONS"][key] = [{"TYPE": ptype, "VALS": [k, n, phi0]}]

            # ── INVERSIONS ────────────────────────────────────────────────────
            # format: type1  type2  type3  type4  PARM_TYPE  k  phi0  n
            # type1 is the central atom (IT_JIKL: i=central, j,k,l=satellites)
            elif section == 7:
                parts = stripped.split()
                if len(parts) < 8:
                    continue
                try:
                    t1, t2, t3, t4, ptype = parts[0], parts[1], parts[2], parts[3], parts[4]
                    k, phi0, n = float(parts[5]), float(parts[6]), float(parts[7])
                except (ValueError, IndexError):
                    continue
                key = inversion_key(t1, t2, t3, t4)
                parms["INVERSIONS"][key] = [{"TYPE": ptype, "VALS": [k, phi0, n]}]

    return parms


def get_vdw_radius(fftype: str, parms: dict) -> Optional[float]:
    """Return the VDW sigma/2 for *fftype*, or None if not found.

    Uses VALS[1] from the DIAGONAL_VDW entry (sigma_LJ after alter),
    which equals the conventional Lennard-Jones sigma.  The contact
    radius is sigma/2.
    """
    vdw = parms.get("VDW", {})
    entry = vdw.get(fftype, {}).get(fftype, {}).get(1)
    if entry and entry.get("VALS") and len(entry["VALS"]) >= 2:
        return entry["VALS"][1] / 2.0
    return None


# ── FF merging ────────────────────────────────────────────────────────────────

def load_ff(specs: str | list | Path, alter: bool = True) -> dict:
    """Load and merge one or more force field files.

    Accepts a space-separated string, a list of paths, or a single path.
    Cerius2 .ff files and AMBER .frcmod files are both supported.
    Later files override earlier files for the same interaction.

    Returns the merged parms dict.
    """
    from atlas_toolkit.io.frcmod import read_frcmod

    if isinstance(specs, str):
        paths = specs.split()
    elif isinstance(specs, Path):
        paths = [specs]
    else:
        paths = list(specs)

    merged: dict = {
        "ATOMTYPES": {},
        "VDW": {},
        "BONDS": {},
        "ANGLES": {},
        "TORSIONS": {},
        "INVERSIONS": {},
        "PARMS": {
            "cut_vdw": 14.0,
            "cut_coul": 15.0,
            "mix_rule": "geometric",
        },
    }

    for path_str in paths:
        p = Path(str(path_str))
        if not p.exists():
            found = find_ff(str(path_str))
            if found:
                p = found
            else:
                raise FileNotFoundError(f"Force field file not found: {path_str!r}")

        suffix = p.suffix.lower()
        if suffix in (".frcmod", ".par"):
            ff = read_frcmod(p, alter=alter)
        else:
            ff = read_ff(p, alter=alter)

        _merge_parms(merged, ff)

    return merged


def _merge_parms(target: dict, source: dict) -> None:
    """Merge *source* parms into *target* in-place. Source overrides target."""
    # Flat-merge sections (last writer wins per key)
    for section in ("ATOMTYPES", "BONDS", "ANGLES", "TORSIONS", "INVERSIONS", "PARMS"):
        if section in source and source[section]:
            target.setdefault(section, {}).update(source[section])

    # VDW is nested two levels deep — merge per (t1, t2) pair
    for t1, inner in source.get("VDW", {}).items():
        for t2, entry in inner.items():
            target["VDW"].setdefault(t1, {}).setdefault(t2, {}).update(entry)


# ── internal helpers ──────────────────────────────────────────────────────────

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _apply_alter(parm_type: str, vals: list[float]) -> list[float]:
    """Apply parseCerius2FF alter=1 transformations to raw VDW values."""
    pt = parm_type.upper()
    if pt == "LJ_6_12":
        # File stores [sigma_cerius, epsilon]
        # After alter: [epsilon, sigma_LJ]  where sigma_LJ = sigma_cerius / 2^(1/6)
        if len(vals) >= 2:
            epsilon = vals[1]
            sigma_lj = vals[0] / _TWO_16
            result = [epsilon, sigma_lj]
            if len(vals) > 2:
                result.append(vals[3] / _TWO_16 if len(vals) > 3 else sigma_lj)
            return result
    return vals
