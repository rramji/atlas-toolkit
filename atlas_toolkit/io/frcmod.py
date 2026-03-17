"""
AMBER .frcmod (force field modification) file parser.

Port of the frcmod reading logic used by createLammpsInput.pl / LoadFFs.

Public API
----------
read_frcmod(path, alter=True) -> dict
    Parse an AMBER .frcmod file.  Returns a parms dict with the same
    key structure as read_ff():
        ATOMTYPES  : {fftype -> {MASS, ATOM, LABEL, CHARGE, ...}}
        VDW        : {t1 -> {t1 -> {1 -> {TYPE, VALS: [epsilon, sigma_lj]}}}}
                     (diagonal only; sigma_lj = 2*rmin_half / 2^(1/6) when alter=True)
        BONDS      : {(t1, t2) -> {TYPE, VALS: [k, r0]}}
        ANGLES     : {(t1, t2, t3) -> {TYPE, VALS: [k, theta0]}}
        TORSIONS   : {(t1, t2, t3, t4) -> [{TYPE, VALS: [k, n, phi0]}, ...]}
                     k already divided by the divider factor.
        INVERSIONS : {(t1, t2, t3, t4) -> [{TYPE, VALS: [k, phi0, n]}]}
        PARMS      : {} (frcmod files carry no global cutoff settings)

AMBER frcmod section conventions
---------------------------------
MASS     type  mass  [polarizability]
BOND     type1-type2  k  r0
ANGLE    type1-type2-type3  k  theta0
DIHE     type1-type2-type3-type4  div  k  phase  n
         Negative n means more terms follow for the same quartet.
IMPROPER type1-type2-type3-type4  k  phase  n   (type3 = central atom in AMBER)
NONBON   type  rmin_half  epsilon
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_TWO_16 = 2.0 ** (1.0 / 6.0)

# Section keywords (first 4 chars, upper-cased)
_SECTION_TAGS = {"MASS", "BOND", "ANGL", "DIHE", "IMPR", "NONB"}


def read_frcmod(path: str | Path, alter: bool = True) -> dict:
    """Parse an AMBER .frcmod file.

    Parameters
    ----------
    path  : path to the .frcmod file
    alter : if True, convert NONBON rmin_half → sigma_LJ (= 2*rmin_half / 2^(1/6))
            and store VDW as [epsilon, sigma_lj] to match read_ff() convention.

    Returns
    -------
    parms dict (same structure as read_ff output)
    """
    from atlas_toolkit.io.ff import bond_key, angle_key, torsion_key, inversion_key

    path = Path(path)
    parms: dict = {
        "ATOMTYPES": {},
        "VDW": {},
        "BONDS": {},
        "ANGLES": {},
        "TORSIONS": {},
        "INVERSIONS": {},
        "PARMS": {},
    }

    section: Optional[str] = None
    _current_dihe_key: Optional[tuple] = None  # for n<0 multi-term dihedrals

    with open(path) as fh:
        for line_no, raw in enumerate(fh):
            # First line is a remark — skip
            if line_no == 0:
                continue

            line = raw.rstrip("\n")
            # Strip AMBER-style comments
            data = re.sub(r"!.*$", "", line).strip()
            if not data:
                continue

            # ── Section header detection ─────────────────────────────────────
            tag4 = data[:4].upper() if len(data) >= 4 else data.upper()
            # A section header line contains only the keyword (no - separators)
            if tag4 in _SECTION_TAGS and "-" not in data.split()[0]:
                section = tag4
                _current_dihe_key = None
                continue

            # ── MASS ─────────────────────────────────────────────────────────
            if section == "MASS":
                parts = data.split()
                if len(parts) >= 2 and _is_float(parts[1]):
                    atype = parts[0]
                    mass = float(parts[1])
                    if atype not in parms["ATOMTYPES"]:
                        parms["ATOMTYPES"][atype] = {
                            "MASS":      mass,
                            "ATOM":      atype,
                            "LABEL":     atype,
                            "CHARGE":    0.0,
                            "NUMBONDS":  0,
                            "LONEPAIRS": 0,
                            "OTHER":     0,
                        }
                    else:
                        parms["ATOMTYPES"][atype]["MASS"] = mass

            # ── BOND ─────────────────────────────────────────────────────────
            # format: type1-type2  k  r0
            elif section == "BOND":
                m = re.match(
                    r"^(\S+?)\s*-\s*(\S+?)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)",
                    data,
                )
                if m:
                    t1, t2 = m.group(1).strip(), m.group(2).strip()
                    k, r0 = float(m.group(3)), float(m.group(4))
                    parms["BONDS"][bond_key(t1, t2)] = {"TYPE": "HARMONIC", "VALS": [k, r0]}

            # ── ANGL / ANGLE ─────────────────────────────────────────────────
            # format: type1-type2-type3  k  theta0
            elif section == "ANGL":
                m = re.match(
                    r"^(\S+?)\s*-\s*(\S+?)\s*-\s*(\S+?)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)",
                    data,
                )
                if m:
                    t1, t2, t3 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                    k, theta0 = float(m.group(4)), float(m.group(5))
                    parms["ANGLES"][angle_key(t1, t2, t3)] = {"TYPE": "THETA_HARM", "VALS": [k, theta0]}

            # ── DIHE ─────────────────────────────────────────────────────────
            # format: type1-type2-type3-type4  div  k  phase  n
            # n < 0 means another term follows for the same quartet
            elif section == "DIHE":
                m = re.match(
                    r"^(\S+?)\s*-\s*(\S+?)\s*-\s*(\S+?)\s*-\s*(\S+?)"
                    r"\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)",
                    data,
                )
                if m:
                    t1 = m.group(1).strip()
                    t2 = m.group(2).strip()
                    t3 = m.group(3).strip()
                    t4 = m.group(4).strip()
                    div = float(m.group(5))
                    k_raw = float(m.group(6))
                    phase = float(m.group(7))
                    n_raw = float(m.group(8))

                    # div is the path count; store effective k per term
                    k = k_raw / div if div != 0 else k_raw
                    n = abs(n_raw)

                    key = torsion_key(t1, t2, t3, t4)
                    term = {"TYPE": "SHFT_DIHDR", "VALS": [k, n, phase]}

                    if _current_dihe_key == key:
                        # Continuation of an ongoing multi-term dihedral
                        parms["TORSIONS"][key].append(term)
                    else:
                        # New dihedral entry
                        parms["TORSIONS"][key] = [term]

                    # n<0 signals more terms follow; n>0 ends this run
                    _current_dihe_key = key if n_raw < 0 else None

            # ── IMPR / IMPROPER ───────────────────────────────────────────────
            # format: type1-type2-type3-type4  k  phase  n
            # In AMBER: type3 is the central (improper center) atom
            elif section == "IMPR":
                m = re.match(
                    r"^(\S+?)\s*-\s*(\S+?)\s*-\s*(\S+?)\s*-\s*(\S+?)"
                    r"\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)",
                    data,
                )
                if m:
                    t1, t2, t3, t4 = (m.group(i).strip() for i in range(1, 5))
                    k, phase, n = float(m.group(5)), float(m.group(6)), float(m.group(7))
                    # AMBER: central atom is t3; Cerius2 IT_JIKL: central is t1
                    # Remap to (central, sat1, sat2, sat3) with t3 as central
                    key = inversion_key(t3, t1, t2, t4)
                    parms["INVERSIONS"][key] = [{"TYPE": "IT_JIKL", "VALS": [k, phase, n]}]

            # ── NONBON ────────────────────────────────────────────────────────
            # format: type  rmin_half  epsilon
            # rmin_half = Rmin/2 where Rmin is where the LJ potential is minimum
            # sigma_lj = Rmin / 2^(1/6) = 2 * rmin_half / 2^(1/6)
            elif section == "NONB":
                parts = data.split()
                if len(parts) >= 3 and _is_float(parts[1]):
                    atype = parts[0]
                    rmin_half = float(parts[1])
                    epsilon = float(parts[2])
                    if alter:
                        sigma_lj = 2.0 * rmin_half / _TWO_16
                        vals = [epsilon, sigma_lj]
                    else:
                        vals = [rmin_half, epsilon]
                    parms["VDW"].setdefault(atype, {}).setdefault(atype, {})[1] = {
                        "TYPE": "LJ_6_12",
                        "VALS": vals,
                        "ATOM": f"{atype} {atype}",
                        "USED": 0,
                    }

    return parms


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
