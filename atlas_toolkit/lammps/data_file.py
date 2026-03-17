"""
Write a LAMMPS data file from BGF atoms + bonds + force-field parameters.

Public API
----------
write_data_file(path, atoms, bonds_dict, ff_parms, box, title="") -> DataFileSummary

The output uses LAMMPS *full* atom style:
    atom-ID  mol-ID  atom-type  charge  x  y  z  [ix iy iz]

Force-field type mapping
------------------------
Bond    HARMONIC   → bond_style harmonic        coeff: K  r0
Angle   THETA_HARM → angle_style harmonic       coeff: K  theta0
Torsion SHFT_DIHDR → dihedral_style charmm      coeff: K  n  d  0.0  0.0
           (multi-term → multiple type IDs, multiple Dihedrals entries)
Improper IT_JIKL   → improper_style cvff         coeff: K  d  n
           (d = +1 if phase==0, -1 if phase==180)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from atlas_toolkit.lammps.topology import (
    enumerate_angles,
    enumerate_bonds,
    enumerate_impropers,
    enumerate_torsions,
)
from atlas_toolkit.io.ff import (
    angle_key,
    bond_key,
    inversion_key,
    lookup_angle,
    lookup_bond,
    lookup_inversion,
    lookup_torsion,
    torsion_key,
)


# ── type registry ──────────────────────────────────────────────────────────────

class _TypeRegistry:
    """Assign sequential 1-based IDs to unique parameter tuples."""

    def __init__(self):
        self._map: dict = {}      # param_key → (type_id, label, vals)
        self._counter = 0

    def register(self, param_key, label: str, vals: list) -> int:
        """Return type_id for param_key, inserting if new."""
        if param_key not in self._map:
            self._counter += 1
            self._map[param_key] = (self._counter, label, vals)
        return self._map[param_key][0]

    def items(self):
        """Sorted list of (type_id, label, vals)."""
        return sorted(self._map.values(), key=lambda x: x[0])

    def __len__(self):
        return self._counter


# ── helpers ────────────────────────────────────────────────────────────────────

def _round_key(vals, ndigits=6):
    return tuple(round(v, ndigits) for v in vals)


def _phase_to_d(phase: float) -> int:
    """Convert SHFT_DIHDR / IT_JIKL phase (degrees) to charmm/cvff d (+1/-1)."""
    if abs(phase) < 1e-3:
        return 1
    if abs(abs(phase) - 180.0) < 1e-3:
        return -1
    # For arbitrary phases we still need to pick a sign; round to nearest
    return 1 if math.cos(math.radians(phase)) >= 0 else -1


def _get_atom_fftype(atom: dict) -> str:
    return str(atom.get("FFTYPE", atom.get("ATMNAME", "UNK")))


# ── DataFileSummary ────────────────────────────────────────────────────────────

@dataclass
class DataFileSummary:
    """Statistics returned by write_data_file."""
    n_atoms: int = 0
    n_bonds: int = 0
    n_angles: int = 0
    n_dihedrals: int = 0
    n_impropers: int = 0
    n_atom_types: int = 0
    n_bond_types: int = 0
    n_angle_types: int = 0
    n_dihedral_types: int = 0
    n_improper_types: int = 0
    missing_bonds: list = field(default_factory=list)
    missing_angles: list = field(default_factory=list)
    missing_torsions: list = field(default_factory=list)
    missing_impropers: list = field(default_factory=list)


# ── main entry point ───────────────────────────────────────────────────────────

def write_data_file(
    path: str | Path,
    atoms: dict,
    bonds_dict: dict,
    ff_parms: dict,
    box: dict,
    title: str = "",
    warn_missing: bool = True,
) -> DataFileSummary:
    """Write a LAMMPS full-style data file.

    Parameters
    ----------
    path        : output file path
    atoms       : atom dict from read_bgf / add_solvent etc.
    bonds_dict  : atom_id → list[atom_id]  (from read_bgf)
    ff_parms    : merged FF parms from load_ff()
    box         : box dict from get_box()
    title       : optional title line
    warn_missing: if True, unknown interactions are skipped (not fatal)

    Returns
    -------
    DataFileSummary with counts and lists of any missing interactions.
    """
    from atlas_toolkit.core.manip_atoms import get_mols

    summary = DataFileSummary()

    # ── 1.  Assign sequential LAMMPS atom IDs (1-based) ─────────────────────
    sorted_ids = sorted(atoms.keys())
    lammps_id: dict[int, int] = {bgf_id: lmp_id
                                  for lmp_id, bgf_id in enumerate(sorted_ids, 1)}

    # ── 2.  Molecule IDs ─────────────────────────────────────────────────────
    get_mols(atoms, bonds_dict)
    mol_of: dict[int, int] = {bgf_id: int(atoms[bgf_id].get("MOLECULEID", 1))
                               for bgf_id in sorted_ids}

    # ── 3.  Atom type IDs ────────────────────────────────────────────────────
    atom_type_reg = _TypeRegistry()
    for bgf_id in sorted_ids:
        fft = _get_atom_fftype(atoms[bgf_id])
        mass = ff_parms.get("ATOMTYPES", {}).get(fft, {}).get("MASS", 1.0)
        atom_type_reg.register(fft, fft, [mass])

    # ── 4.  Enumerate topology ────────────────────────────────────────────────
    raw_bonds    = enumerate_bonds(bonds_dict)
    raw_angles   = enumerate_angles(bonds_dict)
    raw_torsions = enumerate_torsions(bonds_dict)
    raw_impropers = enumerate_impropers(bonds_dict)

    # ── 5.  Look up parameters & assign type IDs ─────────────────────────────
    bond_reg    = _TypeRegistry()
    angle_reg   = _TypeRegistry()
    dihedral_reg = _TypeRegistry()
    improper_reg = _TypeRegistry()

    # Bonds
    bond_entries: list[tuple[int, int, int]] = []   # (type_id, lmp_i, lmp_j)
    for (i, j) in raw_bonds:
        ti = _get_atom_fftype(atoms[i])
        tj = _get_atom_fftype(atoms[j])
        entry = lookup_bond(ti, tj, ff_parms)
        if entry is None:
            summary.missing_bonds.append((ti, tj))
            continue
        k, r0 = entry["VALS"]
        pkey = _round_key([k, r0])
        label = f"{ti}-{tj}"
        tid = bond_reg.register(pkey, label, [k, r0])
        bond_entries.append((tid, lammps_id[i], lammps_id[j]))

    # Angles
    angle_entries: list[tuple[int, int, int, int]] = []
    for (i, j, k) in raw_angles:
        ti = _get_atom_fftype(atoms[i])
        tj = _get_atom_fftype(atoms[j])
        tk = _get_atom_fftype(atoms[k])
        entry = lookup_angle(ti, tj, tk, ff_parms)
        if entry is None:
            summary.missing_angles.append((ti, tj, tk))
            continue
        kc, theta0 = entry["VALS"]
        pkey = _round_key([kc, theta0])
        label = f"{ti}-{tj}-{tk}"
        tid = angle_reg.register(pkey, label, [kc, theta0])
        angle_entries.append((tid, lammps_id[i], lammps_id[j], lammps_id[k]))

    # Torsions (CHARMM style — one entry per Fourier term)
    dihedral_entries: list[tuple[int, int, int, int, int]] = []
    for (i, j, k, l) in raw_torsions:
        ti = _get_atom_fftype(atoms[i])
        tj = _get_atom_fftype(atoms[j])
        tk = _get_atom_fftype(atoms[k])
        tl = _get_atom_fftype(atoms[l])
        terms = lookup_torsion(ti, tj, tk, tl, ff_parms)
        if terms is None:
            summary.missing_torsions.append((ti, tj, tk, tl))
            continue
        for term in terms:
            kd, n, phi0 = term["VALS"]
            n_int = int(round(abs(n)))
            d_deg = int(round(phi0))   # phase in degrees (0 or 180)
            pkey = _round_key([kd, n_int, phi0])
            label = f"{ti}-{tj}-{tk}-{tl}"
            tid = dihedral_reg.register(pkey, label, [kd, n_int, d_deg, 0.0])
            dihedral_entries.append((tid, lammps_id[i], lammps_id[j],
                                      lammps_id[k], lammps_id[l]))

    # Impropers (cvff style)
    improper_entries: list[tuple[int, int, int, int, int]] = []
    for (center, a, b, c) in raw_impropers:
        tc = _get_atom_fftype(atoms[center])
        ta = _get_atom_fftype(atoms[a])
        tb = _get_atom_fftype(atoms[b])
        tv = _get_atom_fftype(atoms[c])
        terms = lookup_inversion(tc, ta, tb, tv, ff_parms)
        if terms is None:
            # Most over-coordinated atoms won't have improper params — skip silently
            continue
        for term in terms:
            ki, phi0, ni = term["VALS"]
            d = _phase_to_d(phi0)
            n_int = int(round(abs(ni)))
            pkey = _round_key([ki, phi0, n_int])
            label = f"{tc}-{ta}-{tb}-{tv}"
            tid = improper_reg.register(pkey, label, [ki, d, n_int])
            improper_entries.append((tid, lammps_id[center], lammps_id[a],
                                      lammps_id[b], lammps_id[c]))

    # ── 6.  Fill summary counts ───────────────────────────────────────────────
    summary.n_atoms        = len(atoms)
    summary.n_bonds        = len(bond_entries)
    summary.n_angles       = len(angle_entries)
    summary.n_dihedrals    = len(dihedral_entries)
    summary.n_impropers    = len(improper_entries)
    summary.n_atom_types   = len(atom_type_reg)
    summary.n_bond_types   = len(bond_reg)
    summary.n_angle_types  = len(angle_reg)
    summary.n_dihedral_types = len(dihedral_reg)
    summary.n_improper_types = len(improper_reg)

    # ── 7.  Write file ────────────────────────────────────────────────────────
    path = Path(path)
    with open(path, "w") as fh:
        _write_header(fh, title, summary, box)
        _write_masses(fh, atom_type_reg, ff_parms)
        _write_pair_coeffs(fh, atom_type_reg, ff_parms)
        _write_bond_coeffs(fh, bond_reg)
        _write_angle_coeffs(fh, angle_reg)
        _write_dihedral_coeffs(fh, dihedral_reg)
        _write_improper_coeffs(fh, improper_reg)
        _write_atoms(fh, sorted_ids, atoms, lammps_id, mol_of, atom_type_reg)
        _write_valence(fh, "Bonds", bond_entries)
        _write_valence(fh, "Angles", angle_entries)
        _write_valence(fh, "Dihedrals", dihedral_entries)
        _write_valence(fh, "Impropers", improper_entries)

    return summary


# ── section writers ────────────────────────────────────────────────────────────

def _write_header(fh, title: str, s: DataFileSummary, box: dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr = title if title else f"Created by atlas_toolkit on {ts}"
    fh.write(f"{hdr}\n\n")
    fh.write(f"{s.n_atoms:12d}  atoms\n")
    fh.write(f"{s.n_bonds:12d}  bonds\n")
    fh.write(f"{s.n_angles:12d}  angles\n")
    fh.write(f"{s.n_dihedrals:12d}  dihedrals\n")
    fh.write(f"{s.n_impropers:12d}  impropers\n\n")
    fh.write(f"{s.n_atom_types:12d}  atom types\n")
    fh.write(f"{s.n_bond_types:12d}  bond types\n")
    fh.write(f"{s.n_angle_types:12d}  angle types\n")
    fh.write(f"{s.n_dihedral_types:12d}  dihedral types\n")
    fh.write(f"{s.n_improper_types:12d}  improper types\n\n")

    xlo = box["X"]["lo"]
    xhi = box["X"]["hi"]
    ylo = box["Y"]["lo"]
    yhi = box["Y"]["hi"]
    zlo = box["Z"]["lo"]
    zhi = box["Z"]["hi"]
    fh.write(f"{xlo:12.6f} {xhi:12.6f}  xlo xhi\n")
    fh.write(f"{ylo:12.6f} {yhi:12.6f}  ylo yhi\n")
    fh.write(f"{zlo:12.6f} {zhi:12.6f}  zlo zhi\n")

    # Triclinic tilt factors (xy, xz, yz)
    H = box.get("H")
    if H is not None:
        xy = H[0][1] if len(H[0]) > 1 else 0.0
        xz = H[0][2] if len(H[0]) > 2 else 0.0
        yz = H[1][2] if len(H[1]) > 2 else 0.0
        if abs(xy) + abs(xz) + abs(yz) > 1e-8:
            fh.write(f"{xy:12.6f} {xz:12.6f} {yz:12.6f}  xy xz yz\n")
    fh.write("\n")


def _write_masses(fh, atom_type_reg: _TypeRegistry, ff_parms: dict):
    fh.write("Masses\n\n")
    for tid, label, vals in atom_type_reg.items():
        mass = vals[0]
        fh.write(f"{tid:6d}  {mass:12.6f}  # {label}\n")
    fh.write("\n")


def _write_pair_coeffs(fh, atom_type_reg: _TypeRegistry, ff_parms: dict):
    """Write diagonal LJ coefficients (LAMMPS mixes off-diagonal automatically)."""
    lines = []
    for tid, label, _mv in atom_type_reg.items():
        vdw_data = ff_parms.get("VDW", {}).get(label, {}).get(label, {}).get(1)
        if vdw_data is None:
            lines.append(f"{tid:6d}  0.000000  0.000000\n")
        else:
            epsilon, sigma = vdw_data["VALS"]
            lines.append(f"{tid:6d}  {epsilon:12.6f}  {sigma:12.6f}\n")
    if lines:
        fh.write("Pair Coeffs\n\n")
        fh.writelines(lines)
        fh.write("\n")


def _write_bond_coeffs(fh, bond_reg: _TypeRegistry):
    if not len(bond_reg):
        return
    fh.write("Bond Coeffs\n\n")
    for tid, label, vals in bond_reg.items():
        k, r0 = vals
        fh.write(f"{tid:6d}  {k:12.4f}  {r0:10.6f}\n")
    fh.write("\n")


def _write_angle_coeffs(fh, angle_reg: _TypeRegistry):
    if not len(angle_reg):
        return
    fh.write("Angle Coeffs\n\n")
    for tid, label, vals in angle_reg.items():
        k, theta0 = vals
        fh.write(f"{tid:6d}  {k:12.4f}  {theta0:10.4f}\n")
    fh.write("\n")


def _write_dihedral_coeffs(fh, dihedral_reg: _TypeRegistry):
    if not len(dihedral_reg):
        return
    fh.write("Dihedral Coeffs\n\n")
    for tid, label, vals in dihedral_reg.items():
        k, n, d, w = vals
        fh.write(f"{tid:6d}  {k:12.6f}  {n:4d}  {d:4d}  {w:.1f}\n")
    fh.write("\n")


def _write_improper_coeffs(fh, improper_reg: _TypeRegistry):
    if not len(improper_reg):
        return
    fh.write("Improper Coeffs\n\n")
    for tid, label, vals in improper_reg.items():
        k, d, n = vals
        fh.write(f"{tid:6d}  {k:12.6f}  {d:3d}  {n:4d}\n")
    fh.write("\n")


def _write_atoms(fh, sorted_ids, atoms, lammps_id, mol_of, atom_type_reg):
    fh.write("Atoms  # full\n\n")
    # Build fftype → type_id lookup
    fft_to_tid = {label: tid for tid, label, _ in atom_type_reg.items()}
    for bgf_id in sorted_ids:
        atom = atoms[bgf_id]
        lid  = lammps_id[bgf_id]
        mid  = mol_of[bgf_id]
        fft  = str(atom.get("FFTYPE", atom.get("ATMNAME", "UNK")))
        tid  = fft_to_tid.get(fft, 1)
        q    = float(atom.get("CHARGE", 0.0))
        x    = float(atom.get("XCOORD", 0.0))
        y    = float(atom.get("YCOORD", 0.0))
        z    = float(atom.get("ZCOORD", 0.0))
        fh.write(f"{lid:8d} {mid:8d} {tid:8d} {q:11.8f} {x:10.5f} {y:10.5f} {z:10.5f}"
                 f"  0  0  0\n")
    fh.write("\n")


def _write_valence(fh, section: str, entries: list):
    if not entries:
        return
    fh.write(f"{section}\n\n")
    for idx, entry in enumerate(entries, 1):
        ids_str = "  ".join(f"{v:8d}" for v in entry)
        fh.write(f"{idx:8d}  {ids_str}\n")
    fh.write("\n")
