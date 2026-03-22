"""
Write a LAMMPS data file directly from a ParmEd Structure.

All bonded and nonbonded parameters are read from the ParmEd object — no
FF lookup dict required.  This is the right path when parameters come from
a prmtop (OpenFF/Sage, GAFF via antechamber, CHARMM, etc.).

Public API
----------
write_data_file_parmed(path, struct, box, title="") -> DataFileSummary

LAMMPS atom style: full
  atom-ID  mol-ID  atom-type  charge  x  y  z

FF style mapping from ParmEd → LAMMPS
--------------------------------------
LJ nonbonded  →  pair_style lj/cut/coul/long
                 pair_coeff  eps  sigma   (sigma = rmin / 2^(1/6))
Bond          →  bond_style harmonic     k  r0
Angle         →  angle_style harmonic    k  theta0
Proper tors   →  dihedral_style charmm  k  n  d  weight
Improper      →  improper_style cvff    k  d  n

Multi-FF systems
----------------
Pass a merged ParmEd Structure built with struct_a + struct_b + ... 
(ParmEd addition preserves all bonded/nonbonded params from each component.)
Parameters from different FFs coexist cleanly since they are keyed by
(epsilon, rmin) tuples, not FF-type strings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import parmed as pmd

__all__ = ["write_data_file_parmed", "DataFileSummaryP"]

# ── constants ──────────────────────────────────────────────────────────────────

_RMIN_TO_SIGMA = 2 ** (-1 / 6)   # sigma = rmin * 2^(-1/6) = rmin / 2^(1/6)


# ── type registry ──────────────────────────────────────────────────────────────

class _TypeRegistry:
    """Map unique parameter tuples to sequential 1-based LAMMPS type IDs."""

    def __init__(self):
        self._map: dict = {}
        self._counter = 0

    def register(self, key, label: str, coeffs: list) -> int:
        if key not in self._map:
            self._counter += 1
            self._map[key] = (self._counter, label, coeffs)
        return self._map[key][0]

    def items(self):
        return sorted(self._map.values(), key=lambda x: x[0])

    def __len__(self):
        return self._counter


def _rkey(*vals, ndigits: int = 6) -> tuple:
    return tuple(round(float(v), ndigits) for v in vals)


# ── summary dataclass ──────────────────────────────────────────────────────────

@dataclass
class DataFileSummaryP:
    n_atoms:          int = 0
    n_bonds:          int = 0
    n_angles:         int = 0
    n_dihedrals:      int = 0
    n_impropers:      int = 0
    n_atom_types:     int = 0
    n_bond_types:     int = 0
    n_angle_types:    int = 0
    n_dihedral_types: int = 0
    n_improper_types: int = 0
    missing_bonds:    list = field(default_factory=list)
    missing_angles:   list = field(default_factory=list)
    missing_torsions: list = field(default_factory=list)

    # expose the same interface as DataFileSummary so write_input_script works
    @property
    def n_lj_types(self):
        return self.n_atom_types


# ── molecule ID assignment ─────────────────────────────────────────────────────

def _assign_mol_ids(struct: pmd.Structure) -> dict[int, int]:
    """Return {atom_idx_0based: mol_id_1based} via BFS over bond graph."""
    n = len(struct.atoms)
    mol_id = [0] * n
    current = 0

    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in struct.bonds:
        i = bond.atom1.idx
        j = bond.atom2.idx
        adj[i].append(j)
        adj[j].append(i)

    for start in range(n):
        if mol_id[start] != 0:
            continue
        current += 1
        queue = [start]
        mol_id[start] = current
        while queue:
            node = queue.pop()
            for nb in adj[node]:
                if mol_id[nb] == 0:
                    mol_id[nb] = current
                    queue.append(nb)

    return {i: mid for i, mid in enumerate(mol_id)}


# ── main writer ────────────────────────────────────────────────────────────────

def write_data_file_parmed(
    path: str | Path,
    struct: pmd.Structure,
    box: dict,
    title: str = "",
    warn_missing: bool = True,
) -> DataFileSummaryP:
    """Write a LAMMPS full-style data file from a ParmEd Structure.

    Parameters
    ----------
    path         : output file path
    struct       : ParmEd Structure (from prmtop load, parmed arithmetic, etc.)
    box          : atlas-toolkit box dict — keys: xlo, xhi, ylo, yhi, zlo, zhi
                   (from get_box(), or build manually for non-periodic systems)
    title        : optional comment line
    warn_missing : if True, log missing params as warnings (not fatal)

    Returns
    -------
    DataFileSummaryP — counts and lists of any missing interactions.
    """
    summary = DataFileSummaryP()
    mol_id_map = _assign_mol_ids(struct)

    # ── 1. Atom type registry ──────────────────────────────────────────────────
    atom_type_reg = _TypeRegistry()
    # Map parmed atom index → lammps atom type ID
    atom_lammps_type: dict[int, int] = {}

    for atom in struct.atoms:
        at = atom.atom_type
        # _UnassignedAtomType has no epsilon/rmin — treat as zero
        try:
            eps  = float(at.epsilon) if at else 0.0
            rmin = float(at.rmin)    if at else 0.0
        except AttributeError:
            eps = rmin = 0.0
        sigma  = rmin * _RMIN_TO_SIGMA
        mass   = float(atom.mass) if atom.mass else 1.0
        # key on (eps, rmin) so identical LJ params collapse to one type
        # regardless of FF-type string
        key = _rkey(eps, rmin)
        label = atom.type or f"T{atom.idx}"
        tid = atom_type_reg.register(key, label, [mass, eps, sigma])
        atom_lammps_type[atom.idx] = tid

    summary.n_atom_types = len(atom_type_reg)

    # ── 2. Bond type registry ──────────────────────────────────────────────────
    bond_type_reg = _TypeRegistry()
    bond_entries: list[tuple] = []

    for bond in struct.bonds:
        bt = bond.type
        if bt is None:
            summary.missing_bonds.append(
                (bond.atom1.type, bond.atom2.type)
            )
            continue
        k   = float(bt.k)
        req = float(bt.req)
        key = _rkey(k, req)
        label = f"{bond.atom1.type}-{bond.atom2.type}"
        tid = bond_type_reg.register(key, label, [k, req])
        bond_entries.append((tid, bond.atom1.idx + 1, bond.atom2.idx + 1))

    summary.n_bond_types = len(bond_type_reg)

    # ── 3. Angle type registry ─────────────────────────────────────────────────
    angle_type_reg = _TypeRegistry()
    angle_entries: list[tuple] = []

    for angle in struct.angles:
        at = angle.type
        if at is None:
            summary.missing_angles.append(
                (angle.atom1.type, angle.atom2.type, angle.atom3.type)
            )
            continue
        k     = float(at.k)
        theta = float(at.theteq)
        key   = _rkey(k, theta)
        label = f"{angle.atom1.type}-{angle.atom2.type}-{angle.atom3.type}"
        tid   = angle_type_reg.register(key, label, [k, theta])
        angle_entries.append((tid,
                               angle.atom1.idx + 1,
                               angle.atom2.idx + 1,
                               angle.atom3.idx + 1))

    summary.n_angle_types = len(angle_type_reg)

    # ── 4. Dihedral type registry (proper + improper) ──────────────────────────
    # LAMMPS harmonic dihedral style: K d n
    #   K  = force constant (kcal/mol)
    #   d  = +1 (phase=0°) or -1 (phase=180°)
    #   n  = periodicity (integer)
    # No 1-4 scaling in harmonic style — handled by special_bonds instead.
    dihedral_type_reg = _TypeRegistry()
    dihedral_entries: list[tuple] = []

    improper_type_reg = _TypeRegistry()
    improper_entries:  list[tuple] = []

    for dih in struct.dihedrals:
        dt = dih.type
        if dt is None:
            if not dih.improper:
                summary.missing_torsions.append((
                    dih.atom1.type, dih.atom2.type,
                    dih.atom3.type, dih.atom4.type,
                ))
            continue

        phi_k = float(dt.phi_k)
        per   = float(dt.per)
        phase = float(dt.phase)

        # ParmEd phase units depend on source:
        # - AMBER prmtop (GAFF/ff14SB): stored in radians
        # - SMIRNOFF/OpenFF prmtop:      stored in degrees
        # Distinguish: if |phase| > 2*pi it must already be degrees.
        if abs(phase) > 7.0:   # > 2π → already degrees
            phase_deg = phase
        else:                   # radians → convert
            phase_deg = math.degrees(phase)

        # Round to nearest valid phase (0° or 180°)
        if abs(phase_deg % 360) < 90 or abs(phase_deg % 360) > 270:
            d_int = 1   # phase ≈ 0°
        else:
            d_int = -1  # phase ≈ 180°
        n_int = max(1, int(round(abs(per))))

        if dih.improper:
            key   = _rkey(phi_k, per, phase)
            label = (f"{dih.atom1.type}-{dih.atom2.type}-"
                     f"{dih.atom3.type}-{dih.atom4.type}")
            tid   = improper_type_reg.register(key, label, [phi_k, d_int, n_int])
            improper_entries.append((tid,
                                     dih.atom1.idx + 1,
                                     dih.atom2.idx + 1,
                                     dih.atom3.idx + 1,
                                     dih.atom4.idx + 1))
        else:
            key   = _rkey(phi_k, per, phase)
            label = (f"{dih.atom1.type}-{dih.atom2.type}-"
                     f"{dih.atom3.type}-{dih.atom4.type}")
            tid   = dihedral_type_reg.register(key, label,
                                                [phi_k, d_int, n_int])
            dihedral_entries.append((tid,
                                     dih.atom1.idx + 1,
                                     dih.atom2.idx + 1,
                                     dih.atom3.idx + 1,
                                     dih.atom4.idx + 1))

    summary.n_dihedral_types = len(dihedral_type_reg)
    summary.n_improper_types = len(improper_type_reg)

    # ── 5. Fill counts ─────────────────────────────────────────────────────────
    summary.n_atoms     = len(struct.atoms)
    summary.n_bonds     = len(bond_entries)
    summary.n_angles    = len(angle_entries)
    summary.n_dihedrals = len(dihedral_entries)
    summary.n_impropers = len(improper_entries)
    summary.n_bond_types    = len(bond_type_reg)
    summary.n_angle_types   = len(angle_type_reg)

    # ── 6. Write file ──────────────────────────────────────────────────────────
    path = Path(path)
    with open(path, "w") as fh:
        _write_header(fh, title, summary, box)
        _write_masses(fh, atom_type_reg)
        _write_pair_coeffs(fh, atom_type_reg)
        _write_bond_coeffs(fh, bond_type_reg)
        _write_angle_coeffs(fh, angle_type_reg)
        _write_dihedral_coeffs(fh, dihedral_type_reg)
        _write_improper_coeffs(fh, improper_type_reg)
        _write_atoms(fh, struct, atom_lammps_type, mol_id_map)
        _write_bonds(fh, bond_entries)
        _write_angles(fh, angle_entries)
        _write_dihedrals(fh, dihedral_entries)
        _write_impropers(fh, improper_entries)

    return summary


# ── section writers ────────────────────────────────────────────────────────────

def _write_header(fh, title: str, s: DataFileSummaryP, box: dict) -> None:
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr = title if title else f"LAMMPS data file — atlas_toolkit {ts}"
    fh.write(f"{hdr}\n\n")
    fh.write(f"{s.n_atoms:12d}  atoms\n")
    fh.write(f"{s.n_bonds:12d}  bonds\n")
    fh.write(f"{s.n_angles:12d}  angles\n")
    fh.write(f"{s.n_dihedrals:12d}  dihedrals\n")
    fh.write(f"{s.n_impropers:12d}  impropers\n\n")
    fh.write(f"{s.n_atom_types:12d}  atom types\n")
    if s.n_bond_types:
        fh.write(f"{s.n_bond_types:12d}  bond types\n")
    if s.n_angle_types:
        fh.write(f"{s.n_angle_types:12d}  angle types\n")
    if s.n_dihedral_types:
        fh.write(f"{s.n_dihedral_types:12d}  dihedral types\n")
    if s.n_improper_types:
        fh.write(f"{s.n_improper_types:12d}  improper types\n")
    fh.write("\n")
    xlo = box.get("xlo", 0.0);  xhi = box.get("xhi", 100.0)
    ylo = box.get("ylo", 0.0);  yhi = box.get("yhi", 100.0)
    zlo = box.get("zlo", 0.0);  zhi = box.get("zhi", 100.0)
    fh.write(f"{xlo:16.8f} {xhi:16.8f}  xlo xhi\n")
    fh.write(f"{ylo:16.8f} {yhi:16.8f}  ylo yhi\n")
    fh.write(f"{zlo:16.8f} {zhi:16.8f}  zlo zhi\n\n")


def _write_masses(fh, atom_type_reg: _TypeRegistry) -> None:
    fh.write("Masses\n\n")
    for tid, label, coeffs in atom_type_reg.items():
        mass = coeffs[0]
        fh.write(f"{tid:6d}  {mass:12.6f}  # {label}\n")
    fh.write("\n")


def _write_pair_coeffs(fh, atom_type_reg: _TypeRegistry) -> None:
    fh.write("Pair Coeffs  # lj/cut/coul/long\n\n")
    for tid, label, coeffs in atom_type_reg.items():
        # coeffs = [mass, epsilon, sigma]
        eps   = coeffs[1] if len(coeffs) > 1 else 0.0
        sigma = coeffs[2] if len(coeffs) > 2 else 0.0
        fh.write(f"{tid:6d}  {eps:12.6f}  {sigma:12.6f}  # {label}\n")
    fh.write("\n")


def _write_bond_coeffs(fh, bond_type_reg: _TypeRegistry) -> None:
    if not len(bond_type_reg):
        return
    fh.write("Bond Coeffs  # harmonic\n\n")
    for tid, label, coeffs in bond_type_reg.items():
        k, r0 = coeffs
        fh.write(f"{tid:6d}  {k:12.6f}  {r0:12.6f}  # {label}\n")
    fh.write("\n")


def _write_angle_coeffs(fh, angle_type_reg: _TypeRegistry) -> None:
    if not len(angle_type_reg):
        return
    fh.write("Angle Coeffs  # harmonic\n\n")
    for tid, label, coeffs in angle_type_reg.items():
        k, theta = coeffs
        fh.write(f"{tid:6d}  {k:12.6f}  {theta:12.6f}  # {label}\n")
    fh.write("\n")


def _write_dihedral_coeffs(fh, dihedral_type_reg: _TypeRegistry) -> None:
    if not len(dihedral_type_reg):
        return
    fh.write("Dihedral Coeffs  # harmonic\n\n")
    for tid, label, coeffs in dihedral_type_reg.items():
        k, d, n = coeffs
        fh.write(f"{tid:6d}  {k:12.6f}  {int(d):4d}  {int(n):4d}  # {label}\n")
    fh.write("\n")


def _write_improper_coeffs(fh, improper_type_reg: _TypeRegistry) -> None:
    if not len(improper_type_reg):
        return
    fh.write("Improper Coeffs  # cvff\n\n")
    for tid, label, coeffs in improper_type_reg.items():
        k, d, n = coeffs
        fh.write(f"{tid:6d}  {k:12.6f}  {int(d):4d}  {int(n):4d}  # {label}\n")
    fh.write("\n")


def _write_atoms(fh, struct: pmd.Structure,
                 atom_lammps_type: dict[int, int],
                 mol_id_map: dict[int, int]) -> None:
    fh.write("Atoms  # full\n\n")
    for atom in struct.atoms:
        lmp_id  = atom.idx + 1
        mol_id  = mol_id_map.get(atom.idx, 1)
        type_id = atom_lammps_type[atom.idx]
        charge  = float(atom.charge) if atom.charge else 0.0
        x = float(atom.xx) if hasattr(atom, 'xx') else 0.0
        y = float(atom.xy) if hasattr(atom, 'xy') else 0.0
        z = float(atom.xz) if hasattr(atom, 'xz') else 0.0
        fh.write(
            f"{lmp_id:8d} {mol_id:6d} {type_id:6d} "
            f"{charge:10.6f} {x:14.6f} {y:14.6f} {z:14.6f}\n"
        )
    fh.write("\n")


def _write_bonds(fh, entries: list) -> None:
    if not entries:
        return
    fh.write("Bonds\n\n")
    for idx, (tid, i, j) in enumerate(entries, 1):
        fh.write(f"{idx:8d} {tid:6d} {i:8d} {j:8d}\n")
    fh.write("\n")


def _write_angles(fh, entries: list) -> None:
    if not entries:
        return
    fh.write("Angles\n\n")
    for idx, (tid, i, j, k) in enumerate(entries, 1):
        fh.write(f"{idx:8d} {tid:6d} {i:8d} {j:8d} {k:8d}\n")
    fh.write("\n")


def _write_dihedrals(fh, entries: list) -> None:
    if not entries:
        return
    fh.write("Dihedrals\n\n")
    for idx, (tid, i, j, k, l) in enumerate(entries, 1):
        fh.write(f"{idx:8d} {tid:6d} {i:8d} {j:8d} {k:8d} {l:8d}\n")
    fh.write("\n")


def _write_impropers(fh, entries: list) -> None:
    if not entries:
        return
    fh.write("Impropers\n\n")
    for idx, (tid, i, j, k, l) in enumerate(entries, 1):
        fh.write(f"{idx:8d} {tid:6d} {i:8d} {j:8d} {k:8d} {l:8d}\n")
    fh.write("\n")
