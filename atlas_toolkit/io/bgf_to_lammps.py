"""
BGF + FF files → parameterized ParmEd Structure.

This is the primary path for existing, already-typed BGF files:
load the structure and FF, populate a ParmEd Structure with all
bonded/nonbonded parameters, then hand off to write_data_file_parmed.

Public API
----------
bgf_ff_to_parmed(bgf_path, ff_specs, ...)  → (pmd.Structure, box, DataFileSummaryP-like)
bgf_ff_to_lammps(bgf_path, ff_specs, ...)  → writes data.* and in.* directly

Supports any combination of .ff and .frcmod files via load_ff().
Parameters are looked up using the existing atlas-toolkit lookup_* functions
(with wildcard fallback for torsions/inversions).

Example
-------
    # Fully automatic for a GAFF+Heinz+JC system:
    struct, box, summary = bgf_ff_to_parmed(
        'system.bgf',
        ['GAFF17.ff', 'heinzAu_oplsIons_softChlorine.ff', 'citrate.frcmod'],
    )

    # Or use auto-detection:
    struct, box, summary = bgf_ff_to_parmed('system.bgf')  # ff auto-detected

    # Write LAMMPS files directly:
    bgf_ff_to_lammps('system.bgf', output_stem='run/system', protocol='nvt')
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "bgf_ff_to_parmed",
    "bgf_ff_to_lammps",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def _load_ff(ff_specs):
    """Accept None, a single path/name, or a list and return merged parms dict."""
    from atlas_toolkit.io.ff import load_ff
    from atlas_toolkit.io.ff_detect import suggest_ff_files

    if ff_specs is None:
        return None   # caller will auto-detect

    if isinstance(ff_specs, (str, Path)):
        ff_specs = [ff_specs]

    return load_ff([str(s) for s in ff_specs])


def _get_lj(ff_parms: dict, fft: str):
    """Return (epsilon, sigma) for fft from ff_parms, or (0, 0) if missing."""
    entry = ff_parms.get("VDW", {}).get(fft, {}).get(fft, {}).get(1)
    if entry and entry.get("VALS"):
        return float(entry["VALS"][0]), float(entry["VALS"][1])
    return 0.0, 0.0


# ── main function ──────────────────────────────────────────────────────────────

def bgf_ff_to_parmed(
    bgf_path: Union[str, Path],
    ff_specs: Union[None, str, Path, list] = None,
    *,
    verbose: bool = False,
) -> tuple:
    """Load a BGF file and FF parameters into a fully parameterized ParmEd Structure.

    Parameters
    ----------
    bgf_path  : path to the BGF file
    ff_specs  : FF file path(s) (.ff or .frcmod).  If None, auto-detected
                from the FF types present in the BGF.
    verbose   : print progress

    Returns
    -------
    (struct, box, report)
    struct  : pmd.Structure with all bonded/nonbonded parameters populated
    box     : atlas-toolkit box dict (xlo/xhi/ylo/yhi/zlo/zhi)
    report  : dict with missing_bonds, missing_angles, missing_torsions lists
    """
    import parmed as pmd
    from parmed.topologyobjects import (
        AtomType, BondType, AngleType, DihedralType, ImproperType,
        Bond, Angle, Dihedral, Improper,
    )
    from parmed.periodic_table import AtomicNum, Mass

    from atlas_toolkit.io.bgf import read_bgf
    from atlas_toolkit.io.bgf_parmed import bgf_to_parmed
    from atlas_toolkit.core.box import get_box
    from atlas_toolkit.lammps.topology import (
        enumerate_bonds, enumerate_angles,
        enumerate_torsions, enumerate_impropers,
    )
    from atlas_toolkit.io.ff import (
        load_ff, lookup_bond, lookup_angle,
        lookup_torsion, lookup_inversion,
    )
    from atlas_toolkit.io.ff_detect import suggest_ff_files, detect_ff

    bgf_path = Path(bgf_path)

    # ── 1. Load BGF ────────────────────────────────────────────────────────────
    if verbose:
        print(f"Reading BGF: {bgf_path.name}")
    atoms_d, bonds_d, headers = read_bgf(bgf_path)
    box = get_box(atoms_d, headers)
    struct = bgf_to_parmed(atoms_d, bonds_d)

    if verbose:
        print(f"  {len(struct.atoms)} atoms, {len(struct.bonds)} bonds")

    # ── 2. Load FF ─────────────────────────────────────────────────────────────
    if ff_specs is None:
        observed = {a['FFTYPE'] for a in atoms_d.values()}
        hits = detect_ff(observed, min_confidence=0.0)
        ff_paths = [h.path for h in hits if h.path and h.confidence >= 0.5]
        if not ff_paths:
            raise RuntimeError(
                "FF auto-detection found no matches. "
                "Pass ff_specs explicitly."
            )
        if verbose:
            print(f"  Auto-detected FFs: {[p.name for p in ff_paths]}")
        ff_parms = load_ff([str(p) for p in ff_paths])
    else:
        ff_parms = _load_ff(ff_specs)
        if verbose:
            fnames = [Path(s).name for s in (
                [ff_specs] if isinstance(ff_specs, (str, Path)) else ff_specs)]
            print(f"  FF files: {fnames}")

    # ── 3. Build atom index map (bgf_index → ParmEd atom) ─────────────────────
    sorted_bgf_ids = sorted(atoms_d.keys())
    bgf_to_pmd: dict[int, pmd.Atom] = {
        bgf_id: struct.atoms[i]
        for i, bgf_id in enumerate(sorted_bgf_ids)
    }

    # ── 4. Inject LJ + mass + charge per atom ─────────────────────────────────
    at_parms = ff_parms.get("ATOMTYPES", {})
    for bgf_id, atom in zip(sorted_bgf_ids, struct.atoms):
        fft = atoms_d[bgf_id]["FFTYPE"]
        at_info = at_parms.get(fft, {})

        # Mass — prefer FF table over bgf_to_parmed's inferred value
        mass_ff = at_info.get("MASS")
        if mass_ff:
            atom.mass = float(mass_ff)

        # Charge — prefer BGF value (already set by bgf_to_parmed) unless FF overrides
        charge_ff = at_info.get("CHARGE")
        if charge_ff is not None and abs(float(charge_ff)) > 0.0:
            atom.charge = float(charge_ff)

        # LJ — create a fresh independent AtomType
        eps, sigma = _get_lj(ff_parms, fft)
        rmin = sigma * (2**(1/6)) if sigma > 0 else 0.0
        elem  = at_info.get("ATOM", fft.rstrip("+-0123456789"))
        anum  = AtomicNum.get(elem, atom.atomic_number or 0)
        new_at = AtomType(fft, None, atom.mass or 1.0, atomic_number=anum)
        if eps > 0 or rmin > 0:
            new_at.set_lj_params(eps, rmin)
        atom.atom_type = new_at

    # ── 5. Enumerate topology and look up parameters ───────────────────────────
    report = {
        "missing_bonds":    [],
        "missing_angles":   [],
        "missing_torsions": [],
        "missing_impropers": [],
    }

    def fft(bgf_id):
        return atoms_d[bgf_id]["FFTYPE"]

    # ── Bonds ──────────────────────────────────────────────────────────────────
    # bonds already set by bgf_to_parmed; we need to add BondType to each
    # Rebuild from bonds_d to ensure we have type info
    struct.bonds.clear()
    for (i, j) in enumerate_bonds(bonds_d):
        ti, tj = fft(i), fft(j)
        entry = lookup_bond(ti, tj, ff_parms)
        if entry is None:
            report["missing_bonds"].append((ti, tj))
            bt = BondType(0.0, 1.5)   # placeholder so topology is intact
        else:
            k, req = float(entry["VALS"][0]), float(entry["VALS"][1])
            bt = BondType(k, req)
        struct.bonds.append(Bond(bgf_to_pmd[i], bgf_to_pmd[j], type=bt))

    # ── Angles ─────────────────────────────────────────────────────────────────
    for (i, j, k) in enumerate_angles(bonds_d):
        ti, tj, tk = fft(i), fft(j), fft(k)
        entry = lookup_angle(ti, tj, tk, ff_parms)
        if entry is None:
            report["missing_angles"].append((ti, tj, tk))
            at = AngleType(0.0, 109.5)
        else:
            kc, theta = float(entry["VALS"][0]), float(entry["VALS"][1])
            at = AngleType(kc, theta)
        struct.angles.append(
            Angle(bgf_to_pmd[i], bgf_to_pmd[j], bgf_to_pmd[k], type=at)
        )

    # ── Proper torsions ────────────────────────────────────────────────────────
    for (i, j, k, l) in enumerate_torsions(bonds_d):
        ti, tj, tk, tl = fft(i), fft(j), fft(k), fft(l)
        terms = lookup_torsion(ti, tj, tk, tl, ff_parms)
        if terms is None:
            report["missing_torsions"].append((ti, tj, tk, tl))
            continue
        for term in terms:
            phi_k = float(term["VALS"][0])
            n     = float(term["VALS"][1])
            phase = float(term["VALS"][2])   # degrees (atlas-toolkit convention)
            dt = DihedralType(phi_k, int(round(abs(n))), phase)
            struct.dihedrals.append(
                Dihedral(bgf_to_pmd[i], bgf_to_pmd[j],
                         bgf_to_pmd[k], bgf_to_pmd[l],
                         type=dt, improper=False)
            )

    # ── Impropers ──────────────────────────────────────────────────────────────
    for (center, a, b, c) in enumerate_impropers(bonds_d):
        tc = fft(center); ta, tb, tv = fft(a), fft(b), fft(c)
        terms = lookup_inversion(tc, ta, tb, tv, ff_parms)
        if terms is None:
            continue   # impropers often have no FF entry; skip silently
        for term in terms:
            ki    = float(term["VALS"][0])
            phase = float(term["VALS"][1])   # degrees
            ni    = float(term["VALS"][2])
            dt = DihedralType(ki, int(round(abs(ni))), phase)
            struct.dihedrals.append(
                Dihedral(bgf_to_pmd[center], bgf_to_pmd[a],
                         bgf_to_pmd[b], bgf_to_pmd[c],
                         type=dt, improper=True)
            )

    if verbose:
        print(f"  bonds={len(struct.bonds)}, angles={len(struct.angles)}, "
              f"dihedrals={len(struct.dihedrals)}")
        if report["missing_bonds"]:
            print(f"  ⚠ missing bonds:    {len(set(report['missing_bonds']))}")
        if report["missing_angles"]:
            print(f"  ⚠ missing angles:   {len(set(report['missing_angles']))}")
        if report["missing_torsions"]:
            print(f"  ⚠ missing torsions: {len(set(report['missing_torsions']))}")
        else:
            print("  ✓ all parameters found")

    return struct, box, report


def bgf_ff_to_lammps(
    bgf_path: Union[str, Path],
    ff_specs: Union[None, str, Path, list] = None,
    *,
    output_stem: Optional[Union[str, Path]] = None,
    protocol: str = "min",
    cutoff: float = 12.0,
    seed: int = 12345,
    verbose: bool = True,
) -> None:
    """Load a BGF + FF files and write LAMMPS data.* and in.* directly.

    Parameters
    ----------
    bgf_path    : path to input BGF
    ff_specs    : FF file(s) — None for auto-detect
    output_stem : stem for output files (default: BGF basename in same dir)
    protocol    : 'min', 'nvt', or 'npt'
    cutoff      : LJ/coul cutoff in Å
    verbose     : print progress
    """
    from atlas_toolkit.lammps.data_file_parmed import write_data_file_parmed, DataFileSummaryP
    from atlas_toolkit.lammps.input_script import write_input_script

    bgf_path = Path(bgf_path)
    if output_stem is None:
        output_stem = bgf_path.parent / bgf_path.stem
    output_stem = Path(output_stem)
    data_path = str(output_stem.parent / f"data.{output_stem.name}")
    in_path   = str(output_stem.parent / f"in.{output_stem.name}")

    struct, box, report = bgf_ff_to_parmed(bgf_path, ff_specs, verbose=verbose)

    if verbose:
        print(f"Writing {data_path} ...")
    summary = write_data_file_parmed(
        data_path, struct, box,
        title=f"Generated from {bgf_path.name}",
    )

    if verbose:
        print(f"  atoms={summary.n_atoms} ({summary.n_atom_types} types), "
              f"bonds={summary.n_bonds}, angles={summary.n_angles}, "
              f"dihedrals={summary.n_dihedrals}, impropers={summary.n_impropers}")
        if summary.missing_bonds or summary.missing_torsions:
            print(f"  ⚠ missing bonds={len(summary.missing_bonds)}, "
                  f"torsions={len(summary.missing_torsions)}")
        else:
            print("  ✓ no missing parameters")
        print(f"Writing {in_path} ...")

    write_input_script(
        in_path, data_path, summary, {}, box,
        protocol=protocol, cutoff=cutoff, seed=seed,
        title=f"Generated from {bgf_path.name}",
    )

    if verbose:
        print("Done.")
