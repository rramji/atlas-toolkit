"""
embed_molecule — port of embedMolecule.pl

Embed a solute into a solvent/membrane structure.  Optionally removes
solvent molecules that overlap with the solute (within 3 Å cutoff).

Usage:
  atlas-embed-molecule -s solute.bgf -m solvent.bgf -w embedded.bgf
  atlas-embed-molecule -s solute.bgf -m solvent.bgf -c com -o 1 -r 0
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.box import cart2frac, get_box, init_box
from atlas_toolkit.core.general import com as _com, file_tester
from atlas_toolkit.core.headers import add_box_to_header, insert_header_remark
from atlas_toolkit.core.manip_atoms import get_atm_data, get_mols
from atlas_toolkit.core.replicate import combine_mols
from atlas_toolkit.io.bgf import get_bgf_atoms, make_seq_atom_index, parse_struct_file, write_bgf

OVERLAP_CUTOFF = 3.0  # Å — hard-coded in the Perl original


# ── core embed logic ──────────────────────────────────────────────────────────

def embed_molecule(
    solu_atoms: dict,
    solu_bonds: dict,
    solv_atoms: dict,
    solv_bonds: dict,
    box: dict,
    center: bool = True,
    check_overlap: bool = True,
    reverse_place: bool = False,
) -> tuple[dict, dict]:
    """Embed solute into solvent; optionally remove overlapping solvent molecules.

    Port of embedMolecule.pl main logic.

    Parameters
    ----------
    solu_atoms, solu_bonds : solute system
    solv_atoms, solv_bonds : solvent system (provides the box dimensions)
    box                    : box dict initialised from solvent headers
    center                 : if True, shift solute CoM onto solvent CoM
    check_overlap          : if True, remove solvent molecules within OVERLAP_CUTOFF Å of any solute atom
    reverse_place          : if True, combine as solvent-first then solute (default: solute-first)

    Returns
    -------
    (atoms, bonds) renumbered sequentially
    """
    n_solu = len(solu_atoms)

    if center:
        _center_mols(solu_atoms, solv_atoms)

    if not reverse_place:
        atoms, bonds = combine_mols(solu_atoms, solu_bonds, solv_atoms, solv_bonds)
    else:
        atoms, bonds = combine_mols(solv_atoms, solv_bonds, solu_atoms, solu_bonds)

    init_box(box, atoms)

    if check_overlap:
        n_removed = _remove_overlaps(atoms, bonds, n_solu, box, reverse_place)
        print(f"Removed {n_removed} overlapping solvent molecule(s).")

    atoms, bonds = make_seq_atom_index(atoms, bonds)
    return atoms, bonds


def _center_mols(solu_atoms: dict, solv_atoms: dict) -> None:
    """Shift solute CoM onto solvent CoM.  Port of embedMolecule.pl::centerMols."""
    solu_com = _com(solu_atoms)
    solv_com = _com(solv_atoms)
    for dim in ("X", "Y", "Z"):
        coord = f"{dim}COORD"
        offset = float(solv_com[coord]) - float(solu_com[coord])
        for a in solu_atoms.values():
            a[coord] = float(a[coord]) + offset


def _min_image_dist(a: dict, b: dict, box: dict) -> float:
    """Minimum-image Cartesian distance between two points (both in XCOORD/YCOORD/ZCOORD)."""
    dist2 = 0.0
    for dim, coord in (("X", "XCOORD"), ("Y", "YCOORD"), ("Z", "ZCOORD")):
        d = abs(float(a[coord]) - float(b[coord]))
        bl = float(box[dim]["len"])
        if d > bl * 0.5:
            d = bl - d
        dist2 += d * d
    return math.sqrt(dist2)


def _remove_overlaps(
    atoms: dict,
    bonds: dict,
    n_solu: int,
    box: dict,
    reverse_place: bool,
) -> int:
    """Remove solvent molecules whose CoM is within OVERLAP_CUTOFF of any solute atom.

    Uses a fractional-coord grid (cell spacing ≈ 2.5 Å) to avoid O(N²).
    Port of embedMolecule.pl::removeOverlaps.

    Returns the number of solvent molecules removed.
    """
    total = max(atoms) if atoms else 0

    if not reverse_place:
        solu_indices = {i for i in atoms if i <= n_solu}
        solv_set = {i: 1 for i in atoms if i > n_solu}
    else:
        solu_indices = {i for i in atoms if i > (total - n_solu)}
        solv_set = {i: 1 for i in atoms if i <= (total - n_solu)}

    # Shallow-copy atom dicts so cart2frac annotations don't pollute the originals
    tmp = {i: copy.copy(atoms[i]) for i in atoms}
    cart2frac(tmp, box)

    # Grid spacing in fractional units — ≈ 2.5 Å per cell
    gs = {dim: 2.5 / float(box[dim]["len"]) for dim in ("X", "Y", "Z")}

    # Build grid of solute atoms (indexed by fractional-coord cell)
    grid: dict[int, dict[int, dict[int, set[int]]]] = {}
    for i in solu_indices:
        if i not in tmp:
            continue
        a = tmp[i]
        ix = int(float(a.get("FA", 0.0)) / gs["X"])
        iy = int(float(a.get("FB", 0.0)) / gs["Y"])
        iz = int(float(a.get("FC", 0.0)) / gs["Z"])
        grid.setdefault(ix, {}).setdefault(iy, {}).setdefault(iz, set()).add(i)

    # Identify solvent molecules
    solv_mols = get_mols(atoms, bonds, solv_set)

    # Track which atoms to keep
    keep: set[int] = set(atoms.keys())
    n_removed = 0

    for mol in solv_mols.values():
        members = mol["MEMBERS"]
        mol_tmp = get_atm_data(tmp, members)
        if not mol_tmp:
            continue

        # Solvent molecule CoM in Cartesian
        mol_com = _com(mol_tmp)

        # Convert CoM to fractional for grid lookup
        com_frac = copy.copy(mol_com)
        cart2frac({1: com_frac}, box)

        gix = int(float(com_frac.get("FA", 0.0)) / gs["X"])
        giy = int(float(com_frac.get("FB", 0.0)) / gs["Y"])
        giz = int(float(com_frac.get("FC", 0.0)) / gs["Z"])

        overlap_found = False
        for di in range(-1, 2):
            if overlap_found:
                break
            nx = gix + di
            if nx not in grid:
                continue
            for dj in range(-1, 2):
                if overlap_found:
                    break
                ny = giy + dj
                if ny not in grid[nx]:
                    continue
                for dk in range(-1, 2):
                    nz = giz + dk
                    if nz not in grid[nx][ny]:
                        continue
                    for solu_idx in grid[nx][ny][nz]:
                        if solu_idx not in keep:
                            continue
                        d = _min_image_dist(mol_com, atoms[solu_idx], box)
                        if d < OVERLAP_CUTOFF:
                            overlap_found = True
                            break
                    if overlap_found:
                        break

        if overlap_found:
            n_removed += 1
            for j in members:
                keep.discard(j)

    # Remove atoms/bonds not in keep
    for i in list(atoms.keys()):
        if i not in keep:
            atoms.pop(i)
            bonds.pop(i, None)

    return n_removed


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    file_tester(args.solute)
    file_tester(args.membrane)

    print(f"Parsing solvent/membrane structure {args.membrane}...")
    solv_atoms, solv_bonds, solv_headers = parse_struct_file(args.membrane, save_headers=True)
    print(f"Parsing solute structure {args.solute}...")
    solu_atoms, solu_bonds, _ = parse_struct_file(args.solute, save_headers=True)
    print("Done")

    box = get_box(solv_atoms, solv_headers)
    do_center = args.center.upper() != "NONE"
    do_overlap = args.overlap

    print("Embedding systems...")
    atoms, bonds = embed_molecule(
        solu_atoms, solu_bonds,
        solv_atoms, solv_bonds,
        box,
        center=do_center,
        check_overlap=do_overlap,
        reverse_place=args.reverse,
    )
    print("Done")

    save_path = args.save or _default_save(args.solute)
    print(f"Creating structure file {save_path}...")
    insert_header_remark(solv_headers, f"REMARK {args.solute} embedded in {args.membrane}")
    add_box_to_header(solv_headers, box)
    write_bgf(atoms, bonds, save_path, solv_headers)
    print("Done")


def _default_save(solu_path: str) -> str:
    p = Path(solu_path)
    return str(p.parent / (p.stem + "_embed.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a solute into a solvent/membrane structure.",
        epilog=__doc__,
    )
    parser.add_argument("-s", "--solute", required=True, help="Solute structure file (BGF)")
    parser.add_argument("-m", "--membrane", required=True,
                        help="Solvent/membrane structure file (BGF)")
    parser.add_argument("-w", "--save", default=None, help="Output file name")
    parser.add_argument("-c", "--center", default="com",
                        help="Centering mode: com (default) | none")
    parser.add_argument("-o", "--overlap", type=lambda x: x.lower() not in ("0", "no"),
                        default=True, help="Remove overlapping solvent? 1/yes (default) or 0/no")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Reverse placement: put solvent first, solute second")
    return parser.parse_args()


if __name__ == "__main__":
    main()
