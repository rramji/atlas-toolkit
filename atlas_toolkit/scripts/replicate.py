"""
replicate — port of replicate.pl

Replicates a periodic unit cell and optionally randomly rotates each molecule.

Usage:
  atlas-replicate -b struct.bgf -d "2 2 1" -s out.bgf
  atlas-replicate -b struct.bgf -d "3 3 3" -r -u -c

Flags mirror the Perl original:
  -b / --bgf         Input structure file (BGF)
  -d / --dims        Replication string  "X Y Z"  e.g. "2 2 1"
  -f / --ff          Force field(s) — accepted but currently unused
                     (bond detection reads from the BGF directly)
  -s / --save        Output file name (default: <stem>.<X>x<Y>x<Z>.bgf)
  -c / --center      Centre the system before replication
  -r / --rotate      Randomly rotate each molecule after replication
  -u / --update-res  Update residue numbers per molecule
  -i / --pbc-bonds   Create PBC bonds (default: True)
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.box import get_box
from atlas_toolkit.core.general import com
from atlas_toolkit.core.headers import add_box_to_header, insert_header_remark
from atlas_toolkit.core.manip_atoms import get_mols
from atlas_toolkit.core.replicate import replicate_cell, rotate
from atlas_toolkit.io.bgf import parse_struct_file, write_bgf


def main() -> None:
    args = _parse_args()
    dims = _parse_dims(args.dims)

    print("Initializing...")
    print("Done")

    print(f"Parsing structure file {args.bgf}...")
    atoms, bonds, headers = parse_struct_file(args.bgf, save_headers=True)
    box = get_box(atoms, headers)
    print("Done")

    rep_str = f"{dims['X']}x{dims['Y']}x{dims['Z']}"
    print(f"Replicating cell {rep_str}...")
    atoms, bonds, box = replicate_cell(
        atoms, bonds, box, dims,
        center=args.center,
        update_res_num=args.update_res,
        pbc=args.pbc_bonds,
    )
    print("Done")

    if args.update_res or args.rotate:
        mols = get_mols(atoms, bonds)
        if args.update_res:
            _update_res_by_mol(atoms, mols)
        if args.rotate:
            print("Randomly rotating molecules...")
            _rotate_mols(atoms, mols)
            print("Done")

    save_path = args.save or _default_save(args.bgf, rep_str)
    print(f"Creating {save_path}...")
    insert_header_remark(headers, f"REMARK {args.bgf} replicated {rep_str}")
    add_box_to_header(headers, box)
    write_bgf(atoms, bonds, save_path, headers)
    print("Done")


# ── helpers ──────────────────────────────────────────────────────────────────

def _rotate_mols(atoms: dict, mols: dict) -> None:
    """Randomly rotate each molecule about its centre of mass.

    Uses the fixed com() that returns a copy for single-atom molecules,
    so monoatomic ions rotate (trivially) without aliasing bugs.
    """
    for mol in mols.values():
        members = mol["MEMBERS"]
        mol_atoms = {idx: atoms[idx] for idx in members if idx in atoms}

        angles = [2 * math.pi * random.random() for _ in range(3)]
        centre = com(mol_atoms)

        # Translate to origin
        for a in mol_atoms.values():
            a["XCOORD"] = float(a["XCOORD"]) - centre["XCOORD"]
            a["YCOORD"] = float(a["YCOORD"]) - centre["YCOORD"]
            a["ZCOORD"] = float(a["ZCOORD"]) - centre["ZCOORD"]

        rotate(mol_atoms, angles, coord=3)

        # Translate back
        for a in mol_atoms.values():
            a["XCOORD"] = float(a["XCOORD"]) + centre["XCOORD"]
            a["YCOORD"] = float(a["YCOORD"]) + centre["YCOORD"]
            a["ZCOORD"] = float(a["ZCOORD"]) + centre["ZCOORD"]


def _update_res_by_mol(atoms: dict, mols: dict) -> None:
    """Set RESNUM = molecule id for every atom.  Port of replicate.pl::updateResByMol."""
    for a in atoms.values():
        mid = a.get("MOLECULEID")
        if mid is not None:
            a["RESNUM"] = int(mid)


def _parse_dims(s: str) -> dict[str, int]:
    parts = s.split()
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"--dims expects three integers e.g. '2 2 1', got: {s!r}")
    x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
    if x == y == z == 1:
        raise ValueError("ERROR: Need at least one dimension > 1")
    return {"X": x, "Y": y, "Z": z}


def _default_save(bgf_path: str, rep_str: str) -> str:
    p = Path(bgf_path)
    return str(p.parent / (p.stem + f".{rep_str}.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate a periodic BGF unit cell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bgf", required=True, help="Input BGF structure file")
    parser.add_argument("-d", "--dims", required=True,
                        help='Replication string e.g. "2 2 1"')
    parser.add_argument("-f", "--ff", default=None,
                        help="Force field(s) — accepted, currently unused")
    parser.add_argument("-s", "--save", default=None, help="Output file name")
    parser.add_argument("-c", "--center", action="store_true",
                        help="Centre system before replication")
    parser.add_argument("-r", "--rotate", action="store_true",
                        help="Randomly rotate each molecule after replication")
    parser.add_argument("-u", "--update-res", action="store_true",
                        help="Update residue numbers per molecule")
    parser.add_argument("-i", "--pbc-bonds", action="store_true", default=True,
                        help="Create PBC bonds (default: on)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
