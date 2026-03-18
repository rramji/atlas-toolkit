"""
get_bgf_atoms — port of getBGFAtoms.pl

Extract a subset of atoms from a BGF file by atom selection, with optional
whole-molecule expansion.

Usage:
  atlas-get-bgf-atoms -b struct.bgf -o "resname eq WAT" -s out.bgf
  atlas-get-bgf-atoms -b struct.bgf -o "fftype eq Au" -m -s gold.bgf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.general import file_tester
from atlas_toolkit.core.manip_atoms import (
    add_mols_to_selection,
    get_mols,
    select_atoms,
)
from atlas_toolkit.core.headers import insert_header_remark
from atlas_toolkit.io.bgf import get_bgf_atoms, parse_struct_file, write_bgf


def main() -> None:
    args = _parse_args()
    file_tester(args.bgf)

    print(f"Getting atom information from {args.bgf}...")
    atoms, bonds, headers = parse_struct_file(args.bgf, save_headers=True)
    get_mols(atoms, bonds)
    print("Done")

    print("Selecting relevant atoms...")
    selection = select_atoms(args.selection, atoms)
    if not selection:
        sys.exit("ERROR: No atoms matched selection")

    if args.mol_opt:
        add_mols_to_selection(selection, atoms)

    sub_atoms, sub_bonds = get_bgf_atoms(selection, atoms, bonds)

    save = args.save or _default_save(args.bgf)
    print(f"Done\nCreating structure file {save}...")
    insert_header_remark(headers, f"REMARK selected atoms: {args.selection}")
    write_bgf(sub_atoms, sub_bonds, save, headers)
    print("Done")


def _default_save(bgf: str) -> str:
    p = Path(bgf)
    return str(p.parent / (p.stem + "_mod.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a subset of atoms from a BGF file.",
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bgf", required=True, help="Input BGF file")
    parser.add_argument("-o", "--selection", required=True,
                        help='Atom selection string e.g. "resname eq WAT"')
    parser.add_argument("-s", "--save", default=None, help="Output file name")
    parser.add_argument("-m", "--mol-opt", action="store_true",
                        help="Expand selection to include entire molecules")
    parser.add_argument("-f", "--ff", default=None,
                        help="Force field file(s) (currently unused for selection)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
