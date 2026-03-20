"""
bgf_to_mol2 — convert a BGF structure file to Tripos MOL2 format.

Usage:
  atlas-bgf-to-mol2 -b struct.bgf -s out.mol2
  atlas-bgf-to-mol2 -b struct.bgf            # writes struct.mol2
  atlas-bgf-to-mol2 -b struct.bgf -o "resname eq LIG"   # subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.manip_atoms import add_mols_to_selection, get_mols, select_atoms
from atlas_toolkit.io.bgf import get_bgf_atoms, parse_struct_file
from atlas_toolkit.io.mol2 import write_mol2


def bgf_to_mol2(
    bgf_path: str,
    save_path: str | None = None,
    selection: str | None = None,
    mol_opt: bool = False,
    title: str | None = None,
) -> None:
    """Convert BGF → MOL2.

    Parameters
    ----------
    bgf_path  : input BGF file
    save_path : output .mol2 path (default: same stem, .mol2 extension)
    selection : optional atom selection string to write a subset
    mol_opt   : expand selection to whole molecules
    title     : molecule name written to @<TRIPOS>MOLECULE
    """
    p = Path(bgf_path)
    save = Path(save_path) if save_path else p.with_suffix(".mol2")

    print(f"Reading {p}...")
    atoms, bonds, _ = parse_struct_file(str(p), save_headers=True)
    get_mols(atoms, bonds)
    print(f"  {len(atoms)} atoms, {sum(len(v) for v in bonds.values()) // 2} bonds")

    if selection:
        print(f"Selecting atoms: {selection}")
        sel = select_atoms(selection, atoms)
        if not sel:
            sys.exit("ERROR: No atoms matched selection")
        if mol_opt:
            add_mols_to_selection(sel, atoms)
        atoms, bonds = get_bgf_atoms(sel, atoms, bonds)
        print(f"  {len(atoms)} atoms selected")

    mol_title = title or p.stem
    print(f"Writing {save}...")
    write_mol2(atoms, bonds, save, title=mol_title)
    print("Done.")


def main() -> None:
    args = _parse_args()
    bgf_to_mol2(
        bgf_path=args.bgf,
        save_path=args.save,
        selection=args.selection,
        mol_opt=args.mol_opt,
        title=args.title,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a BGF structure file to Tripos MOL2 format.",
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bgf", required=True, help="Input BGF file")
    parser.add_argument("-s", "--save", default=None, help="Output .mol2 file")
    parser.add_argument("-o", "--selection", default=None,
                        help='Optional atom selection e.g. "resname eq LIG"')
    parser.add_argument("-m", "--mol-opt", action="store_true",
                        help="Expand selection to whole molecules")
    parser.add_argument("--title", default=None,
                        help="Molecule name in @<TRIPOS>MOLECULE (default: BGF stem)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
