"""
remove_mols — port of removeMols.pl

Remove whole molecules from a BGF file based on atom selection.

Usage:
  atlas-remove-mols -b struct.bgf -a "resname eq WAT" -s out.bgf
  atlas-remove-mols -b struct.bgf -a "resname eq WAT" -m 50 -r
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.general import file_tester
from atlas_toolkit.core.manip_atoms import get_mols, select_atoms
from atlas_toolkit.io.bgf import make_seq_atom_index, parse_struct_file, write_bgf


def remove_mols(
    atoms: dict,
    bonds: dict,
    selection: dict,
    max_atoms: int | None = None,
    max_mols: int | None = None,
    randomize: bool = False,
) -> tuple[int, int]:
    """Delete whole molecules that contain any selected atom.

    Port of removeMols.pl::removeMols.

    Parameters
    ----------
    atoms, bonds : modified in-place
    selection    : {atom_idx: ...} — atoms whose molecules should be removed
    max_atoms    : stop after removing this many atoms (optional)
    max_mols     : stop after removing this many molecules (optional)
    randomize    : shuffle selection order before processing

    Returns
    -------
    (n_atoms_removed, n_mols_removed)
    """
    indices = sorted(selection, reverse=True)
    if randomize:
        random.shuffle(indices)

    atom_count = mol_count = 0
    stop = False

    for i in indices:
        if stop or i not in atoms:
            continue
        if max_mols is not None and mol_count >= max_mols:
            break
        mol_members = list(atoms[i].get("MOLECULE", {}).get("MEMBERS", {i: 1}))
        removed_any = False
        for j in mol_members:
            if j in atoms:
                del atoms[j]
                bonds.pop(j, None)
                atom_count += 1
                removed_any = True
                if max_atoms is not None and atom_count >= max_atoms:
                    stop = True
                    break
        if removed_any:
            mol_count += 1
        if stop:
            break
        if max_mols is not None and mol_count >= max_mols:
            break

    if not atoms:
        raise RuntimeError("ERROR: No atoms remain after deletion!")

    return atom_count, mol_count


def main() -> None:
    args = _parse_args()
    file_tester(args.bgf)

    print(f"Getting atom information from {args.bgf}...")
    atoms, bonds, headers = parse_struct_file(args.bgf, save_headers=True)
    get_mols(atoms, bonds)
    print("Done")

    print("Getting selected atoms...")
    selected = select_atoms(args.atom_sel, atoms)
    print(f"Done\nRemoving {len(selected)} atoms...")

    n_atoms, n_mols = remove_mols(
        atoms, bonds, selected,
        max_atoms=args.num_atoms,
        max_mols=args.num_mols,
        randomize=args.randomize,
    )
    print(f"removed {n_atoms} atoms ({n_mols} mols)...")

    atoms, bonds = make_seq_atom_index(atoms, bonds)

    save_path = args.save or _default_save(args.bgf)
    print(f"Creating BGF file {save_path}...")
    write_bgf(atoms, bonds, save_path, headers)
    print("Done")


def _default_save(bgf_path: str) -> str:
    p = Path(bgf_path)
    return str(p.parent / (p.stem + "_mod.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove whole molecules from a BGF file.",
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bgf", required=True, help="Input BGF file")
    parser.add_argument("-a", "--atom-sel", required=True,
                        help='Atom selection string e.g. "resname eq WAT"')
    parser.add_argument("-s", "--save", default=None, help="Output file name")
    parser.add_argument("-n", "--num-atoms", type=int, default=None,
                        help="Stop after removing this many atoms")
    parser.add_argument("-m", "--num-mols", type=int, default=None,
                        help="Stop after removing this many molecules")
    parser.add_argument("-r", "--randomize", action="store_true",
                        help="Randomize selection order")
    return parser.parse_args()


if __name__ == "__main__":
    main()
