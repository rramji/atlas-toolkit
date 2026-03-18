"""
combine_bgf — port of combineBGF.pl

Merge two or more BGF files into a single structure.  Atom indices are
renumbered sequentially; headers are taken from the last input file.

Usage:
  atlas-combine-bgf -i "file1.bgf file2.bgf" -s combined.bgf
  atlas-combine-bgf -i "a.bgf b.bgf c.bgf" -s out.bgf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.general import file_tester
from atlas_toolkit.core.replicate import combine_mols
from atlas_toolkit.io.bgf import parse_struct_file, write_bgf


def combine_bgf(input_files: list[str]) -> tuple[dict, dict, list]:
    """Merge a list of BGF files into one.

    Returns (atoms, bonds, headers) where headers come from the last file.
    Port of combineBGF.pl::readStructs.
    """
    atoms: dict = {}
    bonds: dict = {}
    headers: list = []

    for f in input_files:
        print(f"Parsing structure file {f}...")
        f_atoms, f_bonds, f_headers = parse_struct_file(f, save_headers=True)
        if not atoms:
            atoms = f_atoms
            bonds = f_bonds
        else:
            atoms, bonds = combine_mols(atoms, bonds, f_atoms, f_bonds)
        headers = f_headers  # keep last file's headers (matches Perl)
        print("Done")

    return atoms, bonds, headers


def main() -> None:
    args = _parse_args()

    files = args.input.split()
    for f in files:
        file_tester(f)

    atoms, bonds, headers = combine_bgf(files)

    save = args.save or _default_save(files[0])
    print(f"Creating {save}...")
    write_bgf(atoms, bonds, save, headers)
    print("Done")


def _default_save(first: str) -> str:
    p = Path(first)
    return str(p.parent / (p.stem + ".combined.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple BGF files into one.",
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input", required=True,
                        help='Space-separated list of BGF files e.g. "a.bgf b.bgf"')
    parser.add_argument("-s", "--save", default=None, help="Output file name")
    parser.add_argument("-f", "--ff", default=None,
                        help="Force field file(s) (currently unused)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
