"""
get_bounds — port of getBounds.pl (static BGF mode)

Print the XYZ coordinate min/max for a selected set of atoms in a BGF file.

Usage:
  atlas-get-bounds -b struct.bgf
  atlas-get-bounds -b struct.bgf -o "resname eq WAT"
  atlas-get-bounds -b struct.bgf -o "fftype eq Au" --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.general import file_tester
from atlas_toolkit.core.manip_atoms import get_bounds, get_mols, select_atoms
from atlas_toolkit.io.bgf import parse_struct_file


def main() -> None:
    args = _parse_args()
    file_tester(args.bgf)

    print(f"Getting atom information from {args.bgf}...", file=sys.stderr)
    atoms, bonds, _ = parse_struct_file(args.bgf)
    get_mols(atoms, bonds)
    print("Done", file=sys.stderr)

    sel_str = args.selection or "index > 0"
    print(f"Selecting relevant atoms ({sel_str})...", file=sys.stderr)
    selection = select_atoms(sel_str, atoms)
    if not selection:
        sys.exit("ERROR: No atoms matched selection")
    print("Done", file=sys.stderr)

    bounds = get_bounds(atoms, selection)

    if args.json:
        print(json.dumps(bounds, indent=2))
    else:
        print("ATOM Bounds")
        for dim in ("X", "Y", "Z"):
            b = bounds[dim]
            print(f"{dim} {b['min']:.5f} {b['max']:.5f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print XYZ coordinate bounds for selected atoms.",
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bgf", required=True, help="Input BGF file")
    parser.add_argument("-o", "--selection", default=None,
                        help='Atom selection (default: all atoms)')
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of plain text")
    return parser.parse_args()


if __name__ == "__main__":
    main()
