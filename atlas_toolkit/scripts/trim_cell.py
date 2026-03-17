"""
trim_cell — port of trimCell.pl

Trim atoms outside a specified new cell.  Keeps whole molecules together.

Usage:
  atlas-trim-cell -b struct.bgf -c "30 30 30" -s trimmed.bgf
  atlas-trim-cell -b struct.bgf -c "20 20 40" --com -o 1 -m
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.box import cart2frac, get_box, get_box_displacement_tensor, init_box
from atlas_toolkit.core.general import com as _com, file_tester
from atlas_toolkit.core.headers import add_box_to_header, insert_header_remark
from atlas_toolkit.core.manip_atoms import (
    get_atm_data, get_mols, reimage_atoms, select_atoms,
)
from atlas_toolkit.io.bgf import make_seq_atom_index, parse_struct_file, write_bgf


# ── core trim logic ──────────────────────────────────────────────────────────

def trim_cell(
    atoms: dict,
    bonds: dict,
    mols: dict,
    new_box: dict,
    selection: dict | None = None,
    use_com: bool = False,
) -> None:
    """Remove molecules (or atoms) whose fractional coords in new_box lie outside [0,1).

    Port of trimCell.pl::trimCell.

    Parameters
    ----------
    atoms, bonds : modified in-place
    mols         : molecule dict from get_mols
    new_box      : target box (must already have H/F matrices from init_box)
    selection    : which atom indices are eligible for removal (None = all)
    use_com      : if True, use the molecule CoM instead of per-atom fractional coords
    """
    if selection is None:
        selection = {idx: 1 for idx in atoms}

    for mol in mols.values():
        members: dict = mol.get("MEMBERS", {})

        # Build a temporary dict with just this molecule's atoms (shallow copy of coords)
        if use_com:
            mol_atoms = get_atm_data(atoms, members)
            centre = _com(mol_atoms)
            curr = {j: centre for j in members}
        else:
            curr = {}
            for j in members:
                if j in atoms:
                    curr[j] = dict(atoms[j])   # copy coords; cart2frac modifies in-place

        cart2frac(curr, new_box)

        is_outside = False
        for j, a in curr.items():
            for fk in ("FA", "FB", "FC"):
                if float(a.get(fk, 0.0)) < 0 or float(a.get(fk, 0.0)) > 1:
                    is_outside = True
                    break
            if is_outside:
                break
            if use_com:
                break  # only check once (all point to same CoM)

        if not is_outside:
            continue

        for j in members:
            if j in selection:
                atoms.pop(j, None)
                bonds.pop(j, None)


def center_sys(atoms: dict, box: dict, start_origin: int = 0) -> None:
    """Shift all atoms so a reference point maps to the origin.

    Port of trimCell.pl::centerSys (numeric modes only).

    start_origin:
        0  — box center (default)
        1  — origin already (no shift)
        2  — box min
        3  — box max
    """
    get_box_displacement_tensor(box)

    if start_origin == 1:
        shift = {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0}
    elif start_origin == 2:
        shift = {f"{d}COORD": box[d]["lo"] for d in ("X", "Y", "Z")}
    elif start_origin == 3:
        shift = {f"{d}COORD": box[d]["hi"] for d in ("X", "Y", "Z")}
    else:  # 0 — box centre
        shift = {f"{d}COORD": box[d]["len"] / 2.0 for d in ("X", "Y", "Z")}

    for a in atoms.values():
        for coord, s in shift.items():
            a[coord] = float(a[coord]) - s


def make_atoms_mols(atoms: dict) -> dict:
    """Treat every atom as its own singleton molecule (no bonds required).

    Port of trimCell.pl::makeAtomsMols.
    """
    return {idx: {"MEMBERS": {idx: 1}, "SIZE": 1} for idx in atoms}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    file_tester(args.bgf)

    print(f"Parsing BGF file {args.bgf}...")
    atoms, bonds, headers = parse_struct_file(args.bgf, save_headers=True)
    box = get_box(atoms, headers)
    print("Done")

    if args.molecule:
        mols = get_mols(atoms, bonds)
    else:
        mols = make_atoms_mols(atoms)

    # Selection (default: all atoms eligible)
    selection: dict
    if args.atom_sel:
        selection = select_atoms(args.atom_sel, atoms)
    else:
        selection = {idx: 1 for idx in atoms}

    print("Removing atoms outside new box...")
    center_sys(atoms, box, args.origin)
    init_box(box, atoms)

    reimage_atoms(atoms, bonds, mols, box, selection)

    # Build new cell from -c string
    new_box = _parse_cell(args.cell, args.padding)
    init_box(new_box, atoms, args.padding)

    trim_cell(atoms, bonds, mols, new_box, selection, args.com)
    atoms, bonds = make_seq_atom_index(atoms, bonds)

    save_path = args.save or _default_save(args.bgf, args.cell)
    print(f"Creating BGF file {save_path}...")
    cell_str = args.cell.replace(" ", "x")
    insert_header_remark(headers, f"REMARK {args.bgf} trimmed {cell_str}")
    add_box_to_header(headers, new_box)
    write_bgf(atoms, bonds, save_path, headers)
    print("Done")


def _parse_cell(cell_str: str, padding: float = 0.0) -> dict:
    parts = cell_str.split()
    if len(parts) < 3:
        raise ValueError(f"--cell expects at least 3 values (x y z), got: {cell_str!r}")
    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    alpha = float(parts[3]) if len(parts) > 3 else 90.0
    beta  = float(parts[4]) if len(parts) > 4 else 90.0
    gamma = float(parts[5]) if len(parts) > 5 else 90.0
    return {
        "X": {"hi": x, "lo": 0.0, "len": x, "angle": alpha},
        "Y": {"hi": y, "lo": 0.0, "len": y, "angle": beta},
        "Z": {"hi": z, "lo": 0.0, "len": z, "angle": gamma},
    }


def _default_save(bgf_path: str, cell_str: str) -> str:
    p = Path(bgf_path)
    tag = cell_str.replace(" ", "x")
    return str(p.parent / (p.stem + f"_trim_{tag}.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim atoms outside a new cell.",
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bgf", required=True, help="Input BGF file")
    parser.add_argument("-c", "--cell", required=True,
                        help='New cell dimensions "x y z [alpha beta gamma]"')
    parser.add_argument("-s", "--save", default=None, help="Output file name")
    parser.add_argument("-p", "--padding", type=float, default=0.0,
                        help="Padding subtracted from cell dimension")
    parser.add_argument("-a", "--atom-sel", default=None,
                        help="Atom selection (default: all)")
    parser.add_argument("-o", "--origin", type=int, default=0,
                        help="Centering: 0=box center (default), 1=origin, 2=box min, 3=box max")
    parser.add_argument("-m", "--molecule", action="store_true",
                        help="Keep molecules together")
    parser.add_argument("--com", action="store_true",
                        help="Use molecule CoM to decide inside/outside")
    return parser.parse_args()


if __name__ == "__main__":
    main()
