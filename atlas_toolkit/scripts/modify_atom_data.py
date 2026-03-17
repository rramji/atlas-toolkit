"""
modify_atom_data — port of modifyAtomData.pl

Reads a structure file, selects atoms, modifies specified fields, writes output.

Usage (mirrors Perl flags):
  atlas-modify-atom-data -s struct.bgf -a "fftype eq Cl-" -f "CHARGE:-1.0" -w out.bgf
  atlas-modify-atom-data -s struct.bgf -a "*" -f "RESNAME:RES CHARGE:+0.5"

Field spec format:  FIELD[:MOD]VALUE
  No modifier : assign         CHARGE:-1.0    → CHARGE = -1.0
  +           : add            CHARGE:+0.5    → CHARGE += 0.5
  -           : subtract       CHARGE:-0.1    → CHARGE -= 0.1
  .           : string append  RESNAME:.X     → RESNAME += "X"
"""
import argparse
import copy
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Allow running as a script without installing the package
if __name__ == "__main__":
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.general import file_tester, has_cell
from atlas_toolkit.core.headers import insert_header_remark
from atlas_toolkit.core.manip_atoms import (
    add_mols_to_selection,
    build_selection,
    get_mols,
    select_atoms,
)
from atlas_toolkit.io.bgf import parse_struct_file, write_bgf
from atlas_toolkit.types import AtomsDict, BondsDict, HeadersList


@dataclass
class FieldSpec:
    mod: str   # "", "+", "-", "."
    val: str   # raw value string


def main() -> None:
    args = _parse_args()

    print("Initializing...")
    file_tester(args.structure)
    field_specs = _parse_field_str(args.fields)
    print("Done")

    print(f"Parsing Structure file {args.structure}...")
    atoms, bonds, headers = parse_struct_file(args.structure, save_headers=True)
    print("Done")

    # Deep-copy bonds before potentially modifying them (for --delete-bonds)
    working_bonds = copy.deepcopy(bonds)

    if args.delete_bonds:
        _remove_bonds(atoms, working_bonds, args.delete_bonds)

    get_mols(atoms, working_bonds)

    # Validate requested fields exist in atoms
    _validate_fields(atoms, field_specs)

    print("Parsing atom/residue selection...")
    selected = select_atoms(args.atom_sel, atoms)
    print("Done")

    if args.random and args.random > 0:
        _select_random(selected, args.random)

    if args.mol_opt:
        add_mols_to_selection(selected, atoms)

    print("Updating fields..")
    _update_atom_fields(atoms, selected, field_specs)
    print("Done")

    save_path = args.save or _default_save_name(args.structure, args.save_type)
    print(f"Creating {save_path}...")
    field_str_summary = " ".join(
        f"{f}:{spec.mod}{spec.val}" for f, spec in field_specs.items()
    )
    insert_header_remark(headers, f"REMARK {args.structure} modified {field_str_summary}")
    write_bgf(atoms, bonds, save_path, headers)   # write with original bonds
    print("Done")


# ── argument parsing ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modify atom fields in a structure file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-s", "--structure", required=True,
                        help="Input structure file (BGF)")
    parser.add_argument("-a", "--atom-sel", default="*",
                        help='Atom selection string (default: "*" = all atoms)')
    parser.add_argument("-f", "--fields", required=True,
                        help='Field spec(s) e.g. "CHARGE:-1.0 RESNAME:RES"')
    parser.add_argument("-w", "--save",
                        help="Output file name (default: <input>_mod.bgf)")
    parser.add_argument("-t", "--save-type",
                        help="Output file type (default: inferred from extension)")
    parser.add_argument("-m", "--mol-opt", action="store_true",
                        help="Expand selection to complete molecules")
    parser.add_argument("-r", "--random", type=int, default=0,
                        help="Modify only N randomly selected atoms/molecules")
    parser.add_argument("-d", "--delete-bonds",
                        help='Temporarily delete bonds between atom groups: "sel1::sel2"')
    return parser.parse_args()


# ── field spec parsing ─────────────────────────────────────────────────────

def _parse_field_str(field_str: str) -> dict[str, FieldSpec]:
    """Parse "-f" argument into a dict of FieldSpec objects.

    Perl regex: /(word+):(+|-|.)?(.*)/
    """
    specs: dict[str, FieldSpec] = {}
    for token in field_str.split():
        m = re.match(r"(\w+):(([+\-.])?(.*))", token)
        if not m:
            raise ValueError(
                f"Invalid field spec '{token}'. "
                "Expected FIELD:[+|-|.]VALUE  e.g. CHARGE:-1.0"
            )
        field = m.group(1).upper()
        mod = m.group(3) or ""
        val = m.group(4)
        specs[field] = FieldSpec(mod=mod, val=val)

    if not specs:
        raise ValueError(f"No valid field specs found in: {field_str!r}")
    return specs


# ── field validation ───────────────────────────────────────────────────────

def _validate_fields(atoms: AtomsDict, specs: dict[str, FieldSpec]) -> None:
    """Remove field specs that don't exist on atoms (Perl parseFieldList).

    Perl silently drops missing fields; we warn but continue.
    """
    if not atoms:
        return
    first_atom = next(iter(atoms.values()))
    to_remove = [f for f in specs if f not in first_atom]
    for f in to_remove:
        print(f"WARNING: Field '{f}' not found in structure — skipping.", file=sys.stderr)
        del specs[f]
    if not specs:
        raise ValueError("ERROR: No valid fields found in BGF file for given field spec.")


# ── atom field update ──────────────────────────────────────────────────────

def _update_atom_fields(
    atoms: AtomsDict,
    selection: dict[int, Any],
    specs: dict[str, FieldSpec],
) -> None:
    """Apply field modifications to selected atoms (Perl updateAtomFields)."""
    for idx in selection:
        atom = atoms[idx]
        for field, spec in specs.items():
            if spec.mod == ".":
                atom[field] = str(atom.get(field, "")) + spec.val
            elif spec.mod == "+":
                atom[field] = float(atom.get(field, 0)) + float(spec.val)
            elif spec.mod == "-":
                atom[field] = float(atom.get(field, 0)) - float(spec.val)
            else:
                # Try numeric conversion; fall back to string
                try:
                    atom[field] = float(spec.val)
                except ValueError:
                    atom[field] = spec.val


# ── random sub-selection ───────────────────────────────────────────────────

def _select_random(selection: dict[int, Any], n: int) -> None:
    """Keep only n randomly chosen entries in selection (Perl selectRandom)."""
    total = len(selection)
    if total <= n:
        return
    keep = set(random.sample(list(selection), n))
    for k in list(selection):
        if k not in keep:
            del selection[k]


# ── bond removal ───────────────────────────────────────────────────────────

def _remove_bonds(
    atoms: AtomsDict,
    bonds: BondsDict,
    delete_str: str,
) -> None:
    """Temporarily remove bonds between two atom groups (Perl removeBonds).

    delete_str format: "selection1::selection2"
    """
    if "::" not in delete_str:
        raise ValueError(f"--delete-bonds expects 'sel1::sel2', got: {delete_str!r}")

    sel1_str, sel2_str = delete_str.split("::", 1)
    a1 = select_atoms(sel1_str.strip(), atoms)
    a2 = select_atoms(sel2_str.strip(), atoms)

    for idx in list(a1):
        bonds[idx] = [b for b in bonds.get(idx, []) if b not in a2]
    for idx in list(a2):
        bonds[idx] = [b for b in bonds.get(idx, []) if b not in a1]


# ── numbonds sync ─────────────────────────────────────────────────────────

def _update_numbonds(atoms: AtomsDict, bonds: BondsDict) -> None:
    """Sync NUMBONDS field from live bond list (Perl updateNumbonds)."""
    for idx in bonds:
        atoms[idx]["NUMBONDS"] = len(bonds[idx]) if bonds[idx] else 0


# ── default save name ─────────────────────────────────────────────────────

def _default_save_name(struct_path: str, save_type: Optional[str]) -> str:
    p = Path(struct_path)
    ext = (save_type or "bgf").lower().lstrip(".")
    return str(p.parent / (p.stem + f"_mod.{ext}"))


if __name__ == "__main__":
    main()
