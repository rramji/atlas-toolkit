"""
add_solvent — port of addSolvent.pl

Solvate a solute structure by:
  1. Replicating a solvent box to cover the solute cell.
  2. Trimming the replicated box to the target cell.
  3. Embedding the solute (with overlap removal).
  4. Optionally reducing to a target molecule count.

Usage:
  atlas-add-solvent -i solute.bgf -f ff.ff -n "total: 1000"
  atlas-add-solvent -i solute.bgf -n "x: 30 y: 30 z: 30" -w spc -s out.bgf
  atlas-add-solvent -i solute.bgf -n "density: 1.0"
"""
from __future__ import annotations

import argparse
import copy
import math
import re
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.box import get_box, init_box
from atlas_toolkit.core.general import com as _com, file_tester
from atlas_toolkit.core.headers import add_box_to_header, insert_header_remark
from atlas_toolkit.core.manip_atoms import get_mols
from atlas_toolkit.core.replicate import replicate_cell
from atlas_toolkit.io.bgf import make_seq_atom_index, parse_struct_file, write_bgf

from atlas_toolkit.scripts.embed_molecule import embed_molecule
from atlas_toolkit.scripts.remove_mols import remove_mols
from atlas_toolkit.scripts.trim_cell import center_sys, init_box as _ib, trim_cell, make_atoms_mols

# ── bundled WAT directory ────────────────────────────────────────────────────

_WAT_DIR = Path(__file__).resolve().parents[2] / "atlas_toolkit" / "data" / "wat"

_SOLVENT_ALIASES: dict[str, str] = {
    "spc":        "spc_box.bgf",
    "tip3":       "tip3_box.bgf",
    "tip3_charmm": "tip3_charmm_box.bgf",
    "f3c":        "f3c_box.bgf",
    "tip4":       "tip4_box.bgf",
    "meso":       "meso_box.bgf",
}


def _resolve_solvent(name: str | None) -> Path:
    """Return path to solvent BGF (alias, file path, or default SPC)."""
    if name is None:
        return _WAT_DIR / "spc_box.bgf"
    p = Path(name)
    if p.exists():
        return p
    key = name.lower().strip()
    fname = _SOLVENT_ALIASES.get(key)
    if fname:
        cand = _WAT_DIR / fname
        if cand.exists():
            return cand
    # Try loose match
    for alias, fname in _SOLVENT_ALIASES.items():
        if alias in key:
            cand = _WAT_DIR / fname
            if cand.exists():
                return cand
    raise FileNotFoundError(f"Cannot find solvent file for: {name!r}")


# ── cell options parser ──────────────────────────────────────────────────────

def parse_cell_opts(opts_str: str) -> dict:
    """Parse the -n / --cell-opts string.

    Supported patterns:
      "total: N"       — target N solvent molecules
      "density: X"     — target density X g/cm³  (assumes solvent density 1)
      "x: +/- V"       — inflate box by V on each side of x
      "x: + V"         — add V on the + x face
      "x: - V"         — add V on the - x face
      "x: = V"         — set x cell length to V
      Axes can be combined: "x: +/- 10 y: =10 z: -12"
      Multiple may coexist with total/density.
    """
    result: dict = {}

    m = re.search(r"total\s*:\s*(\d+)", opts_str, re.IGNORECASE)
    if m:
        result["total"] = int(m.group(1))

    m = re.search(r"density\s*:\s*(\d+\.?\d*)", opts_str, re.IGNORECASE)
    if m:
        result["density"] = float(m.group(1))

    for dim_pat, dim_key in (("[xyz]", None), ):
        for mm in re.finditer(
            r"([xyz]+)\s*:\s*([+\-=+/\-]*)\s*(\d+\.?\d*)",
            opts_str,
            re.IGNORECASE,
        ):
            axes_str = mm.group(1).lower()
            op = mm.group(2).strip()
            val = float(mm.group(3))
            cell = result.setdefault("cell", {})
            for ax in axes_str:
                entry = cell.setdefault(ax, {})
                if op in ("=", ""):
                    entry["lo"] = 0.0
                    entry["hi"] = val
                else:
                    if "+" in op:
                        entry["hi"] = val
                    if "-" in op:
                        entry["lo"] = val

    if not result:
        raise ValueError(f"Cannot parse cell options: {opts_str!r}")
    return result


# ── target box computation ───────────────────────────────────────────────────

def compute_target_cell(solu_box: dict, cell_opts: dict) -> dict:
    """Return target box dimensions dict {"X": len, "Y": len, "Z": len}."""
    target: dict[str, float] = {}
    for dim, ax in (("X", "x"), ("Y", "y"), ("Z", "z")):
        base = float(solu_box[dim]["len"])
        cell = cell_opts.get("cell", {})
        entry = cell.get(ax, {})
        if "hi" in entry and "lo" not in entry:
            target[dim] = base + float(entry["hi"])
        elif "lo" in entry and "hi" not in entry:
            target[dim] = base + float(entry["lo"])
        elif "hi" in entry and "lo" in entry:
            hi_val = entry["hi"]
            lo_val = entry["lo"]
            if hi_val == lo_val:  # +/- case
                target[dim] = base + 2 * hi_val
            else:
                target[dim] = base + hi_val + lo_val
        else:
            target[dim] = base
    return target


# ── replication vector ───────────────────────────────────────────────────────

def compute_rep_dims(target: dict[str, float], solv_box: dict) -> dict[str, int]:
    """How many replicas needed per axis to cover target dimensions."""
    dims: dict[str, int] = {}
    for dim in ("X", "Y", "Z"):
        n = math.ceil(target[dim] / float(solv_box[dim]["len"]))
        dims[dim] = max(n, 1)
    return dims


# ── main solvation ───────────────────────────────────────────────────────────

def add_solvent(
    solu_atoms: dict,
    solu_bonds: dict,
    solu_headers: list,
    cell_opts: dict,
    solvent_path: Path,
    randomize: bool = False,
    random_rotate: bool = False,
) -> tuple[dict, dict, dict]:
    """Solvate a solute structure.

    Returns (atoms, bonds, box).
    """
    # Load solvent
    solv_atoms, solv_bonds, solv_headers = parse_struct_file(
        str(solvent_path), save_headers=True
    )
    solv_box = get_box(solv_atoms, solv_headers)

    # Compute solute bounding box
    solu_box = get_box(solu_atoms, solu_headers)

    # Target cell dimensions
    target = compute_target_cell(solu_box, cell_opts)

    # Replication dims
    dims = compute_rep_dims(target, solv_box)
    rep_needed = any(v > 1 for v in dims.values())

    print(f"  Replicating solvent {dims['X']}x{dims['Y']}x{dims['Z']}...")
    if rep_needed:
        rep_atoms, rep_bonds, rep_box = replicate_cell(
            copy.deepcopy(solv_atoms), copy.deepcopy(solv_bonds),
            copy.deepcopy(solv_box), dims, pbc=False,
        )
    else:
        rep_atoms = copy.deepcopy(solv_atoms)
        rep_bonds = copy.deepcopy(solv_bonds)
        rep_box = copy.deepcopy(solv_box)

    # Center replicated solvent
    center_sys(rep_atoms, rep_box, start_origin=0)
    init_box(rep_box, rep_atoms)

    # Trim to target cell only when explicit cell dimensions were requested
    # (Perl skips trim when replication is 1x1x1, matching this behaviour)
    has_explicit_cell = "cell" in cell_opts
    if rep_needed or has_explicit_cell:
        cell_str = " ".join(str(target[d]) for d in ("X", "Y", "Z"))
        print(f"  Trimming solvent to {cell_str} Å...")
        from atlas_toolkit.scripts.trim_cell import _parse_cell
        new_box = _parse_cell(cell_str)
        init_box(new_box, rep_atoms)

        rep_mols = get_mols(rep_atoms, rep_bonds)
        from atlas_toolkit.core.manip_atoms import reimage_atoms
        reimage_atoms(rep_atoms, rep_bonds, rep_mols, rep_box)
        trim_cell(rep_atoms, rep_bonds, rep_mols, new_box)
        rep_atoms, rep_bonds = make_seq_atom_index(rep_atoms, rep_bonds)

        trim_box = get_box(rep_atoms, [
            f"CRYSTX  {target['X']} {target['Y']} {target['Z']} 90.0 90.0 90.0"
        ])
        center_sys(rep_atoms, trim_box, start_origin=0)
        init_box(trim_box, rep_atoms)
    else:
        # No trim needed — use the full solvent box as-is
        trim_box = rep_box

    # Embed solute into trimmed solvent
    print("  Embedding solute (removing overlaps)...")
    atoms, bonds = embed_molecule(
        copy.deepcopy(solu_atoms), copy.deepcopy(solu_bonds),
        rep_atoms, rep_bonds,
        trim_box,
        center=True,
        check_overlap=True,
        reverse_place=False,
    )

    # Count solvent molecules after embedding
    n_solu = len(solu_atoms)
    solv_set = {i: 1 for i in atoms if i > n_solu}
    solv_mols_after = get_mols(atoms, bonds, solv_set)
    n_solv_mols = len(solv_mols_after)

    # Remove excess to hit target count
    if "total" in cell_opts and n_solv_mols > cell_opts["total"]:
        n_excess = n_solv_mols - cell_opts["total"]
        print(f"  Removing {n_excess} excess solvent molecules to reach {cell_opts['total']}...")
        mol_sel = f"index > {n_solu}"
        remove_mols(atoms, bonds, solv_set, max_mols=n_excess, randomize=randomize)
        atoms, bonds = make_seq_atom_index(atoms, bonds)

    return atoms, bonds, trim_box


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    file_tester(args.input)

    print(f"Parsing solute structure {args.input}...")
    solu_atoms, solu_bonds, solu_headers = parse_struct_file(args.input, save_headers=True)
    print("Done")

    solvent_path = _resolve_solvent(args.solvent)
    print(f"Using solvent: {solvent_path}")

    cell_opts = parse_cell_opts(args.cell_opts)

    print("Creating solvent box...")
    atoms, bonds, box = add_solvent(
        solu_atoms, solu_bonds, solu_headers,
        cell_opts,
        solvent_path,
        randomize=args.random,
        random_rotate=args.rotate,
    )
    print("Done")

    solv_sel = {i: 1 for i in atoms if i > len(solu_atoms)}
    solv_mols_final = get_mols(atoms, bonds, solv_sel)
    print(f"Added {len(solv_mols_final)} solvent molecules ({len(solv_sel)} atoms).")

    save_path = args.save or _default_save(args.input)
    print(f"Creating {save_path}...")
    insert_header_remark(solu_headers, f"REMARK {args.input} solvated with {solvent_path.name}")
    add_box_to_header(solu_headers, box)
    write_bgf(atoms, bonds, save_path, solu_headers)
    print("Done")


def _default_save(path: str) -> str:
    p = Path(path)
    return str(p.parent / (p.stem + "_solvated.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add solvent to a BGF structure.",
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input", required=True, help="Solute structure file (BGF)")
    parser.add_argument("-n", "--cell-opts", required=True,
                        help='Cell options e.g. "total: 1000" or "x: +/-10 y: =30"')
    parser.add_argument("-w", "--solvent", default=None,
                        help="Solvent name (spc, tip3, f3c, meso) or path to BGF file")
    parser.add_argument("-s", "--save", default=None, help="Output file name")
    parser.add_argument("-f", "--ff", default=None, help="Force field file(s) — currently unused")
    parser.add_argument("-r", "--random", action="store_true",
                        help="Randomize which excess solvent molecules are removed")
    parser.add_argument("-a", "--rotate", action="store_true",
                        help="Randomly rotate solvent molecules after replication")
    parser.add_argument("-o", "--reverse", action="store_true",
                        help="Reverse placement: solvent first, solute second")
    return parser.parse_args()


if __name__ == "__main__":
    main()
