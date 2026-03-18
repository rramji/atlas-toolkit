"""
atlas-create-lammps-input — Generate LAMMPS data.* and in.* files from a BGF.

Usage
-----
atlas-create-lammps-input -b struct.bgf -f "ff1.ff ff2.frcmod" \\
    [-t min] [-s stem] [-c 12.0]

Mirrors the essential functionality of createLammpsInput.pl for the
standard AMBER/GAFF/Heinz-Au use case (no QEq, no Drude, no AMOEBA).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def create_lammps_input(bgf_file: str, ff_string: str, save_name: str = None,
                        protocol: str = "min", cutoff: float = 12.0,
                        seed: int = 12345) -> None:
    """Generate LAMMPS data.* and in.* files from a BGF structure.

    Parameters
    ----------
    bgf_file:  Path to input BGF file.
    ff_string: Space-separated force-field file paths (.ff / .frcmod).
    save_name: Output stem (default: BGF basename without extension).
    protocol:  Run protocol — "min", "nvt", or "npt" (default: "min").
    cutoff:    VDW / real-space electrostatic cutoff in Å (default: 12.0).
    seed:      RNG seed for velocity initialisation (default: 12345).
    """
    from atlas_toolkit.io.bgf import read_bgf
    from atlas_toolkit.core.box import get_box
    from atlas_toolkit.io.ff import load_ff
    from atlas_toolkit.lammps.data_file import write_data_file
    from atlas_toolkit.lammps.input_script import write_input_script

    bgf_path = Path(bgf_file)
    if not bgf_path.exists():
        raise FileNotFoundError(f"BGF file not found: {bgf_path}")

    stem_path = Path(save_name) if save_name else bgf_path.parent / bgf_path.stem
    data_name = str(stem_path.parent / f"data.{stem_path.name}")
    in_name   = str(stem_path.parent / f"in.{stem_path.name}")

    print(f"Reading structure: {bgf_path}")
    atoms, bonds_dict, headers = read_bgf(bgf_path)
    box = get_box(atoms, headers)

    print(f"Loading FF: {ff_string}")
    ff_parms = load_ff(ff_string)

    print(f"Writing {data_name} ...")
    summary = write_data_file(
        data_name, atoms, bonds_dict, ff_parms, box,
        title=f"Generated from {bgf_path.name}",
    )
    print(f"  {summary.n_atoms} atoms  ({summary.n_atom_types} types)")
    print(f"  {summary.n_bonds} bonds  ({summary.n_bond_types} types)")
    print(f"  {summary.n_angles} angles  ({summary.n_angle_types} types)")
    print(f"  {summary.n_dihedrals} dihedrals  ({summary.n_dihedral_types} types)")
    print(f"  {summary.n_impropers} impropers  ({summary.n_improper_types} types)")

    if summary.missing_bonds:
        unique = sorted(set(summary.missing_bonds))
        print(f"  WARNING: {len(unique)} bond type(s) missing FF parameters:")
        for t in unique[:10]:
            print(f"    {t[0]}-{t[1]}")

    if summary.missing_angles:
        unique = sorted(set(summary.missing_angles))
        print(f"  WARNING: {len(unique)} angle type(s) missing FF parameters:")
        for t in unique[:10]:
            print(f"    {t[0]}-{t[1]}-{t[2]}")

    if summary.missing_torsions:
        unique = sorted(set(summary.missing_torsions))
        print(f"  WARNING: {len(unique)} torsion type(s) missing FF parameters:")
        for t in unique[:10]:
            print(f"    {t[0]}-{t[1]}-{t[2]}-{t[3]}")

    print(f"Writing {in_name} ...")
    write_input_script(
        in_name, data_name, summary, ff_parms, box,
        protocol=protocol,
        cutoff=cutoff,
        seed=seed,
        title=f"Generated from {bgf_path.name}",
    )
    print("Done.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="atlas-create-lammps-input",
        description="Generate LAMMPS data.* and in.* from a BGF structure.",
    )
    p.add_argument("-b", "--bgf",      required=True,  help="Input BGF file")
    p.add_argument("-f", "--ff",       required=True,
                   help="Force-field file(s): space-separated .ff / .frcmod paths")
    p.add_argument("-t", "--type",     default="min",
                   choices=["min", "nvt", "npt"],
                   help="Run protocol (default: min)")
    p.add_argument("-s", "--stem",     default=None,
                   help="Output stem (default: BGF basename without extension)")
    p.add_argument("-c", "--cutoff",   type=float, default=12.0,
                   help="VDW / real-space electrostatic cutoff in Å (default: 12)")
    p.add_argument("--seed",           type=int,   default=12345,
                   help="RNG seed for velocity initialisation (default: 12345)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        create_lammps_input(
            bgf_file=args.bgf,
            ff_string=args.ff,
            save_name=args.stem,
            protocol=args.type,
            cutoff=args.cutoff,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
