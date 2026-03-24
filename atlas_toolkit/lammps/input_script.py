"""
Write a LAMMPS input script (in.*) for a system described by a data file.

Public API
----------
write_input_script(path, data_file_name, summary, ff_parms, box, protocol="min",
                   cutoff=14.0, seed=12345) -> None

Supported protocols
-------------------
min   : conjugate-gradient minimization then halt
nvt   : NVT MD (Nosé-Hoover thermostat)
npt   : NPT MD (Nosé-Hoover)

Output format matches createLammpsInput.pl
------------------------------------------
Pair     : lj/charmm/coul/long (inner=cutoff-1, outer=cutoff)
Bond     : harmonic
Angle    : harmonic
Dihedral : fourier
Improper : cvff
Kspace   : pppm 0.00001
"""
from __future__ import annotations

from pathlib import Path


def write_input_script(
    path: str | Path,
    data_file_name: str,
    summary,          # DataFileSummary
    ff_parms: dict,
    box: dict,
    protocol: str = "min",
    cutoff: float = 14.0,
    seed: int = 12345,
    title: str = "",
    solute_atoms: int | None = None,
) -> None:
    """Write a LAMMPS in.* script matching createLammpsInput.pl output.

    Parameters
    ----------
    path            : output file path  (e.g. "in.mysystem")
    data_file_name  : name of the associated data.* file
    summary         : DataFileSummary from write_data_file()
    ff_parms        : merged FF parms from load_ff()
    box             : box dict from get_box()
    protocol        : 'min', 'nvt', or 'npt'
    cutoff          : outer LJ cutoff in Å; switch starts at cutoff-1 (default: 14)
    seed            : RNG seed for velocity initialisation
    title           : optional comment written at the top
    solute_atoms    : atom index of the last solute atom for group definitions;
                      defaults to total atom count (all atoms = solute)
    """
    path = Path(path)

    # Derive sname from the data file name
    data_stem = Path(data_file_name).name
    sname = data_stem[5:] if data_stem.startswith("data.") else data_stem
    in_name = f"in.{sname}"
    inner = cutoff - 1.0
    s_atoms = solute_atoms if solute_atoms is not None else summary.n_atoms

    # FF-derived settings (with Perl-compatible defaults)
    p = ff_parms.get("PARMS", {})
    dielectric    = p.get("dielectric", 1)
    kspace_acc    = p.get("coul_accuracy", 0.001)
    vdw_14        = p.get("vdw_14_scale", 1.0)
    coul_14       = p.get("coul_14_scale", 1.0)
    if vdw_14 == coul_14:
        special_bonds = f"lj/coul 0.0 0.0 {vdw_14:g}"
    else:
        special_bonds = f"lj 0.0 0.0 {vdw_14:g} coul 0.0 0.0 {coul_14:g}"

    lines: list[str] = []

    def w(s: str = ""):
        lines.append(s)

    if title:
        w(f"# {title}")

    # Basic settings
    w(f"{'units':<21s}real")
    w(f"{'atom_style':<21s}full")
    w(f"{'boundary':<21s}p p p")
    w(f"{'dielectric':<21s}{dielectric}")
    w(f"{'special_bonds':<21s}{special_bonds}")
    w()

    # Interaction styles
    w(f"{'pair_style':<21s}lj/charmm/coul/long {inner:.5g} {cutoff:.5f}")
    if summary.n_bond_types > 0:
        w(f"{'bond_style':<21s}harmonic")
    if summary.n_angle_types > 0:
        w(f"{'angle_style':<21s}harmonic")
    if summary.n_dihedral_types > 0:
        w(f"{'dihedral_style':<21s}fourier")
    if summary.n_improper_types > 0:
        w(f"{'improper_style':<21s}cvff")
    w(f"{'kspace_style':<21s}pppm {kspace_acc:g}")
    w()

    # Read data
    w(f"{'read_data':<21s}{data_file_name}")
    w()

    # Post-read settings
    w(f"{'pair_modify':<21s}mix geometric")
    w(f"{'neighbor':<21s}2.0 multi")
    w(f"{'neigh_modify':<21s}every 2 delay 4 check yes")
    w(f"{'thermo_style':<21s}multi")
    w(f"{'thermo_modify':<21s}line multi format float %14.6f flush yes")
    w(f"{'variable':<21s}input string {in_name}")
    w(f"{'variable':<21s}sname string {sname}")
    w(f"{'variable':<21s}sAtoms index {s_atoms}")
    w(f"{'group':<21s}solute id <> 1 ${{sAtoms}}")
    w(f"{'group':<21s}solvent subtract all solute")
    w()
    w()

    w(f"{'timestep':<21s}1")
    w()

    if protocol == "min":
        _write_min(w)
    elif protocol == "nvt":
        _write_nvt(w, seed)
    elif protocol == "npt":
        _write_npt(w, seed)
    else:
        raise ValueError(
            f"Unknown protocol: {protocol!r}. Use 'min', 'nvt', or 'npt'."
        )

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_min(w):
    w("print                .")
    w("print                ==========================================")
    w('print                "500 steps CG Minimization"')
    w("print                ==========================================")
    w("print                .")
    w()
    w(f"{'dump':<21s}1 all custom 25 ${{sname}}.min.lammps id type xu yu zu vx vy vz")
    w(f"{'thermo':<21s}10")
    w(f"{'fix':<21s}restraint solute spring/self 500.0")
    w(f"{'min_style':<21s}sd")
    w(f"{'minimize':<21s}1.0e-4 1.0e-4 500 5000")
    w(f"{'min_style':<21s}cg")
    w(f"{'minimize':<21s}1.0e-4 1.0e-4 500 5000")
    w(f"{'#now':<21s}minimize the entire system")
    w(f"{'unfix':<21s}restraint")
    w(f"{'minimize':<21s}1.0e-4 1.0e-4 500 5000")
    w(f"{'undump':<21s}1")


def _write_nvt(w, seed: int):
    w("print                .")
    w("print                =====================================")
    w('print                "NVT dynamics to heat system"')
    w("print                =====================================")
    w("print                .")
    w()
    w(f"{'fix':<21s}shakeH all shake 0.0001 20 500 m 1.008 b 3 6 a 9")
    w(f"{'velocity':<21s}all create 0.0 {seed} dist uniform")
    w(f"{'thermo':<21s}100")
    w(f"{'dump':<21s}1 all custom 1000 ${{sname}}.heat.lammpstrj id type xu yu zu vx vy vz")
    w(f"{'fix':<21s}3 all nvt temp 1.0 ${{rtemp}} 100.0")
    w(f"{'run':<21s}50000")
    w(f"{'unfix':<21s}3")
    w(f"{'undump':<21s}1")
    w()
    w(f"{'fix':<21s}balance all balance 1000 1.0 shift xyz 20 1.0")
    w()
    w("print                .")
    w("print                ================================================")
    w('print                "NVT production dynamics "')
    w("print                ================================================")
    w("print                .")
    w()
    w(f"{'fix':<21s}2 all nvt temp ${{rtemp}} ${{rtemp}} 100.0 tloop 10 ploop 10")
    w(f"{'restart':<21s}1000000 ${{sname}}.${{rtemp}}K.*.restart")
    w(f"{'dump':<21s}1 all custom 1000 ${{sname}}.${{rtemp}}K.equil.lammps id type xu yu zu vx vy vz")
    w(f"{'fix':<21s}recenter all recenter INIT INIT INIT")
    w(f"{'run':<21s}5000000")
    w(f"{'write_restart':<21s}post_prod.restart")
    w(f"{'unfix':<21s}2")
    w(f"{'undump':<21s}1")


def _write_npt(w, seed: int):
    w("print                .")
    w("print                =====================================")
    w('print                "NPT dynamics to equilibrate"')
    w("print                =====================================")
    w("print                .")
    w()
    w(f"{'velocity':<21s}all create 300.0 {seed} dist gaussian")
    w()
    w(f"{'fix':<21s}1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0")
    w(f"{'thermo':<21s}1000")
    w(f"{'dump':<21s}1 all custom 1000 ${{sname}}.npt.lammpstrj id type xu yu zu vx vy vz")
    w(f"{'run':<21s}1000000")
    w(f"{'write_restart':<21s}post_npt.restart")
    w(f"{'unfix':<21s}1")
    w(f"{'undump':<21s}1")
