"""
Write a LAMMPS input script (in.*) for a system described by a data file.

Public API
----------
write_input_script(path, data_file_name, summary, ff_parms, box, protocol="min",
                   cutoff=12.0, seed=12345) -> None

Supported protocols
-------------------
min   : conjugate-gradient minimization
nvt   : NVT MD at 300 K (Nosé-Hoover thermostat)
npt   : NPT MD at 300 K / 1 atm (Nosé-Hoover)

FF style mapping (based on types present in ff_parms)
------------------------------------------------------
Pair     : lj/cut/coul/long  (always; kspace pppm)
Bond     : harmonic           (HARMONIC)
Angle    : harmonic           (THETA_HARM)
Dihedral : charmm             (SHFT_DIHDR)
Improper : cvff               (IT_JIKL)
"""
from __future__ import annotations

from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────
_KCAL_MOL = "kcal/mol"
_METAL    = "metal"
_REAL     = "real"     # LAMMPS units for most MD; kcal/mol, Å, fs

_THERMO_STYLE = (
    "step  temp  press  pe  ke  etotal  vol  density  lx  ly  lz"
)


def write_input_script(
    path: str | Path,
    data_file_name: str,
    summary,          # DataFileSummary
    ff_parms: dict,
    box: dict,
    protocol: str = "min",
    cutoff: float = 12.0,
    seed: int = 12345,
    title: str = "",
) -> None:
    """Write a LAMMPS in.* script.

    Parameters
    ----------
    path            : output file path
    data_file_name  : name of the associated data.* file (relative path)
    summary         : DataFileSummary from write_data_file()
    ff_parms        : merged FF parms from load_ff()
    box             : box dict from get_box()
    protocol        : 'min', 'nvt', or 'npt'
    cutoff          : LJ+real-space coulomb cutoff (Å)
    seed            : RNG seed for velocity initialisation
    title           : optional comment line
    """
    path = Path(path)
    parms = ff_parms.get("PARMS", {})
    cut = parms.get("cut_vdw", cutoff)

    lines: list[str] = []

    def w(*args):
        lines.append(" ".join(str(a) for a in args))

    def blank():
        lines.append("")

    def comment(s):
        lines.append(f"# {s}")

    # ── header ─────────────────────────────────────────────────────────────
    if title:
        comment(title)
    comment(f"LAMMPS input script — protocol: {protocol}")
    blank()

    # ── basic settings ──────────────────────────────────────────────────────
    w("units", "real")
    w("atom_style", "full")
    blank()

    # ── interaction styles ──────────────────────────────────────────────────
    # lj/charmm/coul/long required when dihedral_style charmm is used
    # (charmm dihedrals handle 1-4 scaling internally; incompatible with lj/cut)
    if summary.n_dihedral_types > 0:
        w("pair_style", "lj/charmm/coul/long", cut - 2.0, cut)
    else:
        w("pair_style", "lj/cut/coul/long", cut, cut)

    if summary.n_bond_types > 0:
        w("bond_style", "harmonic")
    if summary.n_angle_types > 0:
        w("angle_style", "harmonic")
    if summary.n_dihedral_types > 0:
        w("dihedral_style", "charmm")
    if summary.n_improper_types > 0:
        w("improper_style", "cvff")
    blank()

    # ── read data ───────────────────────────────────────────────────────────
    w("read_data", data_file_name)
    blank()

    # ── pair mixing & special bonds ─────────────────────────────────────────
    w("pair_modify", "mix", "arithmetic")
    w("special_bonds", "amber")     # 1-4: LJ 0.5, coul 0.8333
    blank()

    # ── kspace ─────────────────────────────────────────────────────────────
    w("kspace_style", "pppm", "1.0e-4")
    blank()

    # ── neighbor ───────────────────────────────────────────────────────────
    w("neighbor", "2.0", "bin")
    w("neigh_modify", "every", "2", "delay", "4", "check", "yes")
    blank()

    # ── thermo ─────────────────────────────────────────────────────────────
    w("thermo", "100")
    w("thermo_style", "custom", _THERMO_STYLE)
    blank()

    # ── protocol-specific commands ──────────────────────────────────────────
    if protocol == "min":
        _write_min(lines, w, blank, comment)
    elif protocol == "nvt":
        _write_nvt(lines, w, blank, comment, seed)
    elif protocol == "npt":
        _write_npt(lines, w, blank, comment, seed)
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}. "
                         "Use 'min', 'nvt', or 'npt'.")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_min(lines, w, blank, comment):
    comment("─── Energy minimisation ───")
    w("min_style", "cg")
    w("minimize", "1.0e-4", "1.0e-4", "10000", "100000")
    blank()
    w("run", "0")


def _write_nvt(lines, w, blank, comment, seed: int):
    temp = 300.0
    comment(f"─── NVT MD at {temp} K ───")
    w("velocity", "all", "create", temp, seed, "dist", "gaussian")
    blank()
    w("fix", "1", "all", "nvt", "temp", temp, temp, "100.0")
    blank()
    w("timestep", "1.0")        # 1 fs
    w("run", "1000000")         # 1 ns


def _write_npt(lines, w, blank, comment, seed: int):
    temp = 300.0
    pres = 1.0
    comment(f"─── NPT MD at {temp} K / {pres} atm ───")
    w("velocity", "all", "create", temp, seed, "dist", "gaussian")
    blank()
    w("fix", "1", "all", "npt",
      "temp", temp, temp, "100.0",
      "iso", pres, pres, "1000.0")
    blank()
    w("timestep", "1.0")
    w("run", "1000000")
