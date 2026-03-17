"""Convert a LAMMPS dump trajectory to other formats.

CLI: atlas-convert-lammps-trj -b struct.bgf -l dump.npt -o lammps -s out.lammpstrj
     atlas-convert-lammps-trj -b struct.bgf -l dump.npt -o bgf    -s frame.bgf
     atlas-convert-lammps-trj -b struct.bgf -l dump.npt -o xyz    -s out.xyz
     atlas-convert-lammps-trj -b struct.bgf -l dump.npt -o amber  -s out.mdcrd
     atlas-convert-lammps-trj -b struct.bgf -l dump.npt -t "1-100:5 200"

Output types
------------
lammps  — LAMMPS dump (default); pass-through with optional unwrapping
bgf     — one BGF file per frame (name.TIMESTEP.bgf)
pdb     — one PDB file per frame
xyz     — single XYZ trajectory file
amber   — AMBER .mdcrd trajectory

Frame selection (-t)
--------------------
*           — all frames (default)
5           — single frame
1 5 10      — explicit list
1-100:5     — range start–end:step
:1-100:5    — same (leading colon allowed for Perl compat)
Mixed:      "1-50:2 100 200-300:10"

Coordinate handling
-------------------
-u 1   unwrap using image flags ix/iy/iz if present (default)
-u 0   keep coordinates as-is (wrapped)
-m 1   reimage into unit cell after applying coordinates (centre-on-origin)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from atlas_toolkit.io.bgf import read_bgf, write_bgf
from atlas_toolkit.lammps.dump import (
    apply_coords_to_atoms,
    iter_frames,
    lammps_box_to_crystx,
    parse_frame_selection,
    recenter_atoms,
    write_amber_coords,
    write_lammps_frame,
    write_xyz_frame,
)


def convert_lammps_trj(
    bgf_path: str | Path,
    dump_path: str | Path,
    save_path: str | Path,
    out_type: str = "lammps",
    frame_sel: str = "*",
    unwrap: bool = True,
    reimage: bool = False,
    recenter: bool = False,
) -> int:
    """Convert a LAMMPS dump trajectory.

    Parameters
    ----------
    bgf_path  : BGF structure file (provides topology & atom metadata)
    dump_path : LAMMPS dump trajectory
    save_path : output path (for bgf/pdb this is the stem; timestep appended)
    out_type  : lammps | bgf | pdb | xyz | amber
    frame_sel : frame selection string (see module docstring)
    unwrap    : unwrap coords using image flags
    reimage   : reimage into unit cell (centre at origin)
    recenter  : shift COM to box centre and re-wrap atoms

    Returns
    -------
    Number of frames written.
    """
    bgf_path = Path(bgf_path)
    dump_path = Path(dump_path)
    save_path = Path(save_path)
    out_type = out_type.lower()

    atoms, bonds, headers = read_bgf(bgf_path)
    selection = parse_frame_selection(frame_sel)

    is_trj = out_type in ("lammps", "xyz", "amber")
    frame_count = 0

    if is_trj:
        mode = "w"
        trj_fh = open(save_path, mode, encoding="utf-8")
        if out_type == "amber":
            trj_fh.write(f"TITLE: AMBER trajectory converted from {dump_path.name}\n")
    else:
        trj_fh = None

    try:
        for ts, dump_atoms, box, columns in iter_frames(dump_path, selection=selection):
            apply_coords_to_atoms(atoms, dump_atoms, box, columns, unwrap=unwrap)

            if recenter:
                recenter_atoms(atoms, box)
            elif reimage:
                _reimage_atoms(atoms, box)

            frame_count += 1

            if out_type == "lammps":
                write_lammps_frame(trj_fh, ts, dump_atoms, box, columns)

            elif out_type == "xyz":
                _xyz_from_bgf(trj_fh, ts, atoms, box)

            elif out_type == "amber":
                _amber_from_bgf(trj_fh, atoms, box)

            elif out_type in ("bgf", "pdb"):
                stem = str(save_path).rsplit(".", 1)[0] if "." in save_path.name else str(save_path)
                ext = out_type
                frame_path = Path(f"{stem}.{ts}.{ext}")
                updated_headers = _replace_crystx(headers, box)
                if out_type == "bgf":
                    write_bgf(atoms, bonds, frame_path, updated_headers)
                else:
                    _write_pdb(atoms, bonds, frame_path)

            if frame_count % 100 == 0:
                print(f"  {frame_count} frames written...", end="\r", flush=True)

    finally:
        if trj_fh is not None:
            trj_fh.close()

    print(f"  {frame_count} frames written.       ")
    return frame_count


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _replace_crystx(headers: list, box: dict) -> list:
    a, b, c, alpha, beta, gamma = lammps_box_to_crystx(box)
    crystx = (
        f"CRYSTX  {a:11.5f}{b:11.5f}{c:11.5f}"
        f"{alpha:11.5f}{beta:11.5f}{gamma:11.5f}"
    )
    if any(h.startswith("CRYSTX") for h in headers):
        return [crystx if h.startswith("CRYSTX") else h for h in headers]
    box_block = [
        "PERIOD 111", "AXES   ZYX",
        "SGNAME P 1                  1    1",
        crystx,
        "CELLS    -1    1   -1    1   -1    1",
    ]
    return box_block + headers


def _reimage_atoms(atoms: dict, box: dict) -> None:
    """Wrap all atoms into [0, len) along each axis (simple image)."""
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]
    for a in atoms.values():
        a["XCOORD"] = a["XCOORD"] % lx
        a["YCOORD"] = a["YCOORD"] % ly
        a["ZCOORD"] = a["ZCOORD"] % lz


def _xyz_from_bgf(fh, ts: int, atoms: dict, box: dict) -> None:
    """Write one XYZ frame using BGF element/fftype as label."""
    fh.write(f"{len(atoms)}\n")
    fh.write(f"Timestep {ts}\n")
    for idx in sorted(atoms):
        a = atoms[idx]
        label = a.get("ELEMENT") or a.get("FFTYPE", "X")
        fh.write(f"{label}  {a['XCOORD']:.6f}  {a['YCOORD']:.6f}  {a['ZCOORD']:.6f}\n")


def _amber_from_bgf(fh, atoms: dict, box: dict) -> None:
    """Write one AMBER .mdcrd frame from BGF atoms."""
    coords: list[float] = []
    for idx in sorted(atoms):
        a = atoms[idx]
        coords += [a["XCOORD"], a["YCOORD"], a["ZCOORD"]]
    for i, val in enumerate(coords):
        fh.write(f"{val:8.3f}")
        if (i + 1) % 10 == 0:
            fh.write("\n")
    if len(coords) % 10 != 0:
        fh.write("\n")
    if box:
        a_val, b_val, c_val, alpha, beta, gamma = lammps_box_to_crystx(box)
        box_vals = [a_val, b_val, c_val, alpha, beta, gamma]
        for i, val in enumerate(box_vals):
            fh.write(f"{val:8.3f}")
            if (i + 1) % 10 == 0:
                fh.write("\n")
        if len(box_vals) % 10 != 0:
            fh.write("\n")


def _write_pdb(atoms: dict, bonds: dict, path: Path) -> None:
    """Minimal PDB writer for trajectory frames."""
    with open(path, "w", encoding="utf-8") as fh:
        for idx in sorted(atoms):
            a = atoms[idx]
            label = a.get("LABEL", "ATOM")
            name = a.get("ATMNAME", "X").ljust(4)
            resname = a.get("RESNAME", "UNK").ljust(3)
            resnum = a.get("RESNUM", 1)
            x, y, z = a["XCOORD"], a["YCOORD"], a["ZCOORD"]
            fh.write(
                f"{label:<6}{idx:5d} {name} {resname}  {resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
        fh.write("END\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert LAMMPS dump trajectory to other formats"
    )
    p.add_argument("-b", dest="bgf", required=True, metavar="BGF",
                   help="BGF structure file")
    p.add_argument("-l", dest="dump", required=True, metavar="DUMP",
                   help="LAMMPS dump trajectory")
    p.add_argument("-o", dest="out_type", default="lammps", metavar="TYPE",
                   help="Output type: lammps (default), bgf, pdb, xyz, amber")
    p.add_argument("-s", dest="save", default=None, metavar="OUT",
                   help="Output file/stem (default: <dump>_out.<ext>)")
    p.add_argument("-t", dest="frames", default="*", metavar="SEL",
                   help="Frame selection: *, N, N-M, N-M:step, or mixed list")
    p.add_argument("-u", dest="unwrap", type=int, default=1, metavar="0|1",
                   help="Unwrap coordinates using image flags (default 1)")
    p.add_argument("-c", dest="recenter", type=int, default=0, metavar="0|1",
                   help="Shift COM to box centre and re-wrap atoms (default 0)")
    p.add_argument("-m", dest="reimage", type=int, default=0, metavar="0|1",
                   help="Reimage atoms into unit cell (default 0)")
    args = p.parse_args(argv)

    out_type = args.out_type.lower()
    ext_map = {"lammps": "lammpstrj", "bgf": "bgf", "pdb": "pdb",
               "xyz": "xyz", "amber": "mdcrd"}
    if out_type not in ext_map:
        p.error(f"Unknown output type: {out_type}. Choose: {', '.join(ext_map)}")

    save = args.save
    if save is None:
        stem = Path(args.dump).stem
        save = f"{stem}_out.{ext_map[out_type]}"

    print(f"Reading structure: {args.bgf}")
    print(f"Converting: {args.dump}  →  {save}  [{out_type}]")
    print(f"Frames: {args.frames}")

    n = convert_lammps_trj(
        args.bgf, args.dump, save,
        out_type=out_type,
        frame_sel=args.frames,
        unwrap=bool(args.unwrap),
        reimage=bool(args.reimage),
        recenter=bool(args.recenter),
    )
    print(f"Done. {n} frames written.")


if __name__ == "__main__":
    main()
