"""Update BGF atom coordinates from the final frame of a LAMMPS dump file.

CLI: atlas-update-bgf-coords -b struct.bgf -l dump.min [-w out.bgf] [-u 0] [-c 1]

By default coordinates are unwrapped using image flags (ix iy iz) if present.
Pass -c 1 to shift the COM to the box centre and re-wrap into the unit cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atlas_toolkit.io.bgf import read_bgf, write_bgf
from atlas_toolkit.lammps.dump import (
    apply_coords_to_atoms,
    lammps_box_to_crystx,
    read_last_frame,
    recenter_atoms,
)


def update_bgf_coords(
    bgf_path: str | Path,
    dump_path: str | Path,
    save_path: str | Path | None = None,
    unwrap: bool = True,
    recenter: bool = False,
) -> None:
    """Update BGF coords from the last frame of a LAMMPS dump.

    Parameters
    ----------
    bgf_path  : input BGF file (topology + original coords)
    dump_path : LAMMPS dump trajectory
    save_path : output BGF (default: overwrite bgf_path)
    unwrap    : apply image flags to unwrap coords across PBC
    recenter  : shift COM to box centre and re-wrap atoms
    """
    bgf_path = Path(bgf_path)
    dump_path = Path(dump_path)
    save_path = Path(save_path) if save_path else bgf_path

    print(f"Reading structure: {bgf_path}")
    atoms, bonds, headers = read_bgf(bgf_path)

    print(f"Reading last frame: {dump_path}")
    _ts, dump_atoms, box, columns = read_last_frame(dump_path)

    apply_coords_to_atoms(atoms, dump_atoms, box, columns, unwrap=unwrap)

    if recenter:
        recenter_atoms(atoms, box)

    a, b, c, alpha, beta, gamma = lammps_box_to_crystx(box)
    crystx = (
        f"CRYSTX  {a:11.5f}{b:11.5f}{c:11.5f}"
        f"{alpha:11.5f}{beta:11.5f}{gamma:11.5f}"
    )
    headers = _replace_box_headers(headers, crystx)

    print(f"Writing: {save_path}")
    write_bgf(atoms, bonds, save_path, headers)


def _replace_box_headers(headers: list, crystx_line: str) -> list:
    has_crystx = any(h.startswith("CRYSTX") for h in headers)
    if has_crystx:
        return [crystx_line if h.startswith("CRYSTX") else h for h in headers]
    box_block = [
        "PERIOD 111",
        "AXES   ZYX",
        "SGNAME P 1                  1    1",
        crystx_line,
        "CELLS    -1    1   -1    1   -1    1",
    ]
    return box_block + headers


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Update BGF coordinates from the last frame of a LAMMPS dump"
    )
    p.add_argument("-b", dest="bgf", required=True, metavar="BGF",
                   help="Input BGF file")
    p.add_argument("-l", dest="dump", required=True, metavar="DUMP",
                   help="LAMMPS dump trajectory")
    p.add_argument("-w", dest="save", default=None, metavar="OUT",
                   help="Output BGF (default: overwrite input BGF)")
    p.add_argument("-u", dest="unwrap", type=int, default=1, metavar="0|1",
                   help="Unwrap coordinates using image flags (default 1)")
    p.add_argument("-c", dest="recenter", type=int, default=0, metavar="0|1",
                   help="Shift COM to box centre and re-wrap atoms (default 0)")
    args = p.parse_args(argv)

    update_bgf_coords(args.bgf, args.dump, args.save,
                      unwrap=bool(args.unwrap), recenter=bool(args.recenter))


if __name__ == "__main__":
    main()
