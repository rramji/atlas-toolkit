"""Read and write LAMMPS dump trajectory files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Generator

__all__ = [
    "iter_frames",
    "read_last_frame",
    "parse_frame_selection",
    "lammps_box_to_crystx",
    "apply_coords_to_atoms",
    "recenter_atoms",
    "write_lammps_frame",
    "write_xyz_frame",
    "write_amber_coords",
]

# ---------------------------------------------------------------------------
# Frame type alias
# ---------------------------------------------------------------------------
# Frame = (timestep, atoms, box, columns)
#   timestep : int
#   atoms    : {atom_id: {col_name: value_str, ...}}
#   box      : {'xlo','xhi','ylo','yhi','zlo','zhi', optionally 'xy','xz','yz'}
#   columns  : list[str]


def iter_frames(
    path: str | Path,
    selection: set[int] | None = None,
) -> Generator[tuple[int, dict, dict, list[str]], None, None]:
    """Iterate over frames in a LAMMPS dump file.

    Parameters
    ----------
    path      : dump file path
    selection : set of 1-based frame indices to yield (None = all frames)

    Yields
    ------
    (timestep, atoms, box, columns)
    """
    path = Path(path)
    frame_index = 0
    with open(path, "r", encoding="utf-8") as fh:
        while True:
            result = _parse_frame(fh)
            if result is None:
                break
            frame_index += 1
            if selection is None or frame_index in selection:
                yield result


def read_last_frame(path: str | Path) -> tuple[int, dict, dict, list[str]]:
    """Return the final frame of a LAMMPS dump file.

    For large files this is done efficiently by seeking to the last
    ITEM: TIMESTEP byte offset rather than iterating all frames.

    Returns
    -------
    (timestep, atoms, box, columns)
    """
    path = Path(path)
    last_offset = _find_last_frame_offset(path)
    if last_offset is None:
        raise ValueError(f"No frames found in {path}")
    with open(path, "r", encoding="utf-8") as fh:
        fh.seek(last_offset)
        result = _parse_frame(fh)
    if result is None:
        raise ValueError(f"Could not parse last frame in {path}")
    return result


def parse_frame_selection(spec: str, n_frames: int | None = None) -> set[int] | None:
    """Parse a Perl-style frame selection string into a set of 1-based indices.

    Formats supported
    -----------------
    ``*``              — all frames (returns None)
    ``5``              — single frame
    ``1 5 10``         — explicit list
    ``:1-100:5``       — range start-end:step (all three parts required when using : form)
    ``1-100:5``        — same without leading colon
    ``1-100``          — range with step=1
    Mixed: ``"1-10 50 90-100:2"``
    """
    spec = spec.strip()
    if spec in ("*", "all"):
        return None

    indices: set[int] = set()
    for token in spec.split():
        token = token.lstrip(":")
        if "-" in token:
            parts = token.split(":")
            bounds = parts[0].split("-")
            start, end = int(bounds[0]), int(bounds[1])
            step = int(parts[1]) if len(parts) > 1 else 1
            indices.update(range(start, end + 1, step))
        else:
            indices.add(int(token))
    return indices


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _find_last_frame_offset(path: Path) -> int | None:
    """Scan file for ITEM: TIMESTEP; return byte offset of last occurrence."""
    last_offset = None
    marker = b"ITEM: TIMESTEP"
    with open(path, "rb") as fh:
        offset = 0
        for line in fh:
            if line.rstrip(b"\r\n") == marker:
                last_offset = offset
            offset += len(line)
    return last_offset


def _parse_frame(fh) -> tuple[int, dict, dict, list[str]] | None:
    """Parse one frame from current file position. Returns None at EOF."""
    timestep: int = 0
    n_atoms: int | None = None
    box: dict = {}
    columns: list[str] = []
    atoms: dict = {}
    seen_timestep = False

    while True:
        line = fh.readline()
        if not line:
            return None if not seen_timestep else None
        line = line.strip()
        if not line:
            continue

        if line == "ITEM: TIMESTEP":
            timestep = int(fh.readline().strip())
            seen_timestep = True

        elif line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(fh.readline().strip())

        elif line.startswith("ITEM: BOX BOUNDS"):
            parts = line.split()
            tilt = "xy" in parts
            _read_box_bounds(fh, box, tilt)

        elif line.startswith("ITEM: ATOMS"):
            columns = line.split()[2:]  # drop "ITEM:" and "ATOMS"
            for _ in range(n_atoms):
                row = fh.readline().split()
                atom_data = {col: row[i] for i, col in enumerate(columns)}
                atom_id = int(atom_data["id"])
                atoms[atom_id] = atom_data
            return timestep, atoms, box, columns

    return None


def _read_box_bounds(fh, box: dict, tilt: bool) -> None:
    """Populate box dict from 3 bound lines, converting to true lo/hi."""
    raw = [[float(v) for v in fh.readline().split()] for _ in range(3)]

    xy = raw[0][2] if tilt else 0.0
    xz = raw[1][2] if tilt else 0.0
    yz = raw[2][2] if tilt else 0.0

    if tilt:
        box["xy"] = xy
        box["xz"] = xz
        box["yz"] = yz
        box["xlo"] = raw[0][0] - min(0.0, xy, xz, xy + xz)
        box["xhi"] = raw[0][1] - max(0.0, xy, xz, xy + xz)
        box["ylo"] = raw[1][0] - min(0.0, yz)
        box["yhi"] = raw[1][1] - max(0.0, yz)
        box["zlo"] = raw[2][0]
        box["zhi"] = raw[2][1]
    else:
        keys = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
        for i, (lo, hi) in enumerate(zip(keys[::2], keys[1::2])):
            box[lo] = raw[i][0]
            box[hi] = raw[i][1]


# ---------------------------------------------------------------------------
# Box conversion
# ---------------------------------------------------------------------------

def lammps_box_to_crystx(box: dict) -> tuple[float, float, float, float, float, float]:
    """Convert LAMMPS dump box to CRYSTX (a, b, c, alpha, beta, gamma in degrees)."""
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]
    xy = box.get("xy", 0.0)
    xz = box.get("xz", 0.0)
    yz = box.get("yz", 0.0)

    a = lx
    b = math.sqrt(ly * ly + xy * xy)
    c = math.sqrt(lz * lz + xz * xz + yz * yz)

    def _acos_deg(cos_val: float) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))

    alpha = _acos_deg((xy * xz + ly * yz) / (b * c)) if (b > 0 and c > 0) else 90.0
    beta  = _acos_deg(xz / c) if c > 0 else 90.0
    gamma = _acos_deg(xy / b) if b > 0 else 90.0

    return a, b, c, alpha, beta, gamma


# ---------------------------------------------------------------------------
# Coordinate application
# ---------------------------------------------------------------------------

def apply_coords_to_atoms(
    bgf_atoms: dict,
    dump_atoms: dict,
    box: dict,
    columns: list[str],
    unwrap: bool = True,
) -> None:
    """Update XCOORD/YCOORD/ZCOORD in bgf_atoms from dump_atoms (in-place).

    Handles all four LAMMPS coordinate column conventions:
      x  y  z     — wrapped Cartesian
      xs ys zs    — wrapped scaled (fractional)
      xu yu zu    — unwrapped Cartesian  (image flags already applied)
      xsu ysu zsu — unwrapped scaled
    Image flag unwrapping (ix iy iz) is applied only when coordinates are
    wrapped (x/xs) and unwrap=True.
    """
    col_set = set(columns)

    if "xsu" in col_set:
        cx, cy, cz = "xsu", "ysu", "zsu"
        scaled, already_unwrapped = True, True
    elif "xu" in col_set:
        cx, cy, cz = "xu", "yu", "zu"
        scaled, already_unwrapped = False, True
    elif "xs" in col_set:
        cx, cy, cz = "xs", "ys", "zs"
        scaled, already_unwrapped = True, False
    else:
        cx, cy, cz = "x", "y", "z"
        scaled, already_unwrapped = False, False

    has_image = all(c in col_set for c in ("ix", "iy", "iz"))

    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]
    xy = box.get("xy", 0.0)
    xz = box.get("xz", 0.0)
    yz = box.get("yz", 0.0)

    for atom_id, d in dump_atoms.items():
        if atom_id not in bgf_atoms:
            continue

        v0, v1, v2 = float(d[cx]), float(d[cy]), float(d[cz])

        if scaled:
            x = v0 * lx + v1 * xy + v2 * xz
            y = v1 * ly + v2 * yz
            z = v2 * lz
        else:
            x, y, z = v0, v1, v2

        if unwrap and has_image and not already_unwrapped:
            ix, iy, iz = int(d["ix"]), int(d["iy"]), int(d["iz"])
            x += ix * lx + iy * xy + iz * xz
            y += iy * ly + iz * yz
            z += iz * lz

        bgf_atoms[atom_id]["XCOORD"] = x
        bgf_atoms[atom_id]["YCOORD"] = y
        bgf_atoms[atom_id]["ZCOORD"] = z


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def recenter_atoms(bgf_atoms: dict, box: dict) -> None:
    """Shift all atoms so their (mass-weighted) COM aligns with the box center (in-place).

    After shifting, each atom is re-wrapped into [lo, hi) independently.
    Uses MASS field for weighting when present; falls back to geometric centre.
    """
    vals = list(bgf_atoms.values())

    total_mass = sum(float(a.get("MASS", 1.0)) for a in vals)
    cx = sum(float(a.get("MASS", 1.0)) * a["XCOORD"] for a in vals) / total_mass
    cy = sum(float(a.get("MASS", 1.0)) * a["YCOORD"] for a in vals) / total_mass
    cz = sum(float(a.get("MASS", 1.0)) * a["ZCOORD"] for a in vals) / total_mass

    box_cx = (box["xlo"] + box["xhi"]) / 2
    box_cy = (box["ylo"] + box["yhi"]) / 2
    box_cz = (box["zlo"] + box["zhi"]) / 2

    dx, dy, dz = box_cx - cx, box_cy - cy, box_cz - cz

    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]

    for a in vals:
        a["XCOORD"] = box["xlo"] + (a["XCOORD"] + dx - box["xlo"]) % lx
        a["YCOORD"] = box["ylo"] + (a["YCOORD"] + dy - box["ylo"]) % ly
        a["ZCOORD"] = box["zlo"] + (a["ZCOORD"] + dz - box["zlo"]) % lz


def write_lammps_frame(
    fh,
    timestep: int,
    atoms: dict,
    box: dict,
    columns: list[str],
) -> None:
    """Write one frame in LAMMPS dump format."""
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]
    xy = box.get("xy", 0.0)
    xz = box.get("xz", 0.0)
    yz = box.get("yz", 0.0)
    tilt = any(v != 0.0 for v in (xy, xz, yz))

    fh.write(f"ITEM: TIMESTEP\n{timestep}\n")
    fh.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")

    if tilt:
        xlo_b = box["xlo"] + min(0.0, xy, xz, xy + xz)
        xhi_b = box["xhi"] + max(0.0, xy, xz, xy + xz)
        ylo_b = box["ylo"] + min(0.0, yz)
        yhi_b = box["yhi"] + max(0.0, yz)
        fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        fh.write(f"{xlo_b:.6f} {xhi_b:.6f} {xy:.6f}\n")
        fh.write(f"{ylo_b:.6f} {yhi_b:.6f} {xz:.6f}\n")
        fh.write(f"{box['zlo']:.6f} {box['zhi']:.6f} {yz:.6f}\n")
    else:
        fh.write("ITEM: BOX BOUNDS pp pp pp\n")
        for lo, hi in (("xlo", "xhi"), ("ylo", "yhi"), ("zlo", "zhi")):
            fh.write(f"{box[lo]:.6f} {box[hi]:.6f}\n")

    fh.write("ITEM: ATOMS " + " ".join(columns) + "\n")
    for atom_id in sorted(atoms):
        d = atoms[atom_id]
        fh.write(" ".join(d[c] for c in columns) + "\n")


def write_xyz_frame(fh, timestep: int, atoms: dict, box: dict, columns: list[str]) -> None:
    """Write one frame in XYZ format using element or type as atom label."""
    scaled = "xs" in columns
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]
    xy = box.get("xy", 0.0)
    xz = box.get("xz", 0.0)
    yz = box.get("yz", 0.0)

    fh.write(f"{len(atoms)}\n")
    fh.write(f"Timestep {timestep}\n")
    for atom_id in sorted(atoms):
        d = atoms[atom_id]
        label = d.get("element", d.get("type", "X"))
        if scaled:
            xs, ys, zs = float(d["xs"]), float(d["ys"]), float(d["zs"])
            x = xs * lx + ys * xy + zs * xz
            y = ys * ly + zs * yz
            z = zs * lz
        else:
            x, y, z = float(d["x"]), float(d["y"]), float(d["z"])
        fh.write(f"{label}  {x:.6f}  {y:.6f}  {z:.6f}\n")


def write_amber_coords(fh, atoms: dict, box: dict | None, columns: list[str]) -> None:
    """Append one frame in AMBER .mdcrd format (60-char wide, 3 coords per atom).

    The AMBER coordinate format writes x1,y1,z1,x2,y2,z2,... 6 values per
    line (each 8.3f in a 10-wide field, 6 per line = 60 chars).
    Box line written only when box is provided.
    """
    scaled = "xs" in columns
    lx = box["xhi"] - box["xlo"] if box else 0.0
    ly = box["yhi"] - box["ylo"] if box else 0.0
    lz = box["zhi"] - box["zlo"] if box else 0.0
    xy = box.get("xy", 0.0) if box else 0.0
    xz = box.get("xz", 0.0) if box else 0.0
    yz = box.get("yz", 0.0) if box else 0.0

    coords: list[float] = []
    for atom_id in sorted(atoms):
        d = atoms[atom_id]
        if scaled:
            xs, ys, zs = float(d["xs"]), float(d["ys"]), float(d["zs"])
            coords += [
                xs * lx + ys * xy + zs * xz,
                ys * ly + zs * yz,
                zs * lz,
            ]
        else:
            coords += [float(d["x"]), float(d["y"]), float(d["z"])]

    for i, val in enumerate(coords):
        fh.write(f"{val:8.3f}")
        if (i + 1) % 10 == 0:
            fh.write("\n")
    if len(coords) % 10 != 0:
        fh.write("\n")

    if box is not None:
        a, b, c, alpha, beta, gamma = lammps_box_to_crystx(box)
        box_line = [a, b, c, alpha, beta, gamma]
        for i, val in enumerate(box_line):
            fh.write(f"{val:8.3f}")
            if (i + 1) % 10 == 0:
                fh.write("\n")
        if len(box_line) % 10 != 0:
            fh.write("\n")
