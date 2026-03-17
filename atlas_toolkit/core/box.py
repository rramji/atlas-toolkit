"""
Box geometry: reading, initialising, and converting between coordinate frames.

Port of BOX.pm (GetBox, InitBox, GetBoxDisplacementTensor, box2H, box2F,
Cart2Frac, Frac2Cart, Map2UnitCell, CenterAtoms).

Box dict layout (all keys always present after get_box + init_box):
    box = {
        "X": {"hi": float, "lo": float, "len": float, "angle": float,
              "DISP_V": {"X": float, "Y": float, "Z": float},
              "center": float},
        "Y": {...},
        "Z": {...},
        "H":    [[3x3]]  # frac → cart  (box2H)
        "Hinv": [[3x3]]
        "F":    [[3x3]]  # cart → frac  (box2F)
        "Finv": [[3x3]]
    }
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atlas_toolkit.types import AtomsDict, HeadersList

# Default VDW radius when no force-field data is available.
_DEFAULT_RADII = 3.0  # Å


# ── public API ──────────────────────────────────────────────────────────────

def get_box(
    atoms: "AtomsDict",
    headers: "HeadersList | None" = None,
    padding: float = 0.0,
) -> dict:
    """Compute the simulation box from CRYSTX headers or atom coordinates.

    Port of BOX::GetBox.  Returns a box dict with X/Y/Z sub-dicts populated
    and displacement-tensor / H / F matrices computed via init_box.
    """
    box: dict = {}
    valid = False

    if headers:
        for line in headers:
            if line and line.startswith("CRYSTX"):
                vals = line.split()[1:]
                # CRYSTX: a b c alpha beta gamma
                a, b, c = float(vals[0]), float(vals[1]), float(vals[2])
                alpha = float(vals[3]) if len(vals) > 3 else 90.0
                beta  = float(vals[4]) if len(vals) > 4 else 90.0
                gamma = float(vals[5]) if len(vals) > 5 else 90.0
                box = {
                    "X": {"hi": a, "lo": 0.0, "len": a, "angle": alpha},
                    "Y": {"hi": b, "lo": 0.0, "len": b, "angle": beta},
                    "Z": {"hi": c, "lo": 0.0, "len": c, "angle": gamma},
                }
                valid = True
                break

    if not valid:
        # Fall back to bounding box of atom coordinates + padding.
        x_hi = y_hi = z_hi = -math.inf
        x_lo = y_lo = z_lo =  math.inf
        for a in atoms.values():
            r = _DEFAULT_RADII + padding
            xc, yc, zc = float(a["XCOORD"]), float(a["YCOORD"]), float(a["ZCOORD"])
            x_hi = max(x_hi, xc + r); x_lo = min(x_lo, xc - r)
            y_hi = max(y_hi, yc + r); y_lo = min(y_lo, yc - r)
            z_hi = max(z_hi, zc + r); z_lo = min(z_lo, zc - r)
        box = {
            "X": {"hi": x_hi, "lo": x_lo, "len": x_hi - x_lo, "angle": 90.0},
            "Y": {"hi": y_hi, "lo": y_lo, "len": y_hi - y_lo, "angle": 90.0},
            "Z": {"hi": z_hi, "lo": z_lo, "len": z_hi - z_lo, "angle": 90.0},
        }

    init_box(box, atoms, padding)
    return box


def init_box(box: dict, atoms: "AtomsDict", padding: float = 0.0) -> None:
    """Compute H/F matrices and fractional coords.  Modifies box and atoms in-place.

    Port of BOX::InitBox.
    """
    if padding > 0:
        for dim in ("X", "Y", "Z"):
            box[dim]["len"] -= padding
            box[dim]["hi"]  -= padding
    _box2h(box)
    _box2f(box)
    cart2frac(atoms, box)


def get_box_displacement_tensor(box: dict) -> None:
    """Compute the LAMMPS-style upper-triangular displacement tensor.

    Stores DISP_V and center in each box dimension sub-dict.
    No-op if already computed.  Port of REPLICATE::GetBoxDisplacementTensor.
    """
    if "DISP_V" in box.get("X", {}):
        return

    lx = box["X"]["len"] or 1.0
    ly = box["Y"]["len"] or 1.0
    lz = box["Z"]["len"] or 1.0
    alpha = math.radians(box["X"]["angle"])
    beta  = math.radians(box["Y"]["angle"])
    gamma = math.radians(box["Z"]["angle"])

    ax = lx
    bx = ly * math.cos(gamma)
    cx = lz * math.cos(beta)
    by = ly * math.sin(gamma)
    cy = (ly * lz * math.cos(alpha) - bx * cx) / by
    cz = math.sqrt(max(lz * lz - cx * cx - cy * cy, 0.0))

    box["X"]["DISP_V"] = {"X": ax, "Y": 0.0, "Z": 0.0}
    box["Y"]["DISP_V"] = {"X": bx, "Y": by,  "Z": 0.0}
    box["Z"]["DISP_V"] = {"X": cx, "Y": cy,  "Z": cz}
    box["X"]["center"] = (ax + bx + cx) / 2.0
    box["Y"]["center"] = (by + cy) / 2.0
    box["Z"]["center"] = cz / 2.0


def cart2frac(atoms: "AtomsDict", box: dict) -> None:
    """Convert Cartesian coords to fractional (FA, FB, FC).  In-place.

    Port of BOX::Cart2Frac.
    """
    F = box["F"]
    for a in atoms.values():
        x, y, z = float(a["XCOORD"]), float(a["YCOORD"]), float(a["ZCOORD"])
        a["FA"] = F[0][0]*x + F[0][1]*y + F[0][2]*z
        a["FB"] = F[1][0]*x + F[1][1]*y + F[1][2]*z
        a["FC"] = F[2][0]*x + F[2][1]*y + F[2][2]*z


def frac2cart(atoms: "AtomsDict", box: dict) -> None:
    """Convert fractional coords (FA, FB, FC) back to Cartesian.  In-place.

    Port of BOX::Frac2Cart.
    """
    H = box["H"]
    for a in atoms.values():
        fa, fb, fc = float(a["FA"]), float(a["FB"]), float(a["FC"])
        a["XCOORD"] = H[0][0]*fa + H[0][1]*fb + H[0][2]*fc
        a["YCOORD"] = H[1][0]*fa + H[1][1]*fb + H[1][2]*fc
        a["ZCOORD"] = H[2][0]*fa + H[2][1]*fb + H[2][2]*fc


def map2unit_cell(atom: dict, box: dict) -> None:
    """Image a single atom into the unit cell [0, len).  Modifies atom in-place.

    Port of BOX::Map2UnitCell.
    """
    Hinv = box["Hinv"]
    H    = box["H"]
    ox, oy, oz = float(atom["XCOORD"]), float(atom["YCOORD"]), float(atom["ZCOORD"])

    # Cartesian → reduced fractional
    n = [
        Hinv[0][0]*ox + Hinv[0][1]*oy + Hinv[0][2]*oz,
        Hinv[1][0]*ox + Hinv[1][1]*oy + Hinv[1][2]*oz,
        Hinv[2][0]*ox + Hinv[2][1]*oy + Hinv[2][2]*oz,
    ]
    for i in range(3):
        while n[i] > 1.0: n[i] -= 1.0
        while n[i] < 0.0: n[i] += 1.0

    # Fractional → Cartesian (mapped position)
    nx = H[0][0]*n[0] + H[0][1]*n[1] + H[0][2]*n[2]
    ny = H[1][0]*n[0] + H[1][1]*n[1] + H[1][2]*n[2]
    nz = H[2][0]*n[0] + H[2][1]*n[1] + H[2][2]*n[2]

    atom["SHIFT"] = {"XCOORD": nx - ox, "YCOORD": ny - oy, "ZCOORD": nz - oz}
    atom["XCOORD"] = nx
    atom["YCOORD"] = ny
    atom["ZCOORD"] = nz


def center_atoms(atoms: "AtomsDict", box: dict) -> None:
    """Translate all atoms so the box lower corner is at the origin.

    Port of BOX::CenterAtoms.
    """
    ox = box["X"]["lo"]
    oy = box["Y"]["lo"]
    oz = box["Z"]["lo"]
    for a in atoms.values():
        a["XCOORD"] = float(a["XCOORD"]) - ox
        a["YCOORD"] = float(a["YCOORD"]) - oy
        a["ZCOORD"] = float(a["ZCOORD"]) - oz
    box["X"]["hi"] -= ox; box["X"]["lo"] = 0.0
    box["Y"]["hi"] -= oy; box["Y"]["lo"] = 0.0
    box["Z"]["hi"] -= oz; box["Z"]["lo"] = 0.0


def get_box_vol(box: dict) -> float:
    """Return the scalar cell volume.  Port of REPLICATE::GetBoxVol."""
    get_box_displacement_tensor(box)
    dv = box["X"]["DISP_V"]
    ax = dv["X"]
    dv = box["Y"]["DISP_V"]
    by = dv["Y"]
    dv = box["Z"]["DISP_V"]
    cz = dv["Z"]
    # For upper-triangular H: vol = ax * by * cz
    return ax * by * cz


# ── private helpers ─────────────────────────────────────────────────────────

def _box2h(box: dict) -> None:
    """Compute H and Hinv matrices.  H maps fractional → Cartesian.

    Convention: c along Z, b in the yz-plane, a fills the rest.
    Port of BOX::box2H.
    """
    la = box["X"]["len"] or 0.00001
    lb = box["Y"]["len"] or 0.00001
    lc = box["Z"]["len"] or 0.00001

    alpha = math.radians(box["X"]["angle"])
    beta  = math.radians(box["Y"]["angle"])
    gamma = math.radians(box["Z"]["angle"])

    # c along Z
    c = [0.0, 0.0, lc]
    # b in yz-plane
    b = [0.0, lb * math.sin(alpha), lb * math.cos(alpha)]
    # a determined by dot products
    a2 = la * math.cos(beta)
    a1 = (la * lb * math.cos(gamma) - a2 * b[2]) / b[1]
    a0 = math.sqrt(max(la*la - a1*a1 - a2*a2, 0.0))
    a = [a0, a1, a2]

    # H[row][col]:  col 0 = a-vector, col 1 = b-vector, col 2 = c-vector
    H = [
        [a[0], b[0], c[0]],
        [a[1], b[1], c[1]],
        [a[2], b[2], c[2]],
    ]
    box["H"] = H

    # Hinv via numpy for numerical stability (3×3 inverse)
    Hn = np.array(H, dtype=float)
    Hinvn = np.linalg.inv(Hn)
    box["Hinv"] = Hinvn.tolist()


def _box2f(box: dict) -> None:
    """Compute F and Finv matrices.  F maps Cartesian → fractional.

    Port of BOX::box2F (crystallographic formula).
    """
    la = box["X"]["len"] or 1.0
    lb = box["Y"]["len"] or 1.0
    lc = box["Z"]["len"] or 1.0

    alpha = math.radians(box["X"]["angle"])
    beta  = math.radians(box["Y"]["angle"])
    gamma = math.radians(box["Z"]["angle"])

    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)

    vol = la * lb * lc * math.sqrt(
        max(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg, 0.0)
    )
    if vol == 0.0:
        vol = 1.0

    F = [
        [1/la,              -cg/(la*sg),            (ca*cg - cb)/(la*vol*sg)],
        [0.0,                1/(lb*sg),              (cb*cg - ca)/(lb*vol*sg)],
        [0.0,                0.0,                    la*lb*sg/vol           ],
    ]
    box["F"] = F

    Finv = [
        [la,   lb*cg,  lc*cb                       ],
        [0.0,  lb*sg,  lc*(ca - cb*cg)/sg           ],
        [0.0,  0.0,    vol/(la*lb*sg)               ],
    ]
    box["Finv"] = Finv
