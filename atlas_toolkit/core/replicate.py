"""
Cell replication and coordinate rotation.

Port of REPLICATE.pm (TransAtom, TransCellAtoms, combineMols,
setPBCbonds, updatePBCbonds, ReplicateCell) and General.pm::Rotate.
"""
from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atlas_toolkit.types import AtomsDict, BondsDict

from atlas_toolkit.core.box import get_box_displacement_tensor


# ── coordinate transforms ───────────────────────────────────────────────────

def trans_atom(atom: dict, box: dict, vec: list[float]) -> dict:
    """Translate a single atom by a lattice-vector multiple.

    new_pos = pos + H_upper_tri @ vec
    Port of REPLICATE::TransAtom.
    """
    get_box_displacement_tensor(box)
    ax = box["X"]["DISP_V"]["X"]; bx = box["Y"]["DISP_V"]["X"]; cx = box["Z"]["DISP_V"]["X"]
    by = box["Y"]["DISP_V"]["Y"]; cy = box["Z"]["DISP_V"]["Y"]
    cz = box["Z"]["DISP_V"]["Z"]

    return {
        "XCOORD": float(atom["XCOORD"]) + ax*vec[0] + bx*vec[1] + cx*vec[2],
        "YCOORD": float(atom["YCOORD"]) +             by*vec[1] + cy*vec[2],
        "ZCOORD": float(atom["ZCOORD"]) +                         cz*vec[2],
    }


def trans_cell_atoms(atoms: "AtomsDict", box: dict, vec: list[float]) -> "AtomsDict":
    """Deep-copy atoms and translate all coordinates by a lattice-vector multiple.

    Port of REPLICATE::TransCellAtoms.
    """
    result = copy.deepcopy(atoms)
    get_box_displacement_tensor(box)
    ax = box["X"]["DISP_V"]["X"]; bx = box["Y"]["DISP_V"]["X"]; cx = box["Z"]["DISP_V"]["X"]
    by = box["Y"]["DISP_V"]["Y"]; cy = box["Z"]["DISP_V"]["Y"]
    cz = box["Z"]["DISP_V"]["Z"]

    for a in result.values():
        x = float(a["XCOORD"]); y = float(a["YCOORD"]); z = float(a["ZCOORD"])
        a["XCOORD"] = x + ax*vec[0] + bx*vec[1] + cx*vec[2]
        a["YCOORD"] = y +             by*vec[1] + cy*vec[2]
        a["ZCOORD"] = z +                         cz*vec[2]
    return result


# ── molecule combining ───────────────────────────────────────────────────────

def combine_mols(
    atoms1: "AtomsDict",
    bonds1: "BondsDict",
    atoms2: "AtomsDict",
    bonds2: "BondsDict",
    update_res_num: bool = False,
) -> tuple["AtomsDict", "BondsDict"]:
    """Merge two atom/bond dicts, renumbering atoms2 by max(atoms1) offset.

    Port of REPLICATE::combineMols.
    """
    if atoms1:
        offset = max(atoms1)
        res_offset = atoms1[max(atoms1)]["RESNUM"] if update_res_num else 0
    else:
        offset = res_offset = 0

    combined_atoms = atoms1
    combined_bonds: "BondsDict" = dict(bonds1)

    for old_idx in sorted(atoms2):
        new_idx = old_idx + offset
        a = copy.deepcopy(atoms2[old_idx])
        a["INDEX"] = new_idx
        if update_res_num:
            a["RESNUM"] = int(a.get("RESNUM", 1)) + res_offset
        combined_atoms[new_idx] = a

        new_bonds = [b + offset for b in bonds2.get(old_idx, [])]
        combined_bonds[new_idx] = new_bonds

    return combined_atoms, combined_bonds


# ── PBC bond handling ────────────────────────────────────────────────────────

def set_pbc_bonds(atoms: "AtomsDict", bonds: "BondsDict", box: dict) -> None:
    """Mark bonds that cross periodic boundaries by storing displacement indices.

    For each bonded pair (i, j), checks if the fractional-coordinate difference
    exceeds 0.5 in any dimension; if so, stores DISPX/DISPY/DISPZ on atom i
    indicating the image offset needed to reconstruct the bond.

    Port of REPLICATE::setPBCbonds.
    """
    F = box.get("F")
    if F is None:
        return  # box not initialised — skip silently

    for i, nbrs in bonds.items():
        if not nbrs:
            continue
        ai = atoms.get(i)
        if ai is None:
            continue
        xi, yi, zi = float(ai["XCOORD"]), float(ai["YCOORD"]), float(ai["ZCOORD"])

        for j in nbrs:
            if i >= j:
                continue  # process each pair once
            aj = atoms.get(j)
            if aj is None:
                continue

            # Fractional displacement
            dx = float(aj["XCOORD"]) - xi
            dy = float(aj["YCOORD"]) - yi
            dz = float(aj["ZCOORD"]) - zi
            dfa = F[0][0]*dx + F[0][1]*dy + F[0][2]*dz
            dfb = F[1][0]*dx + F[1][1]*dy + F[1][2]*dz
            dfc = F[2][0]*dx + F[2][1]*dy + F[2][2]*dz

            ix = round(dfa)
            iy = round(dfb)
            iz = round(dfc)
            if ix == 0 and iy == 0 and iz == 0:
                continue  # same image

            # Annotate atom i with per-neighbour displacement indices
            _append_disp(ai, j, ix, iy, iz)
            _append_disp(aj, i, -ix, -iy, -iz)


def update_pbc_bonds(
    atoms: "AtomsDict",
    bonds: "BondsDict",
    offset: int,
    n_unit: int,
    dim: str,
) -> None:
    """Fix up PBC bonds after replicating along one dimension.

    After adding a replica (with index offset) along *dim*, some bonds in the
    original unit that crossed the +dim boundary now connect to the replica
    rather than the original's periodic image.  Reassign those bond targets.

    Port of REPLICATE::updatePBCbonds.
    """
    dim_idx = {"X": 0, "Y": 1, "Z": 2}[dim]

    for i in list(bonds.keys()):
        if i > offset:
            break  # only process original-unit atoms
        ai = atoms.get(i)
        if ai is None:
            continue

        dispx = ai.get("DISPX", [])
        dispy = ai.get("DISPY", [])
        dispz = ai.get("DISPZ", [])
        if not dispx:
            continue

        new_bonds = list(bonds.get(i, []))
        for k, j in enumerate(list(bonds.get(i, []))):
            disps = [
                dispx[k] if k < len(dispx) else 0,
                dispy[k] if k < len(dispy) else 0,
                dispz[k] if k < len(dispz) else 0,
            ]
            if disps[dim_idx] > 0 and j <= n_unit:
                # Bond crosses +dim boundary → point to replica
                new_j = j + offset
                new_bonds[k] = new_j
                # Add reverse bond on replica atom
                rev = bonds.setdefault(new_j, [])
                if i not in rev:
                    rev.append(i)
        bonds[i] = new_bonds


def _append_disp(atom: dict, neighbour: int, ix: int, iy: int, iz: int) -> None:
    """Append a displacement entry for a cross-boundary bond."""
    nbrs = atom.setdefault("DISPX_NBRS", [])
    nbrs.append(neighbour)
    atom.setdefault("DISPX", []).append(ix)
    atom.setdefault("DISPY", []).append(iy)
    atom.setdefault("DISPZ", []).append(iz)


# ── main replication ─────────────────────────────────────────────────────────

def replicate_cell(
    atoms: "AtomsDict",
    bonds: "BondsDict",
    box: dict,
    dims: dict[str, int],
    center: bool = False,
    update_res_num: bool = False,
    pbc: bool = True,
) -> tuple["AtomsDict", "BondsDict", dict]:
    """Replicate a periodic cell dims["X"] × dims["Y"] × dims["Z"] times.

    Port of REPLICATE::ReplicateCell (axis-by-axis strategy).
    Returns (new_atoms, new_bonds, new_box).
    """
    get_box_displacement_tensor(box)
    if pbc:
        set_pbc_bonds(atoms, bonds, box)

    if center:
        vec = [
            int((dims["X"] - 1) / 2),
            int((dims["Y"] - 1) / 2),
            int((dims["Z"] - 1) / 2),
        ]
        atoms = trans_cell_atoms(atoms, box, vec)

    for dim in ("X", "Y", "Z"):
        n = dims[dim]
        if n <= 1:
            continue

        unit_atoms = copy.deepcopy(atoms)
        unit_bonds = copy.deepcopy(bonds)
        n_unit = len(unit_atoms)

        for step in range(1, n):
            vec = [0.0, 0.0, 0.0]
            vec[{"X": 0, "Y": 1, "Z": 2}[dim]] = float(step)
            cell_atoms = trans_cell_atoms(unit_atoms, box, vec)
            offset = max(atoms)
            atoms, bonds = combine_mols(atoms, bonds, cell_atoms, unit_bonds, update_res_num)
            if pbc:
                update_pbc_bonds(atoms, bonds, offset, n_unit, dim)

        # Expand box along this dimension
        box[dim]["hi"]  = box[dim]["len"] * n
        box[dim]["lo"]  = 0.0
        box[dim]["len"] = box[dim]["len"] * n

    return atoms, bonds, box


# ── Euler rotation ───────────────────────────────────────────────────────────

def rotate(atoms: "AtomsDict", angles: list[float], coord: int = 3) -> None:
    """Apply an Euler rotation to all atom coordinates in-place.

    coord selects the rotation matrix:
        0 = Rx  1 = Ry  2 = Rz  3 = combined Rx*Ry*Rz (default)

    Port of General.pm::Rotate.
    """
    sa = [math.sin(a) for a in angles]
    ca = [math.cos(a) for a in angles]

    matrices = [
        # Rx
        np.array([[1,     0,      0    ],
                  [0,     ca[0],  sa[0]],
                  [0,    -sa[0],  ca[0]]], dtype=float),
        # Ry
        np.array([[ ca[1], 0,    -sa[1]],
                  [ 0,     1,     0    ],
                  [ sa[1], 0,     ca[1]]], dtype=float),
        # Rz
        np.array([[ ca[2], sa[2], 0],
                  [-sa[2], ca[2], 0],
                  [ 0,     0,     1]], dtype=float),
        # combined (Perl's coord=3 matrix — full Rx Ry Rz product)
        np.array([
            [ ca[1]*ca[2],  sa[0]*sa[1]*ca[2]+ca[0]*sa[2], -ca[0]*sa[1]*ca[2]+sa[0]*sa[2]],
            [-ca[1]*sa[2], -sa[0]*sa[1]*sa[2]+ca[0]*ca[2],  ca[0]*sa[1]*sa[2]+sa[0]*ca[2]],
            [ sa[1],        -sa[0]*ca[1],                    ca[0]*ca[1]                  ],
        ], dtype=float),
    ]
    R = matrices[coord]

    for a in atoms.values():
        v = np.array([float(a["XCOORD"]), float(a["YCOORD"]), float(a["ZCOORD"])])
        rv = R @ v
        a["XCOORD"], a["YCOORD"], a["ZCOORD"] = float(rv[0]), float(rv[1]), float(rv[2])
