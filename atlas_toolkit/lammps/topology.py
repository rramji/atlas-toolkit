"""
Enumerate bonded interactions from a BGF connectivity dict.

bonds_dict : dict[int, list[int]]
    atom_id → list of neighbour atom_ids  (as returned by read_bgf)

All returned tuples are in *canonical* order so that the same interaction
is never listed twice.
"""
from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def enumerate_bonds(bonds_dict: dict) -> list[tuple[int, int]]:
    """Return (i, j) pairs with i < j for every covalent bond."""
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for i, nbrs in bonds_dict.items():
        for j in nbrs:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                result.append(key)
    result.sort()
    return result


def enumerate_angles(bonds_dict: dict) -> list[tuple[int, int, int]]:
    """Return (i, j, k) triples where j is the vertex (center) atom.

    Canonical form: i < k (endpoints sorted so each angle appears once).
    """
    seen: set[tuple[int, int, int]] = set()
    result: list[tuple[int, int, int]] = []
    for j, nbrs in bonds_dict.items():
        nbr_list = list(nbrs)
        for idx, i in enumerate(nbr_list):
            for k in nbr_list[idx + 1 :]:
                lo, hi = (i, k) if i < k else (k, i)
                key = (lo, j, hi)
                if key not in seen:
                    seen.add(key)
                    result.append((lo, j, hi))
    result.sort()
    return result


def enumerate_torsions(bonds_dict: dict) -> list[tuple[int, int, int, int]]:
    """Return (i, j, k, l) 4-tuples for every proper dihedral.

    Canonical form: tuple is lexicographically ≤ its reverse.
    """
    seen: set[tuple[int, int, int, int]] = set()
    result: list[tuple[int, int, int, int]] = []
    for j in bonds_dict:
        for k in bonds_dict[j]:
            if k <= j:
                continue  # process each j-k bond once
            for i in bonds_dict[j]:
                if i == k:
                    continue
                for l in bonds_dict[k]:
                    if l == j:
                        continue
                    if l == i:
                        continue  # avoid i==l (degenerate)
                    fwd = (i, j, k, l)
                    rev = (l, k, j, i)
                    canon = min(fwd, rev)
                    if canon not in seen:
                        seen.add(canon)
                        result.append(canon)
    result.sort()
    return result


def enumerate_impropers(bonds_dict: dict) -> list[tuple[int, int, int, int]]:
    """Return (center, a, b, c) 4-tuples for all atoms with ≥ 3 bonds.

    For each such center, all C(n, 3) satellite combinations are returned.
    Satellites are stored in sorted order.
    """
    seen: set[tuple[int, int, int, int]] = set()
    result: list[tuple[int, int, int, int]] = []
    for center, nbrs in bonds_dict.items():
        nbr_list = sorted(nbrs)
        if len(nbr_list) < 3:
            continue
        for trio in combinations(nbr_list, 3):
            key = (center, trio[0], trio[1], trio[2])
            if key not in seen:
                seen.add(key)
                result.append(key)
    result.sort()
    return result
