"""
Atom manipulation — port of ManipAtoms.pm (subset for modifyAtomData).

Key differences from Perl:
  - get_mols uses iterative BFS instead of recursion (avoids Python recursion limit).
  - select_atoms uses a compiled predicate function instead of eval().
  - com() for single atoms returns a copy, not a reference (bug fix).
"""
import re
import random
from collections import deque
from typing import Any, Callable, Optional

from ..types import AtomsDict, BondsDict, MolsDict

__all__ = [
    "get_mols",
    "add_mols_to_selection",
    "build_selection",
    "select_atoms",
    "get_atm_data",
    "get_bounds",
]

# ── field names recognised in selection strings ────────────────────────────
_KNOWN_FIELDS: set[str] = {
    "INDEX", "ATMNAME", "RESNAME", "CHAIN", "RESNUM",
    "XCOORD", "YCOORD", "ZCOORD", "FFTYPE", "NUMBONDS",
    "LONEPAIRS", "CHARGE", "MOLECULEID", "MOLSIZE",
    "FA", "FB", "FC",
}


# ── molecule detection ─────────────────────────────────────────────────────

def get_mols(
    atoms: AtomsDict,
    bonds: BondsDict,
    select: Optional[dict[int, Any]] = None,
) -> MolsDict:
    """Flood-fill bond graph to assign molecules (Perl GetMols).

    Annotates each atom in *atoms* in place with:
      atom['MOLECULE']   = {'INDEX': n, 'MEMBERS': {idx: 1, ...}, 'MOLSIZE': k}
      atom['MOLECULEID'] = n   (int, not a scalar reference as in Perl)
      atom['MOLSIZE']    = k

    Returns mols dict: {mol_index: {'INDEX': n, 'MEMBERS': {...}, 'MOLSIZE': k}}
    """
    if select is None:
        select = atoms

    used: set[int] = set()
    mols: MolsDict = {}
    counter = 0

    for atom_id in sorted(atoms):
        if atom_id in used or atom_id not in select:
            continue

        counter += 1
        mol: dict = {"INDEX": counter, "MEMBERS": {}, "MOLSIZE": 0}
        mols[counter] = mol

        # Iterative BFS (avoids Python recursion limit on large molecules)
        queue: deque[int] = deque([atom_id])
        while queue:
            curr = queue.popleft()
            if curr in used:
                continue
            used.add(curr)
            mol["MEMBERS"][curr] = 1
            mol["MOLSIZE"] += 1

            # Annotate the atom
            for field in ("MOLECULE", "MOLECULEID", "MOLSIZE"):
                atoms[curr].pop(field, None)
            atoms[curr]["MOLECULE"] = mol
            atoms[curr]["MOLECULEID"] = counter
            atoms[curr]["MOLSIZE"] = mol["MOLSIZE"]

            for neighbour in (bonds.get(curr) or []):
                if neighbour not in used:
                    queue.append(neighbour)

        # Update MOLSIZE on all members now that the flood-fill is complete
        for mid in mol["MEMBERS"]:
            atoms[mid]["MOLSIZE"] = mol["MOLSIZE"]

    return mols


def add_mols_to_selection(select: dict[int, Any], atoms: AtomsDict) -> None:
    """Expand selection to include all atoms in any selected molecule (Perl AddMolsToSelection).

    Modifies *select* in place.
    """
    for atom_id in list(select):
        mol = atoms[atom_id].get("MOLECULE")
        if mol:
            for member_id in mol["MEMBERS"]:
                select[member_id] = 1


def get_atm_data(all_atoms: AtomsDict, atom_list: dict[int, Any]) -> AtomsDict:
    """Return a view of atoms restricted to atom_list (Perl GetAtmData).

    Values are references to the original atom dicts (not copies).
    """
    return {idx: all_atoms[idx] for idx in atom_list if idx in all_atoms}


# ── atom selection DSL ─────────────────────────────────────────────────────

def build_selection(sel_str: str) -> Callable[[dict], bool]:
    """Compile a selection string to a predicate (Perl BuildAtomSelectionString).

    Supported syntax:
      *                          — all atoms
      field op value             — simple comparison
      expr and expr              — conjunction
      expr or expr               — disjunction
      not expr                   — negation

    Fields (case-insensitive): INDEX ATMNAME RESNAME CHAIN RESNUM XCOORD
      YCOORD ZCOORD FFTYPE NUMBONDS LONEPAIRS CHARGE MOLECULEID MOLSIZE
      FA FB FC

    Operators: > < >= <= == !=  (numeric)
               eq ne            (string equality, case-sensitive)
               =~ !~            (regex match against string representation)

    Examples:
      "*"
      "fftype eq Cl-"
      "resname eq WAT and charge < 0"
      "index > 0"
      "charge >= -1.0 or charge <= 1.0"
    """
    sel = sel_str.strip()
    if sel == "*":
        return lambda atom: True
    return _compile_or(sel)


def select_atoms(
    sel_str: str,
    atoms: AtomsDict,
    box: dict | None = None,
) -> dict[int, int]:
    """Return {atom_idx: 1} for all atoms matching selection (Perl SelectAtoms).

    Raises ValueError if no atoms match or if the selection string is invalid.
    """
    pred = build_selection(sel_str)
    result = {idx: 1 for idx, atom in atoms.items() if pred(atom)}
    if not result:
        raise ValueError(f"ERROR: No atoms matched selection '{sel_str}'!")
    return result


# ── selection compiler internals ───────────────────────────────────────────

def _compile_or(expr: str) -> Callable[[dict], bool]:
    # Split on bare 'or' (not inside parentheses)
    parts = _split_on_keyword(expr, "or")
    if len(parts) > 1:
        preds = [_compile_and(p) for p in parts]
        return lambda atom, ps=preds: any(p(atom) for p in ps)
    return _compile_and(expr)


def _compile_and(expr: str) -> Callable[[dict], bool]:
    parts = _split_on_keyword(expr, "and")
    if len(parts) > 1:
        preds = [_compile_not(p) for p in parts]
        return lambda atom, ps=preds: all(p(atom) for p in ps)
    return _compile_not(expr)


def _compile_not(expr: str) -> Callable[[dict], bool]:
    expr = expr.strip()
    if re.match(r"^not\s+", expr, re.IGNORECASE):
        inner = _compile_not(expr[3:].strip())
        return lambda atom, p=inner: not p(atom)
    return _compile_predicate(expr)


def _compile_predicate(expr: str) -> Callable[[dict], bool]:
    expr = expr.strip()

    # Parenthesised sub-expression
    if expr.startswith("(") and expr.endswith(")"):
        return _compile_or(expr[1:-1])

    # field op value
    m = re.match(
        r"^([a-zA-Z_]+)\s*(>=|<=|!=|==|>|<|=~|!~|eq|ne)\s*(.+)$",
        expr,
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Cannot parse selection predicate: {expr!r}")

    field = m.group(1).upper()
    op = m.group(2).lower()
    raw_val = m.group(3).strip().strip("\"'")

    # Determine if value is numeric
    try:
        num_val = float(raw_val)
        is_numeric = True
    except ValueError:
        num_val = 0.0
        is_numeric = False

    str_val = raw_val

    def predicate(
        atom: dict,
        _field: str = field,
        _op: str = op,
        _num: float = num_val,
        _str: str = str_val,
        _is_num: bool = is_numeric,
    ) -> bool:
        atom_val = atom.get(_field)
        if atom_val is None:
            return False
        try:
            if _op == ">":
                return float(atom_val) > _num
            if _op == "<":
                return float(atom_val) < _num
            if _op == ">=":
                return float(atom_val) >= _num
            if _op == "<=":
                return float(atom_val) <= _num
            if _op == "==":
                return float(atom_val) == _num if _is_num else str(atom_val) == _str
            if _op == "!=":
                return float(atom_val) != _num if _is_num else str(atom_val) != _str
            if _op == "eq":
                return str(atom_val) == _str
            if _op == "ne":
                return str(atom_val) != _str
            if _op == "=~":
                return bool(re.search(_str, str(atom_val)))
            if _op == "!~":
                return not bool(re.search(_str, str(atom_val)))
        except (TypeError, ValueError):
            return False
        return False

    return predicate


def _split_on_keyword(expr: str, keyword: str) -> list[str]:
    """Split expr on bare 'keyword' (not inside parens), case-insensitive."""
    pattern = re.compile(r"\b" + keyword + r"\b", re.IGNORECASE)
    parts: list[str] = []
    depth = 0
    last = 0
    i = 0
    while i < len(expr):
        if expr[i] == "(":
            depth += 1
            i += 1
        elif expr[i] == ")":
            depth -= 1
            i += 1
        elif depth == 0:
            m = pattern.match(expr, i)
            if m:
                parts.append(expr[last:i].strip())
                last = m.end()
                i = last
            else:
                i += 1
        else:
            i += 1
    parts.append(expr[last:].strip())
    return [p for p in parts if p]


# ── atom data extraction ─────────────────────────────────────────────────────

def get_atm_data(atoms: AtomsDict, index_set: dict) -> AtomsDict:
    """Return a view of atoms restricted to the keys in index_set.

    Port of ManipAtoms.pm::GetAtmData.
    Does NOT deep-copy — shares the underlying atom dicts.
    """
    return {idx: atoms[idx] for idx in index_set if idx in atoms}


def get_bounds(atoms: AtomsDict, selection: dict | None = None) -> dict:
    """Return coordinate min/max for selected atoms.

    Port of getBounds.pl::getBounds.

    Parameters
    ----------
    atoms     : full atoms dict
    selection : keys to include (default: all atoms)

    Returns
    -------
    {"X": {"min": float, "max": float}, "Y": ..., "Z": ...}
    """
    keys = list(selection) if selection is not None else list(atoms)
    result: dict = {}
    for dim in ("X", "Y", "Z"):
        coord = f"{dim}COORD"
        vals = [float(atoms[k][coord]) for k in keys if k in atoms]
        if vals:
            result[dim] = {"min": min(vals), "max": max(vals)}
        else:
            result[dim] = {"min": 0.0, "max": 0.0}
    return result


# ── periodic image repair ────────────────────────────────────────────────────

def reimage_atoms(
    atoms: AtomsDict,
    bonds: BondsDict,
    mols: dict,
    box: dict,
    selection: dict | None = None,
) -> None:
    """Image molecules back into the periodic box.  Modifies atoms in-place.

    For each molecule:
    1. Walk bonded pairs and fold atoms that are more than half a cell length
       away from their bonded partner back by one cell vector.
    2. Compute the molecule CoM; if it lies outside [0, box_len), shift all
       atoms by the appropriate number of cell lengths.

    Port of ManipAtoms.pm::ReimageAtoms.
    """
    from atlas_toolkit.core.general import com as _com

    if selection is None:
        selection = {idx: 1 for idx in atoms}

    for mol in mols.values():
        members: dict = mol.get("MEMBERS", {})

        # Step 1: unfold internal bonds
        for j in members:
            if j not in selection or j not in atoms:
                continue
            for k in bonds.get(j, []):
                if k <= j:
                    continue
                for dim in ("X", "Y", "Z"):
                    coord = f"{dim}COORD"
                    blen = box[dim]["len"]
                    dist = abs(float(atoms[j][coord]) - float(atoms[k][coord]))
                    if dist > 4.0:
                        factor = 1 if float(atoms[k][coord]) > float(atoms[j][coord]) else -1
                        while dist > 4.0:
                            dist -= blen
                            atoms[k][coord] = float(atoms[k][coord]) - factor * blen

        # Step 2: shift CoM into box
        mol_atoms = get_atm_data(atoms, members)
        if not mol_atoms:
            continue
        c = _com(mol_atoms)
        shift = {}
        for dim in ("X", "Y", "Z"):
            coord = f"{dim}COORD"
            blen = box[dim]["len"]
            cv = float(c[coord])
            s = 0.0
            if cv < 0:
                while cv < 0:
                    s += blen
                    cv += blen
            elif cv > blen:
                while cv > blen:
                    s -= blen
                    cv -= blen
            shift[coord] = s

        if any(v != 0.0 for v in shift.values()):
            for j in members:
                if j in atoms:
                    for coord, s in shift.items():
                        atoms[j][coord] = float(atoms[j][coord]) + s
