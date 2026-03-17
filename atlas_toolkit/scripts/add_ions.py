"""
add_ions — port of addIons.pl

Replace randomly selected solvent molecules with single-atom ions.
Ions are specified by element symbol and looked up in the force field.

Usage:
  atlas-add-ions -b struct.bgf -f ff.ff -i Na -n 10 -s "resname eq WAT" -w out.bgf
  atlas-add-ions -b struct.bgf -f tip3p.ff -i "Na Cl" -n "0.15" -w out.bgf
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas_toolkit.core.general import com as _com, file_tester
from atlas_toolkit.core.headers import add_box_to_header, insert_header_remark
from atlas_toolkit.core.manip_atoms import get_atm_data, get_mols, select_atoms, add_mols_to_selection
from atlas_toolkit.core.box import get_box
from atlas_toolkit.io.bgf import make_seq_atom_index, parse_struct_file, write_bgf
from atlas_toolkit.io.ff import read_ff, find_ff


# ── minimal element mass table (for disambiguating multiple FF atomtypes) ────

_ELEMENT_MASSES: dict[str, float] = {
    "H": 1.008, "HE": 4.003, "LI": 6.941, "BE": 9.012, "B": 10.811,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "NE": 20.180,
    "NA": 22.990, "MG": 24.305, "AL": 26.982, "SI": 28.086, "P": 30.974,
    "S": 32.060, "CL": 35.450, "AR": 39.948, "K": 39.098, "CA": 40.078,
    "MN": 54.938, "FE": 55.845, "CO": 58.933, "NI": 58.693, "CU": 63.546,
    "ZN": 65.38,  "BR": 79.904, "SR": 87.62,  "MO": 95.96,  "RH": 102.906,
    "PD": 106.42, "AG": 107.87, "I":  126.904, "CS": 132.905, "BA": 137.327,
    "AU": 196.967, "HG": 200.59,
}


# ── ion parameter lookup ──────────────────────────────────────────────────────

def get_ion_parms(ff_parms: dict, element: str) -> dict:
    """Look up FF parameters for a monatomic ion by element symbol.

    Matches by the ATMNAME field in ATOMTYPES that contains the element symbol.
    Returns a dict with keys: FFTYPE, MASS, CHARGE, EPSILON, SIGMA.

    Raises ValueError if the element is not found.
    """
    atomtypes = ff_parms.get("ATOMTYPES", {})
    vdw = ff_parms.get("VDW", {})

    # Search by the ATOM field (chemical element symbol stored in ATOMTYPES),
    # then fall back to matching the fftype key itself.
    ele_upper = element.upper()
    candidates = []
    for atype, data in atomtypes.items():
        # The 'ATOM' field stores the chemical element symbol (e.g. 'Na', 'Cl')
        atom_field = str(data.get("ATOM", "")).upper()
        if atom_field == ele_upper:
            candidates.append(atype)

    if not candidates:
        # Fallback: match by fftype key name
        for atype in atomtypes:
            if atype.upper() == ele_upper:
                candidates.append(atype)

    if not candidates:
        raise ValueError(
            f"Ion element '{element}' not found in force field ATOMTYPES. "
            f"Available: {sorted(atomtypes.keys())}"
        )

    # If multiple candidates, pick the one whose MASS is closest to the known
    # atomic mass (disambiguates, e.g., Na vs I in AMBER when both have ATOM='Na')
    ref_mass = _ELEMENT_MASSES.get(ele_upper)
    if ref_mass is not None and len(candidates) > 1:
        candidates.sort(key=lambda t: abs(float(atomtypes[t].get("MASS", 0)) - ref_mass))

    fftype = candidates[0]
    data = atomtypes[fftype]
    mass = float(data.get("MASS", 0.0))
    charge = float(data.get("CHARGE", 0.0))

    # VDW parameters (sigma, epsilon) from DIAGONAL_VDW
    sigma = eps = 0.0
    vdw_entry = vdw.get(fftype, {}).get(fftype, {}).get(1, {})
    vals = vdw_entry.get("VALS", [])
    if len(vals) >= 2:
        eps, sigma = float(vals[0]), float(vals[1])

    return {
        "FFTYPE":  fftype,
        "ATMNAME": str(data.get("ATMNAME", element)),
        "RESNAME": element,
        "MASS":    mass,
        "CHARGE":  charge,
        "EPSILON": eps,
        "SIGMA":   sigma,
    }


# ── ion count resolution ──────────────────────────────────────────────────────

def resolve_ion_count(
    n_spec: str,
    box: dict,
    sys_charge: float,
    ion_charge: float,
) -> int:
    """Convert --nion spec to an integer count.

    n_spec:
      "0"   → neutralise: place enough ions to cancel sys_charge
      "N"   → integer N
      "X.x" → X.x molar concentration (using box volume in Å³)
    """
    try:
        val = float(n_spec)
    except ValueError:
        raise ValueError(f"Cannot parse --nion value: {n_spec!r}")

    if val == 0.0:
        # neutralise
        if ion_charge == 0:
            raise ValueError("Ion charge is zero — cannot neutralize.")
        n = int(round(-sys_charge / ion_charge))
        if n < 0:
            raise ValueError(
                f"Ion charge ({ion_charge:+.2f}) has the same sign as system charge "
                f"({sys_charge:+.2f}) — cannot neutralize."
            )
        return n

    if val != int(val):
        # molar concentration: N = c * V * N_A * 1e-27
        # V in Å³, c in mol/L, N_A = 6.022e23, 1 L = 1e27 Å³
        vol = 1.0
        for dim in ("X", "Y", "Z"):
            vol *= float(box[dim]["len"])
        n = int(round(6.022e-4 * val * vol))  # 6.022e23 / 1e27 = 6.022e-4
        return max(n, 0)

    return int(val)


# ── placement helpers ─────────────────────────────────────────────────────────

def _sys_charge(atoms: dict, solv_set: set[int]) -> float:
    """Sum charge of all non-solvent atoms."""
    return sum(float(atoms[i].get("CHARGE", 0.0)) for i in atoms if i not in solv_set)


def _place_ion_at_com(
    atoms: dict,
    bonds: dict,
    mol_members: dict,
    ion_parms: dict,
    new_idx: int,
    resnum: int,
) -> None:
    """Replace a solvent molecule with a single-atom ion placed at the molecule CoM.

    Modifies atoms and bonds in-place.
    """
    mol_data = get_atm_data(atoms, mol_members)
    centre = _com(mol_data)

    # Remove solvent atoms/bonds
    for j in mol_members:
        atoms.pop(j, None)
        bonds.pop(j, None)

    # Insert ion
    atoms[new_idx] = {
        "INDEX":     new_idx,
        "LABEL":     "HETATM",
        "ATMNAME":   ion_parms["ATMNAME"],
        "RESNAME":   ion_parms["RESNAME"],
        "RESNUM":    resnum,
        "CHAIN":     "X",
        "FFTYPE":    ion_parms["FFTYPE"],
        "CHARGE":    ion_parms["CHARGE"],
        "MASS":      ion_parms["MASS"],
        "NUMBONDS":  0,
        "LONEPAIRS": 0,
        "XCOORD":    centre["XCOORD"],
        "YCOORD":    centre["YCOORD"],
        "ZCOORD":    centre["ZCOORD"],
    }
    bonds[new_idx] = []


# ── main add_ions function ────────────────────────────────────────────────────

def add_ions(
    atoms: dict,
    bonds: dict,
    box: dict,
    ion_specs: list[tuple[str, str]],   # [(element, n_spec), ...]
    ff_parms: dict,
    solv_select: str = "resname eq WAT",
    randomize: bool = True,
) -> int:
    """Replace solvent molecules with ions.

    Parameters
    ----------
    atoms, bonds : modified in-place
    box          : cell box dict
    ion_specs    : list of (element, n_spec) pairs
    ff_parms     : parsed force field from read_ff()
    solv_select  : selection string identifying solvent atoms
    randomize    : if True, select random solvent molecules (else highest-energy)

    Returns
    -------
    Total number of ions placed.
    """
    from atlas_toolkit.core.manip_atoms import build_selection

    # Identify solvent atoms and molecules
    pred = build_selection(solv_select)
    solv_set: set[int] = {idx for idx, a in atoms.items() if pred(a)}
    add_mols_to_selection({i: 1 for i in solv_set}, atoms)
    solv_set_full = {idx for idx, a in atoms.items() if a.get("IS_SOLVENT") or idx in solv_set}

    # Re-select cleanly
    solv_dict = {i: 1 for i in solv_set}
    add_mols_to_selection(solv_dict, atoms)
    solv_set_full = set(solv_dict.keys())
    solv_mols = get_mols(atoms, bonds, {i: 1 for i in solv_set_full})

    sys_q = _sys_charge(atoms, solv_set_full)

    # Build ordered list of solvent molecule IDs
    mol_ids = list(solv_mols.keys())
    if randomize:
        random.shuffle(mol_ids)

    # Maximum existing index
    max_idx = max(atoms) if atoms else 0
    max_resnum = max(int(a.get("RESNUM", 0)) for a in atoms.values()) if atoms else 0

    total_placed = 0

    for element, n_spec in ion_specs:
        ion_parms = get_ion_parms(ff_parms, element)
        ion_q = ion_parms["CHARGE"]
        n_ions = resolve_ion_count(n_spec, box, sys_q, ion_q)

        print(f"  Placing {n_ions} {element} (charge {ion_q:+.2f}) ions...")

        placed = 0
        for mol_id in list(mol_ids):
            if placed >= n_ions:
                break
            if mol_id not in solv_mols:
                continue
            members = solv_mols[mol_id].get("MEMBERS", {})
            # Check members still in atoms
            if not all(j in atoms for j in members):
                continue

            max_idx += 1
            max_resnum += 1
            _place_ion_at_com(atoms, bonds, members, ion_parms, max_idx, max_resnum)
            mol_ids.remove(mol_id)
            placed += 1
            total_placed += 1
            sys_q += ion_q

        if placed < n_ions:
            print(f"  WARNING: Could only place {placed}/{n_ions} {element} ions "
                  f"(not enough solvent molecules).")

    return total_placed


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    file_tester(args.bgf)

    print(f"Parsing structure file {args.bgf}...")
    atoms, bonds, headers = parse_struct_file(args.bgf, save_headers=True)
    box = get_box(atoms, headers)
    print("Done")

    # Load force field
    if args.ff:
        ff_path = Path(args.ff)
        if not ff_path.exists():
            cand = find_ff(args.ff)
            if cand is None:
                raise FileNotFoundError(f"Cannot find force field: {args.ff!r}")
            ff_path = cand
        print(f"Loading force field {ff_path}...")
        ff_parms = read_ff(ff_path)
        print("Done")
    else:
        ff_parms = {"ATOMTYPES": {}, "VDW": {}, "PARMS": {}}

    # Parse ion specs: parallel lists from --ion-type and --nion
    ion_types = args.ion_type.split()
    nion_vals = args.nion.split()
    if len(nion_vals) == 1:
        nion_vals = nion_vals * len(ion_types)
    if len(nion_vals) != len(ion_types):
        raise ValueError("--ion-type and --nion must have the same number of entries.")
    ion_specs = list(zip(ion_types, nion_vals))

    print("Setting Ion options...Done")
    print("Placing ions...")
    n_placed = add_ions(
        atoms, bonds, box,
        ion_specs=ion_specs,
        ff_parms=ff_parms,
        solv_select=args.solvent or "resname eq WAT",
        randomize=args.random,
    )
    print(f"Placed {n_placed} ions total.")

    atoms, bonds = make_seq_atom_index(atoms, bonds)
    print(f"Creating {args.save or _default_save(args.bgf)}...")
    save_path = args.save or _default_save(args.bgf)
    insert_header_remark(headers, f"REMARK {args.bgf} + ions")
    write_bgf(atoms, bonds, save_path, headers)
    print("Done")


def _default_save(path: str) -> str:
    p = Path(path)
    return str(p.parent / (p.stem + "_ion.bgf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace solvent molecules with ions.",
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bgf", required=True, help="Input BGF structure file")
    parser.add_argument("-f", "--ff", required=True, help="Force field file or name")
    parser.add_argument("-i", "--ion-type", required=True,
                        help="Ion element symbol(s) e.g. 'Na' or 'Na Cl'")
    parser.add_argument("-n", "--nion", default="0",
                        help="Number/concentration of ions: 0=neutralize, N=integer, X.x=molar")
    parser.add_argument("-s", "--solvent", default=None,
                        help="Solvent selection string (default: 'resname eq WAT')")
    parser.add_argument("-w", "--save", default=None, help="Output file name")
    parser.add_argument("-r", "--random", action="store_true",
                        help="Select solvent molecules at random (default)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
