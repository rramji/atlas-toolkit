# ATLAS Toolkit — Python Port

Python port of the [ATLAS molecular simulation toolkit](https://github.com/atlas-nano/ATLAS-toolkit), originally written in Perl. Targets maintainability, extensibility, and ease of use for research groups working with BGF-format molecular structure files and LAMMPS-based MD simulations.

---

## Installation

```bash
cd atlas_toolkit
pip install -e ".[dev]"   # editable install with test dependencies
```

Requires Python ≥ 3.10. Core dependency: `numpy`. No other non-stdlib dependencies.

### Force field files

`atlas_toolkit/data/ff/` bundles Cerius2 `.ff` files and AMBER/DREIDING/GAFF/OPLS/UFF parameter sets (~37 MB). The large CHARMM directories (~140 MB) are excluded; to use them set:

```bash
export ATLAS_FF_DIR=/path/to/ATLAS-toolkit/ff
```

---

## CLI reference

All scripts are available as `atlas-*` entry points after installation.

| Command | Perl equivalent | Description |
|---------|----------------|-------------|
| `atlas-modify-atom-data` | `modifyAtomData.pl` | Assign/add/subtract atom field values by selection |
| `atlas-replicate` | `replicate.pl` | Replicate a unit cell |
| `atlas-trim-cell` | `trimCell.pl` | Trim a replicated cell to target dimensions |
| `atlas-remove-mols` | `removeMols.pl` | Remove molecules by atom selection |
| `atlas-embed-molecule` | `embedMolecule.pl` | Embed a solute into a solvent box (overlap removal) |
| `atlas-add-solvent` | `addSolvent.pl` | Solvate a structure with a given water model or molecule |
| `atlas-add-ions` | `addIons.pl` | Add counterions by replacing solvent molecules |
| `atlas-add-box-to-bgf` | `addBoxToBGF.pl` | Add/update CRYSTX box header from atom coordinates |
| `atlas-create-lammps-input` | `createLammpsInput.pl` | Generate LAMMPS data file + input script from BGF + FF |
| `atlas-update-bgf-coords` | `convertLammpsTrj.pl` (subset) | Update BGF coords from last frame of a LAMMPS dump |
| `atlas-convert-lammps-trj` | `convertLammpsTrj.pl` | Convert LAMMPS dump → LAMMPS/AMBER/XYZ/BGF/PDB |
| `atlas-get-bgf-atoms` | `getBGFAtoms.pl` | Extract atom subset by selection, optionally whole molecules |
| `atlas-combine-bgf` | `combineBGF.pl` | Merge two or more BGF files into one |
| `atlas-get-bounds` | `getBounds.pl` | Print XYZ coordinate min/max for selected atoms |

### Quick examples

```bash
# Replicate 2×2×1
atlas-replicate -b struct.bgf -d "2 2 1" -w replicated.bgf

# Solvate with SPC water to 1000 total molecules
atlas-add-solvent -i solute.bgf -n "total: 1000" -w spc -s solvated.bgf

# Neutralise with Na+ ions
atlas-add-ions -b solvated.bgf -f AMBER99.ff -i Na -n 0 -w neutralised.bgf

# Generate LAMMPS input (minimisation)
atlas-create-lammps-input -b solvated.bgf -f "AMBER99.ff custom.frcmod" -t min -s job

# Extract final frame of a LAMMPS run back into BGF
atlas-update-bgf-coords -b solvated.bgf -l dump.npt -w final.bgf -c 1

# Convert full trajectory to AMBER mdcrd (every 5th frame)
atlas-convert-lammps-trj -b solvated.bgf -l dump.npt -o amber -t "1-1000:5" -s out.mdcrd
```

### `atlas-modify-atom-data`

```bash
atlas-modify-atom-data -s input.bgf -a "fftype eq Cl-" -f "CHARGE:-1.0" -w output.bgf
atlas-modify-atom-data -s input.bgf -a "resname eq WAT" -f "CHARGE:+0.1"
atlas-modify-atom-data -s input.bgf -a "index>0" -f "RESNAME:RES CHARGE:-1.0"
atlas-modify-atom-data -s input.bgf -a "fftype eq OW" -f "RESNAME:SOL" --mol-opt
atlas-modify-atom-data -s input.bgf -a "resname eq WAT" -f "CHARGE:+0.5" --random 50
```

**Field spec** (`-f`): `FIELD:VALUE` assign · `FIELD:+VALUE` add · `FIELD:-VALUE` subtract · `FIELD:.VALUE` string append

### `atlas-create-lammps-input`

```bash
atlas-create-lammps-input -b struct.bgf -f "ff.ff custom.frcmod" -t min -s job
atlas-create-lammps-input -b struct.bgf -f "ff.ff custom.frcmod" -t nvt -s job
atlas-create-lammps-input -b struct.bgf -f "ff.ff custom.frcmod" -t npt -s job
```

Writes `data.<stem>` (LAMMPS full atom-style data file) and `in.<stem>` (input script).

### `atlas-update-bgf-coords` / `atlas-convert-lammps-trj`

| Flag | Default | Description |
|------|---------|-------------|
| `-u 0\|1` | 1 | Unwrap coordinates using image flags (ix iy iz) |
| `-c 0\|1` | 0 | Shift mass-weighted COM to box centre, re-wrap |
| `-t SEL` | `*` | Frame selection: `*`, `5`, `1-100:5`, `"1-50 200"` |
| `-o TYPE` | lammps | Output: `lammps`, `bgf`, `pdb`, `xyz`, `amber` |

**Coordinate column auto-detection:** `x y z` · `xs ys zs` (scaled) · `xu yu zu` (pre-unwrapped) · `xsu ysu zsu` (pre-unwrapped scaled)

---

## Atom selection syntax

Used by `-a` in most scripts:

```
*                              # all atoms
index > 100                    # numeric: > < >= <= == !=
fftype eq CT                   # string equality
resname ne WAT                 # string not-equal
fftype =~ ^H                   # regex match
resname eq WAT and charge < 0  # AND
fftype eq OW or fftype eq HW   # OR
```

Fields (case-insensitive): `INDEX` `ATMNAME` `RESNAME` `CHAIN` `RESNUM`
`XCOORD` `YCOORD` `ZCOORD` `FFTYPE` `NUMBONDS` `LONEPAIRS` `CHARGE`
`MOLECULEID` `MOLSIZE` `FA` `FB` `FC`

---

## Library API

```python
from atlas_toolkit.io.bgf import read_bgf, write_bgf
from atlas_toolkit.core.manip_atoms import get_mols, select_atoms
from atlas_toolkit.core.box import get_box
from atlas_toolkit.io.ff import load_ff
from atlas_toolkit.lammps.dump import read_last_frame, apply_coords_to_atoms, recenter_atoms

atoms, bonds, headers = read_bgf("struct.bgf")
write_bgf(atoms, bonds, "out.bgf", headers)

selected = select_atoms("fftype eq OW", atoms)
mols = get_mols(atoms, bonds)   # annotates MOLECULEID, MOLSIZE in-place
box  = get_box(atoms, headers=headers)

parms = load_ff("AMBER99.ff custom.frcmod")
# parms keys: ATOMTYPES, VDW, BONDS, ANGLES, TORSIONS, INVERSIONS

ts, dump_atoms, box, columns = read_last_frame("dump.npt")
apply_coords_to_atoms(atoms, dump_atoms, box, columns, unwrap=True)
recenter_atoms(atoms, box)   # optional: align COM with box centre
```

---

## Data model

```python
atoms: dict[int, dict] = {
    1: {
        "INDEX": 1, "ATMNAME": "O1", "RESNAME": "WAT",
        "XCOORD": -0.239, "YCOORD": -0.309, "ZCOORD": 0.0,
        "FFTYPE": "OW", "CHARGE": -0.834, "NUMBONDS": 2, "LONEPAIRS": 0,
        "MOLECULEID": 1, "MOLSIZE": 3,   # populated by get_mols()
    },
    ...
}
bonds: dict[int, list[int]] = {1: [2, 3], 2: [1], 3: [1]}
headers: list[str] = ["BIOGRF 200", "CRYSTX  30.0 ...", ...]
```

---

## Running tests

```bash
pytest tests/ -v                       # all tests
pytest tests/ -v -m "not perl_oracle"  # skip Perl comparison tests
pytest tests/oracle/ -v                # comparison tests only
```

**277 tests** passing (262 unit + 15 oracle). Perl oracle/comparison tests are automatically
skipped when `~/ATLAS-toolkit` is not found or Perl is not installed.

---

## Project status

### Complete

| Milestone | Key modules | Tests |
|-----------|-------------|-------|
| **1** BGF I/O + `modify_atom_data` | `io/bgf.py`, `core/manip_atoms.py`, `scripts/modify_atom_data.py` | 49 |
| **2** Box geometry + replication | `core/box.py`, `core/replicate.py`, `scripts/replicate.py` | +30 |
| **3** Solvent/ion placement | `io/ff.py` (VDW), `scripts/trim_cell.py`, `remove_mols.py`, `embed_molecule.py`, `add_solvent.py`, `add_ions.py`, `add_box_to_bgf.py`, `data/wat/` | +60 |
| **5a** FF bonded terms + frcmod | `io/ff.py` (bonded + wildcards + `load_ff`), `io/frcmod.py` | +30 |
| **5b** LAMMPS input generation | `lammps/topology.py`, `lammps/data_file.py`, `lammps/input_script.py`, `scripts/create_lammps_input.py` | +36 |
| **5c** `add_box_to_bgf` | `scripts/add_box_to_bgf.py` | +8 |
| **6** LAMMPS trajectory I/O | `lammps/dump.py`, `scripts/update_bgf_coords.py`, `scripts/convert_lammps_trj.py` | +35 |
| **7** BGF utilities | `scripts/get_bgf_atoms.py`, `scripts/combine_bgf.py`, `scripts/get_bounds.py`; `core/manip_atoms.get_bounds` | +14 |
| **Oracle** | `tests/oracle/` — 15 Perl-vs-Python comparison tests (replicate, trim, remove_mols, modify_atom_data, add_box, add_solvent) | +15 |
| **Total** | | **277** |

### Not yet ported

| Item | Notes |
|------|-------|
| Milestone 4 — PDB/MOL2 readers | Not required for current pipeline |
| Analysis scripts (`rdf.py`, `get_bounds.py`, etc.) | Post-processing; lower priority |
| CHARMM FF bundle | Too large for package; use `ATLAS_FF_DIR` |
| GitHub Actions CI | Not yet set up |

---

## Improvements over the Perl version

| Issue in Perl | Python fix |
|---------------|-----------|
| `CoM()` for single-atom molecules returns the atom hash directly → aliasing zeros coords on rotate | Always returns a copy; fixes the Cl- counterion placement bug |
| `getMolList` is recursive → hits stack limits on large polymers | Iterative BFS with `collections.deque` |
| `BuildAtomSelectionString` uses `eval` on user input | Compiled predicate parser — no `eval` |
| `HEADER` key mixed into the atoms dict and stripped on write | Headers kept as a separate `list[str]` throughout |
| No support for `xu yu zu` / `xsu ysu zsu` dump columns | Auto-detects all four LAMMPS coordinate conventions |
| CHARMM dihedral written as `K d n wlj wcoul` (two 1-4 weights) | Correctly written as `K n d w` (single weight, phase in degrees) |
| `addSolvent.pl` with `total: N` on a box-less solute trims solvent to solute bounding box (nearly empty) | Skips trim when no replication is needed; uses full solvent box then removes excess |
