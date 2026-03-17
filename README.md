# ATLAS Toolkit — Python Port

Python port of the [ATLAS molecular simulation toolkit](https://github.com/atlas-nano/ATLAS-toolkit), originally written in Perl. Targets maintainability, extensibility, and ease of use for research groups working with BGF-format molecular structure files and LAMMPS-based MD simulations.

---

## Installation

```bash
cd atlas_toolkit
pip install -e ".[dev]"   # editable install with test dependencies
```

Requires Python 3.10+. No non-stdlib dependencies for the core library.

---

## Usage

### As a library

```python
from atlas_toolkit import read_bgf, write_bgf, select_atoms, get_mols

atoms, bonds, headers = read_bgf("my_structure.bgf")

# Select atoms
selected = select_atoms("fftype eq Cl-", atoms)
selected = select_atoms("charge < 0 and resname eq WAT", atoms)
selected = select_atoms("index > 100", atoms)

# Detect molecules (annotates atoms in-place with MOLECULEID, MOLSIZE)
mols = get_mols(atoms, bonds)

# Modify and save
for idx in selected:
    atoms[idx]["CHARGE"] = -1.5
write_bgf(atoms, bonds, "output.bgf", headers)
```

### CLI — `modify_atom_data` (port of `modifyAtomData.pl`)

```bash
# Assign a field value
atlas-modify-atom-data -s input.bgf -a "fftype eq Cl-" -f "CHARGE:-1.0" -w output.bgf

# Add to a field
atlas-modify-atom-data -s input.bgf -a "resname eq WAT" -f "CHARGE:+0.1"

# Multiple fields at once
atlas-modify-atom-data -s input.bgf -a "index>0" -f "RESNAME:RES CHARGE:-1.0"

# Expand selection to whole molecules
atlas-modify-atom-data -s input.bgf -a "fftype eq OW" -f "RESNAME:SOL" --mol-opt

# Modify N randomly chosen atoms
atlas-modify-atom-data -s input.bgf -a "resname eq WAT" -f "CHARGE:+0.5" --random 50

# Remove bonds between groups before modifying
atlas-modify-atom-data -s input.bgf -a "*" -f "CHARGE:0" \
    --delete-bonds "resname eq AU::resname eq WAT"

# Run directly without installing
python atlas_toolkit/scripts/modify_atom_data.py -s input.bgf -a "index>0" -f "FFTYPE:Cl-"
```

### Field spec format (`-f`)

| Syntax | Effect | Example |
|--------|--------|---------|
| `FIELD:VALUE` | Assign | `RESNAME:RES` |
| `FIELD:-VALUE` | Subtract | `CHARGE:-0.1` |
| `FIELD:+VALUE` | Add | `CHARGE:+0.5` |
| `FIELD:.VALUE` | String append | `RESNAME:.2` |

### Selection syntax (`-a`)

```
*                                    # all atoms
index > 0                            # numeric comparison: > < >= <= == !=
fftype eq Cl-                        # string equality (case-sensitive value)
resname ne WAT                       # string not-equal
fftype =~ ^H                         # regex match
resname eq WAT and charge < 0        # conjunction
fftype eq OW or fftype eq HW         # disjunction
```

Available fields (case-insensitive): `INDEX` `ATMNAME` `RESNAME` `CHAIN` `RESNUM`
`XCOORD` `YCOORD` `ZCOORD` `FFTYPE` `NUMBONDS` `LONEPAIRS` `CHARGE`
`MOLECULEID` `MOLSIZE` `FA` `FB` `FC`

---

## Running tests

```bash
pytest tests/ -v
```

The oracle test (`test_oracle_charge_assignment`) compares Python output against Perl output line-by-line. It is skipped automatically if Perl is not installed.

---

## Data model

The core data model is a plain `dict[int, dict]`, mirroring the Perl hash-of-hashes:

```python
atoms: dict[int, dict] = {
    1: {
        "INDEX": 1, "ATMNAME": "O1", "RESNAME": "WAT",
        "XCOORD": -0.239, "YCOORD": -0.309, "ZCOORD": 0.0,
        "FFTYPE": "OW", "CHARGE": -0.834, "NUMBONDS": 2, ...
    },
    ...
}
bonds: dict[int, list[int]] = {1: [2, 3], 2: [1], 3: [1]}
headers: list[str] = ["BIOGRF 200", "DESCRP tip3", "CRYSTX ...", ...]
```

Atom keys are 1-based integers. Fields are strings matching the BGF column names.
`TypedDict` definitions in `atlas_toolkit/types.py` document all known fields.

### Intentional improvements over the Perl version

| Issue in Perl | Fix in Python |
|---------------|---------------|
| `CoM()` for single-atom molecules returns the atom hash directly, causing aliasing that zeros coordinates on rotate | Returns a copy — fixes the Cl- counterion stacking bug |
| `getMolList` is recursive — hits Python's recursion limit on large polymers | Replaced with iterative BFS using `collections.deque` |
| `BuildAtomSelectionString` uses `eval` on user-supplied strings | Replaced with a compiled predicate parser — no `eval` |
| `HEADER` key stuffed into the atoms dict and stripped on write | Headers kept as a separate list throughout |
| `TransCellAtomsOld` shallow-copies atoms, mutating the source as a side effect | Documented; fix planned for `replicate.py` port |

---

## Project status

### Done (Milestone 1 — BGF reader/writer + `modify_atom_data`)

| Module | Perl source | Notes |
|--------|-------------|-------|
| `atlas_toolkit/types.py` | — | `AtomRecord` TypedDict, type aliases |
| `atlas_toolkit/core/general.py` | `General.pm` (subset) | `trim`, `file_tester`, `has_cell`, `com` |
| `atlas_toolkit/core/headers.py` | `FileFormats.pm` | `create_headers`, `insert_header_remark`, `add_box_to_header` |
| `atlas_toolkit/core/manip_atoms.py` | `ManipAtoms.pm` (subset) | `get_mols`, `select_atoms`, `build_selection`, `add_mols_to_selection` |
| `atlas_toolkit/io/bgf.py` | `FileFormats.pm` | `read_bgf`, `write_bgf`, `parse_struct_file` (BGF only) |
| `atlas_toolkit/scripts/modify_atom_data.py` | `modifyAtomData.pl` | Full CLI port |
| `tests/` | — | 49/49 passing incl. Perl oracle test |

### Done (Milestone 2 — Core geometry & replication)

| Module | Perl source | Notes |
|--------|-------------|-------|
| `atlas_toolkit/core/box.py` | `BOX.pm` | `get_box`, `init_box`, `cart2frac`, `frac2cart`, `map2unit_cell`, `center_atoms`, displacement tensor, H/F matrices (numpy) |
| `atlas_toolkit/core/replicate.py` | `REPLICATE.pm`, `General.pm::Rotate` | `replicate_cell`, `trans_cell_atoms`, `combine_mols`, `set_pbc_bonds`, `rotate` |
| `atlas_toolkit/scripts/replicate.py` | `replicate.pl` | Full CLI port; `atlas-replicate` entry point |
| `atlas_toolkit/data/ff/` | `ATLAS-toolkit/ff/` | Bundled force field files (~37 MB; CHARMM dirs excluded — see below) |
| `tests/test_box.py`, `tests/test_replicate.py` | — | 30/30 passing incl. Perl oracle test |

**Dependencies added:** `numpy>=1.22` (used by geometry/rotation code; BGF I/O remains zero-dependency).

**Force field directory:** The `atlas_toolkit/data/ff/` directory contains the Cerius2 `.ff` files and AMBER/DREIDING/GAFF/OPLS/UFF parameter sets bundled with the package. The large CHARMM directories (~140 MB) are excluded from the bundle; to use them point the `ATLAS_FF_DIR` environment variable at your full ATLAS-toolkit `ff/` directory.

### Done (Milestone 3 — Solvent/ion placement)

| Module | Perl source | Notes |
|--------|-------------|-------|
| `atlas_toolkit/io/ff.py` | `CERIUS2.pm` | `read_ff`, `find_ff`, `get_vdw_radius`; parses ATOMTYPES + DIAGONAL/OFF_DIAGONAL_VDW |
| `atlas_toolkit/io/bgf.py` additions | `FileFormats.pm` | `get_bgf_atoms`, `make_seq_atom_index` |
| `atlas_toolkit/core/manip_atoms.py` additions | `ManipAtoms.pm` | `get_atm_data`, `reimage_atoms` |
| `atlas_toolkit/scripts/remove_mols.py` | `removeMols.pl` | `atlas-remove-mols`; selection + max_atoms/max_mols/randomize |
| `atlas_toolkit/scripts/trim_cell.py` | `trimCell.pl` | `atlas-trim-cell`; trim to new cell with optional CoM mode |
| `atlas_toolkit/scripts/embed_molecule.py` | `embedMolecule.pl` | `atlas-embed-molecule`; fractional-coord grid overlap removal (3 Å cutoff) |
| `atlas_toolkit/scripts/add_solvent.py` | `addSolvent.pl` | `atlas-add-solvent`; replicate → trim → embed → remove excess; bundled WAT boxes |
| `atlas_toolkit/scripts/add_ions.py` | `addIons.pl` | `atlas-add-ions`; replace solvent with ions; FF lookup by element + mass matching |
| `atlas_toolkit/data/wat/` | `scripts/dat/WAT/` | Bundled water box BGFs (SPC, TIP3, F3C, TIP4, MESO, CHARMM TIP3, ...) |
| `tests/test_ff.py` … `tests/test_add_ions.py` | — | 60 new tests; 139/139 total passing |

**New CLI entry points:** `atlas-trim-cell`, `atlas-remove-mols`, `atlas-embed-molecule`, `atlas-add-solvent`, `atlas-add-ions`

### Planned

**Milestone 4 — Additional file formats**
- `io/pdb.py` — PDB reader/writer
- `io/mol2.py` — MOL2 reader/writer
- `io/amber.py` — AMBER trajectory reader

**Milestone 5 — LAMMPS input generation**
- Force field loading (`LoadFFs`, `ReadFFs`, `MolData.pm`)
- `scripts/create_lammps_input.py` — port of `createLammpsInput.pl` (4876 lines)

**Milestone 6 — Analysis scripts**
- `scripts/get_bounds.py`, `scripts/rdf.py`, `scripts/analyze_solvation.py`, etc.
- Replace `Graph.pm` with `networkx`
- Replace `CurveFit.pm` with `scipy`

**Infrastructure**
- GitHub repo + CI (GitHub Actions, pytest on push)
- `ManipAtoms::ImageAtoms`, `UnwrapAtoms`, `ReimageAtoms` for periodic systems
