"""
OpenFF / SMIRNOFF parameterization bridge.

Takes a molecule (mol2, SDF, SMILES, or RDKit mol) and a SMIRNOFF force field,
assigns partial charges, and returns either:
  - a parameterized ParmEd Structure (for downstream LAMMPS/BGF writing)
  - or writes AMBER prmtop/inpcrd directly

Charge methods (in order of preference):
  1. 'am1bcc'          — AmberTools antechamber (most accurate, slowest)
  2. 'nagl'            — OpenFF NAGL neural net surrogate (fast, nearly AM1-BCC quality)
  3. 'am1bccelf10'     — OpenEye ELF10 variant (requires openeye licence, skipped)
  4. 'gasteiger'       — RDKit Gasteiger (fast, less accurate, always available)

Force fields:
  - 'openff-2.1.0'    — Sage 2.1 (recommended default)
  - 'openff-2.2.0'    — Sage 2.2 (latest)
  - 'openff-2.0.0'    — Sage 2.0
  - Any .offxml path

Route: mol → RDKit → OpenFF Molecule → charges → Interchange
       → AMBER prmtop/inpcrd (temp files) → ParmEd Structure

Notes
-----
- Non-periodic (vacuum) systems are handled correctly; the LAMMPS interchange
  exporter is NOT used (it requires a periodic box + PME).
- mol2 files must be read via RDKit (openff-toolkit doesn't support mol2 directly).
- Existing coordinates from the input file are preserved where possible.
- AmberTools antechamber must be on PATH for 'am1bcc'; the molsim env has it at
  ~/miniforge3/envs/molsim/bin/antechamber.

Usage
-----
    from atlas_toolkit.io.param_openff import param_openff, mol_to_openff

    # from mol2 file
    struct = param_openff('molecule.mol2', ff='openff-2.1.0', charges='nagl')

    # from SMILES
    struct = param_openff('CCO', ff='openff-2.1.0', is_smiles=True)

    # write AMBER files and load with parmed yourself
    param_openff('molecule.mol2', output_stem='/tmp/mol', write_amber=True)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "param_openff",
    "mol_to_openff",
    "CHARGE_METHODS",
    "DEFAULT_FF",
    "DEFAULT_CHARGES",
]

DEFAULT_FF      = "openff-2.1.0"
DEFAULT_CHARGES = "nagl"

# Canonical NAGL model name (bundled with openff-nagl-models)
_NAGL_MODEL = "openff-gnn-am1bcc-0.1.0-rc.3.pt"

CHARGE_METHODS = {
    "nagl":     "OpenFF NAGL neural-net AM1-BCC surrogate (fast, no AmberTools needed)",
    "am1bcc":   "AmberTools antechamber AM1-BCC (accurate, requires antechamber on PATH)",
    "gasteiger": "RDKit Gasteiger (always available, less accurate)",
}


# ── PATH fix for molsim env ────────────────────────────────────────────────

def _ensure_ambertools_on_path() -> None:
    """Add molsim/bin to PATH if antechamber isn't already findable."""
    import shutil
    if shutil.which("antechamber"):
        return
    # Try the molsim env
    import sys
    molsim_bin = Path(sys.executable).parent
    candidate = molsim_bin / "antechamber"
    if candidate.exists():
        os.environ["PATH"] = str(molsim_bin) + os.pathsep + os.environ.get("PATH", "")


# ── mol2/SDF/SMILES → OpenFF Molecule ─────────────────────────────────────

def mol_to_openff(
    source: Union[str, Path, "Chem.Mol"],
    *,
    is_smiles: bool = False,
    allow_undefined_stereo: bool = True,
) -> "openff.toolkit.Molecule":
    """Load a molecule from mol2/SDF/SMILES/RDKit mol into an OpenFF Molecule.

    Parameters
    ----------
    source               : file path, SMILES string, or RDKit Mol object
    is_smiles            : if True, treat source as a SMILES string
    allow_undefined_stereo : passed to OpenFF (default True for simulation inputs)

    Returns
    -------
    openff.toolkit.Molecule
    """
    from openff.toolkit import Molecule
    from rdkit import Chem

    # RDKit Mol passed directly
    if hasattr(source, "GetNumAtoms"):
        return Molecule.from_rdkit(source,
                                   allow_undefined_stereo=allow_undefined_stereo)

    # SMILES string
    if is_smiles:
        mol = Molecule.from_smiles(str(source),
                                   allow_undefined_stereo=allow_undefined_stereo)
        mol.generate_conformers(n_conformers=1)
        return mol

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in (".sdf", ".mol"):
        # OpenFF can read SDF directly
        return Molecule.from_file(str(path),
                                  allow_undefined_stereo=allow_undefined_stereo)

    if suffix == ".mol2":
        # Must go via RDKit (OpenFF doesn't support mol2 without OpenEye)
        rdmol = Chem.MolFromMol2File(str(path), removeHs=False)
        if rdmol is None:
            raise ValueError(f"RDKit could not parse mol2 file: {path}")
        return Molecule.from_rdkit(rdmol,
                                   allow_undefined_stereo=allow_undefined_stereo)

    raise ValueError(
        f"Unsupported file format: {suffix}. "
        "Supported: .mol2, .sdf, .mol, or pass is_smiles=True."
    )


# ── charge assignment ──────────────────────────────────────────────────────

def _assign_charges(
    mol: "openff.toolkit.Molecule",
    method: str,
) -> "openff.toolkit.Molecule":
    """Assign partial charges to mol in-place, return mol."""

    if method == "nagl":
        mol.assign_partial_charges(_NAGL_MODEL)

    elif method == "am1bcc":
        _ensure_ambertools_on_path()
        from openff.toolkit.utils.ambertools_wrapper import AmberToolsToolkitWrapper
        from openff.toolkit.utils import ToolkitRegistry
        registry = ToolkitRegistry([AmberToolsToolkitWrapper])
        mol.assign_partial_charges("am1bcc", toolkit_registry=registry)

    elif method == "gasteiger":
        mol.assign_partial_charges("gasteiger")

    else:
        raise ValueError(
            f"Unknown charge method {method!r}. "
            f"Choose from: {list(CHARGE_METHODS)}"
        )

    return mol


# ── main API ───────────────────────────────────────────────────────────────

def param_openff(
    source: Union[str, Path, "Chem.Mol"],
    *,
    ff: str = DEFAULT_FF,
    charges: str = DEFAULT_CHARGES,
    is_smiles: bool = False,
    output_stem: Optional[str] = None,
    write_amber: bool = False,
    allow_undefined_stereo: bool = True,
    verbose: bool = False,
) -> "parmed.Structure":
    """Parameterize a molecule with an OpenFF SMIRNOFF force field.

    Parameters
    ----------
    source       : mol2/SDF file path, SMILES string, or RDKit Mol
    ff           : SMIRNOFF FF name or .offxml path (default: 'openff-2.1.0')
                   Append '.offxml' automatically if not present.
    charges      : charge method — 'nagl' | 'am1bcc' | 'gasteiger'
    is_smiles    : if True, treat source as a SMILES string
    output_stem  : if given, write <stem>.prmtop and <stem>.inpcrd
    write_amber  : if True, write AMBER files even when output_stem is given
    allow_undefined_stereo : passed to mol loading (default True)
    verbose      : print progress info

    Returns
    -------
    parmed.Structure — fully parameterized, with coordinates and FF terms
    """
    import parmed as pmd
    from openff.toolkit import ForceField, Topology

    # ── resolve FF name ────────────────────────────────────────────────────
    ff_name = ff if ff.endswith(".offxml") else f"{ff}.offxml"

    if verbose:
        print(f"Loading FF: {ff_name}")
    forcefield = ForceField(ff_name)

    # ── load molecule ──────────────────────────────────────────────────────
    if verbose:
        print(f"Loading molecule: {source!r}")
    mol = mol_to_openff(source, is_smiles=is_smiles,
                        allow_undefined_stereo=allow_undefined_stereo)

    if verbose:
        print(f"  {mol.n_atoms} atoms, {mol.n_bonds} bonds")

    # ── ensure conformer ───────────────────────────────────────────────────
    if mol.n_conformers == 0:
        if verbose:
            print("  Generating conformer...")
        mol.generate_conformers(n_conformers=1)

    # ── assign charges ─────────────────────────────────────────────────────
    if verbose:
        print(f"  Assigning charges ({charges})...")
    _assign_charges(mol, charges)

    # ── build interchange ──────────────────────────────────────────────────
    if verbose:
        print("  Building interchange...")
    top = Topology.from_molecules([mol])
    interchange = forcefield.create_interchange(top, charge_from_molecules=[mol])

    # ── export to AMBER then load via parmed ───────────────────────────────
    # Use temp files unless output_stem given
    if output_stem:
        prmtop_path = f"{output_stem}.prmtop"
        inpcrd_path = f"{output_stem}.inpcrd"
        interchange.to_prmtop(prmtop_path)
        interchange.to_inpcrd(inpcrd_path)
        if verbose:
            print(f"  Wrote {prmtop_path}")
            print(f"  Wrote {inpcrd_path}")
        struct = pmd.load_file(prmtop_path, inpcrd_path)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            prmtop = os.path.join(tmp, "mol.prmtop")
            inpcrd  = os.path.join(tmp, "mol.inpcrd")
            interchange.to_prmtop(prmtop)
            interchange.to_inpcrd(inpcrd)
            struct = pmd.load_file(prmtop, inpcrd)

    if verbose:
        print(f"  ParmEd structure: {len(struct.atoms)} atoms, "
              f"{len(struct.bonds)} bonds, "
              f"{len(struct.dihedrals)} dihedrals")
    return struct
