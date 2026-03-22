"""
Wrapper around AmberTools parmchk2 for auto-generating missing GAFF parameters.

When atlas-toolkit's write_data_file detects missing bond/angle/torsion types,
this module:
  1. Takes the mol2 file(s) for the affected molecules
  2. Calls parmchk2 to estimate parameters for unknown types (by analogy)
  3. Parses the resulting frcmod
  4. Returns it ready to merge with load_ff()

Usage
-----
    from atlas_toolkit.io.parmchk import run_parmchk, parmchk_for_missing

    # From a mol2 directly
    frcmod_path = run_parmchk('molecule.mol2', gaff_version='gaff2')

    # Auto-fill missing params from a DataFileSummary
    from atlas_toolkit.io.parmchk import parmchk_for_missing
    extra_parms = parmchk_for_missing(summary, mol2_files=['RR.mol2', 'citrate.mol2'])
    ff_parms = load_ff([...existing ffs...])
    ff_parms = merge_parms(ff_parms, extra_parms)

Notes
-----
- parmchk2 must be on PATH (it's in the molsim env at bin/parmchk2)
- Input must be mol2 or prepi format
- parmchk2 uses parameter analogy: if c3-nh-cz-nh is missing, it finds the
  closest known type combination and uses those constants with a penalty flag
- The output frcmod is parsed by atlas-toolkit's existing frcmod reader
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "run_parmchk",
    "parmchk_for_missing",
    "PARMCHK2_PATH",
]

# ── locate parmchk2 ────────────────────────────────────────────────────────

def _find_parmchk2() -> Optional[str]:
    """Find parmchk2 binary, checking PATH and common conda env locations."""
    # Check PATH first
    found = shutil.which("parmchk2")
    if found:
        return found

    # Check molsim env explicitly
    import sys
    molsim_bin = Path(sys.executable).parent / "parmchk2"
    if molsim_bin.exists():
        return str(molsim_bin)

    # Check AMBERHOME
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        candidate = Path(amberhome) / "bin" / "parmchk2"
        if candidate.exists():
            return str(candidate)

    return None


PARMCHK2_PATH: Optional[str] = _find_parmchk2()


def _ensure_parmchk2() -> str:
    """Return path to parmchk2 or raise a clear error."""
    path = _find_parmchk2()
    if path is None:
        raise RuntimeError(
            "parmchk2 not found. Add AmberTools to PATH or activate the molsim env:\n"
            "  export PATH=/home/rosalind/miniforge3/envs/molsim/bin:$PATH"
        )
    return path


# ── main API ───────────────────────────────────────────────────────────────

def run_parmchk(
    mol2_path: Union[str, Path],
    *,
    gaff_version: str = "gaff2",
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> Path:
    """Run parmchk2 on a mol2 file and return the path to the generated frcmod.

    Parameters
    ----------
    mol2_path    : input mol2 file (must have GAFF atom types assigned)
    gaff_version : 'gaff' or 'gaff2' (default: gaff2)
    output_path  : where to write the frcmod (default: same dir as mol2)
    verbose      : print parmchk2 output

    Returns
    -------
    Path to the generated .frcmod file
    """
    parmchk2 = _ensure_parmchk2()

    mol2_path = Path(mol2_path)
    if not mol2_path.exists():
        raise FileNotFoundError(f"mol2 file not found: {mol2_path}")

    if output_path is None:
        output_path = mol2_path.parent / (mol2_path.stem + ".parmchk2.frcmod")
    output_path = Path(output_path)

    # gaff_version → parmchk2 -s flag
    ff_flag = {"gaff": "1", "gaff2": "2", "1": "1", "2": "2"}.get(
        str(gaff_version).lower(), "2"
    )

    cmd = [
        parmchk2,
        "-i", str(mol2_path),
        "-f", "mol2",
        "-o", str(output_path),
        "-s", ff_flag,
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose and result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"parmchk2 failed (exit {result.returncode}):\n{result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"parmchk2 ran but produced no output at {output_path}\n"
            f"stderr: {result.stderr}"
        )

    if verbose:
        print(f"  Written: {output_path}")

    return output_path


def run_parmchk_from_struct(
    struct,
    output_dir: Union[str, Path],
    stem: str = "mol",
    *,
    gaff_version: str = "gaff2",
    verbose: bool = False,
) -> Path:
    """Run parmchk2 on a ParmEd Structure by first writing a mol2.

    Requires openbabel or parmed's mol2 writer to be available.

    Parameters
    ----------
    struct     : ParmEd Structure (must have GAFF atom types)
    output_dir : directory for intermediate mol2 and output frcmod
    stem       : filename stem
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = output_dir / f"{stem}.mol2"

    # Write mol2 via parmed
    import parmed as pmd
    struct.save(str(mol2_path), overwrite=True)

    return run_parmchk(mol2_path, gaff_version=gaff_version,
                       output_path=output_dir / f"{stem}.parmchk2.frcmod",
                       verbose=verbose)


def parmchk_for_missing(
    summary,
    mol2_files: list[Union[str, Path]],
    *,
    gaff_version: str = "gaff2",
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> dict:
    """Auto-generate FF parameters for missing types detected in a DataFileSummary.

    Runs parmchk2 on each provided mol2, merges the resulting frcmods, and
    returns a parms dict ready to be merged with load_ff() output.

    Parameters
    ----------
    summary      : DataFileSummary from write_data_file (has .missing_torsions etc.)
    mol2_files   : mol2 files for the molecules with missing params
    gaff_version : 'gaff' or 'gaff2'
    output_dir   : where to write intermediate frcmod files (default: tempdir)
    verbose      : print progress

    Returns
    -------
    Merged parms dict from all generated frcmods (pass to load_ff or merge directly)
    """
    from atlas_toolkit.io.ff import load_ff

    missing = (
        list(getattr(summary, 'missing_bonds', []))
        + list(getattr(summary, 'missing_angles', []))
        + list(getattr(summary, 'missing_torsions', []))
    )

    if not missing:
        if verbose:
            print("No missing parameters — parmchk2 not needed.")
        return {}

    if verbose:
        print(f"Missing param types: "
              f"{len(set(summary.missing_bonds))} bond, "
              f"{len(set(getattr(summary,'missing_angles',[])))} angle, "
              f"{len(set(getattr(summary,'missing_torsions',[])))} torsion")

    use_tmp = output_dir is None
    tmp = tempfile.mkdtemp() if use_tmp else None
    out_dir = Path(tmp if use_tmp else output_dir)

    frcmod_paths = []
    for mol2 in mol2_files:
        mol2 = Path(mol2)
        if not mol2.exists():
            if verbose:
                print(f"  Skipping missing file: {mol2}")
            continue
        frcmod = run_parmchk(
            mol2,
            gaff_version=gaff_version,
            output_path=out_dir / (mol2.stem + ".parmchk2.frcmod"),
            verbose=verbose,
        )
        frcmod_paths.append(str(frcmod))
        if verbose:
            print(f"  {mol2.name} → {frcmod.name}")

    if not frcmod_paths:
        return {}

    merged = load_ff(frcmod_paths)

    if use_tmp:
        import shutil as _shutil
        _shutil.rmtree(tmp, ignore_errors=True)

    return merged
