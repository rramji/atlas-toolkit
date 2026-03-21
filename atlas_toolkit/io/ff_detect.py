"""
FF type detection — infer which force-field file(s) to load from the set of
FF type labels present in a BGF atoms dict or ParmEd Structure.

Design
------
Detection is rule-based, not exhaustive lookup.  Each FF family has a
distinctive type-naming convention; we score every candidate FF against the
observed type set and return the best match(es).

Supported families (in priority order):
  1. GAFF / GAFF2      — lowercase 1-2 char + optional digit (c3, hn, oh, …)
  2. AMBER              — UPPERCASE 1-2 char + optional digit/star (CT, CA, N*, …)
  3. DREIDING           — Element + underscore + hybridisation (C_3, N_R, O_2, …)
  4. OPLS-AA            — Letter + exactly 3 digits (C135, H046, …)
  5. Heinz / metal NPs  — Element only, 1-2 chars, capitalised (Au, Ag, …)
  6. Ions (AMBER style) — Na+, Cl-, K+, Li+, Mg2+, Ca2+, …

Multiple families can be active at once (e.g. GAFF + Heinz Au + ion frcmod).
The function returns a list of (ff_name, confidence, path) tuples sorted by
confidence descending.

Usage
-----
    from atlas_toolkit.io.ff_detect import detect_ff, suggest_ff_files

    # from a BGF atoms dict
    from atlas_toolkit.io.bgf import read_bgf
    atoms, bonds, _ = read_bgf("system.bgf")
    hits = detect_ff({a['FFTYPE'] for a in atoms.values()})

    # from a ParmEd Structure
    import parmed as pmd
    struct = pmd.load_file("system.prmtop", "system.inpcrd")
    hits = detect_ff({a.type for a in struct.atoms})

    # one-call convenience: returns ordered list of .ff paths to load
    ff_paths = suggest_ff_files(struct)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .ff import find_ff

__all__ = [
    "FFHit",
    "detect_ff",
    "suggest_ff_files",
    "fftype_family",
]

# ── family patterns ────────────────────────────────────────────────────────

# GAFF: all-lowercase, 1-2 letters + optional single digit
# Covers GAFF, GAFF2 (GAFF11/13/17 variants)
_GAFF_RE = re.compile(r"^[a-z]{1,2}\d?$")

# AMBER (classic / AMBER94–AMBER99 / ff14SB etc.)
# UPPERCASE letters + optional digit or special char
_AMBER_RE = re.compile(r"^[A-Z]{1,2}[\d\*]?$")

# DREIDING: element symbol + underscore + hybridisation code
# e.g. C_3, N_R, O_2, H_, H___A
_DREIDING_RE = re.compile(r"^[A-Z][a-z]?_")

# OPLS-AA: one capital letter + exactly 3 digits
_OPLS_RE = re.compile(r"^[A-Z]\d{3}$")

# Ion types: Na+, Cl-, K+, Li+, Mg2+, Ca2+, Fe2+, Zn2+, etc.
_ION_RE = re.compile(r"^[A-Z][a-z]?\d*[+\-]$")

# Heinz / metal NP FF: bare element symbol (1-2 chars, capitalised) that is
# not a valid AMBER or DREIDING type.  We check against a known metal set.
_METALS = frozenset(
    "Au Ag Cu Pt Pd Fe Co Ni Ti Zr Mo W Cr Mn V Nb Ta Re Ru Rh Os Ir "
    "Al Si Ge Sn Pb Bi Sb Te Se Po At".split()
)


# ── FF candidate registry ─────────────────────────────────────────────────

@dataclass
class _Candidate:
    """Internal: a single FF file candidate."""
    name: str            # logical name used with find_ff()
    family: str          # 'gaff', 'amber', 'dreiding', 'opls', 'heinz', 'ion'
    # Types this FF is known to cover (populated at module load from bundled FFs)
    known_types: frozenset = field(default_factory=frozenset)


def _load_types_from_ff(ff_name: str) -> frozenset:
    """Read ATOMTYPES section from a bundled .ff and return the set of labels."""
    path = find_ff(ff_name)
    if path is None:
        return frozenset()
    types: set = set()
    in_section = False
    try:
        with open(path) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped == "ATOMTYPES":
                    in_section = True
                    continue
                if in_section and stripped == "END":
                    break
                if in_section and stripped and not stripped.startswith("#"):
                    types.add(stripped.split()[0])
    except OSError:
        pass
    return frozenset(types)


# Ordered list of candidates.  More specific entries first within each family.
_CANDIDATES: list[_Candidate] = [
    # GAFF family — GAFF17 is the most complete; prefer it
    _Candidate("GAFF17",     "gaff"),
    _Candidate("GAFF13",     "gaff"),
    _Candidate("GAFF11",     "gaff"),
    _Candidate("GAFF",       "gaff"),
    # AMBER family
    _Candidate("AMBER99",    "amber"),
    _Candidate("AMBER03",    "amber"),
    _Candidate("AMBER94",    "amber"),
    # DREIDING family
    _Candidate("DREIDING2.21", "dreiding"),
    _Candidate("DREIDING-UT",  "dreiding"),
    # OPLS-AA
    _Candidate("oplsaa",     "opls"),
    # Heinz metal NP FFs
    _Candidate("Au_heinzFCC", "heinz"),
    # Ion supplement files (not full FFs, but carry Na+/Cl- etc.)
    _Candidate("AMBER99",    "ion"),   # AMBER99 has K, Na, Cl as bare symbols
]

# Populate known_types at import time (lazy would be nicer but this is fast)
_known: dict[str, frozenset] = {}


def _candidate_types(name: str) -> frozenset:
    if name not in _known:
        _known[name] = _load_types_from_ff(name)
    return _known[name]


# ── public API ─────────────────────────────────────────────────────────────

@dataclass
class FFHit:
    """A detected force-field match."""
    ff_name:    str           # logical name (e.g. 'GAFF17')
    family:     str           # 'gaff' | 'amber' | 'dreiding' | 'opls' | 'heinz' | 'ion' | 'unknown'
    confidence: float         # 0..1 — fraction of observed types covered
    covered:    frozenset     # types this FF covers from the observed set
    uncovered:  frozenset     # observed types NOT covered by this FF
    path:       Optional[Path] = None  # absolute path to the .ff file, or None

    def __repr__(self) -> str:
        return (
            f"FFHit({self.ff_name!r}, family={self.family!r}, "
            f"confidence={self.confidence:.2f}, "
            f"covered={len(self.covered)}/{len(self.covered)+len(self.uncovered)})"
        )


def fftype_family(fftype: str) -> str:
    """Classify a single FF type string into a family name.

    Returns one of: 'gaff', 'amber', 'dreiding', 'opls', 'heinz', 'ion', 'unknown'.
    """
    if _ION_RE.match(fftype):
        return "ion"
    if fftype in _METALS:
        return "heinz"
    if _GAFF_RE.match(fftype):
        return "gaff"
    if _DREIDING_RE.match(fftype):
        return "dreiding"
    if _OPLS_RE.match(fftype):
        return "opls"
    if _AMBER_RE.match(fftype):
        return "amber"
    return "unknown"


def detect_ff(
    observed_types: Iterable[str],
    *,
    min_confidence: float = 0.05,
    max_hits: int = 6,
    include_path: bool = True,
) -> list[FFHit]:
    """Detect which FF file(s) best match a set of FF type labels.

    Parameters
    ----------
    observed_types  : iterable of FF type strings from a structure
    min_confidence  : minimum coverage fraction to include a hit (default 0.05)
    max_hits        : maximum number of hits to return (default 6)
    include_path    : resolve and attach .ff file paths (default True)

    Returns
    -------
    List of FFHit, sorted by confidence descending.  Typically the first
    entry is the primary FF; subsequent entries are supplements.

    Notes
    -----
    - Ion types (Na+, Cl-, K+…) are scored separately: they don't penalise
      the confidence of the main organic FF.
    - 'Au' and other bare metal types are matched against Heinz/metal FFs.
    - Types that match no known FF are reported in the first hit's `uncovered`.
    """
    obs: frozenset = frozenset(t for t in observed_types if t)

    if not obs:
        return []

    # Partition observed types by family for targeted scoring
    ion_types     = frozenset(t for t in obs if _ION_RE.match(t))
    metal_types   = frozenset(t for t in obs if t in _METALS)
    organic_types = obs - ion_types - metal_types

    hits: list[FFHit] = []
    seen_names: set[str] = set()

    # ── score organic FF candidates ───────────────────────────────────────
    for cand in _CANDIDATES:
        if cand.family in ("ion",):
            continue  # handled separately
        if cand.name in seen_names:
            continue

        known = _candidate_types(cand.name)
        if not known:
            continue

        # Only score against the relevant partition
        if cand.family == "heinz":
            target = metal_types
        else:
            target = organic_types

        if not target:
            continue

        covered   = target & known
        uncovered = target - known
        confidence = len(covered) / len(target) if target else 0.0

        if confidence < min_confidence:
            continue

        # Attach path
        path = find_ff(cand.name) if include_path else None

        hits.append(FFHit(
            ff_name    = cand.name,
            family     = cand.family,
            confidence = confidence,
            covered    = covered,
            uncovered  = uncovered,
            path       = path,
        ))
        seen_names.add(cand.name)

    # Sort by (family priority, confidence desc)
    _fam_order = {"gaff": 0, "amber": 1, "dreiding": 2, "opls": 3, "heinz": 4}
    hits.sort(key=lambda h: (_fam_order.get(h.family, 9), -h.confidence))

    # Deduplicate by family: keep only the best-scoring per family
    best_per_family: dict[str, FFHit] = {}
    for h in hits:
        if h.family not in best_per_family:
            best_per_family[h.family] = h

    hits = list(best_per_family.values())

    # ── ion supplement ────────────────────────────────────────────────────
    if ion_types:
        # Try to match each ion type; report as a supplemental hit
        # frcmod.ionsjc_tip3p covers the standard Joung-Cheatham set
        _jc_ions = frozenset(["Na+", "Cl-", "K+", "Li+", "Rb+", "Cs+",
                               "F-", "Br-", "I-", "Mg2+", "Ca2+"])
        covered_jc   = ion_types & _jc_ions
        uncovered_jc = ion_types - _jc_ions

        # Also check AMBER99 for bare-symbol ions (Na, K, Cl as AMBER types)
        _amber_ions = frozenset(["Na", "K", "Cl", "Li", "Mg", "Ca", "Zn", "Fe"])
        covered_amb   = ion_types & _amber_ions
        uncovered_amb = ion_types - _amber_ions

        # Choose whichever covers more
        if len(covered_jc) >= len(covered_amb):
            ion_covered   = covered_jc
            ion_uncovered = uncovered_jc
            ion_name      = "frcmod.ionsjc_tip3p"
            ion_path      = find_ff("frcmod.ionsjc_tip3p") if include_path else None
        else:
            ion_covered   = covered_amb
            ion_uncovered = uncovered_amb
            ion_name      = "AMBER99"
            ion_path      = find_ff("AMBER99") if include_path else None

        if ion_covered:
            ion_conf = len(ion_covered) / len(ion_types)
            hits.append(FFHit(
                ff_name    = ion_name,
                family     = "ion",
                confidence = ion_conf,
                covered    = ion_covered,
                uncovered  = ion_uncovered,
                path       = ion_path,
            ))

    # ── unknown types summary ─────────────────────────────────────────────
    # Collect everything uncovered by all hits
    all_covered = frozenset().union(*(h.covered for h in hits))
    truly_unknown = obs - all_covered
    if truly_unknown and hits:
        hits[0] = FFHit(
            ff_name    = hits[0].ff_name,
            family     = hits[0].family,
            confidence = hits[0].confidence,
            covered    = hits[0].covered,
            uncovered  = hits[0].uncovered | truly_unknown,
            path       = hits[0].path,
        )

    return hits[:max_hits]


def suggest_ff_files(
    source,
    *,
    min_confidence: float = 0.5,
    verbose: bool = False,
) -> list[Path]:
    """Return an ordered list of .ff file Paths to load for a structure.

    Parameters
    ----------
    source          : ParmEd Structure, BGF atoms dict, or iterable of type strings
    min_confidence  : only include FFs with coverage above this threshold
    verbose         : if True, print a detection summary

    Returns
    -------
    List of Path objects suitable for passing to load_ff() / read_ff().
    Returns [] if nothing is detected or paths are unavailable.
    """
    # Extract type set from various input types
    if hasattr(source, "atoms"):
        # ParmEd Structure
        types = {a.type for a in source.atoms if a.type}
    elif isinstance(source, dict):
        # BGF atoms dict
        types = {v.get("FFTYPE", "") for v in source.values()}
    else:
        types = set(source)

    hits = detect_ff(types, min_confidence=0.0)

    if verbose:
        print(f"Observed {len(types)} unique FF types")
        print()
        for h in hits:
            if h.confidence == 0.0:
                continue  # skip zero-coverage candidates
            bar = "█" * int(h.confidence * 20)
            print(f"  {h.ff_name:22s} [{bar:<20s}] {h.confidence*100:5.1f}%  "
                  f"({len(h.covered)} covered, {len(h.uncovered)} uncovered)  "
                  f"family={h.family}")
            if h.uncovered:
                unc = sorted(h.uncovered)[:8]
                ellipsis = "…" if len(h.uncovered) > 8 else ""
                print(f"    uncovered: {', '.join(unc)}{ellipsis}")
        print()

    paths = []
    for h in hits:
        if h.confidence >= min_confidence and h.path is not None:
            paths.append(h.path)

    return paths
