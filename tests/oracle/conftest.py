"""Shared fixtures and helpers for Perl oracle comparison tests.

All tests in this package are automatically skipped when:
  - perl is not found in PATH, or
  - ~/ATLAS-toolkit/scripts/ does not exist

BGF comparison utilities are in bgf_compare.py.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ATLAS_PERL = Path.home() / "ATLAS-toolkit" / "scripts"
FIXTURES = Path(__file__).parent.parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "perl_oracle: requires Perl and ~/ATLAS-toolkit"
    )


# ---------------------------------------------------------------------------
# Session-scoped availability checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def perl_available():
    return shutil.which("perl") is not None


@pytest.fixture(scope="session")
def atlas_perl_available():
    return ATLAS_PERL.is_dir()


@pytest.fixture(autouse=True)
def skip_if_no_perl(request, perl_available, atlas_perl_available):
    """Auto-skip any test in the oracle package when Perl/ATLAS is unavailable."""
    if not perl_available or not atlas_perl_available:
        pytest.skip("Perl or ~/ATLAS-toolkit not available")


# ---------------------------------------------------------------------------
# Helper: run a Perl ATLAS script
# ---------------------------------------------------------------------------

def run_perl(script_name: str, args: list[str], cwd=None) -> subprocess.CompletedProcess:
    """Run an ATLAS Perl script and return the CompletedProcess."""
    cmd = ["perl", str(ATLAS_PERL / script_name)] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def run_python(module: str, args: list[str], cwd=None) -> subprocess.CompletedProcess:
    """Run an atlas_toolkit script via `python -m` and return the CompletedProcess."""
    cmd = [sys.executable, "-m", module] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
