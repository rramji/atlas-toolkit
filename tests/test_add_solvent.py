"""Tests for scripts/add_solvent.py — cell opts parser and target cell logic."""
import pytest

from atlas_toolkit.scripts.add_solvent import (
    compute_rep_dims,
    compute_target_cell,
    parse_cell_opts,
)


def _box(lx, ly, lz):
    return {
        "X": {"len": lx, "lo": 0.0, "hi": lx, "angle": 90.0},
        "Y": {"len": ly, "lo": 0.0, "hi": ly, "angle": 90.0},
        "Z": {"len": lz, "lo": 0.0, "hi": lz, "angle": 90.0},
    }


# ── parse_cell_opts ────────────────────────────────────────────────────────────

def test_parse_cell_opts_total():
    opts = parse_cell_opts("total: 500")
    assert opts["total"] == 500


def test_parse_cell_opts_density():
    opts = parse_cell_opts("density: 0.9")
    assert abs(opts["density"] - 0.9) < 1e-6


def test_parse_cell_opts_inflate_symmetric():
    opts = parse_cell_opts("x: +/- 10")
    assert "cell" in opts
    entry = opts["cell"]["x"]
    assert entry.get("hi") == pytest.approx(10.0)
    assert entry.get("lo") == pytest.approx(10.0)


def test_parse_cell_opts_set_dim():
    opts = parse_cell_opts("y: = 30")
    entry = opts["cell"]["y"]
    assert entry.get("hi") == pytest.approx(30.0)


def test_parse_cell_opts_plus_only():
    opts = parse_cell_opts("z: + 5")
    entry = opts["cell"]["z"]
    assert entry.get("hi") == pytest.approx(5.0)
    assert "lo" not in entry


def test_parse_cell_opts_total_with_cell():
    opts = parse_cell_opts("total: 200 x: +/- 5")
    assert opts["total"] == 200
    assert "cell" in opts


def test_parse_cell_opts_empty_raises():
    with pytest.raises(ValueError):
        parse_cell_opts("   ")


# ── compute_target_cell ────────────────────────────────────────────────────────

def test_compute_target_cell_no_inflation():
    solu_box = _box(20.0, 20.0, 20.0)
    opts = {}  # no cell modification
    target = compute_target_cell(solu_box, opts)
    assert target["X"] == pytest.approx(20.0)
    assert target["Y"] == pytest.approx(20.0)
    assert target["Z"] == pytest.approx(20.0)


def test_compute_target_cell_set_dim():
    solu_box = _box(20.0, 20.0, 20.0)
    opts = {"cell": {"x": {"hi": 30.0, "lo": 0.0}}}
    target = compute_target_cell(solu_box, opts)
    # Both hi and lo given with same value → inflate by 2*hi? No: len = base + hi + lo
    # hi=30, lo=0 → 20 + 30 + 0 = 50
    assert target["X"] == pytest.approx(50.0)


def test_compute_target_cell_symmetric_inflate():
    solu_box = _box(20.0, 20.0, 20.0)
    opts = {"cell": {"x": {"hi": 10.0, "lo": 10.0}}}  # +/- 10
    target = compute_target_cell(solu_box, opts)
    # +/- 10 → hi==lo → base + 2*10
    assert target["X"] == pytest.approx(40.0)


# ── compute_rep_dims ───────────────────────────────────────────────────────────

def test_compute_rep_dims_exact_fit():
    target = {"X": 20.0, "Y": 20.0, "Z": 20.0}
    solv_box = _box(10.0, 10.0, 10.0)
    dims = compute_rep_dims(target, solv_box)
    assert dims["X"] == 2
    assert dims["Y"] == 2
    assert dims["Z"] == 2


def test_compute_rep_dims_non_exact():
    target = {"X": 25.0, "Y": 10.0, "Z": 5.0}
    solv_box = _box(10.0, 10.0, 10.0)
    dims = compute_rep_dims(target, solv_box)
    assert dims["X"] == 3   # ceil(25/10)
    assert dims["Y"] == 1   # ceil(10/10)
    assert dims["Z"] == 1   # ceil(5/10) = 1


def test_compute_rep_dims_minimum_one():
    target = {"X": 3.0, "Y": 3.0, "Z": 3.0}
    solv_box = _box(10.0, 10.0, 10.0)
    dims = compute_rep_dims(target, solv_box)
    assert dims["X"] == 1
    assert dims["Y"] == 1
    assert dims["Z"] == 1
