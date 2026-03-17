"""Tests for lammps/dump.py and scripts/update_bgf_coords.py"""

import math
from pathlib import Path

import pytest

from atlas_toolkit.lammps.dump import (
    apply_coords_to_atoms,
    iter_frames,
    lammps_box_to_crystx,
    parse_frame_selection,
    read_last_frame,
    recenter_atoms,
    write_lammps_frame,
    write_xyz_frame,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Dump content helpers
# ---------------------------------------------------------------------------

def _ortho_dump(frames: list[list[tuple]]) -> str:
    """Build a minimal orthogonal dump string with multiple frames."""
    lines = []
    for ts, atoms in enumerate(frames):
        lines += [
            "ITEM: TIMESTEP", str(ts * 1000),
            "ITEM: NUMBER OF ATOMS", str(len(atoms)),
            "ITEM: BOX BOUNDS pp pp pp",
            "0.0 10.0", "0.0 20.0", "0.0 30.0",
            "ITEM: ATOMS id x y z",
        ]
        for atom_id, x, y, z in atoms:
            lines.append(f"{atom_id} {x} {y} {z}")
    return "\n".join(lines) + "\n"


def _write_tmp(tmp_path, content, name="test.dump"):
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# read_last_frame
# ---------------------------------------------------------------------------

class TestReadLastFrame:
    def test_single_frame(self, tmp_path):
        content = _ortho_dump([[(1, 1.0, 2.0, 3.0), (2, 4.0, 5.0, 6.0)]])
        p = _write_tmp(tmp_path, content)
        ts, atoms, box, columns = read_last_frame(p)
        assert set(atoms.keys()) == {1, 2}
        assert float(atoms[1]["x"]) == pytest.approx(1.0)
        assert float(atoms[2]["z"]) == pytest.approx(6.0)
        assert ts == 0

    def test_last_frame_returned(self, tmp_path):
        frame0 = [(1, 0.0, 0.0, 0.0)]
        frame1 = [(1, 9.9, 8.8, 7.7)]
        content = _ortho_dump([frame0, frame1])
        p = _write_tmp(tmp_path, content)
        ts, atoms, box, columns = read_last_frame(p)
        assert float(atoms[1]["x"]) == pytest.approx(9.9)
        assert ts == 1000  # second frame ts = 1*1000

    def test_box_orthogonal(self, tmp_path):
        content = _ortho_dump([[(1, 1.0, 1.0, 1.0)]])
        p = _write_tmp(tmp_path, content)
        _, _, box, _ = read_last_frame(p)
        assert box["xlo"] == pytest.approx(0.0)
        assert box["xhi"] == pytest.approx(10.0)
        assert box["yhi"] == pytest.approx(20.0)
        assert box["zhi"] == pytest.approx(30.0)
        assert "xy" not in box

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.dump"
        p.write_text("")
        with pytest.raises(ValueError, match="No frames"):
            read_last_frame(p)

    def test_many_frames_last_only(self, tmp_path):
        frames = [[(1, float(i), 0.0, 0.0)] for i in range(10)]
        content = _ortho_dump(frames)
        p = _write_tmp(tmp_path, content)
        _, atoms, _, _ = read_last_frame(p)
        assert float(atoms[1]["x"]) == pytest.approx(9.0)

    def test_image_flags_in_columns(self, tmp_path):
        lines = [
            "ITEM: TIMESTEP", "0",
            "ITEM: NUMBER OF ATOMS", "1",
            "ITEM: BOX BOUNDS pp pp pp",
            "0.0 10.0", "0.0 10.0", "0.0 10.0",
            "ITEM: ATOMS id x y z ix iy iz",
            "1 1.0 2.0 3.0 1 -1 0",
        ]
        p = _write_tmp(tmp_path, "\n".join(lines) + "\n")
        _, atoms, _, columns = read_last_frame(p)
        assert "ix" in columns
        assert atoms[1]["ix"] == "1"

    def test_scaled_coords_in_columns(self, tmp_path):
        lines = [
            "ITEM: TIMESTEP", "0",
            "ITEM: NUMBER OF ATOMS", "1",
            "ITEM: BOX BOUNDS pp pp pp",
            "0.0 10.0", "0.0 10.0", "0.0 10.0",
            "ITEM: ATOMS id xs ys zs",
            "1 0.1 0.2 0.3",
        ]
        p = _write_tmp(tmp_path, "\n".join(lines) + "\n")
        _, atoms, _, columns = read_last_frame(p)
        assert "xs" in columns
        assert float(atoms[1]["xs"]) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# iter_frames
# ---------------------------------------------------------------------------

class TestIterFrames:
    def test_yields_all_frames(self, tmp_path):
        content = _ortho_dump([[(1, float(i), 0.0, 0.0)] for i in range(5)])
        p = _write_tmp(tmp_path, content)
        frames = list(iter_frames(p))
        assert len(frames) == 5

    def test_frame_selection_single(self, tmp_path):
        content = _ortho_dump([[(1, float(i), 0.0, 0.0)] for i in range(5)])
        p = _write_tmp(tmp_path, content)
        frames = list(iter_frames(p, selection={3}))
        assert len(frames) == 1
        _, atoms, _, _ = frames[0]
        assert float(atoms[1]["x"]) == pytest.approx(2.0)  # 0-based index 2 = frame 3

    def test_frame_selection_range(self, tmp_path):
        content = _ortho_dump([[(1, float(i), 0.0, 0.0)] for i in range(10)])
        p = _write_tmp(tmp_path, content)
        frames = list(iter_frames(p, selection={1, 3, 5}))
        assert len(frames) == 3

    def test_timestep_returned(self, tmp_path):
        content = _ortho_dump([[(1, 0.0, 0.0, 0.0)], [(1, 1.0, 0.0, 0.0)]])
        p = _write_tmp(tmp_path, content)
        frames = list(iter_frames(p))
        assert frames[0][0] == 0
        assert frames[1][0] == 1000


# ---------------------------------------------------------------------------
# Triclinic box parsing
# ---------------------------------------------------------------------------

class TestTriclinicBox:
    def _tilt_dump(self, tmp_path, xy=1.0, xz=0.5, yz=0.5):
        xlo_b = 0.0 + min(0.0, xy, xz, xy + xz)
        xhi_b = 10.0 + max(0.0, xy, xz, xy + xz)
        ylo_b = 0.0 + min(0.0, yz)
        yhi_b = 10.0 + max(0.0, yz)
        lines = [
            "ITEM: TIMESTEP", "0",
            "ITEM: NUMBER OF ATOMS", "1",
            "ITEM: BOX BOUNDS xy xz yz pp pp pp",
            f"{xlo_b} {xhi_b} {xy}",
            f"{ylo_b} {yhi_b} {xz}",
            f"0.0 10.0 {yz}",
            "ITEM: ATOMS id x y z",
            "1 1.0 1.0 1.0",
        ]
        p = tmp_path / "tilt.dump"
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_tilt_keys_present(self, tmp_path):
        p = self._tilt_dump(tmp_path)
        _, _, box, _ = read_last_frame(p)
        assert "xy" in box and "xz" in box and "yz" in box

    def test_true_lo_hi(self, tmp_path):
        p = self._tilt_dump(tmp_path, xy=1.0, xz=0.0, yz=0.0)
        _, _, box, _ = read_last_frame(p)
        assert box["xlo"] == pytest.approx(0.0)
        assert box["xhi"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# parse_frame_selection
# ---------------------------------------------------------------------------

class TestParseFrameSelection:
    def test_star_returns_none(self):
        assert parse_frame_selection("*") is None
        assert parse_frame_selection("all") is None

    def test_single_int(self):
        assert parse_frame_selection("5") == {5}

    def test_space_separated(self):
        assert parse_frame_selection("1 5 10") == {1, 5, 10}

    def test_range_with_step(self):
        assert parse_frame_selection("1-10:2") == {1, 3, 5, 7, 9}

    def test_range_no_step(self):
        assert parse_frame_selection("1-5") == {1, 2, 3, 4, 5}

    def test_leading_colon(self):
        assert parse_frame_selection(":1-5:2") == {1, 3, 5}

    def test_mixed(self):
        result = parse_frame_selection("1-3 10 20-22")
        assert result == {1, 2, 3, 10, 20, 21, 22}


# ---------------------------------------------------------------------------
# lammps_box_to_crystx
# ---------------------------------------------------------------------------

class TestLammpsBoxToCrystx:
    def test_orthogonal(self):
        box = {"xlo": 0.0, "xhi": 10.0, "ylo": 0.0, "yhi": 20.0, "zlo": 0.0, "zhi": 30.0}
        a, b, c, alpha, beta, gamma = lammps_box_to_crystx(box)
        assert a == pytest.approx(10.0)
        assert b == pytest.approx(20.0)
        assert c == pytest.approx(30.0)
        assert alpha == pytest.approx(90.0)
        assert beta  == pytest.approx(90.0)
        assert gamma == pytest.approx(90.0)

    def test_triclinic_gamma(self):
        box = {
            "xlo": 0.0, "xhi": 10.0,
            "ylo": 0.0, "yhi": 10.0,
            "zlo": 0.0, "zhi": 10.0,
            "xy": 5.0, "xz": 0.0, "yz": 0.0,
        }
        a, b, c, alpha, beta, gamma = lammps_box_to_crystx(box)
        assert b == pytest.approx(math.sqrt(125))
        assert gamma == pytest.approx(math.degrees(math.acos(5.0 / math.sqrt(125))))
        assert alpha == pytest.approx(90.0)
        assert beta  == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# apply_coords_to_atoms
# ---------------------------------------------------------------------------

class TestApplyCoords:
    def _bgf(self, ids):
        return {i: {"XCOORD": 0.0, "YCOORD": 0.0, "ZCOORD": 0.0} for i in ids}

    def _box(self, lx=10.0, ly=10.0, lz=10.0):
        return {"xlo": 0.0, "xhi": lx, "ylo": 0.0, "yhi": ly, "zlo": 0.0, "zhi": lz}

    def test_cartesian(self):
        bgf = self._bgf([1, 2])
        dump = {1: {"id": "1", "x": "3.5", "y": "4.5", "z": "5.5"},
                2: {"id": "2", "x": "1.0", "y": "2.0", "z": "3.0"}}
        apply_coords_to_atoms(bgf, dump, self._box(), ["id", "x", "y", "z"])
        assert bgf[1]["XCOORD"] == pytest.approx(3.5)
        assert bgf[2]["ZCOORD"] == pytest.approx(3.0)

    def test_scaled(self):
        bgf = self._bgf([1])
        dump = {1: {"id": "1", "xs": "0.5", "ys": "0.5", "zs": "0.5"}}
        box = self._box(10.0, 20.0, 30.0)
        apply_coords_to_atoms(bgf, dump, box, ["id", "xs", "ys", "zs"])
        assert bgf[1]["XCOORD"] == pytest.approx(5.0)
        assert bgf[1]["YCOORD"] == pytest.approx(10.0)
        assert bgf[1]["ZCOORD"] == pytest.approx(15.0)

    def test_unwrap_image_flags(self):
        bgf = self._bgf([1])
        dump = {1: {"id": "1", "x": "1.0", "y": "0.0", "z": "0.0",
                    "ix": "1", "iy": "0", "iz": "0"}}
        apply_coords_to_atoms(bgf, dump, self._box(10.0), ["id", "x", "y", "z", "ix", "iy", "iz"])
        assert bgf[1]["XCOORD"] == pytest.approx(11.0)

    def test_no_unwrap_when_disabled(self):
        bgf = self._bgf([1])
        dump = {1: {"id": "1", "x": "1.0", "y": "0.0", "z": "0.0",
                    "ix": "2", "iy": "0", "iz": "0"}}
        apply_coords_to_atoms(bgf, dump, self._box(), ["id", "x", "y", "z", "ix", "iy", "iz"],
                              unwrap=False)
        assert bgf[1]["XCOORD"] == pytest.approx(1.0)

    def test_unwrapped_cartesian_xu(self):
        """xu yu zu are already unwrapped — image flags must not be re-applied."""
        bgf = self._bgf([1])
        dump = {1: {"id": "1", "xu": "11.0", "yu": "0.0", "zu": "0.0",
                    "ix": "1", "iy": "0", "iz": "0"}}
        apply_coords_to_atoms(bgf, dump, self._box(10.0), ["id", "xu", "yu", "zu", "ix", "iy", "iz"])
        # xu is already unwrapped; image flags must NOT add another 10 Å
        assert bgf[1]["XCOORD"] == pytest.approx(11.0)

    def test_missing_atom_id_skipped(self):
        bgf = self._bgf([1])
        dump = {1: {"id": "1", "x": "5.0", "y": "0.0", "z": "0.0"},
                99: {"id": "99", "x": "9.0", "y": "0.0", "z": "0.0"}}
        apply_coords_to_atoms(bgf, dump, self._box(), ["id", "x", "y", "z"])
        assert 99 not in bgf
        assert bgf[1]["XCOORD"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Writers (smoke tests)
# ---------------------------------------------------------------------------

class TestRecenterAtoms:
    def _box(self, lx=10.0, ly=10.0, lz=10.0):
        return {"xlo": 0.0, "xhi": lx, "ylo": 0.0, "yhi": ly, "zlo": 0.0, "zhi": lz}

    def test_com_moves_to_box_centre(self):
        box = self._box()
        atoms = {
            1: {"XCOORD": 1.0, "YCOORD": 1.0, "ZCOORD": 1.0, "MASS": 1.0},
            2: {"XCOORD": 3.0, "YCOORD": 3.0, "ZCOORD": 3.0, "MASS": 1.0},
        }
        recenter_atoms(atoms, box)
        cx = (atoms[1]["XCOORD"] + atoms[2]["XCOORD"]) / 2
        cy = (atoms[1]["YCOORD"] + atoms[2]["YCOORD"]) / 2
        cz = (atoms[1]["ZCOORD"] + atoms[2]["ZCOORD"]) / 2
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)
        assert cz == pytest.approx(5.0)

    def test_atoms_stay_in_box(self):
        """After recentering, all coords should be within [lo, hi)."""
        box = self._box(10.0, 10.0, 10.0)
        atoms = {i: {"XCOORD": float(i), "YCOORD": float(i), "ZCOORD": float(i)}
                 for i in range(1, 9)}
        recenter_atoms(atoms, box)
        for a in atoms.values():
            assert 0.0 <= a["XCOORD"] < 10.0
            assert 0.0 <= a["YCOORD"] < 10.0
            assert 0.0 <= a["ZCOORD"] < 10.0

    def test_mass_weighted(self):
        """Heavier atom should pull COM toward itself."""
        box = self._box(20.0, 20.0, 20.0)
        atoms = {
            1: {"XCOORD": 2.0, "YCOORD": 10.0, "ZCOORD": 10.0, "MASS": 1.0},
            2: {"XCOORD": 6.0, "YCOORD": 10.0, "ZCOORD": 10.0, "MASS": 3.0},
        }
        # COM before = (1*2 + 3*6)/4 = 20/4 = 5.0; box centre = 10.0; shift = +5
        recenter_atoms(atoms, box)
        new_cx = (1.0 * atoms[1]["XCOORD"] + 3.0 * atoms[2]["XCOORD"]) / 4.0
        assert new_cx == pytest.approx(10.0)

    def test_no_mass_field_uses_geometric(self):
        box = self._box()
        atoms = {
            1: {"XCOORD": 2.0, "YCOORD": 5.0, "ZCOORD": 5.0},
            2: {"XCOORD": 4.0, "YCOORD": 5.0, "ZCOORD": 5.0},
        }
        recenter_atoms(atoms, box)
        cx = (atoms[1]["XCOORD"] + atoms[2]["XCOORD"]) / 2
        assert cx == pytest.approx(5.0)


class TestWriters:
    def _simple_frame(self):
        atoms = {
            1: {"id": "1", "x": "1.0", "y": "2.0", "z": "3.0"},
            2: {"id": "2", "x": "4.0", "y": "5.0", "z": "6.0"},
        }
        box = {"xlo": 0.0, "xhi": 10.0, "ylo": 0.0, "yhi": 10.0, "zlo": 0.0, "zhi": 10.0}
        columns = ["id", "x", "y", "z"]
        return atoms, box, columns

    def test_write_lammps_frame(self, tmp_path):
        atoms, box, columns = self._simple_frame()
        p = tmp_path / "out.dump"
        with open(p, "w") as fh:
            write_lammps_frame(fh, 500, atoms, box, columns)
        text = p.read_text()
        assert "ITEM: TIMESTEP" in text
        assert "500" in text
        assert "ITEM: ATOMS id x y z" in text
        assert "1.0" in text

    def test_write_lammps_roundtrip(self, tmp_path):
        """Write then re-read — should recover same coords."""
        atoms, box, columns = self._simple_frame()
        p = tmp_path / "rt.dump"
        with open(p, "w") as fh:
            write_lammps_frame(fh, 0, atoms, box, columns)
        _, rt_atoms, _, _ = read_last_frame(p)
        assert float(rt_atoms[1]["x"]) == pytest.approx(1.0)
        assert float(rt_atoms[2]["z"]) == pytest.approx(6.0)

    def test_write_xyz_frame(self, tmp_path):
        atoms = {
            1: {"id": "1", "x": "1.0", "y": "2.0", "z": "3.0", "element": "O"},
        }
        box = {"xlo": 0.0, "xhi": 10.0, "ylo": 0.0, "yhi": 10.0, "zlo": 0.0, "zhi": 10.0}
        p = tmp_path / "out.xyz"
        with open(p, "w") as fh:
            write_xyz_frame(fh, 0, atoms, box, ["id", "x", "y", "z", "element"])
        lines = p.read_text().splitlines()
        assert lines[0] == "1"
        assert "O" in lines[2]
        assert "1.000000" in lines[2]
