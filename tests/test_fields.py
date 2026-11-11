"""Field pictures, on small files built the way OpenSn builds them.

meshio writes the same zlib-compressed appended VTU that OpenSn does, so these
files exercise the real reading path: a 1D line mesh, a 2D multigroup flux
split across two rank pieces with a .pvtu index, a 3D mesh with two material
blocks, and a CSV line profile.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
pytest.importorskip("matplotlib")

from autoopensn.viz import fields  # noqa: E402

PNG = b"\x89PNG"


def quad_piece(x0, x1, groups=3):
    """A one-row strip of discontinuous quads: each cell has its own four points."""
    points, cells = [], []
    for i in range(4):
        a, b = x0 + (x1 - x0) * i / 4, x0 + (x1 - x0) * (i + 1) / 4
        base = len(points)
        points += [[a, 0, 0], [b, 0, 0], [b, 1, 0], [a, 1, 0]]
        cells.append([base, base + 1, base + 2, base + 3])
    points = np.array(points, dtype=float)
    data = {f"phi_g{g:03d}_m00": np.exp(-points[:, 0]) / (g + 1) for g in range(groups)}
    return meshio.Mesh(points, [("quad", np.array(cells))], point_data=data,
                       cell_data={"Block": [np.zeros(4, dtype=np.int32)]})


@pytest.fixture
def run_dir(tmp_path):
    out = tmp_path / "case"
    flux = out / "Flux"
    flux.mkdir(parents=True)
    quad_piece(0, 1).write(flux / "flux_0.vtu")
    quad_piece(1, 2).write(flux / "flux_1.vtu")
    (flux / "flux.pvtu").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PUnstructuredGrid" version="0.1">\n'
        '<PUnstructuredGrid GhostLevel="0">\n<PPointData>\n'
        + "".join(f'<PDataArray type="Float64" Name="phi_g{g:03d}_m00"/>\n' for g in range(3))
        + '</PPointData>\n<PCellData>\n<PDataArray type="Int32" Name="Block"/>\n</PCellData>\n'
        '<Piece Source="flux_0.vtu"/>\n<Piece Source="flux_1.vtu"/>\n'
        "</PUnstructuredGrid>\n</VTKFile>\n"
    )
    z = np.linspace(0, 2, 9)
    line_points = np.array([[0, 0, v] for v in z])
    meshio.Mesh(line_points, [("line", np.array([[i, i + 1] for i in range(8)]))],
                point_data={"phi_g000_m00": np.sin(z)}).write(out / "line.vtu")
    grid = np.array([[x, y, zz] for zz in (0, 1, 2) for y in (0, 1, 2) for x in (0, 1, 2)], dtype=float)
    index = lambda x, y, zz: zz * 9 + y * 3 + x  # noqa: E731
    hexes = [[index(x, y, zz), index(x + 1, y, zz), index(x + 1, y + 1, zz), index(x, y + 1, zz),
              index(x, y, zz + 1), index(x + 1, y, zz + 1), index(x + 1, y + 1, zz + 1), index(x, y + 1, zz + 1)]
             for zz in (0, 1) for y in (0, 1) for x in (0, 1)]
    blocks = np.array([1 if h[0] % 3 == 0 else 2 for h in hexes], dtype=np.int32)
    meshio.Mesh(grid, [("hexahedron", np.array(hexes))],
                cell_data={"Block": [blocks], "Partition": [np.zeros(8, dtype=np.int32)]}).write(out / "mesh3d.vtu")
    (out / "detector.csv").write_text("x,y,z,phi_g000_m00\n" + "".join(f"0,0,{v},{v*v}\n" for v in z))
    (out / "notes.csv").write_text("a,b\n1,2\n")
    return out


def test_a_parallel_index_is_one_field_not_three(run_dir):
    paths = [f.path for f in fields.find_fields(run_dir)]
    assert "Flux/flux.pvtu" in paths
    assert "Flux/flux_0.vtu" not in paths and "Flux/flux_1.vtu" not in paths


def test_arrays_come_from_the_header_alone(run_dir):
    flux = next(f for f in fields.find_fields(run_dir) if f.path == "Flux/flux.pvtu")
    assert flux.point_arrays == ["phi_g000_m00", "phi_g001_m00", "phi_g002_m00"]
    assert flux.cell_arrays == ["Block"]
    assert flux.groups == 3 and flux.pieces == 2
    assert flux.default_array() == "phi_g000_m00"


def test_a_csv_is_a_profile_only_with_coordinates(run_dir):
    kinds = {f.path: f.kind for f in fields.find_fields(run_dir)}
    assert kinds["detector.csv"] == "profile"
    assert "notes.csv" not in kinds


def test_pieces_are_joined_into_one_field(run_dir):
    whole = fields.load(run_dir / "Flux" / "flux.pvtu")
    assert len(whole.points) == 32, "two pieces of 16 points"
    assert sum(len(c) for _, c in whole.cells) == 8
    assert whole.cells[1][1].min() >= 16, "the second piece's connectivity is offset"


@pytest.mark.parametrize("file,array", [
    ("line.vtu", None),
    ("Flux/flux.pvtu", "phi_g001_m00"),
    ("Flux/flux.pvtu", "Block"),
    ("mesh3d.vtu", None),
])
def test_every_kind_of_field_draws(run_dir, file, array):
    assert fields.render(run_dir / file, array).startswith(PNG)


def test_a_missing_array_is_an_error_not_a_blank_picture(run_dir):
    with pytest.raises(fields.FieldError):
        fields.render(run_dir / "line.vtu", "phi_g009_m00")


def test_a_region_field_that_varies_is_the_default_over_one_that_does_not(run_dir):
    """A uniform Block is the one picture with nothing in it."""
    assert fields._choose_array(fields.load(run_dir / "mesh3d.vtu")) == "Block"
    data = fields.load(run_dir / "mesh3d.vtu")
    data.cell_data["Block"] = np.zeros(8)
    data.cell_data["Partition"] = np.arange(8) % 2
    assert fields._choose_array(data) == "Partition"


def test_region_colours_are_one_per_value_present():
    """Labels, not quantities: no colour for a block 1.5 that cannot exist."""
    codes, _, _, present = fields._regions(np.array([3, 1, 3, 7]))
    assert list(present) == [1, 3, 7]
    assert list(codes) == [1, 0, 1, 2]


def test_a_path_outside_the_run_directory_is_refused(run_dir):
    with pytest.raises(fields.FieldError, match="outside"):
        fields.resolve(run_dir, "../../etc/passwd")


def test_a_multigroup_flux_has_a_spectrum(run_dir):
    chart = fields.spectrum_chart(run_dir / "Flux" / "flux.pvtu")
    assert [v["group"] for v in chart["data"]["values"]] == [0, 1, 2]
    vlc = pytest.importorskip("vl_convert")
    assert vlc.vegalite_to_svg(json.dumps(chart)).startswith("<svg")


def test_one_group_has_no_spectrum(run_dir):
    assert fields.spectrum_chart(run_dir / "line.vtu") is None


def test_a_profile_is_charted_along_the_axis_that_varies(run_dir):
    chart = fields.profile_chart(run_dir / "detector.csv")
    assert chart["encoding"]["x"]["field"] == "z"
    assert len(chart["data"]["values"]) == 9
