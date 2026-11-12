"""Pictures of the fields OpenSn writes: flux on the mesh, material regions, profiles.

OpenSn exports fields as VTK unstructured grids — one ``.vtu`` per MPI rank and
a ``.pvtu`` index tying them together — and some tutorials write line profiles
as CSV. Exploring every file the cluster verification produced settled what to
draw and how:

**1D** (line cells, one axis varying): the value along that axis.

**2D** (quads, triangles, polygons in a plane): each cell split into triangles
and shaded from its own corner values. OpenSn's flux is linear discontinuous —
every cell keeps its own copy of its corner values, 48 points for 12 quads — so
smoothing across cells would draw a solution OpenSn did not compute. Shading
cell by cell from its own points draws the one it did.

**3D** (hexahedra, tetrahedra): a slice through the middle of the longest axis,
drawn as the cells that cross it. A full 3D view belongs in ParaView; a slice
answers "what does it look like inside" in a picture a page can show.

**Region fields** (``Block``, ``Partition``) are integers naming a material or
an MPI rank, drawn as discrete colours, not a gradient: block 2 is not "more"
than block 1.

**Multigroup flux** (``phi_g000_m00`` to ``phi_g171_m00``) also gets a spectrum:
the flux in each energy group, averaged over the mesh, as a Vega-Lite chart like
the study charts.

**Profiles** (CSV with ``x, y, z`` and value columns) become line charts.

Reading uses meshio, because the files are zlib-compressed appended binary and a
hand-written reader is how one gets that subtly wrong. Listing a file's arrays
reads only its XML header, which matters: one rank's file in the logical-volume
tutorial is 1.1 million points.
"""

from __future__ import annotations

import functools
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
GROUP_ARRAY = re.compile(r"^phi_g(?P<group>\d+)_m(?P<moment>\d+)$")
REGION_ARRAYS = {"Block", "Partition"}
OUTLINE_LIMIT = 4000
"""Cells beyond which outlines are left off: past this they merge into a grey
smear that hides the field instead of showing the mesh."""
_PIECE = re.compile(r'<Piece\s+Source="(?P<source>[^"]+)"')
MAX_HEADER = 256 * 1024
"""Bytes read to list a file's arrays. The XML header precedes the appended
binary data, and is a few kilobytes even for the largest file seen."""


class FieldError(Exception):
    """A field file that cannot be read or drawn."""


# --- finding and describing files ---------------------------------------------------


@dataclass
class FieldFile:
    """One field output of one run, as found on disk."""

    path: str
    """Relative to the run directory."""
    kind: str
    """``vtk`` or ``profile``."""
    point_arrays: list[str] = field(default_factory=list)
    cell_arrays: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    pieces: int = 1

    @property
    def arrays(self) -> list[str]:
        return self.point_arrays + [a for a in self.cell_arrays if a not in self.point_arrays]

    @property
    def groups(self) -> int:
        return sum(1 for name in self.arrays if (m := GROUP_ARRAY.match(name)) and m["moment"] == "00")

    def default_array(self) -> Optional[str]:
        """The flux in the first group if there is one, else the material regions."""
        for name in self.arrays:
            if GROUP_ARRAY.match(name):
                return name
        for name in ("Block", "Partition"):
            if name in self.arrays:
                return name
        return self.arrays[0] if self.arrays else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "kind": self.kind, "arrays": self.arrays,
            "columns": self.columns, "pieces": self.pieces, "groups": self.groups,
            "default_array": self.default_array(),
        }


def _header_arrays(path: Path) -> tuple[list[str], list[str]]:
    """Point and cell array names, from the XML header alone."""
    with path.open("rb") as handle:
        head = handle.read(MAX_HEADER).decode("latin-1")
    cut = head.find("<AppendedData")
    head = head[:cut] if cut >= 0 else head
    point, cell, section = [], [], None
    for match in re.finditer(r"<(/?)(PointData|CellData|PPointData|PCellData)\b|<P?DataArray[^>]*\bName=\"([^\"]+)\"", head):
        closing, tag, name = match.group(1), match.group(2), match.group(3)
        if tag:
            section = None if closing else ("point" if "Point" in tag else "cell")
        elif name and section:
            (point if section == "point" else cell).append(name)
    return point, cell


def _profile_columns(path: Path) -> list[str]:
    with path.open() as handle:
        header = handle.readline().strip()
    return [column.strip() for column in header.split(",")]


def find_fields(directory: Path) -> list[FieldFile]:
    """Every field output under a run directory.

    A ``.vtu`` already listed by a ``.pvtu`` is one piece of it, not a field of
    its own, and is not listed separately.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[FieldFile] = []
    claimed: set[Path] = set()
    for index in sorted(directory.rglob("*.pvtu")):
        sources = _PIECE.findall(index.read_text(errors="replace"))
        pieces = [index.parent / source for source in sources]
        claimed.update(piece.resolve() for piece in pieces)
        point, cell = _header_arrays(index)
        found.append(FieldFile(str(index.relative_to(directory)), "vtk", point, cell, pieces=len(pieces)))
    for piece in sorted(directory.rglob("*.vtu")):
        if piece.resolve() in claimed:
            continue
        point, cell = _header_arrays(piece)
        found.append(FieldFile(str(piece.relative_to(directory)), "vtk", point, cell))
    for table in sorted(directory.rglob("*.csv")):
        columns = _profile_columns(table)
        if {"x", "y", "z"} <= set(columns) and len(columns) > 3:
            found.append(FieldFile(str(table.relative_to(directory)), "profile", columns=columns))
    return found


def resolve(directory: Path, relative: str) -> Path:
    """A field file's absolute path, refusing anything outside the run directory."""
    base = Path(directory).resolve()
    target = (base / relative).resolve()
    if base != target and base not in target.parents:
        raise FieldError(f"{relative!r} is outside the run directory")
    if not target.is_file():
        raise FieldError(f"no field file {relative!r}")
    return target


# --- reading -------------------------------------------------------------------------


@dataclass
class Field:
    """A whole field, all ranks' pieces joined."""

    points: np.ndarray
    cells: list[tuple[str, np.ndarray]]
    point_data: dict[str, np.ndarray]
    cell_data: dict[str, np.ndarray]


def _read_one(path: Path) -> Field:
    import meshio  # noqa: PLC0415

    mesh = meshio.read(str(path))
    cell_data = {name: np.concatenate([np.asarray(block).ravel() for block in blocks])
                 for name, blocks in mesh.cell_data.items()}
    return Field(np.asarray(mesh.points, dtype=float),
                 [(block.type, np.asarray(block.data)) for block in mesh.cells],
                 {name: np.asarray(values).ravel() for name, values in mesh.point_data.items()},
                 cell_data)


@functools.lru_cache(maxsize=8)
def _load_cached(path: str, mtime: float) -> Field:
    file = Path(path)
    if file.suffix != ".pvtu":
        return _read_one(file)
    pieces = [_read_one(file.parent / source) for source in _PIECE.findall(file.read_text(errors="replace"))]
    if not pieces:
        raise FieldError(f"{file.name} lists no pieces")
    offset, points, cells = 0, [], []
    point_data: dict[str, list[np.ndarray]] = {}
    cell_data: dict[str, list[np.ndarray]] = {}
    for piece in pieces:
        points.append(piece.points)
        cells.extend((kind, data + offset) for kind, data in piece.cells)
        for name, values in piece.point_data.items():
            point_data.setdefault(name, []).append(values)
        for name, values in piece.cell_data.items():
            cell_data.setdefault(name, []).append(values)
        offset += len(piece.points)
    return Field(np.vstack(points), cells,
                 {k: np.concatenate(v) for k, v in point_data.items()},
                 {k: np.concatenate(v) for k, v in cell_data.items()})


def load(path: Path) -> Field:
    """A field file, read once and kept: rendering several groups of one file
    should not re-read a million points each time."""
    path = Path(path)
    return _load_cached(str(path), path.stat().st_mtime)


# --- drawing -----------------------------------------------------------------------------


def _spatial_axes(points: np.ndarray) -> list[int]:
    extent = np.ptp(points, axis=0) if len(points) else np.zeros(3)
    scale = max(float(extent.max()), 1e-30)
    return [axis for axis in range(3) if extent[axis] > 1e-9 * scale]


def _cell_values(field: Field, array: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Each cell's corner indices, and one value per cell for ``array``."""
    connectivity = [row for _, data in field.cells for row in data]
    if array in field.cell_data:
        return connectivity, np.asarray(field.cell_data[array], dtype=float)
    values = field.point_data[array]
    return connectivity, np.array([float(np.mean(values[row])) for row in connectivity])


def _triangles(field: Field) -> tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate every planar cell; returns triangles and their cell index."""
    triangles, owners, index = [], [], 0
    for kind, data in field.cells:
        for row in data:
            if kind in ("triangle", "quad", "polygon") or len(row) >= 3:
                for k in range(1, len(row) - 1):
                    triangles.append((row[0], row[k], row[k + 1]))
                    owners.append(index)
            index += 1
    return np.array(triangles, dtype=int), np.array(owners, dtype=int)


def _colour_map(array: str):
    import matplotlib  # noqa: PLC0415

    return matplotlib.colormaps["tab20" if array in REGION_ARRAYS else "inferno"]


def _regions(values: np.ndarray):
    """Discrete colours for an integer field: one per value actually present.

    A material block or an MPI rank is a label, not a quantity. A continuous
    colour scale puts ticks at 1.25 and 1.5 between blocks that cannot exist,
    and a field with a single block becomes a bar from -0.1 to 0.1.
    """
    import matplotlib  # noqa: PLC0415
    from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: PLC0415

    present = np.unique(np.rint(values).astype(int))
    codes = np.searchsorted(present, np.rint(values).astype(int))
    palette = matplotlib.colormaps["tab20"](np.arange(len(present)) % 20)
    colours = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(len(present) + 1) - 0.5, len(present))
    return codes, colours, norm, present


def _choose_array(field: "Field") -> Optional[str]:
    """What to draw when nobody said: the flux, else a region field that varies.

    A mesh tutorial with one material is uniform in Block and is about how the
    mesh is split across ranks, so Partition is the informative picture there.
    """
    names = list(field.point_data) + list(field.cell_data)
    for name in names:
        if GROUP_ARRAY.match(name):
            return name
    for name in ("Block", "Partition"):
        data = field.cell_data.get(name, field.point_data.get(name))
        if data is not None and len(np.unique(data)) > 1:
            return name
    for name in ("Block", "Partition"):
        if name in names:
            return name
    return names[0] if names else None


def render(path: Path, array: Optional[str] = None, *, dpi: int = 110) -> bytes:
    """A PNG of one array of one field file."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.tri import Triangulation  # noqa: PLC0415

    field = load(path)
    array = array or _choose_array(field)
    if array is None or (array not in field.point_data and array not in field.cell_data):
        raise FieldError(f"{Path(path).name} has no array {array!r}")
    axes = _spatial_axes(field.points)
    labels = "xyz"
    region = array in REGION_ARRAYS
    figure, plot = plt.subplots(figsize=(6.2, 4.6), dpi=dpi)

    if len(axes) <= 1:
        axis = axes[0] if axes else 0
        if array in field.point_data:
            coordinate, value = field.points[:, axis], field.point_data[array]
        else:
            connectivity, value = _cell_values(field, array)
            coordinate = np.array([field.points[row, axis].mean() for row in connectivity])
        order = np.argsort(coordinate, kind="stable")
        plot.plot(coordinate[order], value[order], color="#7c1d1d", linewidth=1.4)
        plot.set_xlabel(labels[axis])
        plot.set_ylabel(array)
        plot.grid(alpha=0.3)
        title = f"{array} along {labels[axis]}"
    elif len(axes) == 2:
        a, b = axes
        triangles, owners = _triangles(field)
        if not len(triangles):
            raise FieldError("no planar cells to draw")
        mesh = Triangulation(field.points[:, a], field.points[:, b], triangles)
        if array in field.point_data and not region:
            shaded = plot.tripcolor(mesh, field.point_data[array], shading="gouraud", cmap=_colour_map(array))
            figure.colorbar(shaded, ax=plot, label=array)
        else:
            _, per_cell = _cell_values(field, array)
            if region:
                codes, colours, norm, present = _regions(per_cell)
                shaded = plot.tripcolor(mesh, facecolors=codes[owners], cmap=colours, norm=norm)
                bar = figure.colorbar(shaded, ax=plot, label=array, ticks=np.arange(len(present)))
                bar.set_ticklabels([str(v) for v in present])
            else:
                shaded = plot.tripcolor(mesh, facecolors=per_cell[owners], cmap=_colour_map(array))
                figure.colorbar(shaded, ax=plot, label=array)
        cells = [row for _, data in field.cells for row in data]
        if len(cells) <= OUTLINE_LIMIT:
            from matplotlib.collections import PolyCollection  # noqa: PLC0415

            outlines = [field.points[row][:, [a, b]] for row in cells]
            plot.add_collection(PolyCollection(outlines, facecolors="none",
                                               edgecolors=(0, 0, 0, 0.22), linewidths=0.4))
        plot.set_aspect("equal")
        plot.set_xlabel(labels[a])
        plot.set_ylabel(labels[b])
        title = array
    else:
        extent = np.ptp(field.points, axis=0)
        normal = int(np.argmax(extent))
        a, b = [axis for axis in range(3) if axis != normal]
        middle = float(field.points[:, normal].min() + extent[normal] / 2)
        connectivity, per_cell = _cell_values(field, array)
        low = np.array([field.points[row, normal].min() for row in connectivity])
        high = np.array([field.points[row, normal].max() for row in connectivity])
        # Half-open: a mid-plane on a cell boundary, as on any even-numbered
        # orthogonal mesh, otherwise selects both neighbouring layers and draws
        # one over the other.
        crossing = np.flatnonzero((low <= middle) & (high > middle))
        if not len(crossing):
            raise FieldError("no cells cross the mid-plane")
        centres = np.array([field.points[connectivity[i]].mean(axis=0) for i in crossing])
        widths = np.array([np.ptp(field.points[connectivity[i]][:, a]) for i in crossing])
        size = (np.median(widths) / max(np.ptp(centres[:, a]), 1e-30) * 300) ** 2
        # Few enough cells to see one by one: outline them, so a mesh tutorial
        # shows its mesh even when every cell is the same block.
        edges = ({"linewidths": 0.5, "edgecolors": (1, 1, 1, 0.7)}
                 if len(crossing) <= OUTLINE_LIMIT else {"linewidths": 0})
        if region:
            codes, colours, norm, present = _regions(per_cell[crossing])
            shaded = plot.scatter(centres[:, a], centres[:, b], c=codes, s=max(size, 2),
                                  marker="s", cmap=colours, norm=norm, **edges)
            bar = figure.colorbar(shaded, ax=plot, label=array, ticks=np.arange(len(present)))
            bar.set_ticklabels([str(v) for v in present])
        else:
            shaded = plot.scatter(centres[:, a], centres[:, b], c=per_cell[crossing], s=max(size, 2),
                                  marker="s", cmap=_colour_map(array), **edges)
            figure.colorbar(shaded, ax=plot, label=array)
        plot.set_aspect("equal")
        plot.set_xlabel(labels[a])
        plot.set_ylabel(labels[b])
        title = f"{array}, slice at {labels[normal]} = {middle:.3g}"

    if region:
        _, values = _cell_values(field, array)
        distinct = np.unique(np.rint(values).astype(int))
        if len(distinct) == 1:
            title += f" (every cell is {array} {distinct[0]})"
    plot.set_title(title, fontsize=11)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return buffer.getvalue()


# --- charts from fields -------------------------------------------------------------------


def spectrum_chart(path: Path, moment: str = "00") -> Optional[dict[str, Any]]:
    """Flux per energy group, averaged over the mesh, as Vega-Lite.

    An average over points, which for a mesh of similar-sized cells tracks the
    volume average closely and is labelled as what it is.
    """
    field = load(path)
    groups = []
    for name, values in field.point_data.items():
        match = GROUP_ARRAY.match(name)
        if match and match["moment"] == moment:
            mean = float(np.mean(values))
            if math.isfinite(mean):
                groups.append({"group": int(match["group"]), "flux": mean})
    if len(groups) < 2:
        return None
    positive = all(g["flux"] > 0 for g in groups)
    return {
        "$schema": SCHEMA,
        "title": {"text": f"Flux by energy group ({len(groups)} groups)", "anchor": "start"},
        "width": 520, "height": 220,
        "data": {"values": sorted(groups, key=lambda g: g["group"])},
        "mark": {"type": "line", "color": "#7c1d1d",
                 "point": {"filled": True, "size": 25, "color": "#7c1d1d"}},
        "encoding": {
            "x": {"field": "group", "type": "quantitative", "title": "Energy group (0 = highest energy)"},
            "y": {"field": "flux", "type": "quantitative", "title": "Mean scalar flux over points",
                  "scale": {"type": "log"} if positive else {"zero": False}},
            "tooltip": [{"field": "group"}, {"field": "flux", "format": ".4~g"}],
        },
    }


def profile_chart(path: Path) -> Optional[dict[str, Any]]:
    """A CSV line profile as a Vega-Lite chart, along whichever axis varies."""
    import pandas as pd  # noqa: PLC0415

    table = pd.read_csv(path)
    coordinates = [c for c in ("x", "y", "z") if c in table.columns]
    varying = [c for c in coordinates if table[c].nunique() > 1]
    if not varying:
        return None
    axis = varying[0]
    values = [c for c in table.columns if c not in coordinates]
    if not values:
        return None
    long = table.melt(id_vars=[axis], value_vars=values, var_name="quantity", value_name="value")
    long = long.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "$schema": SCHEMA,
        "title": {"text": f"{Path(path).name}: along {axis}", "anchor": "start"},
        "width": 520, "height": 220,
        "data": {"values": long.to_dict(orient="records")},
        "mark": {"type": "line", "strokeWidth": 1.5},
        "encoding": {
            "x": {"field": axis, "type": "quantitative", "title": axis},
            "y": {"field": "value", "type": "quantitative", "title": "value", "scale": {"zero": False}},
            "color": {"field": "quantity", "type": "nominal", "title": None},
            "tooltip": [{"field": axis}, {"field": "quantity"}, {"field": "value", "format": ".4~g"}],
        },
    }


# --- describing fields in words --------------------------------------------------


def _g(value: float) -> str:
    return f"{value:.4g}"


def _place(points: np.ndarray, index: int, axes: list[int]) -> str:
    """Where a point sits in the domain, in words: centre, edge or corner.

    Computed rather than left to the reader of the coordinates, who cannot
    tell that (0, 0) is the centre of a mesh from -10 to 10 and not a corner.
    """
    low, high = points[:, axes].min(axis=0), points[:, axes].max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    relative = (points[index, axes] - low) / span
    at_boundary = [r <= 0.05 or r >= 0.95 for r in relative]
    if all(abs(r - 0.5) <= 0.1 for r in relative):
        return "the centre"
    if len(axes) >= 2 and all(at_boundary):
        return "a corner"
    if any(at_boundary):
        return "a boundary" if len(axes) == 1 else "an edge"
    return "the interior"


def _where(points: np.ndarray, index: int, axes: list[int]) -> str:
    coordinates = ", ".join(f"{'xyz'[a]}={_g(points[index, a])}" for a in axes)
    return f"({coordinates}, {_place(points, index, axes)})"


def _extent(points: np.ndarray, axes: list[int]) -> str:
    return ", ".join(f"{'xyz'[a]} from {_g(points[:, a].min())} to {_g(points[:, a].max())}" for a in axes)


def describe(path: Path, relative: Optional[str] = None) -> list[str]:
    """What a field file's pictures show, as sentences computed from the data.

    The narration never sees an image. It sees this: the mesh, where the flux
    is highest and lowest, and for multigroup flux where the spectrum peaks,
    every figure calculated here rather than read off a picture.
    """
    path = Path(path)
    name = relative or path.name
    if path.suffix == ".csv":
        import pandas as pd  # noqa: PLC0415

        table = pd.read_csv(path)
        varying = [c for c in ("x", "y", "z") if c in table.columns and table[c].nunique() > 1]
        if not varying:
            return []
        axis = varying[0]
        lines = [f"Field file {name}: a line profile along {axis}, {len(table)} points."]
        for column in [c for c in table.columns if c not in ("x", "y", "z")]:
            series = pd.to_numeric(table[column], errors="coerce")
            if series.notna().sum() < 2:
                continue
            high, low = int(series.idxmax()), int(series.idxmin())
            lines.append(f"- {column}: highest {_g(series[high])} at {axis}={_g(table[axis][high])}, "
                         f"lowest {_g(series[low])} at {axis}={_g(table[axis][low])}.")
        return lines

    field = load(path)
    axes = _spatial_axes(field.points)
    cells = sum(len(data) for _, data in field.cells)
    lines = [f"Field file {name}: a {max(len(axes), 1)}D mesh of {cells} cells, "
             f"{_extent(field.points, axes or [0])}."]

    for region in ("Block", "Partition"):
        data = field.cell_data.get(region)
        if data is not None:
            present = np.unique(np.rint(data).astype(int))
            what = "material regions" if region == "Block" else "MPI ranks owning cells"
            lines.append(f"- {region} ({what}): {len(present)} distinct value"
                         f"{'s' if len(present) != 1 else ''}, drawn one colour each.")

    groups = sorted((int(m["group"]), name_) for name_ in field.point_data
                    if (m := GROUP_ARRAY.match(name_)) and m["moment"] == "00")
    if groups:
        first = groups[0][1]
        values = field.point_data[first]
        high, low = int(np.argmax(values)), int(np.argmin(values))
        picture = ("a line along " + "xyz"[axes[0]]) if len(axes) <= 1 else (
            "a colour map, darker is lower" if len(axes) == 2 else "a slice through the middle")
        lines.append(f"- Default picture: scalar flux in group {groups[0][0]} as {picture}. "
                     f"Highest {_g(values[high])} at {_where(field.points, high, axes)}, "
                     f"lowest {_g(values[low])} at {_where(field.points, low, axes)}, "
                     f"mean over points {_g(float(np.mean(values)))}.")
    if len(groups) >= 2:
        means = [(g, float(np.mean(field.point_data[n]))) for g, n in groups]
        means = [(g, m) for g, m in means if math.isfinite(m)]
        peak = max(means, key=lambda gm: gm[1])
        trough = min(means, key=lambda gm: gm[1])
        last = len(groups) - 1
        position = ("the low-energy (thermal) end" if peak[0] >= 0.75 * last
                    else "the high-energy (fast) end" if peak[0] <= 0.25 * last else "the middle of the range")
        lines.append(f"- Energy spectrum chart: mean flux per group for {len(groups)} groups, group 0 the "
                     f"highest energy. It peaks at group {peak[0]} ({_g(peak[1])}), toward {position}; "
                     f"the lowest is group {trough[0]} ({_g(trough[1])}).")
    return lines
