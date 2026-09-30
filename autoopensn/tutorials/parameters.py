"""Finding the knobs in a tutorial script, and deciding what they may be set to.

This is the module that does automatically what a person did by hand for the
Reed template: read a script, work out which constants are worth varying, and
record what values are legal.

It works on the syntax tree, not on text, so a knob is found by what it *is*
rather than by matching a name. Three shapes qualify:

1. A module-level assignment of a literal, ``num_cells = 40``.
2. A literal keyword argument, ``GLProductQuadrature1DSlab(n_polar=8)``.
3. A literal value in a solver dictionary, ``"l_abs_tol": 1.0e-10``.

Every knob carries the exact source span it occupies, which is what lets the
template generator splice a placeholder in without reformatting anything else.
That matters: the generated template has to render back to the original script
character for character, and a generator that reprints the syntax tree cannot.

**On the ranges.** A bound is only as good as its source, so each one records
where it came from:

``opensn``
    Read from OpenSn's own parameter constraints at the pinned commit. These
    are real limits; violating one is an error from the solver.
``physics``
    A property of the quantity rather than of the code. A scattering ratio
    above one is a multiplying medium, which is a different problem.
``inferred``
    A guard scaled from the value the tutorial itself uses. Not a code limit,
    and not verified. It exists so that a study cannot accidentally request a
    mesh a thousand times finer than anyone intended, and it is the first thing
    to correct by hand when a scenario needs a wider range.

Marking the third kind honestly is the point. A sidecar that presented all
three as equally authoritative would be worse than one with no bounds at all,
because it would invite trust it has not earned.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

# Lists longer than this are data, not a knob. The 41 node positions of a mesh
# are computed from the cell count that sits right above them, and exposing
# them as a parameter would offer a study the chance to hand-write a mesh.
MAX_LIST_LENGTH = 12

# Keys that say nothing on their own. Always qualified by the call or the
# dictionary they were found in.
GENERIC_KEYS = {"block_ids", "type", "name", "id", "xs", "value", "path", "filename"}

# Names that are never knobs: loop variables, accumulators, and the handles the
# script passes between stages.
SKIP_NAMES = {"rank", "size", "i", "j", "k", "n", "_"}

# Arguments that define what an object *is* rather than how it is configured.
# RPPLogicalVolume(infx=True) declares that the volume is unbounded in x; a
# study that flipped it would not be tuning the problem, it would be solving a
# different one, and the mesh it was written against would no longer match.
EXCLUDE_NAMES = {"infx", "infy", "infz"}


@dataclass(frozen=True)
class Constraint:
    """What a parameter may be set to, and who says so."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[tuple] = None
    provenance: str = "inferred"
    note: str = ""


# --- what OpenSn itself constrains ------------------------------------------
#
# Read from the source at commit 2fd4a19ceade4f8da581a5029c68f474d3333ccb. The
# citation is in the note, because a bound whose origin nobody can check is a
# bound nobody will dare to widen.

_GROUPSET = "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.cc"
_DO_PROBLEM = "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.cc"

OPENSN_CONSTRAINTS: dict[str, Constraint] = {
    "l_abs_tol": Constraint(1.0e-18, 1.0e-2, None, "opensn", f"lower bound from {_GROUPSET}"),
    "l_max_its": Constraint(1, 100000, None, "opensn", f"lower bound from {_GROUPSET}"),
    "gmres_restart_interval": Constraint(1, 1000, None, "opensn", f"lower bound from {_GROUPSET}"),
    "inner_linear_method": Constraint(
        None, None,
        ("classic_richardson", "petsc_richardson", "petsc_gmres", "petsc_bicgstab"),
        "opensn", f"allowable list from {_GROUPSET}",
    ),
    "sweep_type": Constraint(None, None, ("AAH", "CBC"), "opensn", f"allowable list from {_DO_PROBLEM}"),
    "type": Constraint(
        None, None, ("vacuum", "isotropic", "reflecting", "arbitrary"), "opensn",
        f"boundary condition types from {_DO_PROBLEM}",
    ),
    "ags_convergence_check": Constraint(None, None, ("l2", "pointwise"), "opensn", "user guide"),
    "max_ags_iterations": Constraint(1, 10000, None, "opensn", "user guide default 100"),
    "ags_tolerance": Constraint(1.0e-18, 1.0e-2, None, "opensn", "user guide default 1e-6"),
    "wgdsa_l_abs_tol": Constraint(1.0e-18, 1.0e-1, None, "opensn", "user guide default 1e-4"),
    "tgdsa_l_abs_tol": Constraint(1.0e-18, 1.0e-1, None, "opensn", "user guide default 1e-4"),
    "wgdsa_l_max_its": Constraint(1, 10000, None, "opensn", "user guide default 30"),
    "tgdsa_l_max_its": Constraint(1, 10000, None, "opensn", "user guide default 30"),
    "k_tol": Constraint(1.0e-16, 1.0e-2, None, "opensn", "user guide range 1e-8 to 1e-10"),
    "max_iters": Constraint(1, 100000, None, "opensn", "power iteration cap"),
    "scattering_order": Constraint(0, 8, None, "opensn", "moment order"),
    "n_polar": Constraint(2, 1024, None, "opensn", "polar ordinates; must be even"),
    "n_azimuthal": Constraint(2, 1024, None, "opensn", "azimuthal ordinates; must be even"),
    "num_groups": Constraint(1, 10000, None, "opensn", "energy group count"),
}

PHYSICS_CONSTRAINTS: dict[str, Constraint] = {
    "c": Constraint(0.0, 1.0, None, "physics", "scattering ratio; above 1 the medium multiplies"),
    "sigma_t": Constraint(0.0, None, None, "physics", "a total cross section cannot be negative"),
    "sigma_a": Constraint(0.0, None, None, "physics", "an absorption cross section cannot be negative"),
    "group_strength": Constraint(0.0, None, None, "physics", "a source strength cannot be negative"),
    "scattering_ratio": Constraint(0.0, 1.0, None, "physics", "above 1 the medium multiplies"),
}

# --- grouping ---------------------------------------------------------------
#
# The group decides which half of a spec a parameter is written in, and how the
# interface lays the sidecar out. Matching is on substrings of the name because
# these scripts name things plainly.

GROUP_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("problem", ("num_groups", "num_moments", "num_materials", "dimension")),
    ("solver", (
        "l_abs_tol", "l_max_its", "gmres_restart", "inner_linear_method", "sweep_type",
        "ags_", "max_iters", "k_tol", "wgdsa", "tgdsa", "nl_", "tolerance", "max_its",
        "restart", "verbose_inner", "verbose_outer", "allow_cycles",
    )),
    ("quadrature", ("polar", "azimuthal", "quadrature", "scattering_order", "sn_order", "aquad")),
    ("materials", ("sigma", "xs", "cross_section", "density", "ratio", "material")),
    ("source", ("source", "strength", "emission", "intensity")),
    ("boundary", ("boundary", "bc_", "_bc", "reflect", "vacuum")),
    ("mesh", (
        "node", "cell", "mesh", "length", "width", "refine", "nref", "nx", "ny", "nz",
        "zmin", "zmax", "xmin", "xmax", "ymin", "ymax", "extrude", "layer", "partition",
    )),
    ("output", ("print", "emit", "export", "plot", "write", "output")),
)


def group_for(name: str) -> str:
    """Which section of a spec this parameter belongs in."""
    lowered = name.lower()
    for group, hints in GROUP_HINTS:
        if any(hint in lowered for hint in hints):
            return group
    if lowered == "c":
        return "materials"
    return "misc"


def constraint_for(name: str, default: Any) -> Constraint:
    """The bound for a parameter, preferring a real limit over a guard."""
    if name in OPENSN_CONSTRAINTS:
        return OPENSN_CONSTRAINTS[name]
    if name in PHYSICS_CONSTRAINTS:
        return PHYSICS_CONSTRAINTS[name]
    return _inferred(default)


def _inferred(default: Any) -> Constraint:
    """A guard scaled from the tutorial's own value.

    Deliberately generous: three orders of magnitude above whatever the
    tutorial chose. The purpose is to catch a study that asks for something
    absurd, not to encode an opinion about what is physically interesting.
    """
    values = default if isinstance(default, list) else [default]
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers:
        return Constraint(provenance="inferred", note="no numeric bound applies")

    largest = max(abs(v) for v in numbers)
    if largest == 0:
        return Constraint(0, 1000, None, "inferred", "guard scaled from a zero default")
    if all(v >= 0 for v in numbers):
        return Constraint(0, _round_up(largest * 1000), None, "inferred", "guard, not a code limit")
    bound = _round_up(largest * 1000)
    return Constraint(-bound, bound, None, "inferred", "guard, not a code limit")


def _round_up(value: float) -> float:
    """A tidy upper bound, so sidecars read like decisions rather than arithmetic."""
    import math

    if value <= 0:
        return 0.0
    exponent = math.floor(math.log10(value))
    return float(10 ** (exponent + 1))


# --- detection --------------------------------------------------------------


@dataclass
class Detected:
    """One knob found in a script."""

    name: str
    default: Any
    start: int
    """Absolute offset of the literal in the source."""
    end: int
    kind: str
    """``assignment``, ``keyword``, or ``dict_entry``."""
    context: str = ""
    """The call or dictionary it was found in, for the description."""

    @property
    def is_list(self) -> bool:
        return isinstance(self.default, list)

    @property
    def type_name(self) -> str:
        value = self.default[0] if self.is_list and self.default else self.default
        if isinstance(value, bool):
            base = "bool"
        elif isinstance(value, int):
            base = "int"
        elif isinstance(value, float):
            base = "float"
        else:
            base = "str"
        return f"{base}_list" if self.is_list else base


def _literal(node: ast.AST) -> tuple[bool, Any]:
    """Whether a node is a literal we can vary, and its value."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        ok, value = _literal(node.operand)
        if ok and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, -value
        return False, None
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for element in node.elts:
            ok, value = _literal(element)
            if not ok or isinstance(value, str):
                return False, None
            values.append(value)
        if not values or len(values) > MAX_LIST_LENGTH:
            return False, None
        if isinstance(node, ast.Tuple):
            # A tuple is usually a coordinate or a group range, where varying
            # one end alone is meaningless. Left alone deliberately.
            return False, None
        return True, values
    return False, None


def _offsets(source: str) -> list[int]:
    """Absolute offset at which each line begins."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _span(node: ast.AST, line_offsets: list[int]) -> tuple[int, int]:
    start = line_offsets[node.lineno - 1] + node.col_offset
    end = line_offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "call"


def detect(source: str) -> list[Detected]:
    """Every knob in a script, in source order.

    Names are made unique by qualifying a collision with the call it came from,
    so a script with two quadratures gets two distinct polar-angle knobs rather
    than one that silently controls both.
    """
    tree = ast.parse(source)
    line_offsets = _offsets(source)
    found: list[Detected] = []
    taken: set[str] = set()

    def claim(preferred: str, context: str) -> Optional[str]:
        if preferred in EXCLUDE_NAMES:
            return None
        # A generic key is qualified by where it was found even when nothing has
        # claimed it yet. Two boundary conditions both have a "type", and a
        # single knob controlling both faces would be a trap.
        name = f"{_snake(context)}_{preferred}" if (context and preferred in GENERIC_KEYS) else preferred
        if name in taken:
            name = f"{_snake(context)}_{preferred}" if context else preferred
        if name in taken:
            index = 2
            while f"{name}_{index}" in taken:
                index += 1
            name = f"{name}_{index}"
        if not name.isidentifier() or name in SKIP_NAMES:
            return None
        taken.add(name)
        return name

    for node in ast.walk(tree):
        # 1. assignments of a literal
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id not in SKIP_NAMES:
                ok, value = _literal(node.value)
                if ok:
                    name = claim(target.id, "")
                    if name:
                        start, end = _span(node.value, line_offsets)
                        found.append(Detected(name, value, start, end, "assignment"))
            continue

        # 2. literal keyword arguments
        if isinstance(node, ast.Call):
            callee = _callee(node)
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                ok, value = _literal(keyword.value)
                if not ok:
                    continue
                name = claim(keyword.arg, callee)
                if name:
                    start, end = _span(keyword.value, line_offsets)
                    found.append(Detected(name, value, start, end, "keyword", callee))
            continue

        # 3. literal values in a dictionary keyed by strings
        if isinstance(node, ast.Dict):
            context = _dict_context(node)
            for key, value_node in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                ok, value = _literal(value_node)
                if not ok:
                    continue
                name = claim(key.value, context)
                if name:
                    start, end = _span(value_node, line_offsets)
                    found.append(Detected(name, value, start, end, "dict_entry", context))

    found.sort(key=lambda item: item.start)
    return _drop_overlaps(found)


def _dict_context(node: ast.Dict) -> str:
    """A label for a dictionary, from its own keys.

    A groupset is recognisable by ``groups_from_to``; a boundary condition by
    its ``name``. This is what turns a bare ``"type"`` key into a knob called
    ``zmin_type`` rather than one called ``type`` that controls both faces.
    """
    keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    if "groups_from_to" in keys:
        return "groupset"
    for key, value in zip(node.keys, node.values):
        if (
            isinstance(key, ast.Constant)
            and key.value == "name"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    if "xs" in keys and "block_ids" in keys:
        return "xs_map"
    if "block_ids" in keys:
        return "block"
    return ""


def _drop_overlaps(found: list[Detected]) -> list[Detected]:
    """Keep the outermost knob where two overlap.

    ``group_strength=[1.0]`` is both a keyword argument holding a list and, if
    we were careless, a list of one constant. Splicing both would corrupt the
    source, so the inner one goes.
    """
    kept: list[Detected] = []
    for item in found:
        if kept and item.start < kept[-1].end:
            continue
        kept.append(item)
    return kept


def _snake(text: str) -> str:
    out = []
    for index, character in enumerate(text):
        if character.isupper() and index and not text[index - 1].isupper():
            out.append("_")
        out.append(character.lower())
    return "".join(out).strip("_")
