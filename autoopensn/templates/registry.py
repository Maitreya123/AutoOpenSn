"""Loading, validating against, and rendering problem templates.

The sidecar is the single source of truth for what a template can be asked to
do. The spec validator reads it, the renderer reads it, and the CLI prints it.
Nothing may set a parameter the sidecar does not declare, and nothing may set
one outside its declared range.

That rule is not bureaucracy. The generated script is executed, and the values
in it came from a language model reading a user's sentence. A restart interval
of zero or a scattering ratio of 1.4 is not a simulation that fails loudly; it
is one that produces numbers.

Two notational details are preserved deliberately, because the goal is a render
that matches the source script byte for byte:

* Material and geometry lists in the OpenSn test suite are written ``[2., 1.]``,
  not ``[2.0, 1.0]``. The ``dotlist`` filter reproduces that.
* Tolerances are written ``1.0e-9``, which is neither ``repr`` nor ``%g`` output.
  The ``sci`` filter reproduces that.

Getting these right is what makes "diff the render against the original" a test
that can actually pass, and therefore a test worth having.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_ROOT = Path(__file__).resolve().parent

PARAMETER_TYPES = {"int", "float", "str", "bool", "int_list", "float_list"}

PARAMETER_GROUPS = {
    "mesh",
    "materials",
    "source",
    "quadrature",
    "solver",
    "boundary",
    "output",
    "runtime",
    "problem",
    "misc",
}
"""Groups a parameter can belong to.

``problem`` and ``misc`` exist for templates generated from tutorials, where a
knob is found by its shape in the syntax tree and does not always fall into one
of the categories a hand-written sidecar uses.

``solver`` is special: the spec carries those in its ``solver`` field and
everything else in ``template_parameters``, which keeps a spec readable as a
description of a physics problem plus a description of how it is solved.
"""


class TemplateError(Exception):
    """A template is malformed, missing, or asked for something it cannot do."""


# --- number formatting ------------------------------------------------------


def format_dot_float(value: float) -> str:
    """Format as the OpenSn test suite writes material and geometry constants.

    ``2.0`` becomes ``2.``, ``0.9`` stays ``0.9``, ``0.0`` becomes ``0.``. This
    is the style used throughout ``test/python/modules/linear_boltzmann_solvers``.
    """
    value = float(value)
    if not math.isfinite(value):
        raise TemplateError(f"cannot render non-finite value {value!r} into a script")
    text = f"{value:g}"
    if "e" in text or "E" in text:
        return format_sci_float(value)
    if "." not in text:
        text += "."
    return text


def format_sci_float(value: float) -> str:
    """Format as the OpenSn test suite writes tolerances: ``1.0e-9``.

    Neither ``repr`` (``1e-09``) nor ``%g`` (``1e-09``) produces this, and the
    exponent is not zero-padded.
    """
    value = float(value)
    if not math.isfinite(value):
        raise TemplateError(f"cannot render non-finite value {value!r} into a script")
    if value == 0.0:
        return "0.0"
    text = f"{value:.10g}"
    if "e" not in text.lower():
        text = f"{value:.10e}"
    mantissa, _, exponent = text.lower().partition("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    if "." not in mantissa:
        mantissa += ".0"
    return f"{mantissa}e{int(exponent)}"


def format_int(value: Any) -> str:
    """Format an integral value, rejecting anything that would silently truncate."""
    as_int = int(value)
    if as_int != float(value):
        raise TemplateError(f"expected an integer, got {value!r}")
    return str(as_int)


def _dot_list(values: Iterable[float]) -> str:
    return ", ".join(format_dot_float(v) for v in values)


def _int_list(values: Iterable[int]) -> str:
    return ", ".join(format_int(v) for v in values)


def _py_str(value: str) -> str:
    """A double-quoted Python string literal, as the test suite writes them."""
    if '"' in value or "\\" in value or "\n" in value:
        raise TemplateError(
            f"refusing to render {value!r} into a script: it contains a quote, a "
            f"backslash, or a newline, and this template quotes values naively"
        )
    return f'"{value}"'


def _py_comment(value: str) -> str:
    """Text safe to place after a ``#`` on one line.

    A newline here would end the comment and inject whatever followed as code,
    which is the one way a purely decorative parameter can change what a script
    computes.
    """
    if "\n" in value or "\r" in value:
        raise TemplateError(
            f"refusing to render {value!r} as a comment: it spans more than one "
            f"line, and the text after the newline would become code"
        )
    return value


def format_literal(value: Any, style: str) -> str:
    """Render a value as a Python literal in a named notation.

    Generated templates need this because byte-exact rendering is byte-exact
    about *notation*, and the tutorials are not uniform. One writes ``4.0``,
    another ``2.``, a third ``1.0e-10``; all three are the same number and only
    one of them reproduces the file it came from. So the generator records which
    notation each literal was written in, and this reproduces it.

    Styles: ``int``, ``plain`` (``4.0``), ``sci`` (``1.0e-10``), ``dot``
    (``2.``), ``dq`` and ``sq`` for the two quote characters, ``bool``, and
    ``list:<style>`` for a bracketed list of any of the above.
    """
    if style.startswith("list:"):
        element_style = style.split(":", 1)[1]
        if not isinstance(value, (list, tuple)):
            raise TemplateError(f"style {style!r} needs a list, got {value!r}")
        return "[" + ", ".join(format_literal(item, element_style) for item in value) + "]"
    if style == "int":
        return format_int(value)
    if style == "sci":
        return format_sci_float(value)
    if style == "dot":
        return format_dot_float(value)
    if style == "plain":
        number = float(value)
        if not math.isfinite(number):
            raise TemplateError(f"cannot render non-finite value {value!r} into a script")
        return repr(number)
    if style == "bool":
        if not isinstance(value, bool):
            raise TemplateError(f"style 'bool' needs true or false, got {value!r}")
        return "True" if value else "False"
    if style in ("dq", "sq"):
        quote = '"' if style == "dq" else "'"
        text = str(value)
        if quote in text or "\\" in text or "\n" in text:
            raise TemplateError(
                f"refusing to render {text!r} into a script: it contains a quote, a "
                f"backslash, or a newline, and this template quotes values naively"
            )
        return f"{quote}{text}{quote}"
    raise TemplateError(f"unknown literal style {style!r}")


# --- parameter declarations -------------------------------------------------


@dataclass(frozen=True)
class ParameterDeclaration:
    """One tunable parameter, as declared by a template's sidecar."""

    name: str
    group: str
    type: str
    default: Any
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[tuple] = None
    length: Optional[int] = None
    provenance: str = "declared"
    """Where the bound came from: opensn, physics, inferred, or declared.

    A bound read from OpenSn's own constraints and a guard scaled from whatever
    value a tutorial happened to use are not the same kind of claim, and a
    sidecar that presented them identically would invite trust it has not
    earned. The interface shows this, and it is the first thing to look at when
    a study is rejected for a value that ought to be legal.
    """

    @property
    def is_list(self) -> bool:
        return self.type.endswith("_list")

    @property
    def element_type(self) -> str:
        return self.type.removesuffix("_list")

    def coerce(self, value: Any) -> Any:
        """Convert ``value`` to this parameter's type, or raise TemplateError.

        YAML gives ``1`` where a float is wanted and ``1.0e-9`` as a string in
        some writings, so coercion is explicit rather than trusting the loader.
        """
        if self.is_list:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TemplateError(f"{self.name}: expected a list, got {value!r}")
            return [self._coerce_scalar(v) for v in value]
        return self._coerce_scalar(value)

    def _coerce_scalar(self, value: Any) -> Any:
        kind = self.element_type
        if kind == "bool":
            if not isinstance(value, bool):
                raise TemplateError(f"{self.name}: expected true or false, got {value!r}")
            return value
        if kind == "str":
            if not isinstance(value, str):
                raise TemplateError(f"{self.name}: expected a string, got {value!r}")
            return value
        # bool is a subclass of int in Python, and "l_max_its: true" must not
        # quietly become 1.
        if isinstance(value, bool):
            raise TemplateError(f"{self.name}: expected a number, got {value!r}")
        if kind == "int":
            try:
                as_int = int(value)
            except (TypeError, ValueError) as exc:
                raise TemplateError(f"{self.name}: expected an integer, got {value!r}") from exc
            if as_int != float(value):
                raise TemplateError(f"{self.name}: expected an integer, got {value!r}")
            return as_int
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TemplateError(f"{self.name}: expected a number, got {value!r}") from exc

    def validate(self, value: Any) -> Any:
        """Coerce and range-check ``value``. Returns the coerced value."""
        coerced = self.coerce(value)
        items = coerced if self.is_list else [coerced]

        if self.is_list and self.length is not None and len(coerced) != self.length:
            raise TemplateError(
                f"{self.name}: expected {self.length} values, got {len(coerced)}"
            )

        for item in items:
            if self.choices is not None and item not in self.choices:
                raise TemplateError(
                    f"{self.name}: {item!r} is not one of {', '.join(map(repr, self.choices))}"
                )
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                if self.minimum is not None and item < self.minimum:
                    raise TemplateError(
                        f"{self.name}: {item!r} is below the declared minimum {self.minimum!r}"
                    )
                if self.maximum is not None and item > self.maximum:
                    raise TemplateError(
                        f"{self.name}: {item!r} is above the declared maximum {self.maximum!r}"
                    )
        return coerced


# --- templates --------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """A problem template and its declared parameters."""

    name: str
    directory: Path
    script: str
    description: str
    source: str
    opensn_commit: str
    parameters: dict[str, ParameterDeclaration]
    observables: tuple[str, ...]
    gold: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    generated_from: str = ""
    """Path of the tutorial notebook this template was generated from, if any."""
    data_files: tuple[str, ...] = ()
    """Files the script reads, relative to the notebook, copied into each run."""
    num_procs: int = 1
    """MPI ranks the upstream test uses for this scenario."""

    # --- introspection ---

    def declaration(self, name: str) -> ParameterDeclaration:
        try:
            return self.parameters[name]
        except KeyError:
            raise TemplateError(
                f"template {self.name!r} declares no parameter {name!r}. "
                f"Declared: {', '.join(sorted(self.parameters))}"
            ) from None

    def names_in_group(self, group: str) -> tuple[str, ...]:
        return tuple(
            sorted(n for n, d in self.parameters.items() if d.group == group)
        )

    @property
    def solver_parameters(self) -> tuple[str, ...]:
        return self.names_in_group("solver")

    @property
    def non_solver_parameters(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, d in self.parameters.items() if d.group != "solver"))

    def defaults(self) -> dict[str, Any]:
        """Every declared parameter at its default value."""
        return {name: decl.default for name, decl in self.parameters.items()}

    # --- validation and rendering ---

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validate a partial set of overrides. Returns the coerced overrides."""
        return {name: self.declaration(name).validate(value) for name, value in values.items()}

    def merge(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Defaults with ``overrides`` applied, all validated."""
        merged = self.defaults()
        merged.update(self.validate(overrides))
        return merged

    def render(self, overrides: Optional[dict[str, Any]] = None) -> str:
        """Render the script. Pure: same parameters, same bytes, always.

        With no overrides this returns the OpenSn regression script the template
        was derived from, character for character. ``tests/test_template_reed.py``
        asserts exactly that.
        """
        values = self.merge(overrides or {})
        environment = _environment(self.directory)
        return environment.get_template(self.script).render(**values)


def _environment(directory: Path) -> Environment:
    """A Jinja environment configured for generating Python source.

    ``trim_blocks`` and ``lstrip_blocks`` make a control-flow tag on its own line
    disappear completely, which is what allows an optional block to be added
    without disturbing the surrounding whitespace. ``StrictUndefined`` turns a
    typo in the template into an error instead of an empty string in a script
    that then runs and computes the wrong problem.
    """
    environment = Environment(
        loader=FileSystemLoader(str(directory)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    environment.filters["dotf"] = format_dot_float
    environment.filters["sci"] = format_sci_float
    environment.filters["intf"] = format_int
    environment.filters["dotlist"] = _dot_list
    environment.filters["intlist"] = _int_list
    environment.filters["pystr"] = _py_str
    environment.filters["pycomment"] = _py_comment
    environment.filters["lit"] = format_literal
    return environment


def _parse_declaration(name: str, raw: Any) -> ParameterDeclaration:
    if not isinstance(raw, dict):
        raise TemplateError(f"parameter {name!r}: declaration must be a mapping")

    unknown = set(raw) - {
        "group",
        "type",
        "default",
        "description",
        "minimum",
        "maximum",
        "choices",
        "length",
        "provenance",
    }
    if unknown:
        raise TemplateError(
            f"parameter {name!r}: unknown declaration keys {', '.join(sorted(unknown))}"
        )

    for required in ("group", "type", "default"):
        if required not in raw:
            raise TemplateError(f"parameter {name!r}: missing {required!r}")

    if raw["type"] not in PARAMETER_TYPES:
        raise TemplateError(
            f"parameter {name!r}: unknown type {raw['type']!r}; "
            f"expected one of {', '.join(sorted(PARAMETER_TYPES))}"
        )
    if raw["group"] not in PARAMETER_GROUPS:
        raise TemplateError(
            f"parameter {name!r}: unknown group {raw['group']!r}; "
            f"expected one of {', '.join(sorted(PARAMETER_GROUPS))}"
        )

    # YAML 1.1 requires a signed exponent, so "1.0e6" loads as the string
    # "1.0e6" while "1.0e+6" loads as a float. Left alone, an unsigned bound
    # silently becomes a string and every comparison against it raises at some
    # later, less obvious moment.
    for bound in ("minimum", "maximum"):
        if raw.get(bound) is not None and not isinstance(raw[bound], (int, float)):
            raise TemplateError(
                f"parameter {name!r}: {bound} is {raw[bound]!r}, not a number. "
                f"In YAML an exponent needs a sign: write 1.0e+6, not 1.0e6."
            )

    choices = raw.get("choices")
    declaration = ParameterDeclaration(
        name=name,
        group=raw["group"],
        type=raw["type"],
        default=None,
        description=raw.get("description", ""),
        minimum=raw.get("minimum"),
        maximum=raw.get("maximum"),
        choices=tuple(choices) if choices is not None else None,
        length=raw.get("length"),
        provenance=raw.get("provenance", "declared"),
    )
    # A default that violates the declaration is a bug in the sidecar, and it
    # would make "render with defaults" produce an invalid script.
    default = declaration.validate(raw["default"])
    return ParameterDeclaration(
        name=name,
        group=declaration.group,
        type=declaration.type,
        default=default,
        description=declaration.description,
        minimum=declaration.minimum,
        maximum=declaration.maximum,
        choices=declaration.choices,
        length=declaration.length,
        provenance=declaration.provenance,
    )


def load_template(name: str, root: Optional[Path] = None) -> Template:
    """Load a template and its sidecar by name."""
    root = Path(root) if root else TEMPLATE_ROOT
    directory = root / name
    sidecar = directory / f"{name}.yaml"
    if not sidecar.exists():
        raise TemplateError(
            f"no template named {name!r} at {directory}. "
            f"Available: {', '.join(list_templates(root)) or 'none'}"
        )

    raw = yaml.safe_load(sidecar.read_text()) or {}
    unknown = set(raw) - {
        "name",
        "description",
        "source",
        "opensn_commit",
        "script",
        "parameters",
        "observables",
        "gold",
        "notes",
        "generated_from",
        "data_files",
        "num_procs",
    }
    if unknown:
        raise TemplateError(f"{sidecar}: unknown keys {', '.join(sorted(unknown))}")

    if raw.get("name", name) != name:
        raise TemplateError(
            f"{sidecar}: declares name {raw['name']!r} but lives in directory {name!r}"
        )

    script = raw.get("script", f"{name}.py.j2")
    if not (directory / script).exists():
        raise TemplateError(f"{sidecar}: script {script!r} not found in {directory}")

    parameters = {
        param_name: _parse_declaration(param_name, decl)
        for param_name, decl in (raw.get("parameters") or {}).items()
    }

    return Template(
        name=name,
        directory=directory,
        script=script,
        description=raw.get("description", ""),
        source=raw.get("source", ""),
        opensn_commit=raw.get("opensn_commit", ""),
        parameters=parameters,
        observables=tuple(raw.get("observables") or ()),
        gold=raw.get("gold") or {},
        notes=raw.get("notes", ""),
        generated_from=raw.get("generated_from", ""),
        data_files=tuple(raw.get("data_files") or ()),
        num_procs=int(raw.get("num_procs", 1) or 1),
    )


def list_templates(root: Optional[Path] = None) -> list[str]:
    """Names of every template available."""
    root = Path(root) if root else TEMPLATE_ROOT
    names = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / f"{child.name}.yaml").exists():
            names.append(child.name)
    return names


# --- provenance -------------------------------------------------------------


def provenance_header(
    *,
    template: str,
    pack_commit: str,
    spec_name: str,
    case_id: str,
    varied: Optional[dict[str, Any]] = None,
) -> str:
    """The comment block prepended to every script AutoOpenSn actually runs.

    Kept out of ``Template.render`` on purpose. The render is compared against
    the original regression script byte for byte, and a header carrying a case
    id would make that test impossible to write. So the pure function stays
    pure, and the impure part, the part that knows which study this is, lives
    here and is prepended by the caller.

    The pack commit is the load-bearing line. A script generated against one
    OpenSn commit and run against another is the failure this whole project is
    arranged to prevent, and the only defence that survives someone copying the
    file out of ``runs/`` is a record inside the file.
    """
    lines = [
        "# Generated by AutoOpenSn. Do not edit; edit the spec and re-render.",
        f"# template:    {template}",
        f"# pack commit: {pack_commit}",
        f"# study:       {spec_name}",
        f"# case:        {case_id}",
    ]
    for name, value in sorted((varied or {}).items()):
        lines.append(f"#   {name} = {value!r}")
    return "\n".join(lines) + "\n\n"


def render_run_script(
    template: Template,
    overrides: dict[str, Any],
    *,
    pack_commit: str,
    spec_name: str,
    case_id: str,
    varied: Optional[dict[str, Any]] = None,
) -> str:
    """The script as written into a run directory: provenance header plus render."""
    header = provenance_header(
        template=template.name,
        pack_commit=pack_commit,
        spec_name=spec_name,
        case_id=case_id,
        varied=varied,
    )
    return header + template.render(overrides)
