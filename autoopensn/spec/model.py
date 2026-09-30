"""The scenario spec model and the expansion of a spec into sweep points.

Two decisions here are worth explaining, because both were forced by the
acceptance test rather than chosen for elegance.

**Sweeps are one-at-a-time by default, not a grid.** The prompt behind Milestone
1 asks to compare three GMRES tolerances *and* three restart intervals against a
reference. That is seven runs, not ten: three tolerances at the baseline restart,
three restarts at the baseline tolerance, one reference, minus the duplicate
where a swept value equals the baseline. A full grid would be nine plus a
reference, and would answer a question nobody asked. Both modes exist; the
default is the one that matches how people actually describe parameter studies.

**Parameters are split into ``template_parameters`` and ``solver``.** The split
is not cosmetic: it is checked. A parameter the template declares in the solver
group must appear under ``solver``, and everything else under
``template_parameters``. This makes a spec readable as "here is the physics
problem, and here is how it is solved", and it makes a spec that puts the mesh
refinement under solver settings fail loudly instead of quietly working.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from autoopensn import PINNED_OPENSN_COMMIT
from autoopensn.templates import Template, TemplateError, load_template
from autoopensn.templates.registry import format_sci_float

ParameterValue = Union[bool, int, float, str, list[int], list[float]]


class SpecError(Exception):
    """A spec is malformed, or asks a template for something it cannot do."""


def _slug(value: Any) -> str:
    """A filesystem-safe, stable rendering of one parameter value.

    Small and large magnitudes go through the scientific formatter so that a
    tolerance sweep produces case ids that line up: ``1.0e-4``, ``1.0e-6``,
    ``1.0e-8``, rather than ``0.0001``, ``1e-06``, ``1e-08``. The case id ends
    up in a directory name, a fixture name, and a results table, so it is worth
    it being readable.
    """
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        magnitude = abs(value)
        if value == 0.0 or 1e-3 <= magnitude < 1e5:
            text = f"{value:g}"
        else:
            text = format_sci_float(value)
    elif isinstance(value, (list, tuple)):
        text = "_".join(_slug(item) for item in value)
    else:
        text = str(value)
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text).strip("_") or "none"


class SweepDimension(BaseModel):
    """One parameter varied over a list of values."""

    model_config = ConfigDict(extra="forbid")

    parameter: str = Field(description="Name of a parameter the template declares.")
    values: list[ParameterValue] = Field(
        min_length=1, description="Values to run this parameter at."
    )

    @field_validator("values")
    @classmethod
    def _reject_duplicates(cls, values: list[ParameterValue]) -> list[ParameterValue]:
        seen: list[ParameterValue] = []
        for value in values:
            if value in seen:
                raise ValueError(f"duplicate value {value!r}")
            seen.append(value)
        return values


class SweepPoint(BaseModel):
    """One run: a case id, the parameters it uses, and what makes it distinct."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    parameters: dict[str, Any]
    varied: dict[str, Any] = Field(
        default_factory=dict,
        description="The parameters that differ from the study baseline.",
    )
    is_reference: bool = False

    def label(self) -> str:
        """A short human label, for the Case column of a results table."""
        if self.is_reference:
            return "reference"
        if not self.varied:
            return "baseline"
        return ", ".join(f"{name}={_slug(value)}" for name, value in sorted(self.varied.items()))


class Spec(BaseModel):
    """A complete description of a parameter study.

    Serializes to YAML, is the only thing the ``render`` and ``run`` stages
    need, and is what a user is expected to inspect before anything executes.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short name for the study; used in run directories.")
    template: str = Field(description="Name of the problem template to render.")
    pack_commit: str = Field(
        default=PINNED_OPENSN_COMMIT,
        description="OpenSn commit this study is generated against. Recorded in every script.",
    )
    description: str = Field(default="", description="What this study is for, in one sentence.")

    template_parameters: dict[str, ParameterValue] = Field(
        default_factory=dict,
        description="Baseline values for non-solver parameters: mesh, materials, quadrature, output.",
    )
    solver: dict[str, ParameterValue] = Field(
        default_factory=dict,
        description="Baseline values for solver parameters.",
    )

    sweep: list[SweepDimension] = Field(
        default_factory=list, description="Parameters to vary, and over what values."
    )
    sweep_mode: Literal["one_at_a_time", "grid"] = Field(
        default="one_at_a_time",
        description=(
            "one_at_a_time varies each dimension alone from the baseline. "
            "grid runs the full cross product."
        ),
    )
    reference: Optional[dict[str, ParameterValue]] = Field(
        default=None,
        description=(
            "Parameter overrides defining a reference case that other cases are "
            "compared against, such as a much tighter tolerance."
        ),
    )

    observables: list[str] = Field(
        default_factory=lambda: ["avg_flux", "sweeps", "iterations", "wall_time", "balance_residual"],
        description="Observables to extract from each run.",
    )

    num_procs: int = Field(default=1, ge=1, le=1024, description="MPI ranks per run.")
    timeout_seconds: float = Field(
        default=1800.0, gt=0, description="Wall-clock ceiling on one run."
    )
    retry_budget: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Times a failed run may be retried before the point is recorded as failed.",
    )

    # --- structural validation (template-independent) ---

    @field_validator("name", "template")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _sweep_parameters_are_distinct(self) -> "Spec":
        names = [dimension.parameter for dimension in self.sweep]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"parameter swept more than once: {', '.join(sorted(duplicates))}"
            )
        return self

    # --- template-dependent validation ---

    def load_template(self) -> Template:
        try:
            return load_template(self.template)
        except TemplateError as exc:
            raise SpecError(str(exc)) from exc

    def validate_against(self, template: Optional[Template] = None) -> Template:
        """Check every parameter against what the template declares.

        Raises ``SpecError`` on an unknown parameter, a value outside a declared
        range, a parameter in the wrong section, or a sweep over something the
        template does not have. Returns the template, because every caller wants
        it next.
        """
        template = template or self.load_template()

        if self.pack_commit != template.opensn_commit and template.opensn_commit:
            raise SpecError(
                f"spec targets pack commit {self.pack_commit} but template "
                f"{template.name!r} was derived from {template.opensn_commit}. "
                f"The PyOpenSn API changes across versions; rendering across a "
                f"commit boundary is not supported."
            )

        try:
            for name, value in self.solver.items():
                declaration = template.declaration(name)
                if declaration.group != "solver":
                    raise SpecError(
                        f"{name!r} is a {declaration.group} parameter; move it from "
                        f"'solver' to 'template_parameters'"
                    )
                declaration.validate(value)

            for name, value in self.template_parameters.items():
                declaration = template.declaration(name)
                if declaration.group == "solver":
                    raise SpecError(
                        f"{name!r} is a solver parameter; move it from "
                        f"'template_parameters' to 'solver'"
                    )
                declaration.validate(value)

            for dimension in self.sweep:
                declaration = template.declaration(dimension.parameter)
                for value in dimension.values:
                    declaration.validate(value)

            for name, value in (self.reference or {}).items():
                template.declaration(name).validate(value)
        except TemplateError as exc:
            raise SpecError(str(exc)) from exc

        overlap = set(self.solver) & set(self.template_parameters)
        if overlap:
            raise SpecError(
                f"parameter set in both sections: {', '.join(sorted(overlap))}"
            )

        unknown_observables = set(self.observables) - set(template.observables)
        if unknown_observables:
            raise SpecError(
                f"template {template.name!r} does not produce "
                f"{', '.join(sorted(unknown_observables))}. "
                f"It produces: {', '.join(template.observables)}"
            )

        # A study that measures a flux difference but never asks the script to
        # print a flux produces a table of blanks, and does so silently. Catch it
        # here, where the fix is one line in the spec.
        if "avg_flux" in self.observables:
            emit = self.template_parameters.get("emit_avg_flux")
            if "emit_avg_flux" in template.parameters and not emit:
                raise SpecError(
                    "observable 'avg_flux' requires template parameter "
                    "emit_avg_flux: true, otherwise the script prints no flux"
                )

        return template

    # --- expansion ---

    def baseline(self, template: Optional[Template] = None) -> dict[str, Any]:
        """Template defaults with the spec's baseline overrides applied."""
        template = template or self.validate_against()
        overrides = {**self.template_parameters, **self.solver}
        return template.merge(overrides)

    def expand(self, template: Optional[Template] = None) -> list[SweepPoint]:
        """Every run this spec asks for, in a stable order.

        Duplicate points are dropped. That matters for the acceptance test: a
        tolerance sweep that includes the baseline tolerance would otherwise run
        the baseline case twice and tabulate it twice.
        """
        template = template or self.validate_against()
        baseline = self.baseline(template)

        combinations: list[dict[str, Any]] = []
        if not self.sweep:
            combinations.append({})
        elif self.sweep_mode == "grid":
            names = [dimension.parameter for dimension in self.sweep]
            for values in itertools.product(*(d.values for d in self.sweep)):
                combinations.append(dict(zip(names, values)))
        else:
            for dimension in self.sweep:
                for value in dimension.values:
                    combinations.append({dimension.parameter: value})

        points: list[SweepPoint] = []
        seen: set[str] = set()

        for varied in combinations:
            coerced = template.validate(varied)
            parameters = {**baseline, **coerced}
            fingerprint = _fingerprint(parameters)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            points.append(
                SweepPoint(
                    case_id=_case_id(coerced),
                    parameters=parameters,
                    varied=coerced,
                )
            )

        if self.reference is not None:
            coerced = template.validate(self.reference)
            parameters = {**baseline, **coerced}
            fingerprint = _fingerprint(parameters)
            if fingerprint not in seen:
                seen.add(fingerprint)
                points.append(
                    SweepPoint(
                        case_id="reference",
                        parameters=parameters,
                        varied=coerced,
                        is_reference=True,
                    )
                )
            else:
                # The reference coincides with a swept point. Mark that point as
                # the reference rather than adding a duplicate run.
                for index, point in enumerate(points):
                    if _fingerprint(point.parameters) == fingerprint:
                        points[index] = point.model_copy(update={"is_reference": True})
                        break

        return points

    # --- serialization ---

    def to_yaml(self) -> str:
        """YAML, ordered as written above rather than alphabetically."""
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=False),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def _fingerprint(parameters: dict[str, Any]) -> str:
    """A stable identity for a fully resolved parameter set."""
    material = repr(sorted((k, _canonical(v)) for k, v in parameters.items()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Make 30 and 30.0 compare equal, and lists hashable in repr."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    return value


def _case_id(varied: dict[str, Any]) -> str:
    """A short, filesystem-safe, deterministic id for one sweep point."""
    if not varied:
        return "baseline"
    return "__".join(f"{name}-{_slug(value)}" for name, value in sorted(varied.items()))


# --- loading and saving -----------------------------------------------------


def load_spec(path: Union[str, Path], validate: bool = True) -> Spec:
    """Load a spec from YAML and, by default, validate it against its template."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: expected a mapping at the top level")

    try:
        spec = Spec(**raw)
    except Exception as exc:  # pydantic ValidationError, with a readable message
        raise SpecError(f"{path}: {exc}") from exc

    if validate:
        spec.validate_against()
    return spec


def save_spec(spec: Spec, path: Union[str, Path]) -> Path:
    """Write a spec to YAML. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_yaml())
    return path
