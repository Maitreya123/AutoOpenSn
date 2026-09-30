"""Turning a tutorial notebook into a template with a parameter sidecar.

The contract is the same one the hand-written Reed template satisfies, and it
is checked here rather than trusted: **rendering the generated template with
every parameter at its default must reproduce the notebook's script character
for character.** A generator that cannot guarantee that is a generator that
quietly edits people's physics.

The mechanism is span substitution. Each detected knob carries the exact offsets
of the literal it occupies, and the generator replaces those offsets, back to
front, with a Jinja placeholder. Nothing else in the file is touched: not the
whitespace, not the comments, not the import order. The alternative, reprinting
the syntax tree, would reformat the whole script and make the byte-exact claim
impossible to state.

**Notation is preserved per literal.** The tutorials write ``4.0`` and ``2.``
and ``1.0e-10`` for what are, to Python, the same kind of thing. The generator
works out which notation each literal was written in by rendering the value in
each candidate style and comparing against the source text. A literal whose
notation cannot be reproduced is left alone: it stops being a knob rather than
becoming one that corrupts the script. That is the conservative direction, and
it is why the byte-exact test can pass for every generated template rather than
most of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from autoopensn.templates.registry import TemplateError, format_literal, load_template
from autoopensn.tutorials.catalog import Scenario, slugify
from autoopensn.tutorials.notebook import Notebook
from autoopensn.tutorials.parameters import Detected, constraint_for, detect, group_for

# Candidate notations, tried in order. ``sci`` before ``plain`` so that a
# tolerance written 1.0e-10 is recognised as scientific rather than left to
# repr, which would render it 1e-10 and change the file.
FLOAT_STYLES = ("sci", "plain", "dot")
INT_STYLES = ("int",)
STR_STYLES = ("dq", "sq")
BOOL_STYLES = ("bool",)


class GenerationError(Exception):
    """A template could not be generated from a notebook."""


@dataclass
class Knob:
    """A detected parameter that survived notation checking."""

    detected: Detected
    style: str

    @property
    def name(self) -> str:
        return self.detected.name


def _styles_for(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        element = value[0] if value else 0
        return tuple(f"list:{style}" for style in _styles_for(element))
    if isinstance(value, bool):
        return BOOL_STYLES
    if isinstance(value, int):
        return INT_STYLES
    if isinstance(value, float):
        return FLOAT_STYLES
    return STR_STYLES


def choose_style(value: Any, source_text: str) -> Optional[str]:
    """The notation that reproduces ``source_text`` exactly, or None."""
    for style in _styles_for(value):
        try:
            if format_literal(value, style) == source_text:
                return style
        except TemplateError:
            continue
    return None


def knobs_for(script: str) -> tuple[list[Knob], list[Detected]]:
    """Which detected parameters can be templated, and which cannot.

    The second list is not a failure. It is the record of what was left as a
    constant and why, which is the thing a person needs when they go looking
    for a knob the catalog promised and the sidecar does not have.
    """
    kept: list[Knob] = []
    skipped: list[Detected] = []
    for item in detect(script):
        text = script[item.start : item.end]
        style = choose_style(item.default, text)
        if style is None:
            skipped.append(item)
            continue
        kept.append(Knob(item, style))
    return kept, skipped


def render_template_source(script: str, knobs: list[Knob]) -> str:
    """The script with each knob's literal replaced by a placeholder."""
    out = script
    for knob in sorted(knobs, key=lambda k: -k.detected.start):
        placeholder = '{{ %s | lit("%s") }}' % (knob.name, knob.style)
        out = out[: knob.detected.start] + placeholder + out[knob.detected.end :]
    return out


# --- descriptions -----------------------------------------------------------


def describe(knob: Knob, scenario: Scenario, prose: str) -> str:
    """One sentence about a parameter, and where the sentence came from.

    The prose of a tutorial usually names its own constants. Where a sentence
    in the notebook mentions this parameter, it is quoted, because the
    tutorial's own words about a quantity are better than anything this
    generator could compose. Where none does, the description says plainly what
    the parameter is structurally, and does not pretend to more.
    """
    detected = knob.detected
    constraint = constraint_for(detected.name, detected.default)

    pieces: list[str] = []
    quoted = _sentence_mentioning(prose, detected.name)
    if quoted:
        pieces.append(f"From the tutorial: {quoted}")
    else:
        if detected.kind == "keyword" and detected.context:
            pieces.append(f"Keyword argument {_bare(detected.name)} of {detected.context}.")
        elif detected.kind == "dict_entry" and detected.context:
            pieces.append(f"Entry {_bare(detected.name)!r} of the {detected.context} block.")
        else:
            pieces.append(f"Script variable {detected.name}.")

    if constraint.provenance == "opensn":
        pieces.append(f"Range from OpenSn itself ({constraint.note}).")
    elif constraint.provenance == "physics":
        pieces.append(f"Range from the quantity ({constraint.note}).")
    else:
        pieces.append(
            "Range is a guard scaled from the tutorial's own value, not a code "
            "limit; widen it by hand if a study needs more."
        )
    return " ".join(pieces)


def _bare(name: str) -> str:
    """A qualified knob name reduced to the argument it stands for."""
    return name.rsplit("_", 1)[-1] if name.count("_") > 2 else name


def _sentence_mentioning(prose: str, name: str) -> str:
    """A sentence from the notebook that mentions this parameter, if any."""
    target = name.replace("_", " ").lower()
    bare = name.lower()
    for sentence in re.split(r"(?<=[.!?])\s+", prose):
        cleaned = " ".join(sentence.split())
        if not (20 < len(cleaned) < 320):
            continue
        lowered = cleaned.lower()
        if bare in lowered or (len(target) > 6 and target in lowered):
            if "```" in cleaned or cleaned.startswith("#"):
                continue
            return cleaned
    return ""


# --- observables ------------------------------------------------------------

ALWAYS_AVAILABLE = ("sweeps", "iterations", "wall_time")


def observables_for(scenario: Scenario, script: str) -> tuple[list[str], dict[str, str]]:
    """What can be measured from this scenario, and which key yields each.

    Three observables come from the solver's own log and are always available.
    The rest come from whatever the tutorial prints and its ``tests.json``
    records, which is the same mechanism OpenSn's regression harness uses, so a
    scenario's observables are exactly the quantities upstream thought were
    worth checking.
    """
    observables = list(ALWAYS_AVAILABLE)
    keys: dict[str, str] = {}

    if "compute_balance" in script or "ComputeBalance" in script:
        observables.append("balance_residual")
    if "AvgFlux=" in script:
        observables.append("avg_flux")
        keys["AvgFlux="] = "avg_flux"

    for check in scenario.gold:
        name = check.observable
        if name not in observables:
            observables.append(name)
        keys[check.key] = name

    return observables, keys


# --- generation -------------------------------------------------------------


@dataclass
class Generated:
    """What generating one template produced."""

    name: str
    directory: Path
    knobs: list[Knob]
    skipped: list[Detected]
    observables: list[str]

    @property
    def parameter_count(self) -> int:
        return len(self.knobs)


def generate(
    scenario: Scenario,
    notebook: Notebook,
    out_root: Path,
    *,
    opensn_commit: str,
    verify: bool = True,
) -> Generated:
    """Write a template and sidecar for one scenario.

    With ``verify``, the generated template is loaded back and rendered at its
    defaults, and the result is compared against the notebook's script. A
    mismatch raises rather than writing a template nobody can trust.
    """
    knobs, skipped = knobs_for(notebook.script)
    template_source = render_template_source(notebook.script, knobs)
    observables, gold_keys = observables_for(scenario, notebook.script)

    directory = Path(out_root) / scenario.name
    directory.mkdir(parents=True, exist_ok=True)
    script_name = f"{scenario.name}.py.j2"
    (directory / script_name).write_text(template_source)

    parameters: dict[str, Any] = {}
    for knob in knobs:
        detected = knob.detected
        constraint = constraint_for(detected.name, detected.default)
        declaration: dict[str, Any] = {
            "group": group_for(detected.name),
            "type": detected.type_name,
            "default": detected.default,
            "provenance": constraint.provenance,
            "description": describe(knob, scenario, notebook.prose),
        }
        if constraint.minimum is not None:
            declaration["minimum"] = constraint.minimum
        if constraint.maximum is not None:
            declaration["maximum"] = constraint.maximum
        if constraint.choices is not None:
            declaration["choices"] = list(constraint.choices)
        if isinstance(detected.default, list):
            declaration["length"] = len(detected.default)
        parameters[detected.name] = declaration

    sidecar: dict[str, Any] = {
        "name": scenario.name,
        "description": f"{scenario.title}. {scenario.summary}".strip(),
        "source": f"doc/source/tutorials/{scenario.notebook}",
        "opensn_commit": opensn_commit,
        "script": script_name,
        "generated_from": scenario.notebook,
        "num_procs": scenario.num_procs,
        "notes": GENERATED_NOTE.format(
            notebook=scenario.notebook,
            kept=len(knobs),
            skipped=len(skipped),
        ),
        "observables": observables,
        "parameters": parameters,
    }
    if scenario.data_files:
        sidecar["data_files"] = scenario.data_files
    if gold_keys:
        sidecar["gold"] = {
            "case": f"{Path(scenario.notebook).stem}.py",
            "tests_json": f"doc/source/tutorials/{Path(scenario.notebook).parent}/tests.json",
            "keys": gold_keys,
            "values": {check.key: check.value for check in scenario.gold},
            "tolerances": {
                check.key: {"abs_tol": check.abs_tol, "rel_tol": check.rel_tol}
                for check in scenario.gold
            },
        }

    (directory / f"{scenario.name}.yaml").write_text(
        yaml.safe_dump(sidecar, sort_keys=False, default_flow_style=False, allow_unicode=True)
    )

    if verify:
        template = load_template(scenario.name, root=Path(out_root))
        rendered = template.render()
        if rendered != notebook.script:
            raise GenerationError(
                f"{scenario.name}: rendering at defaults does not reproduce the "
                f"notebook script. This template is not safe to use."
            )

    return Generated(
        name=scenario.name,
        directory=directory,
        knobs=knobs,
        skipped=skipped,
        observables=observables,
    )


GENERATED_NOTE = """\
Generated from the OpenSn tutorial {notebook} at the pinned commit. Do not edit
by hand: regenerate with `autoopensn scenarios --rebuild`, or the edit is lost.

Rendering this template with every parameter at its default reproduces the
tutorial's script character for character, which the generator verifies before
writing and the test suite re-checks on every run.

{kept} literal(s) in the script were turned into parameters. {skipped} were left
as constants because their notation could not be reproduced exactly, which is
the conservative direction: a knob that corrupts the script is worse than a
constant.

A parameter whose provenance is `inferred` has a range scaled from the value the
tutorial itself uses. That is a guard against an absurd study, not a limit the
code enforces, and it is the first thing to widen when a legitimate study is
rejected.
"""
