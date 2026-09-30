"""Comparing a run against the OpenSn regression suite's gold values.

``tests.json`` in the OpenSn checkout records, for each regression script, a set
of ``KeyValuePair`` checks: a key to find in stdout, the expected value, and a
tolerance. When a spec's parameters are the template's defaults, the rendered
script *is* one of those regression scripts, so its gold values apply directly
and a disagreement means something is wrong with this tool rather than with the
physics.

The comparison refuses to run when the parameters differ. A gold value for a
different problem is worse than no gold value: it invites a reader to treat a
large relative difference as a bug in AutoOpenSn when it is simply a different
simulation. ``physics_parameters`` names the parameters that make a run a
different problem; the output-only and runtime knobs are excluded, because
printing an extra flux does not change the absorption rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from autoopensn.parse.observables import key_value
from autoopensn.templates import Template

# Parameters whose value cannot change the physics, and therefore cannot
# invalidate a gold comparison.
NON_PHYSICAL_GROUPS = {"output", "runtime"}


class GoldError(Exception):
    """Gold values could not be located or read."""


@dataclass(frozen=True)
class GoldCheck:
    """One gold value: a key to find in stdout and what it should be."""

    key: str
    expected: float
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    def tolerance(self) -> float:
        return self.abs_tol + self.rel_tol * abs(self.expected)


@dataclass(frozen=True)
class GoldComparison:
    """The result of checking one run against its gold values."""

    case: str
    checks: tuple[dict[str, Any], ...]
    applicable: bool
    reason: str = ""

    @property
    def passed(self) -> Optional[bool]:
        """True when every check passed, None when no comparison applies."""
        if not self.applicable or not self.checks:
            return None
        return all(check["passed"] for check in self.checks)

    def worst_relative_difference(self) -> Optional[float]:
        differences = [
            check["relative_difference"]
            for check in self.checks
            if check["relative_difference"] is not None
        ]
        return max(differences) if differences else None

    def as_row(self) -> dict[str, Any]:
        return {
            "gold_case": self.case if self.applicable else None,
            "gold_passed": self.passed,
            "gold_worst_rel_diff": self.worst_relative_difference(),
            "gold_reason": self.reason or None,
        }


def gold_from_sidecar(gold_config: dict[str, Any]) -> tuple[GoldCheck, ...]:
    """Gold checks recorded in the template's own sidecar.

    Generated templates copy the values out of the tutorial's ``tests.json`` at
    generation time. That makes a gold comparison possible on a machine with no
    OpenSn checkout, which is most machines, and it means the values that were
    current when the template was generated travel with the template rather than
    drifting under it.
    """
    values = gold_config.get("values") or {}
    tolerances = gold_config.get("tolerances") or {}
    checks = []
    for key, expected in values.items():
        bounds = tolerances.get(key, {})
        checks.append(
            GoldCheck(
                key=key,
                expected=float(expected),
                abs_tol=float(bounds.get("abs_tol", 0.0) or 0.0),
                rel_tol=float(bounds.get("rel_tol", 0.0) or 0.0),
            )
        )
    return tuple(checks)


def load_gold(tests_json: Path, case: str) -> tuple[GoldCheck, ...]:
    """Load the gold checks recorded for one regression script."""
    path = Path(tests_json)
    if not path.exists():
        raise GoldError(f"no gold file at {path}")
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise GoldError(f"{path}: not valid JSON: {exc}") from exc

    for entry in entries:
        if entry.get("file") != case:
            continue
        checks = []
        for check in entry.get("checks", []):
            if check.get("type") != "KeyValuePair":
                continue
            checks.append(
                GoldCheck(
                    key=check["key"],
                    expected=float(check["goldvalue"]),
                    abs_tol=float(check.get("abs_tol", 0.0)),
                    rel_tol=float(check.get("rel_tol", 0.0)),
                )
            )
        return tuple(checks)

    raise GoldError(f"{path} records no case named {case!r}")


def physics_parameters(template: Template, parameters: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``parameters`` that can change what is computed."""
    return {
        name: value
        for name, value in parameters.items()
        if name in template.parameters
        and template.parameters[name].group not in NON_PHYSICAL_GROUPS
    }


def matches_gold_case(template: Template, parameters: dict[str, Any]) -> tuple[bool, str]:
    """Whether this parameter set is the regression case the gold values describe."""
    defaults = physics_parameters(template, template.defaults())
    actual = physics_parameters(template, parameters)
    differing = sorted(
        name for name in defaults if _differs(defaults[name], actual.get(name, defaults[name]))
    )
    if differing:
        return False, "differs from the regression case in " + ", ".join(differing)
    return True, ""


def _differs(left: Any, right: Any) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) != len(right) or any(_differs(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return left is not right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) != float(right)
    return left != right


def compare_to_gold(
    stdout: str,
    template: Template,
    parameters: dict[str, Any],
    *,
    tests_json: Optional[Path] = None,
) -> GoldComparison:
    """Check one run's stdout against the gold values for its regression case.

    Returns a comparison marked not applicable, with a reason, when the run is
    not that case or when the gold file is unavailable. It never raises for a
    missing checkout: a machine without the sister repository can still build a
    results table, it just cannot cross-check one.
    """
    gold_config = template.gold or {}
    case = gold_config.get("case")
    if not case:
        return GoldComparison(case="", checks=(), applicable=False, reason="template declares no gold case")

    applicable, reason = matches_gold_case(template, parameters)
    if not applicable:
        return GoldComparison(case=case, checks=(), applicable=False, reason=reason)

    checks: tuple[GoldCheck, ...] = ()
    path = Path(tests_json) if tests_json else _default_tests_json(gold_config)
    if path is not None and Path(path).exists():
        try:
            checks = load_gold(path, case)
        except GoldError:
            checks = ()
    if not checks:
        checks = gold_from_sidecar(gold_config)
    if not checks:
        return GoldComparison(
            case=case,
            checks=(),
            applicable=False,
            reason="no gold values available for this case",
        )

    results = []
    for check in checks:
        actual = key_value(stdout, check.key)
        if actual is None:
            results.append(
                {
                    "key": check.key,
                    "expected": check.expected,
                    "actual": None,
                    "difference": None,
                    "relative_difference": None,
                    "tolerance": check.tolerance(),
                    "passed": False,
                    "reason": "key not found in output",
                }
            )
            continue
        difference = abs(actual - check.expected)
        relative = difference / abs(check.expected) if check.expected else None
        results.append(
            {
                "key": check.key,
                "expected": check.expected,
                "actual": actual,
                "difference": difference,
                "relative_difference": relative,
                "tolerance": check.tolerance(),
                "passed": difference <= check.tolerance(),
                "reason": "",
            }
        )

    return GoldComparison(case=case, checks=tuple(results), applicable=True)


def _default_tests_json(gold_config: dict[str, Any]) -> Optional[Path]:
    """Locate tests.json in the pinned checkout, if the sister repository is here."""
    relative = gold_config.get("tests_json")
    if not relative:
        return None
    # Imported late and defensively: parsing must work on a machine that has
    # only this repository.
    try:
        from autoopensn import kp_bridge  # noqa: PLC0415

        if not kp_bridge.available():
            return None
        return kp_bridge.checkout_path() / relative
    except Exception:
        return None
