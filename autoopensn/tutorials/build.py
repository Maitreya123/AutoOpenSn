"""Rebuilding the scenario catalog and the generated templates.

One entry point, used by the ``scenarios`` command and by the test that checks
the committed output has not drifted from the checkout.

The generated templates are committed rather than built on demand. Two reasons.
A machine with only this repository can still run a study, which is the same
reason the fixtures and the Reed script are committed. And a generated template
is reviewable: when the generator changes, the diff shows exactly which physics
scripts changed and how, which is not visible when the artifact is produced at
import time.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from autoopensn import PINNED_OPENSN_COMMIT, kp_bridge
from autoopensn.templates.registry import TEMPLATE_ROOT
from autoopensn.tutorials import catalog
from autoopensn.tutorials.catalog import Scenario
from autoopensn.tutorials.generate import Generated, GenerationError, generate
from autoopensn.tutorials.notebook import Notebook

TUTORIALS_RELATIVE = Path("doc/source/tutorials")

# Templates written by hand are not regenerated and not deleted. There is
# exactly one, and it predates the generator; it stays because it is the
# worked example the generator was checked against.
HAND_WRITTEN = {"reed_1d"}


@dataclass
class BuildReport:
    """What a rebuild produced."""

    scenarios: list[Scenario] = field(default_factory=list)
    generated: list[Generated] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def knob_count(self) -> int:
        return sum(item.parameter_count for item in self.generated)

    def summary(self) -> str:
        parts = [
            f"{len(self.generated)} template(s) from {len(self.scenarios)} scenario(s)",
            f"{self.knob_count} parameter(s)",
        ]
        if self.failures:
            parts.append(f"{len(self.failures)} failed")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        return ", ".join(parts)


def tutorials_root(checkout: Optional[Path] = None) -> Path:
    """Where the tutorials live in the pinned OpenSn checkout."""
    root = Path(checkout) if checkout else kp_bridge.checkout_path()
    path = root / TUTORIALS_RELATIVE
    if not path.is_dir():
        raise GenerationError(f"no tutorials directory at {path}")
    return path


def rebuild(
    *,
    checkout: Optional[Path] = None,
    template_root: Optional[Path] = None,
    catalog_path: Optional[Path] = None,
    commit: str = PINNED_OPENSN_COMMIT,
    prune: bool = True,
) -> BuildReport:
    """Read every tutorial, write a template for each, and save the catalog."""
    root = tutorials_root(checkout)
    template_root = Path(template_root) if template_root else TEMPLATE_ROOT
    report = BuildReport(scenarios=catalog.scan(root))

    wanted = {scenario.name for scenario in report.scenarios}

    for scenario in report.scenarios:
        try:
            notebook = Notebook.read(root / scenario.notebook)
            report.generated.append(
                generate(scenario, notebook, template_root, opensn_commit=commit)
            )
        except Exception as exc:
            report.failures.append((scenario.name, f"{type(exc).__name__}: {exc}"))

    if prune:
        for child in sorted(template_root.iterdir()):
            if not child.is_dir() or child.name in HAND_WRITTEN or child.name in wanted:
                continue
            if not (child / f"{child.name}.yaml").exists():
                continue
            shutil.rmtree(child)
            report.removed.append(child.name)

    # Hand-written templates belong in the catalog too. It is the menu a request
    # is matched against, and a scenario missing from it is a scenario nobody
    # can ask for.
    for name in sorted(HAND_WRITTEN):
        try:
            from autoopensn.templates import load_template  # noqa: PLC0415

            template = load_template(name, root=template_root)
        except Exception as exc:
            report.failures.append((name, f"could not catalogue: {exc}"))
            continue
        report.scenarios.append(catalog.scenario_from_template(template))

    catalog.save(report.scenarios, catalog_path or catalog.CATALOG_PATH)
    return report
