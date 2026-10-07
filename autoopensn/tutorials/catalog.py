"""The scenario catalog: every OpenSn tutorial, as something we can run.

The tutorials are the scenario library. Each is a notebook with prose, a
runnable script, and, for most of them, a ``tests.json`` recording what the
answer should be. That combination is unusual and valuable: a worked example
that is also a regression test.

This module reads them into one list. It is deliberately dumb. It does not
decide which scenario answers a question, it does not run anything, and it does
not interpret the physics. It answers "what is available, what is it about, and
what is the right answer" so that the stage which does choose has something
grounded to choose from.

The catalog is cached to JSON because parsing forty notebooks on every page load
is a second the interface should not spend, and because the catalog is a
function of a pinned checkout and therefore does not change under us.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoopensn.tutorials.notebook import Notebook, NotebookError
from autoopensn.tutorials.parameters import detect, group_for

# Files a tutorial reads: meshes, cross sections, geometry.
DATA_SUFFIXES = (".msh", ".cxs", ".obj", ".vtu", ".xs", ".h5", ".exo", ".csv", ".txt")

_DATA_REFERENCE = re.compile(
    r"""["']([^"'\n]+\.(?:msh|cxs|obj|vtu|xs|h5|exo))["']"""
)


@dataclass
class GoldCheck:
    """One recorded correct answer for a scenario."""

    key: str
    value: float
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    @property
    def observable(self) -> str:
        """A column name for this check: ``MAX_FLUX=`` becomes ``max_flux``."""
        return slugify(self.key.rstrip("=")) or "gold_value"


@dataclass
class Scenario:
    """One tutorial, described well enough to choose and to run."""

    name: str
    title: str
    summary: str
    notebook: str
    """Path relative to the tutorials root."""
    section: str
    """Top-level area: foundations, modeling, numerical_methods, problems, …"""
    gold: list[GoldCheck] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    num_procs: int = 1
    has_gold: bool = False
    script_lines: int = 0
    keywords: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    """Knobs found in the scenario's own script, in source order.

    This is the list a request is matched against when someone asks to "vary
    the restart interval": the question names a parameter, and the parameter
    belongs to a scenario. Reading them from the script rather than from a
    hand-maintained list is what lets the catalog cover every tutorial without
    anyone curating forty of them.
    """
    solver_parameters: list[str] = field(default_factory=list)

    @property
    def sweepable(self) -> bool:
        """Whether there is anything in this scenario to vary.

        A few tutorials demonstrate rather than compute: they read a fixed file
        and print what is in it. Those are worth cataloguing and worth reading,
        and a parameter study over them is not a thing. Saying so here keeps
        them out of a shortlist where they could only produce a spec with an
        empty sweep.
        """
        return bool(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["gold"] = [asdict(check) for check in self.gold]
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Scenario":
        record = dict(record)
        record["gold"] = [GoldCheck(**check) for check in record.get("gold", [])]
        return cls(**record)

    def catalog_entry(self) -> str:
        """One scenario as a model reads it when choosing.

        Kept short on purpose. A catalog of forty entries at four hundred
        characters each is a prompt a model can hold; one at four thousand is a
        prompt it skims.
        """
        parts = [f"{self.name}: {self.title}", f"  area: {self.section}"]
        if self.summary:
            parts.append(f"  about: {self.summary[:280]}")
        if self.has_gold:
            parts.append(f"  verified: yes ({len(self.gold)} recorded value(s))")
        if self.data_files:
            parts.append(f"  needs data files: {', '.join(self.data_files[:4])}")
        if not self.sweepable:
            parts.append("  note: demonstration only; no tunable parameters")
        if self.solver_parameters:
            parts.append(f"  solver knobs: {', '.join(self.solver_parameters)}")
        others = [p for p in self.parameters if p not in self.solver_parameters]
        if others:
            parts.append(f"  other knobs: {', '.join(others[:12])}")
        return "\n".join(parts)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def _gold_for(directory: Path, stem: str) -> tuple[list[GoldCheck], int]:
    """The recorded answers for one notebook, from the tests.json beside it.

    The harness runs a notebook by converting it to a ``.py`` of the same stem,
    so that is the name to match on.
    """
    path = directory / "tests.json"
    if not path.exists():
        return [], 1
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError:
        return [], 1

    for entry in entries:
        if Path(str(entry.get("file", ""))).stem != stem:
            continue
        checks = [
            GoldCheck(
                key=check["key"],
                value=float(check["goldvalue"]),
                abs_tol=float(check.get("abs_tol", 0.0)),
                rel_tol=float(check.get("rel_tol", 0.0)),
            )
            for check in entry.get("checks", [])
            if check.get("type") == "KeyValuePair"
        ]
        return checks, int(entry.get("num_procs", 1) or 1)
    return [], 1


def _data_files(directory: Path, script: str) -> list[str]:
    """Files the script reads, as paths relative to the notebook's directory.

    Only files that actually exist are recorded. A string that looks like a
    path but resolves to nothing is far more likely to be an output filename
    than a missing input, and listing it would make every run fail to stage.
    """
    found: list[str] = []
    for reference in _DATA_REFERENCE.findall(script):
        candidate = (directory / reference).resolve()
        if candidate.exists() and candidate.is_file():
            try:
                found.append(str(Path(reference)))
            except ValueError:
                continue
    found.extend(_helper_modules(directory, script))
    found.extend(_data_directories(directory, script))
    found.extend(_root_relative_files(directory, script))
    # Directories of cross sections are referenced one file at a time, so a
    # tutorial with sixty-nine of them would otherwise list sixty-nine paths.
    return sorted(set(found))


_SYS_PATH_APPEND = re.compile(r"""sys\.path\.(?:append|insert)\((?:\d+,\s*)?["']([^"']+)["']\)""")
_IMPORTED = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)\s+import|import\s+([A-Za-z_]\w*))", re.MULTILINE)


_STRING = re.compile(r"""["']([^"'\n]{1,200})["']""")


def _source_root(directory: Path) -> Optional[Path]:
    """The top of the OpenSn checkout a notebook directory sits in."""
    for candidate in (directory, *directory.parents):
        if (candidate / "doc" / "source" / "tutorials").is_dir():
            return candidate
    return None


def _data_directories(directory: Path, script: str) -> list[str]:
    """Folders of data named by a bare string, like ``xs_dir = "WIMS69"``.

    The detector tutorials build each cross-section path at run time, as
    ``xs_dir + "/Air.cxs"``, so no single string in the script is a file that
    exists — but the folder name is, and the whole folder has to travel.
    """
    found = []
    for literal in _STRING.findall(script):
        if literal in (".", "..") or literal.startswith("/") or ".." in Path(literal).parts:
            continue
        if re.search(rf"""(?:makedirs|mkdir)\(\s*["']{re.escape(literal)}["']""", script):
            # A folder the script creates is an output that happens to share
            # its name with something beside the notebook — the forward-peaked
            # tutorial writes plots into "images", which is also where the
            # documentation keeps its pictures.
            continue
        candidate = directory / literal
        if candidate.is_dir() and any(child.is_file() for child in candidate.rglob("*")):
            found.append(str(Path(literal)))
    return found


def _root_relative_files(directory: Path, script: str) -> list[str]:
    """Data files named relative to the top of the source tree.

    The uncollided tutorial walks up from where it runs until it finds
    ``test/assets/mesh``, then opens a mesh below that. Recorded relative to the
    notebook, like every other data file, so the run lays it out at its true
    source path and the tutorial's own search finds it.
    """
    root = _source_root(directory)
    if root is None:
        return []
    found = []
    for reference in _DATA_REFERENCE.findall(script):
        if (directory / reference).exists() or reference.startswith("/"):
            continue
        candidate = root / reference
        if candidate.is_file():
            found.append(os.path.relpath(candidate, directory))
    return found


def _helper_modules(directory: Path, script: str) -> list[str]:
    """Local Python modules a tutorial reaches by putting a directory on sys.path.

    The sLDFE tutorial appends ``../../../../../tools/ang_quad_plotting`` to the
    module search path and then imports a plotting helper from there. (Spelled
    out in words: the architecture test that confines search-path changes to
    the bridge reads source text, and quoting the call would trip it.) That path only resolves from
    inside the OpenSn source tree, and a run happens in its own directory, so
    the helper has to travel with the run like any mesh does. Recorded relative
    to the notebook, exactly as the tutorial spells it, and only when the module
    file is really there.
    """
    found: list[str] = []
    imported = {a or b for a, b in _IMPORTED.findall(script)}
    for added in _SYS_PATH_APPEND.findall(script):
        for module in imported:
            candidate = directory / added / f"{module}.py"
            if candidate.is_file():
                found.append(str(Path(added) / f"{module}.py"))
    return found


def _keywords(notebook: Notebook, section: str) -> list[str]:
    """Terms a person might use to ask for this scenario.

    Lexical, not semantic. The point is a cheap first pass that narrows forty
    scenarios to a handful before anything expensive happens.
    """
    text = f"{notebook.title} {notebook.summary} {section}".lower()
    words = re.findall(r"[a-z][a-z0-9_-]{2,}", text)
    stop = {
        "the", "this", "that", "and", "for", "with", "from", "are", "its", "one",
        "you", "will", "able", "notebook", "tutorial", "example", "opensn",
        "how", "what", "which", "each", "their", "than", "then", "here", "have",
        "has", "was", "were", "can", "not", "but", "into", "over", "also",
    }
    seen: list[str] = []
    for word in words:
        if word in stop or word in seen:
            continue
        seen.append(word)
    return seen[:40]


def scan(tutorials_root: Path) -> list[Scenario]:
    """Read every tutorial under ``tutorials_root``."""
    root = Path(tutorials_root)
    scenarios: list[Scenario] = []
    seen_names: set[str] = set()

    for path in sorted(root.rglob("*.ipynb")):
        if "checkpoint" in str(path) or "_templates" in path.parts:
            continue
        try:
            notebook = Notebook.read(path)
        except NotebookError:
            continue

        relative = path.relative_to(root)
        stem = path.stem
        name = slugify(stem)
        if name in seen_names:
            name = slugify(f"{relative.parts[0]}_{stem}")
        seen_names.add(name)

        gold, num_procs = _gold_for(path.parent, stem)
        section = relative.parts[0] if len(relative.parts) > 1 else "general"

        try:
            knobs = detect(notebook.script)
        except SyntaxError:
            # A notebook whose cells do not concatenate into valid Python is
            # catalogued without knobs rather than dropped: it is still worth
            # finding, it just cannot be templated yet.
            knobs = []
        parameter_names = [knob.name for knob in knobs]
        solver_names = [name for name in parameter_names if group_for(name) == "solver"]

        scenarios.append(
            Scenario(
                name=name,
                title=notebook.title,
                summary=notebook.summary,
                notebook=str(relative),
                section=section,
                gold=gold,
                data_files=_data_files(path.parent, notebook.script),
                num_procs=num_procs,
                has_gold=bool(gold),
                script_lines=len(notebook.script.splitlines()),
                keywords=_keywords(notebook, section),
                parameters=parameter_names,
                solver_parameters=solver_names,
            )
        )
    return scenarios


def scenario_from_template(template) -> Scenario:
    """A catalog entry for a hand-written template.

    The catalog is the menu a request is matched against, so it has to hold
    every scenario that can actually be run, not only the ones derived from
    tutorials. A hand-written template has the same things a tutorial does, a
    description, parameters and gold values, so it gets the same kind of entry.
    """
    gold = [
        GoldCheck(
            key=key,
            value=float(value),
            abs_tol=float((template.gold.get("tolerances", {}).get(key, {}) or {}).get("abs_tol", 0.0)),
            rel_tol=float((template.gold.get("tolerances", {}).get(key, {}) or {}).get("rel_tol", 0.0)),
        )
        for key, value in (template.gold.get("values") or {}).items()
    ]
    summary = " ".join(template.description.split())
    solver = list(template.solver_parameters)
    return Scenario(
        name=template.name,
        title=summary.split(".")[0][:90] or template.name,
        summary=summary[:600],
        notebook="",
        section="benchmarks",
        gold=gold,
        data_files=list(template.data_files),
        num_procs=template.num_procs,
        has_gold=bool(gold),
        script_lines=len(template.render().splitlines()),
        keywords=re.findall(r"[a-z][a-z0-9_-]{2,}", summary.lower())[:40],
        parameters=list(template.parameters),
        solver_parameters=solver,
    )


# --- persistence ------------------------------------------------------------

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


def save(scenarios: list[Scenario], path: Path = CATALOG_PATH) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(
            {
                "scenarios": [scenario.to_dict() for scenario in scenarios],
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    return path


def load(path: Path = CATALOG_PATH) -> list[Scenario]:
    path = Path(path)
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    return [Scenario.from_dict(item) for item in record.get("scenarios", [])]


def by_name(name: str, scenarios: Optional[list[Scenario]] = None) -> Optional[Scenario]:
    for scenario in scenarios if scenarios is not None else load():
        if scenario.name == name:
            return scenario
    return None


# --- narrowing --------------------------------------------------------------


def rank(query: str, scenarios: list[Scenario], limit: int = 5) -> list[Scenario]:
    """Scenarios most likely to be about ``query``, best first.

    A deliberately simple lexical score. This runs before the model call, to
    cut forty scenarios to a handful, and it only has to be roughly right: the
    model sees the shortlist and the full text of each, and a scenario ranked
    sixth is still reachable because the shortlist is generous.
    """
    terms = set(re.findall(r"[a-z][a-z0-9_-]{2,}", query.lower()))
    if not terms:
        return scenarios[:limit]

    scored: list[tuple[float, Scenario]] = []
    for scenario in scenarios:
        haystack = f"{scenario.name} {scenario.title} {scenario.summary} {scenario.section}".lower()
        knobs = " ".join(scenario.parameters).lower()
        score = sum(3.0 for term in terms if term in scenario.name)
        score += sum(2.0 for term in terms if term in scenario.title.lower())
        score += sum(1.0 for term in terms if term in haystack)
        score += sum(0.5 for term in terms if term in scenario.keywords)
        # A question usually names the knob it wants varied, and the knob is
        # the most specific signal available: "restart interval" belongs to
        # exactly the scenarios whose script has gmres_restart_interval in it.
        score += sum(2.5 for term in terms if term in knobs)
        # A scenario with a recorded correct answer is worth more than one
        # without, all else equal: it can be checked.
        if scenario.has_gold:
            score += 0.5
        # A scenario that needs data files is harder to run anywhere else.
        if scenario.data_files:
            score -= 0.25
        # A scenario with nothing to vary cannot answer a question about how
        # something varies, which is the only kind of question asked here.
        if not scenario.sweepable:
            score -= 5.0
        scored.append((score, scenario))

    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    return [scenario for score, scenario in scored[:limit] if score > 0] or scenarios[:limit]
