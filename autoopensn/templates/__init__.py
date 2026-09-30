"""Problem templates: PyOpenSn scripts with declared, range-checked knobs.

A template is two files in a directory named after it:

    <name>/<name>.py.j2   the Jinja2 script
    <name>/<name>.yaml    the sidecar declaring its tunable parameters

Each template is derived from a script in the OpenSn regression suite at the
pinned commit, and renders byte-for-byte identically to that script when every
parameter is left at its default. That property is the point of the whole
directory: it is a test, it runs on every ``pytest`` invocation, and it is what
lets the rest of the system treat a rendered script as trustworthy without
anyone reading it.

Rendering is a pure function of (template, parameters). It touches no clock, no
filesystem outside the template directory, and no network.
"""

from autoopensn.templates.registry import (
    ParameterDeclaration,
    Template,
    TemplateError,
    list_templates,
    load_template,
    provenance_header,
    render_run_script,
)

__all__ = [
    "ParameterDeclaration",
    "Template",
    "TemplateError",
    "list_templates",
    "load_template",
    "provenance_header",
    "render_run_script",
]
