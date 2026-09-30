"""The scenario spec: the one artifact a user is expected to read and edit.

A spec says what study to run, in terms a template understands. It is the
boundary between the language-model half of this tool and the deterministic
half: everything upstream of it is a sentence, everything downstream is a
function of these fields.

That is why it is strict. Unknown fields are rejected rather than ignored, and
every parameter is checked against the range its template declares. A spec that
loads is a spec that renders, and a spec that renders is a script that is at
least as valid as the regression test it was derived from.
"""

from autoopensn.spec.model import (
    Spec,
    SpecError,
    SweepDimension,
    SweepPoint,
    load_spec,
    save_spec,
)

__all__ = [
    "Spec",
    "SpecError",
    "SweepDimension",
    "SweepPoint",
    "load_spec",
    "save_spec",
]
