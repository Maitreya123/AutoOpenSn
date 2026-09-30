# Test data

## Vendored from OpenSn

Three files are copied verbatim from the OpenSn repository at commit
`2fd4a19ceade4f8da581a5029c68f474d3333ccb`, from
`test/python/modules/linear_boltzmann_solvers/transport_steady/`:

| File here | File there |
| --- | --- |
| `opensn_reed_balance.py` | `reed_balance.py` |
| `opensn_reed_balance_cbc.py` | `reed_balance_cbc.py` |
| `opensn_tests.json` | `tests.json` |

OpenSn is MIT licensed, copyright 2023 The Texas A&M University System. The
copies carry no added header, because the point of having them is a byte-for-byte
comparison against what the Reed template renders, and a header would defeat it.

They are vendored rather than read from the sister repository's checkout so that
the template test is hermetic: it fails when this repository's template drifts,
on a machine that has only this repository. `tests/test_template_reed.py` also
compares against the live checkout when it is present, which is what catches
drift in the other direction, the vendored copy going stale against upstream.

## Written here

`gmres_convergence.yaml` is the Milestone 1 spec, written by hand.
