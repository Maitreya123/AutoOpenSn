# Cluster verification

Every scenario, run once at its default parameters on real OpenSn, with its
output checked against the gold values OpenSn's own regression suite records.

| | |
| --- | --- |
| Machine | class01, TAMU NUEN Orchard cluster |
| OpenSn | commit `2fd4a19`, built with `-DOPENSN_WITH_PYTHON_MODULE=ON` |
| Toolchain | GCC 15.2, MPICH 4.3.2, Python 3.12.3 |
| Runner | `RemoteRunner`, the same path the web page uses |
| Date | 2026-10-07 |

## Result

**37 of 41 reproduce OpenSn's recorded answers and exit cleanly. None produced a wrong answer.**
The largest relative difference from gold across all of them is 2.2e-07.

| Outcome | Scenarios |
| --- | --- |
| matches gold | 37 |
| ran; no gold recorded | 1 |
| matches gold, then failed | 1 |
| did not finish | 2 |

The four that are not a clean match, and why:

- **glovebox** — matches gold, then failed. Answers match; the tutorial then writes to `Flux/`, a folder it never creates, and its 8 ranks hung until the time limit.
- **keigen_to_transient** — did not finish. The site Python 3.12 was built without `_ctypes`.
- **operator_methods** — ran; no gold recorded. Prints no gold values; the run itself completed.
- **source_step_steady_to_transient_to_steady** — did not finish. The site Python 3.12 was built without `_ctypes`.

None of the four is a fault in AutoOpenSn's rendering, staging or parsing.
Two need the cluster's Python rebuilt with `libffi`; one is a bug in the
upstream tutorial; one has nothing recorded to compare against.

## What the first pass found

The first pass over all 41 found no wrong answers either, but most runs did
not finish, and each cause was a bug that could only show up on a real build:

- The login node refused bursts of new SSH connections, nine in a row. Fixed
  by reusing one connection per host and retrying connections that failed
  before anything ran.
- Tutorials end with a Jupyter-only shutdown block that imports IPython,
  which the site Python lacked. Installed IPython and matplotlib to the user
  site rather than editing templates, which would break their character-for-
  character match with upstream.
- Data files were copied flat into the run folder, so any reference with a
  directory in it — `LANL30/OpenMC/Concrete.h5`, `../../../../../test/assets/…`
  — could not be found. A run is now laid out as a mirror of the source tree
  and executed from its notebook's folder.
- Folders of cross sections named by a bare string (`xs_dir = "WIMS69"`), a
  helper module reached through the module search path, and a mesh named
  relative to the source root were not detected as data at all.

Found while reviewing the results: a run that hung was reported as timed out
here and kept running on the node, because only this side was keeping time.
The node now enforces the limit itself.

## Per scenario

| Scenario | Ranks | Outcome | Time on node (s) |
| --- | --- | --- | --- |
| `1g_xs_simple` | 1 | matches gold | 0.522 |
| `adjoint_example` | 4 | matches gold | 3.268 |
| `adjoint_logical_volume_source` | 1 | matches gold | 0.378 |
| `arbitrary_boundary_source` | 1 | matches gold | 1.443 |
| `block_id_source` | 1 | matches gold | 0.387 |
| `cmfd` | 1 | matches gold | 36.344 |
| `delta_forward` | 1 | matches gold | 272.034 |
| `detector` | 4 | matches gold | 21.004 |
| `detector_adj` | 4 | matches gold | 21.348 |
| `first_1d_fixed_source` | 1 | matches gold | 0.414 |
| `first_2d_fixed_source` | 1 | matches gold | 0.418 |
| `forward_example` | 4 | matches gold | 0.489 |
| `forward_peaked_g099` | 1 | matches gold | 10.808 |
| `glovebox` | 8 | matches gold, then failed | 600.008 |
| `glovebox_adj` | 8 | matches gold | 181.44 |
| `hdpe_example` | 8 | matches gold | 2.714 |
| `isotropic_boundary_source` | 1 | matches gold | 1.417 |
| `keigen_to_transient` | 4 | did not finish | 0.05 |
| `lebedev` | 1 | matches gold | 1.573 |
| `logical_volume_source` | 1 | matches gold | 0.385 |
| `logvol_examples` | 1 | matches gold | 65.24 |
| `mesh_extrude` | 1 | matches gold | 0.71 |
| `mesh_ortho_2d` | 1 | matches gold | 0.403 |
| `mesh_ortho_3d` | 1 | matches gold | 0.47 |
| `mesh_ortho_3d_blockids` | 1 | matches gold | 6.365 |
| `mesh_read_gmsh_2d` | 1 | matches gold | 3.013 |
| `mesh_read_gmsh_3d` | 1 | matches gold | 0.832 |
| `mesh_read_obj_2d` | 1 | matches gold | 1.229 |
| `mesh_read_vtu_3d` | 1 | matches gold | 1.147 |
| `mg_xs_read_openmc` | 1 | matches gold | 0.589 |
| `multiple_groupsets` | 1 | matches gold | 0.586 |
| `operator_methods` | 1 | ran; no gold recorded | 3.555 |
| `pincell_example` | 8 | matches gold | 167.416 |
| `point_source` | 1 | matches gold | 0.457 |
| `product_glc` | 1 | matches gold | 1.506 |
| `reed_1d` | 1 | matches gold | 0.885 |
| `sldfe` | 1 | matches gold | 6.925 |
| `source_step_steady_to_transient_to_steady` | 1 | did not finish | 0.049 |
| `triangular_glc` | 1 | matches gold | 1.526 |
| `uncollided` | 1 | matches gold | 1.623 |
| `xs_read_ascii` | 1 | matches gold | 0.498 |

## Repeating it

The harness runs one scenario at a time, checks the node's load before each,
and waits rather than adding work while the node is busy. It is not part of
the test suite, which must never reach a cluster.
