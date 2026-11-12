# Visualising AutoOpenSn's results

Two kinds of data come out of every study, and they need different tools:

- **Results tables**: one row per case, with what was varied and what came
  out (sweeps, convergence, flux difference, timings, memory). These are small,
  tabular, and answer "how did changing X change Y?"
- **Fields**: the solution on the mesh, which OpenSn writes as VTK files, plus
  the occasional CSV profile. These are large and spatial, and answer "what
  does the flux actually look like?"

This document compares the systematic ways to visualise each, records what
exploring the real outputs turned up, and says what AutoOpenSn does and why.

## The options

A *systematic* approach means a chart follows from the data and a fixed rule,
not from someone deciding by hand each time. Five candidates were considered.

| Approach | Good at | Weak at | Fit here |
| --- | --- | --- | --- |
| **Declarative chart grammar** (Vega-Lite) | Tables. A chart is a JSON description (data, marks, encodings) that any Vega renderer draws, in a browser or offline | Spatial fields | **Chosen for results** |
| **Scripted plotting** (matplotlib) | Anything, server-side, to a static image; no browser needed | Not interactive; every plot is code | **Chosen for fields** |
| **Scientific 3D tools** (ParaView, VTK, PyVista) | Full 3D exploration of VTK output: slicing, contours, volume rendering, scripted batch export with `pvbatch` | Heavy; ParaView is a desktop app, and PyVista needs VTK and an OpenGL context on the server | **Recommended for deep 3D work**, not embedded |
| **Notebooks** (Jupyter with pandas, hvPlot, xarray) | Open-ended exploration by one person | Not repeatable for the next study; nothing is derived from the study's structure | A complement, not the system |
| **In situ** (ParaView Catalyst, Ascent) | Visualising during very large runs without writing full fields | Needs OpenSn built with it, and changes to the input scripts | Out of scope at current problem sizes |

## What exploring the outputs found

Every output the cluster verification produced (41 scenarios on class01) was
examined before choosing anything.

- **1D flux** on line cells, for example `adjoint_example`.
- **2D multigroup flux** on quadrilaterals. `hdpe_example` has 172 energy
  groups (`phi_g000_m00` to `phi_g171_m00`), written as one file per MPI rank
  under a `.pvtu` index.
- **3D meshes** with only `Block` (material region) and `Partition` (which MPI
  rank owns each cell). One `logvol_examples` rank file is 1.1 million points.
- **CSV line profiles** with `x, y, z` and a value column.

Two properties of the files decided how they had to be drawn:

1. **The flux is linear discontinuous.** Every cell stores its own copy of its
   corner values (48 points for 12 quadrilaterals, four each). Interpolating
   across cells would draw a smooth solution OpenSn did not compute.
2. **The files are zlib-compressed appended binary.** A hand-written reader is
   the easy way to get this subtly wrong, so they are read with meshio, a small
   pure-Python library.

## What AutoOpenSn does

### Results: derived charts

Each study's charts are worked out from what it varied and what it produced
(`autoopensn/viz/charts.py`):

| The study varied | Chart |
| --- | --- |
| A number (a tolerance, a cell count) | Line chart; log axis when the values span two decades or more |
| A choice (an inner solver) | Bar chart |
| Two parameters on a grid | Heatmap per measure |

A measure is plotted only if it is numeric, present for two or more cases, and
actually varies. A case that **did not converge** is drawn in its own colour
and shape, because its sweep count is the cap it hit rather than what the
method needed. Studies that varied the same parameters on the same scenario
can be **overlaid** on one chart from the catalogue.

### Fields: drawn by kind

`autoopensn/viz/fields.py` picks the picture from the field's geometry:

| Field | Picture |
| --- | --- |
| 1D | Value along the axis that varies |
| 2D | Each cell shaded from its own corner values, so the discontinuous solution is drawn exactly |
| 3D | A slice through the middle of the longest axis |
| `Block`, `Partition` | Discrete colours, one per value present, because they are labels, not quantities |
| Multigroup flux | Also an energy spectrum (flux per group), as a Vega-Lite chart |
| CSV profile | A line chart |

Small meshes are outlined so a mesh tutorial shows its mesh. With no flux in a
file, the default picture is whichever region field actually varies.

### Where they appear

On the web page, under each study's results, and for any earlier study in the
catalogue. Over HTTP: `/api/studies/{id}/charts`, `/api/compare?group=…`,
`/api/studies/{id}/fields`, `…/fields/image` and `…/fields/chart`.

The charts are plain Vega-Lite JSON, so the same description renders in the
page, in a report (to SVG or PNG with vl-convert, no browser needed), or for an
agent over MCP.

### Explained in words

The analysis step ("Analyse these results") never sees an image. With the
study's id, the server describes every chart and field picture in sentences
computed from the same data: what each panel plots and on what axes, each
point's value, which cases did not converge, the mesh's extent, where the flux
is highest and lowest (with the place named: centre, edge, corner), and where
the energy spectrum peaks (`describe_chart` in `viz/charts.py`, `describe` in
`viz/fields.py`). The model explains the figures from those sentences, and the
number check accepts a figure only if it is in the table or in them. Up to six
field files per study are described.

## Checks

Every chart the tests make is drawn by vl-convert, the reference Vega-Lite
renderer, so a malformed chart fails the build rather than the page. Field
pictures were tested against small VTU files written the way OpenSn writes them,
and the pictures from real cluster output were looked at, not just generated.
That inspection caught a single-material mesh drawn as one colour with a
meaningless scale, a 3D slice drawing two cell layers over each other, and
region IDs coloured on a continuous scale.

## What is left

- **Full 3D views.** A slice answers "what does it look like inside"; rotating
  a volume, contouring or volume rendering belongs in ParaView. The VTU files
  open there directly, with `File → Open` on the `.pvtu`.
- **Volume-weighted averages.** The spectrum averages over mesh points, which
  tracks the volume average on meshes of similar-sized cells and is labelled as
  such. Weighting by cell volume would make it exact on graded meshes.
- **Time-dependent output.** No scenario here writes field time series yet;
  when one does, the same rules extend to a frame per step.
