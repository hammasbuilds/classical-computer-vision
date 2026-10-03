# 58 — Marching cubes, against exact ground truth

Marching cubes turns a scalar field into a triangle mesh. The usual way to show it is a
picture of the mesh, which proves nothing: a surface that looks right can be the wrong
size.

Here the surfaces are a **sphere** and a **torus**, whose volume and area are known in
closed form, so the error of every extracted mesh is exact rather than relative to
another algorithm's output.

| | |
|---|---|
| ground truth | closed form — sphere `4/3·π·r³`, `4·π·r²`; torus `2·π²·R·a²`, `4·π²·R·a` |
| grids | 16³ to 128³, seven resolutions |
| variants | `lewiner`, `lorensen` |
| measurements | volume error, area error, vertex distance to the true surface, time |

## Results

Sphere, `lewiner`:

| grid | h | volume error | area error | mean vertex distance | time |
|---|---|---|---|---|---|
| 16³ | 0.173 | 1.8139% | 0.9526% | 0.00123 | 4.0 ms |
| 32³ | 0.084 | 0.4174% | 0.2206% | 0.00029 | 1.6 ms |
| 64³ | 0.041 | 0.1014% | 0.0535% | 0.00007 | 8.6 ms |
| 128³ | 0.020 | 0.0249% | 0.0131% | 0.00002 | 53.3 ms |

The 16³ timing is larger than 32³ because the first extraction in a process pays
one-off setup. Timings are single-threaded wall clock and are meaningful as a ratio
across the larger grids, not at the smallest one.

## The finding: the error is exactly second order, and that fixes the cost

Fitting `log(error)` against `log(h)` gives the observed order of convergence:

| surface | variant | volume | area | vertex distance |
|---|---|---|---|---|
| sphere | lewiner | **2.00** | **2.00** | 2.01 |
| sphere | lorensen | **2.00** | **2.00** | 2.01 |
| torus | lewiner | 2.05 | 2.03 | 2.04 |
| torus | lorensen | 2.03 | 2.03 | 2.04 |

All six fits land within 0.05 of 2.00. Halving the grid spacing quarters the error —
and costs eight times the memory and roughly six times the time. That is the whole
trade, and it means accuracy here is bought at a known, unflattering rate: **ten times
the time buys about three times the accuracy**, not ten.

Vertex distance converging at the same order as volume is the useful check. If the mesh
were in the right place but the wrong size, the two would disagree.

## The two variants are the same algorithm above a resolution that matters

`lewiner` and `lorensen` are identical to seven decimal places at every resolution
tested except the coarsest torus, where they differ by 0.63 percentage points
(9.34% against 8.70%):

| grid | lewiner | lorensen | difference |
|---|---|---|---|
| 16³ | 9.3355% | 8.7049% | 0.63 points |
| 24³ | 3.5673% | 3.5673% | 0 |
| 32³ and finer | — | — | 0 |

Lewiner's extra topology handling only has anything to do when the grid fails to resolve
the shape. At 16³ the torus's hole is four cells across. Choosing between the two
variants for accuracy is choosing between identical numbers; choose on licence or
availability instead.

## Input / Output

Input: analytic signed distance functions, sampled onto a regular grid. No data files.
Output: `results/results.json` (28 rows plus fitted orders), three figures.

```
python run.py        # 28 extractions, writes results.json
python figures.py    # convergence, cost and mesh figures
pytest -q            # 8 tests
```

## Figures

![convergence](results/convergence.png)

![cost](results/cost.png)

![meshes](results/meshes.png)

## Limitations

- Two surfaces, both smooth and closed. Marching cubes is at its worst on thin
  structures, sharp creases and surfaces that touch the grid boundary, none of which
  appear here. The second-order result applies to smooth closed surfaces.
- The field is sampled exactly from an SDF. Real volumes are noisy and quantised, which
  adds an error term this experiment does not contain.
- Timing is single-threaded CPU on one machine, useful as a ratio between rows and not
  as an absolute.
- Both variants come from scikit-image. A different implementation of the same
  algorithm could differ, and that was not tested.

## Tests

`pytest -q` — 8 tests. They assert properties rather than stored numbers: the closed
forms, the SDF being zero on the surface, a unit cube measuring 1.0, error falling
monotonically under refinement, the fitted order landing near 2, the order estimator
recovering a planted exponent, every vertex lying within one grid spacing of the true
surface, and the two variants agreeing once the grid resolves the shape.
