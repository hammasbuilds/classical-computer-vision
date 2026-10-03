# 62 — Point-cloud decimation: what the usual metric cannot see

Three ways to throw away points — **voxel grid**, **farthest-point sampling** and
**random** — compared at matched retained counts on the same 42000-point cloud. The
cloud is sampled from an analytic torus, so distance to the true surface is exact, and
one region of it is deliberately over-sampled with a **closed-form area share**, so
"did the decimation reproduce the shape or the scanner's dwell time" is a ratio against
a planted answer.

| | |
|---|---|
| cloud | 42000 points on a torus (R = 1, a = 0.35), of which 2222 lie in an over-sampled patch |
| planted ground truth | the patch's area is `a·2wu·(R·2wv + a(sin(v0+wv) − sin(v0−wv)))` = 0.08482102920624893, a 0.61% share of the surface; the input holds 5.2904761904761906% of its points there, a bias of **8.618248400757404** |
| measured | Chamfer back to the original, covering radius, distance off the true surface, density bias, wall clock |
| retained | 300 to 20000 points, matched across methods by bisecting the voxel size |

The voxel grid takes a size, not a count, so the size is bisected to hit each target.
Without that the three methods would be plotted against different budgets and the
comparison would mean nothing.

## Results at 1250 kept points

| method | covering radius | Chamfer to original | density bias | time |
|---|---|---|---|---|
| random | 0.19607 | 0.02557 | 10.686317728967957 | 0.07 ms |
| farthest point | **0.07994** | **0.02013** | 1.3032094791424336 | 1103.65 ms |
| voxel (nearest) | 0.09839 | 0.02022 | 2.0868046103161464 | 38.41 ms |
| voxel (centroid) | 0.08944 | 0.02454 | 2.0868046103161464 | 34.59 ms |

## Finding 1 — the metric usually quoted is the one that cannot tell these apart

Chamfer distance back to the original cloud is what decimation is normally scored on.
Across every matched count, the spread between the best and worst method on that metric
is between **1.24× and 1.87×**. On covering radius — the largest distance from any
original point to its nearest kept point — the same four methods differ by up to
**3.28×**, and on density bias by nearly **10×**.

| points kept | 300 | 1250 | 5000 | 20000 |
|---|---|---|---|---|
| Chamfer spread (max ÷ min) | 1.24 | 1.27 | 1.43 | 1.87 |
| random's covering radius ÷ farthest point's | 2.27 | 2.45 | 2.52 | 3.28 |

Chamfer is a mean over every point, so a handful of badly served regions barely move it.
The covering radius is the worst case, and the worst case is what decides whether a
later surface fit, normal estimate or collision test has anything local to work with.

## Finding 2 — random decimation preserves the capture, not the shape

The input's patch is over-represented by a factor of 8.618248400757404 relative to its
area. After decimation:

| points kept | random | farthest point | voxel (nearest) |
|---|---|---|---|
| 20000 | 8.54416714762758 | 1.7837679745762058 | 1.4339893036338398 |
| 10000 | 8.698923273275744 | 1.3357897161209946 | 1.0136364375104236 |
| 2500 | 10.09987346335386 | 1.1728885312281903 | 0.7797424087409853 |
| 600 | 11.131580967674953 | 1.0860078992853615 | 1.629011848928042 |

Random keeps the patch at **7.6 to 11.1 times its area share** at every budget: it
preserves whatever density the scanner happened to produce, because that is exactly what
uniform random selection does. Farthest point lands between 1.0860078992853615 and
1.7837679745762058, and the voxel grid between 0.7797424087409853 and
2.0868046103161464 — both re-normalise the cloud onto the geometry.

Which is right depends on the question. If the extra points are measurement redundancy,
removing the bias is the point. If they are a region someone deliberately scanned
closely, random is the only one of the three that respects that, and the other two will
throw the detail away.

## Finding 3 — the covering radius falls as n^−0.5, and farthest point is the only one that beats it

| method | covering radius vs points kept | Chamfer vs points kept |
|---|---|---|
| random | −0.4894031378211345 | −0.6300478235653548 |
| farthest point | **−0.5716381678139639** | −0.650119041458106 |
| voxel (nearest) | −0.44016913301210747 | −0.6419677027822337 |
| voxel (centroid) | −0.4940860506478998 | −0.5360790259880305 |

A surface is two-dimensional, so n points placed anywhere sensible cover it to about
`n^−0.5`, and random and the voxel centroid sit on that line. Farthest-point sampling is
the greedy 2-approximation to the k-centre problem and its exponent is steeper at
−0.5716381678139639: its advantage **grows** with the budget, from 2.05× at 600 points
to 3.28× at 20000.

## Finding 4 — the cost, and the one method that moves points off the surface

| | random | voxel (nearest) | farthest point |
|---|---|---|---|
| time at 84000 input points | 0.22 ms | 86.19 ms | 22199.88 ms |
| fitted exponent (n ≥ 21000) | 0.7297158093186499 | 1.1398507344871387 | **2.2566813508227397** |

Farthest point is quadratic and it shows: **22199.88 ms** for 84000 points, against 86.19
milliseconds for the voxel grid — more than two hundred times the cost for an advantage of
about 1.6× in covering radius. Below roughly 20000 points, random's measured exponent is held down by
fixed overhead rather than work, which is why the fit above uses the larger sizes only.

The three index-based methods return **original** points, so their worst distance off
the true surface is 3.9e-16 — floating-point dust. A voxel *centroid* is
the average of the points in a cell and sits inside the surface wherever it curves: up
to 0.013379138450649919 off it, with the mean offset growing as `voxel^2.380669287091361`
across voxel sizes from 0.03 to 0.12. Curvature alone predicts an exponent of 2; the fit
is steeper because occupancy is also rising over that range, from 2.69 to 34.91 points
per cell. Keeping the original point nearest the centroid costs nothing and removes the
error entirely.

## Input / Output

Input: points sampled from an analytic torus with a deliberately over-sampled patch. No
data files.
Output: `results/results.json` — one row per method and setting, the fitted exponents,
the covering-radius ratios, the Chamfer spread and the density bias — and three figures.

```
python run.py                                      # all sweeps, writes results.json
python figures.py                                  # three figures
pytest -q                                          # 10 tests
python ../../tools/check_readme_numbers.py .       # every README number vs results.json
```

## Figures

![metrics](results/metrics.png)

![cost](results/cost.png)

![scatter](results/scatter.png)

## Limitations

- One shape and one kind of density bias: a single compact over-sampled patch. Real
  scans are biased by viewing angle and range, which produces a smooth gradient of
  density rather than a patch, and the voxel grid handles that case less cleanly than it
  handles this one.
- The cloud is noise-free. With noisy points the voxel centroid's averaging becomes an
  advantage that this experiment, by construction, cannot show.
- Farthest-point sampling is the plain O(n·k) greedy version. Tree-accelerated and
  approximate variants exist and would change the cost finding, though not the quality
  one.
- The covering radius is measured against the original cloud, not against the continuous
  surface, so it is a lower bound on the true covering radius by about the original
  sample spacing.
- Timings are single-threaded CPU on one machine. Each measurement repeats the operation
  until at least 50 ms has elapsed, which fixed an earlier version where random
  selection was timed once at under a tenth of a millisecond and produced a cost
  exponent of 0.5390484264919814 — a measurement of the clock rather than of the method.
- Only one voxel variant keeps an original point. Other classical choices (the point
  closest to the cell centre, the first point seen) are implemented but not swept.

## Tests

`pytest -q` — 10 tests. Two guard against the experiment measuring nothing:

- `test_the_patch_area_share_matches_an_independent_monte_carlo` checks the planted
  ground truth itself. The area share is a closed-form integral and membership is
  recovered from xyz coordinates by different code; a 400000-point Monte Carlo must
  agree with the formula to within four standard errors.
- `test_the_three_methods_do_not_return_the_same_cloud` fails if any two methods overlap
  on more than 60% of their kept points.

The rest: every sampled point lying exactly on the torus, index-based methods keeping
original points while the centroid does not, farthest point winning on the quantity it
greedily minimises, random preserving the input bias while the others remove it, the
Chamfer spread being far smaller than the coverage spread, the voxel bisection hitting
its target count, the covering radius falling as `n^−0.5`, and the exponent fitter
recovering a planted power law.
