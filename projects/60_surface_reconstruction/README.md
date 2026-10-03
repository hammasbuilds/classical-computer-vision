# 60 — Surface reconstruction, scored against the surface itself

Four classical reconstruction methods are run on points sampled from an analytic
surface, and scored on **distance to the true surface** (exact, from its signed
distance function) and on **topology** (`V − E + F` against the known Euler
characteristic: 2 for a sphere, 0 for a torus). No reference mesh, no other
algorithm's output, nothing to argue with.

| | |
|---|---|
| surfaces | unit sphere, torus (R = 1, a = 0.35) |
| ground truth | `|sdf|` at area-weighted samples of the output triangles; `V − E + F` |
| methods | convex hull · alpha shape (3-D Delaunay) · Hoppe tangent plane · Poisson (FFT) |
| varied | 500–8000 points, noise σ = 0 to 0.04, alpha, grid resolution |
| normals | estimated by PCA and oriented by spanning-tree propagation — not given |

The two implicit methods are **not** handed the analytic normals; they estimate them
and orient them without ground truth, and the orientation accuracy is reported
separately so the quality of that input is measured rather than assumed.

## Results — Chamfer distance to the true surface, by input size

Torus, noise floor σ = 0.001:

| method | 500 | 1000 | 2000 | 4000 | 8000 | fitted exponent | topology |
|---|---|---|---|---|---|---|---|
| convex hull | 0.06827 | 0.06605 | 0.06480 | 0.06488 | 0.06503 | **−0.02** | wrong at every size |
| alpha shape | 0.02968 | 0.00517 | 0.00611 | 0.00288 | 0.00221 | −0.83 | wrong at 4 of 5 |
| Hoppe tangent plane | 0.00794 | 0.00337 | 0.00184 | 0.00106 | 0.00088 | −0.80 | correct |
| Poisson (FFT) | 0.01279 | 0.00611 | 0.00432 | 0.00316 | 0.00232 | −0.59 | correct |

Sphere:

| method | 500 | 8000 | fitted exponent | topology |
|---|---|---|---|---|
| convex hull | 0.00793 | 0.00052 | **−1.00** | correct |
| alpha shape | 0.03521 | 0.00318 | −0.87 | wrong at every size |
| Hoppe tangent plane | 0.00362 | 0.00057 | −0.68 | correct |
| Poisson (FFT) | 0.00485 | 0.00064 | −0.71 | correct |

## Finding 1 — the convex hull is the most accurate method on the sphere and the only one that cannot be fixed on the torus

On a convex surface the hull converges as `n^−1.00` — faster than every other method
tested, and at 8000 points it is the most accurate of the four (0.00052) while taking
18.1 ms against Hoppe's 782.1 ms.

On the torus the same method is stuck at 0.065 and **sixteen times more points change it
by −4.7%**: the fitted exponent is −0.02, which is a flat line with measurement noise on
it. Its Euler characteristic is 2 at every size, because a hull has no way to represent
a hole. This is the clean case of a method that is not slowly wrong but structurally
wrong, and the exponent is how you tell the two apart without looking at a picture.

## Finding 2 — the method with the better clean score is the one that breaks first

At 4000 points, Chamfer distance against noise, with the ratio Hoppe ÷ Poisson:

| σ | Hoppe (sphere) | Poisson (sphere) | ratio | Hoppe (torus) | Poisson (torus) | ratio |
|---|---|---|---|---|---|---|
| 0 | 0.00030 | 0.00119 | 0.25 | 0.00089 | 0.00313 | 0.28 |
| 0.005 | 0.00288 | 0.00148 | 1.94 | 0.00290 | 0.00323 | 0.9 |
| 0.01 | 0.00583 | 0.00211 | 2.77 | 0.00557 | 0.00348 | 1.6 |
| 0.02 | 0.01190 | 0.00369 | 3.23 | 0.01186 | 0.00441 | 2.69 |
| 0.04 | 0.02488 | 0.00735 | **3.39** | 0.03459 | 0.00792 | **4.36** |

On clean points Hoppe is four times *better* (ratio 0.25). By σ = 0.04 it is **4.36×
worse on the torus** and its output is no longer a torus at all — `V − E + F` is −107
there, while Poisson holds 0 at every noise level on both shapes. The reason is in the
structure of the two methods rather than in their parameters: Hoppe reads the nearest
single oriented point, so each noisy normal moves the surface locally, while the Poisson
solve is a global least-squares fit in which independent normal errors cancel. The same
normals feed both — their angular error rises from 0.82° to 14.4° across this sweep.

Largest σ at which Chamfer stays under 0.02: alpha shape 0.01, Hoppe 0.02, Poisson 0.04.

## Finding 3 — alpha shapes get the topology right at three settings out of sixteen

Sweeping alpha from 1.5 to 20 times the mean sample spacing, on both shapes:

| alpha / spacing | 1.5 | 3 | 4 | 6 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|
| torus, Chamfer | 0.01410 | 0.00427 | 0.00288 | **0.00143** | 0.00147 | 0.02274 | 0.04040 |
| torus, `V − E + F` | 1039 | 749 | 461 | **0** | **0** | 2 | 2 |

Too small and the surface is a cloud of disconnected slivers; too large and the hole
fills in and the result is the convex hull again. The window where the torus comes out
as a torus is between 6 and 8 times the sample spacing, and it is found by sweeping,
not by a rule. The accuracy inside it is good — 0.00143, second only to Hoppe — which is
exactly what makes the method dangerous: the wrong settings do not look obviously wrong.

## The degeneracy that forced a noise floor

Four points on a sphere are cospherical, so **every** Delaunay tetrahedron of an exactly
spherical sample is inscribed in that sphere. Measured over all 3993 tetrahedra of a
2000-point sample, the minimum circumradius is 0.9999999999964 — the radius itself, to
twelve digits. The alpha test `circumradius < alpha` therefore keeps either nothing or
everything, and alpha shapes return no surface at all.

Adding σ = 0.001 (0.1% of the radius) drops the median circumradius to 0.726 and puts
28.9% of tetrahedra below half the radius. The density sweep is therefore run at that
noise floor; the noise sweep keeps σ = 0 so the degeneracy is visible rather than
hidden.

## Cost

Implicit methods at a matched grid, 4000 points on the torus:

| grid | Hoppe | Poisson (FFT) | speed-up | Poisson's Chamfer ÷ Hoppe's |
|---|---|---|---|---|
| 32³ | 36.7 ms | 9.3 ms | 3.95 | 3.42 |
| 64³ | 275.5 ms | 68.8 ms | 4.0 | 2.98 |
| 96³ | 971.3 ms | 272.6 ms | 3.56 | 1.83 |

The FFT solve is 2.47 to 6.01 times faster across every grid and shape tested, and 1.16
to 3.6 times less accurate. Hoppe stops improving past 64³ on the torus (0.00106 at both
64³ and 96³) because the 4000 input points, not the grid, are then the limit.

## Input / Output

Input: points sampled from analytic surfaces. No data files.
Output: `results/results.json` — one row per setting, plus fitted exponents, the
topology summary, the alpha and grid trade-offs and the degeneracy measurement — and
four figures.

```
python run.py                                      # all sweeps, writes results.json
python figures.py                                  # four figures
pytest -q                                          # 11 tests
python ../../tools/check_readme_numbers.py .       # every README number vs results.json
```

## Figures

![density](results/density.png)

![noise](results/noise.png)

![alpha and grid](results/alpha_and_grid.png)

![meshes](results/meshes.png)

## Limitations

- Two surfaces, both smooth, closed and convex or near-convex. Sharp creases, thin
  sheets, open boundaries and self-occlusion are absent, and all four methods behave
  differently on those.
- No ball-pivoting. A faithful implementation is a substantial piece of geometry code
  and the comparison here is between a hull, a Delaunay filter and two implicit solvers,
  which already spans the families; ball pivoting would sit beside the alpha shape.
- The Poisson solve uses an FFT with periodic boundaries, not Kazhdan's adaptive octree
  with B-spline bases. The structure of the method is the same and the comparison
  against Hoppe is fair, but the absolute numbers would differ from a reference
  implementation.
- The point samples are uniform over area. Real scans are dense where the sensor looked
  and sparse elsewhere, which hurts the implicit methods more than this shows.
- Completeness is estimated from 20000 analytic surface samples, so it is exact to the
  triangle but Monte Carlo in the averaging.
- Timing is single-threaded CPU on one machine and is meaningful as a ratio between
  rows, not as an absolute.

## Tests

`pytest -q` — 11 tests, all properties rather than stored numbers. The two that matter
most are the guards against the experiment measuring nothing:

- `test_the_error_metric_is_not_floored_by_its_own_sampler` plants a known radial
  displacement and requires the metric to report it. An earlier version estimated
  completeness as the nearest neighbour in a dense point sample of the mesh, which has a
  floor near `0.5·sqrt(area/m)` — about 0.005 on the unit sphere, larger than the error
  of every method here. All four scored the same and the comparison was measuring the
  sampler. It is now an exact point-to-triangle distance.
- `test_more_points_actually_change_the_answer` fails if a sweep's rows are identical.

The rest: analytic samples lying on the surface and normals matching the SDF gradient,
the Euler characteristic of a tetrahedron and a hand-built torus, point-to-triangle
distance against three cases solved by hand, the convex hull's genus-0 failure, the
implicit methods' topology on both shapes, normal orientation reaching above 0.98
without ground truth, Poisson beating Hoppe under heavy noise, and the exponent fitter
recovering a planted power law.
