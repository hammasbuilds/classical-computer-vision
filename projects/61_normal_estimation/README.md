# 61 — Normal estimation by PCA, against normals known in closed form

Surface normals are estimated as the smallest-eigenvalue direction of the covariance of
each point's k nearest neighbours. The points come from analytic surfaces, so the true
normal is the normalised gradient of the signed distance function and **the error is an
unsigned angle in degrees against the right answer**, not against another estimator.

| | |
|---|---|
| surfaces | plane (zero curvature), sphere (curvature 1.0), torus tube (curvature 2.857) |
| ground truth | analytic normals; error is the unsigned angle, so an unoriented PCA direction is scored fairly |
| swept | k from 4 to 2048, σ from 0 to 0.02, 2500 to 40000 points |
| estimators | PCA over k nearest · PCA over a fixed-radius ball · quadratic (jet) fit |

The plane is in the experiment for a reason: it is the one surface whose answer is known
before the run. With no curvature and no noise the error must be **exactly** zero, and
it is — 0.0 degrees at every k, as a maximum and not merely a mean.

## Three predictions from the geometry, three fitted exponents

Each regime has an exponent that follows from the geometry rather than from the data, so
the run either reproduces it or contradicts it.

| regime | predicted | measured |
|---|---|---|
| plane, error against k at fixed σ | −1 | −1.051, −1.051, −1.053, −1.075, −1.107, −1.092 (σ = 0.0005 … 0.02) |
| plane, error against σ at fixed k | +1 | +1.108 (k = 16), +1.041 (k = 64), +1.010 (k = 256) |
| curved and clean, error against point count | −0.5 | −0.496 (sphere), −0.520 (torus) |
| curved and clean, error against k | 0 | −0.045 (sphere), +0.038 (torus), over k = 32 … 256 |

## Finding 1 — the error falls as k⁻¹, not k⁻⁰·⁵

The usual reasoning is that averaging k noisy neighbours suppresses the error as
`1/sqrt(k)`. Measured across six noise levels on the plane, the exponent is between
**−1.051 and −1.107**, roughly twice as steep.

The reason is that a *k-nearest* neighbourhood is not a fixed patch being filled in more
densely — it is a patch that grows. The tilt of a least-squares plane through k points
scattered by σ over radius r goes as `σ / (sqrt(k)·r)`, and at a fixed sample density
`r ~ sqrt(k)`, so the two factors compound. Doubling k halves the error; it does not cut
it by 30%.

| k | 16 | 64 | 256 |
|---|---|---|---|
| plane, σ = 0.005 | 7.0558° | 1.5053° | 0.3754° |

## Finding 2 — on clean points, more neighbours buy nothing at all

| k | 4 | 16 | 32 | 96 | 256 | 512 | 2048 |
|---|---|---|---|---|---|---|---|
| sphere, σ = 0 | 0.8940° | 0.3701° | 0.3300° | 0.3022° | 0.2995° | 0.2919° | 0.3198° |
| torus, σ = 0 | 1.6721° | 1.0619° | 0.9756° | **0.9354°** | 1.0589° | 1.4153° | 5.1427° |

Between k = 32 and k = 256 the fitted exponent is −0.045 on the sphere and +0.038 on the
torus: a flat line. Going from 32 to 256 neighbours costs eight times the work and
changes the torus error by less than a tenth of a degree.

What does move it is **density**: at a fixed k = 32 the clean error falls as `n^−0.496`
(sphere) and `n^−0.520` (torus), matching the predicted −0.5. The clean floor is a
property of how finely the surface was sampled, not of how many of those samples the
estimator looks at. Past k = 512 the patch finally becomes comparable to the torus tube
and the error climbs with exponent **+0.931** — 5.1427° at k = 2048, worse than k = 4.

## Finding 3 — the best k grows 2.67× with noise, and the curvature decides where it starts

The torus brackets its optimum inside the sweep:

| σ | 0 | 0.0005 | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|---|---|---|
| best k | 96 | 96 | 96 | 96 | 128 | 192 | 256 |
| error there | 0.9354° | 0.9376° | 0.9395° | 0.9608° | 1.0401° | 1.1517° | 1.3882° |
| patch radius | 0.1453 | 0.1453 | 0.1453 | 0.1453 | 0.1678 | 0.2055 | 0.2373 |

The optimum is where the k⁻¹ noise term meets the curvature floor, so it moves right as
noise rises: **96 at σ = 0 and 256 at σ = 0.02, a factor of 2.67**. On the sphere, whose
curvature is 2.857 times gentler, the floor sits lower and the optimum is at k = 512
even with clean points. A single default k cannot be right for both: at σ = 0.02 the
torus's best setting of k = 256 gives 1.3882°, while k = 32 gives 7.5038°.

## Finding 4 — the quadratic fit removes the curvature term and nothing else

Fitting `z = ax² + bxy + cy² + dx + ey + f` in the local frame and reading the normal at
the centre is the classical cure for curvature bias. On clean points it works almost
perfectly:

| | plane fit | quadratic jet | ratio |
|---|---|---|---|
| sphere, σ = 0, best k | 0.2919° | **0.0001°** | 2089.93 |
| torus, σ = 0, best k | 0.9401° | 0.0217° | 43.23 |
| sphere, σ = 0.005, best k | 0.3104° | 0.0515° | 6.02 |
| torus, σ = 0.005, best k | 1.0401° | 0.4775° | 2.18 |
| sphere, σ = 0.02, best k | 0.3748° | 0.2009° | 1.87 |
| torus, σ = 0.02, best k | 1.3882° | 0.9736° | 1.43 |

Three orders of magnitude on clean data collapses to 1.43× once σ reaches 0.02, and at
k = 32 on the torus at that noise it is actually *worse* than the plane fit (ratio 0.96)
— six parameters fitted to noisy points overfit. It costs 17.9× the time at k = 32. The
quadratic fit is the right choice for accurate scans of curved objects and a waste
everywhere else.

## The fixed-radius ball is the same estimator with a worse failure mode

Plotted against the mean number of neighbours actually used, a fixed-radius ball traces
the same curve as a fixed count. The difference is what happens when the setting is
wrong: a 0.02 radius holds 4.2 points on average, and at σ = 0.005 that gives
**30.320°** on the sphere — a normal pointing in an unrelated direction. k nearest
neighbours cannot land there, because k is the thing being held fixed.

## Input / Output

Input: points sampled from analytic surfaces. No data files.
Output: `results/results.json` — one row per setting, the fitted exponents, the best k
per noise level and the jet comparison — and three figures.

```
python run.py                                      # all sweeps, writes results.json
python figures.py                                  # three figures
pytest -q                                          # 11 tests
python ../../tools/check_readme_numbers.py .       # every README number vs results.json
```

## Figures

![k sweep](results/k_sweep.png)

![laws](results/laws.png)

![jet and radius](results/jet_and_radius.png)

## Limitations

- Three surfaces, all with constant or near-constant curvature and no boundary. The
  error at a crease, a corner or the edge of a scan is a different problem and is the
  one that usually matters; nothing here measures it.
- Noise is isotropic Gaussian on every coordinate. Depth sensors put most of their error
  along the view ray, which biases the fit in a direction this does not model.
- The sample is uniform over area. On a non-uniform cloud "k nearest" and "fixed radius"
  stop tracing the same curve, which is exactly when the choice between them matters,
  and that case is not covered.
- Normals are evaluated at the clean point positions, so the measurement isolates the
  normal from the positional error. A pipeline that also has to place the point sees
  both.
- Only the second-order jet is tested against the plane fit; higher orders, weighted
  fits and robust (RANSAC-style) variants are not.
- Timings are single-threaded CPU on one machine and the radius estimator runs a Python
  loop, so its absolute times are not comparable with the vectorised ones.

## Tests

`pytest -q` — 11 tests. Two exist specifically to stop the project reporting a number
that means nothing:

- `test_zero_on_the_plane_is_not_the_estimator_always_returning_zero` — exactly 0.000°
  is the correct answer on a plane and is also what a broken estimator would print, so
  the same code path is required to produce ordered, non-zero errors on the sphere and
  a larger one on the tighter-curved torus.
- `test_a_sweep_that_produces_identical_rows_is_broken`.

The rest: analytic normals being unit vectors and matching the SDF gradient on all three
shapes, the angle metric reporting a planted rotation to nine decimal places and being
invariant to a sign flip, the noise-free plane being exactly zero at four values of k, the k⁻¹ law, the
linearity in σ, the clean error being flat in k and falling as `n^−0.5`, the jet fit
beating the plane fit on clean curved data, the jet losing that advantage under noise,
and the exponent fitter recovering a planted power law.
