# 63 — Two-view triangulation, against 3-D points that are written down

Two cameras, 2000 points, four classical triangulators. The 3-D points and both camera
matrices are specified, so the only thing added is pixel noise of a known standard
deviation and **the 3-D error is exact** — not a residual, which only says the
optimiser stopped.

| | |
|---|---|
| cameras | 1280 × 960, focal length 1000 px, second camera verged on the scene centre |
| scene | 2000 points at depths 4.0 to 8.0, mean depth 6 |
| ground truth | the 3-D points themselves; only the pixel measurements are perturbed |
| methods | `cv2.triangulatePoints` (DLT) · midpoint of rays · `cv2.correctMatches` + DLT · Gauss-Newton on reprojection error (the ML estimate) |
| swept | pixel noise σ = 0 to 10 px, baseline 0.02 to 4.0 scene units |

With no noise all four recover the points to between 4.486245164479558e-15 and
1.1187609459260126e-13 scene units, which is the arithmetic and nothing else.

## Two predictions from the geometry, two fitted exponents

| law | predicted | measured (median error) |
|---|---|---|
| error against pixel noise, at a 0.5 baseline | +1 | +0.997, +1.000, +0.997, +0.997 |
| error against baseline, at 1 px noise | −1 | −1.006, −1.010, −1.012, −1.012 |

Both hold to within 1.2%. The baseline fit uses baselines of 0.1 and above: below that
the error is a sizeable fraction of the scene depth, points start landing behind the
camera and the error saturates instead of growing, so a fit across that boundary would
be measuring the saturation.

## Finding 1 — a longer baseline buys depth accuracy and essentially no lateral accuracy

Resolving the 3-D error into a component along the viewing ray and one across it:

| | exponent against baseline |
|---|---|
| error along the view ray (depth) | **−1.071** |
| error across it (lateral) | **−0.021** |

The lateral error is flat. Doubling the baseline halves the depth error and changes the
lateral error by about 1%, because sideways position is already fixed by where the point
lands in the first image and a second view adds almost nothing to it.

That makes the error strongly anisotropic, and the anisotropy is itself a function of
the baseline:

| baseline | 0.02 | 0.1 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| depth error ÷ lateral error | **512.9** | 65.3 | 12.9 | 6.3 | 3.0 | 1.4 |

At a 2 cm baseline on a 6-unit scene, depth is **513 times** worse than lateral
position. A single RMS number averages those together and reports neither.

## Finding 2 — the reprojection error says almost nothing about the 3-D error

Scene units of 3-D error per pixel of reprojection residual, at 1 px noise:

| baseline | 0.02 | 0.05 | 0.1 | 0.5 | 1.0 | 4.0 |
|---|---|---|---|---|---|---|
| error per pixel of residual | **13.1761** | 1.5941 | 0.7542 | 0.1485 | 0.0736 | **0.0189** |

The same optimiser, the same noise, and residuals of 0.5655 px and 0.5267 px
respectively — but a point at the short baseline is **1.49193 units** out while one at
the long baseline is **0.00789 units** out, a ratio of 697 in error per pixel of
residual. When the two rays are nearly parallel the reprojection cost is flat along
them, so a point can slide a long way down the ray while both images still agree to
within half a pixel.

At the 0.02 baseline, **22.8%** of points come out more than half the scene depth away
from the truth while the mean residual is still under 0.6 px. A reprojection threshold
cannot detect that; a baseline check can.

## Finding 3 — which triangulator you use barely matters, except in the tail

Across all sixteen sweep settings, the spread between the best and worst of the four
methods is:

| | largest spread (max ÷ min) |
|---|---|
| median error, baseline sweep | **1.056** |
| median error, noise sweep | 1.037 |
| p95 error, baseline sweep | **2.363** |

Typically the four agree to within 5.6%, which is the honest answer to "which
triangulator should I use": the choice is not where the error comes from. The one place
they separate is the tail at a short baseline:

| baseline 0.02 | median | p95 |
|---|---|---|
| DLT | 1.49509 | 14.25351 |
| midpoint of rays | 1.5749 | **6.04447** |
| optimal correction + DLT | 1.49193 | 14.28177 |
| Gauss-Newton (ML) | 1.49193 | 14.28177 |

The maximum-likelihood estimate is the best in the median, by 5.3%, and is **2.4 times
worse at the 95th percentile**. The midpoint of two rays is bounded by construction —
it cannot place a point outside the segment joining the closest approaches — so when the
rays are nearly parallel it degrades instead of diverging. The ML estimate has no such
bound and slides down the flat direction of the cost surface. It also fails outright
less often: 20.25% of points off by more than half the scene depth against 22.8%.

## Input / Output

Input: a synthetic scene and two camera matrices. No data files.
Output: `results/results.json` — one row per method and setting, the fitted exponents,
the anisotropy and reprojection ratios, the spread across methods and the gross-failure
fractions — and three figures.

```
python run.py                                      # all sweeps, writes results.json
python figures.py                                  # three figures
pytest -q                                          # 11 tests
python ../../tools/check_readme_numbers.py .       # every README number vs results.json
```

## Figures

![laws](results/laws.png)

![anisotropy](results/anisotropy.png)

![reprojection](results/reprojection.png)

## Limitations

- The calibration is exact. Real two-view geometry is limited by errors in `K`, `R` and
  `t` far more often than by pixel noise, and nothing here measures that; the exponents
  above are the best case.
- Noise is independent isotropic Gaussian on each image coordinate. Real feature
  detectors have correlated, structure-dependent localisation error, and no outliers are
  injected at all — which is precisely the regime where the choice of triangulator, with
  a robust loss, would start to matter.
- One scene geometry: points in a shallow slab in front of a verged rig, with every
  point visible in both views. Wide-angle, forward-motion and grazing configurations
  behave differently.
- `cv2.correctMatches` implements Hartley and Sturm's optimal correction for the
  two-view case only. It is not the same thing as a bundle adjustment and should not be
  read as one.
- The Gauss-Newton refinement runs a fixed eight iterations with no line search, which
  is ample here because it starts from the DLT, but it is not a hardened solver.
- Timing is recorded per row but is not compared: all four are dominated by the same
  per-point linear algebra and the differences are not the interesting axis.

## Tests

`pytest -q` — 11 tests. Two guard against the experiment measuring nothing:

- `test_a_deliberately_wrong_triangulation_is_detected`. The four methods agreeing to
  within a few per cent is the finding — and is also exactly what one method called four
  times would look like. A triangulation displaced by a planted amount must be reported
  as that amount, to six significant figures.
- `test_the_two_cameras_see_the_scene_from_different_places` requires a mean disparity
  above 2 px at every baseline, so the two views are never effectively the same camera.

The rest: projection round-tripping through `K`, `R` and `t` by hand with every point in
front of both cameras, exact recovery without noise, the two power laws, the baseline
buying depth and not lateral accuracy, the reprojection residual failing as a proxy, the
midpoint's lighter tail at a short baseline, the error decomposition satisfying
`depth² + lateral² = total²`, and the exponent fitter recovering a planted power law.
