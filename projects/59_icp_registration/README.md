# 59 — ICP registration, against a transform that is known exactly

Iterative Closest Point aligns two point clouds. Shown as a before/after picture it
always looks like it worked. Here a rigid transform is applied **on purpose**, so the
answer is known to machine precision and the error is a rotation in degrees and a
translation in scene units — not a residual, which only says the algorithm stopped.

Point-to-point and point-to-plane are both implemented from the SVD and least-squares
primitives, because the difference between them is the thing being measured.

| | |
|---|---|
| shape | three fused blobs, 2,000 points — asymmetric, so rotation is unambiguous |
| ground truth | a known rigid transform; error is the geodesic angle to it |
| varied | initial misalignment, noise, overlap, trimming |
| success | recovered rotation within 1° |

A sphere would have been the obvious test shape and is the wrong one: it is rotationally
symmetric, so ICP can "succeed" at any rotation and the rotation error would mean nothing.

## Finding 1 — point-to-plane tolerates twice the misalignment

| initial misalignment | point-to-point | point-to-plane |
|---|---|---|
| 20° | 0.000° | 0.000° |
| 30° | 0.000° | 0.000° |
| 45° | 10.967° ✗ | 0.000° |
| 60° | 40.145° ✗ | 0.000° |
| 90° | 91.642° ✗ | 92.856° ✗ |

The convergence basin is **30° for point-to-point and 60° for point-to-plane** — exactly
double. Past its basin ICP does not fail loudly; it converges confidently to the wrong
alignment, which is why a residual cannot be used to detect the failure.

## Finding 2 — and pays for it under noise

| noise σ | point-to-point | point-to-plane |
|---|---|---|
| 0.002 | 0.043° | 0.039° |
| 0.01 | 0.212° | 0.311° |
| 0.02 | 0.409° | **1.642°** ✗ |
| 0.05 | 0.972° | **5.258°** ✗ |

At σ = 0.05 point-to-plane is **5.4× worse**. Its advantage comes from using surface
normals, and normals are estimated from the same noisy neighbourhood — so the thing that
widens the basin is also what breaks first. Point-to-plane is the right default when the
initial guess is poor and the data is clean; point-to-point when it is the other way
round.

## Finding 3 — trimming buys 15 points of overlap

Keeping only the closest 80% of correspondences each iteration:

| overlap | all pairs | closest 80% kept |
|---|---|---|
| 100% | 0.000° | 0.000° |
| 90% | 3.780° ✗ | **0.000°** |
| 75% | 18.797° ✗ | **0.451°** |
| 60% | 42.046° ✗ | 45.738° ✗ |
| 50% | 31.089° ✗ | 39.484° ✗ |

Untrimmed ICP breaks as soon as the clouds stop matching exactly — at 90% overlap it is
already 3.780° out. Trimming holds to 75%. Below 60% both fail, and no amount of trimming
rescues it, because the shared geometry no longer determines the transform.

## Input / Output

Input: synthetic point clouds and a known transform. No data files.
Output: `results/results.json` (29 rows plus the fitted basin limits), one figure.

```
python run.py        # 29 registrations, writes results.json
python figures.py    # the three-panel figure
pytest -q            # 8 tests
```

![icp](results/icp.png)

## Limitations

- One shape. The convergence basin depends on how distinctive the geometry is; a
  flatter or more symmetric object would have a narrower basin than the number here.
- Noise is isotropic Gaussian on every point. Real depth sensors have range-dependent,
  structured error, which ICP responds to differently.
- Overlap is cropped along a single axis. A real second viewpoint also changes which
  surfaces are visible, which this does not model.
- No point-to-plane result for the overlap sweep — the trimming comparison is
  point-to-point only, so "trimming buys 15 points" is established for that variant.
- Timing is recorded but not compared; both variants are dominated by the same
  nearest-neighbour query.

## A bug worth recording

The overlap experiment reported **0.000° at every setting** in two earlier versions.
First it removed random points, which is decimation — a thinner cloud still covers the
whole shape, so every point keeps a correct counterpart. Then it cropped only the source,
making it a strict subset of the destination, which is the easy case rather than partial
overlap. Both looked like a clean result. A test now asserts that a real crop leaves more
than 20% of source points with no counterpart at all.

## Tests

`pytest -q` — 8 tests: rotation matrices orthonormal, the angle metric recovering a
planted angle, exact recovery from a small offset, each of the three findings, the
partial-overlap guard above, and PCA normals agreeing with the analytic normals of a
sphere to 0.97 mean alignment.
