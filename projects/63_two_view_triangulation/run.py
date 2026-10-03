"""Two-view triangulation, against 3-D points that are known exactly.

The scene is synthetic on purpose: the 3-D points, the two camera matrices and the
projections are all written down, so the only thing added is pixel noise of a known
standard deviation. That makes the 3-D error exact, and it makes two things measurable
that a reprojection residual cannot show.

  * The error is anisotropic. Depth along the viewing direction is recovered far worse
    than position across it, by a factor that depends on the baseline, and quoting a
    single RMS hides that completely. Here the error is resolved into a depth component
    and a lateral one.
  * The reprojection error is not the 3-D error. At a short baseline a triangulated
    point can sit metres away from the truth while reprojecting to within a pixel in
    both images, because the two rays are nearly parallel and the cost surface along
    them is flat. The ratio between the two is measured rather than asserted.

Both are predicted by the geometry: the depth error should be linear in pixel noise and
inversely proportional to the baseline, so the fitted exponents are +1 and -1 and either
the run reproduces them or something is wrong.

Four classical triangulators, all from cv2 or a few lines of linear algebra:

  dlt             cv2.triangulatePoints - the direct linear transform, an algebraic
                  least squares in homogeneous coordinates
  midpoint        the point closest to both viewing rays, in closed form
  optimal         cv2.correctMatches (Hartley and Sturm's optimal correction, the exact
                  minimiser of reprojection error under the epipolar constraint) and
                  then the DLT
  iterative_lm    cv2.solvePnP-free Gauss-Newton refinement of the reprojection error,
                  started from the DLT

No neural network, no training, no GPU.
"""
from __future__ import annotations

import io
import json
import os
import platform
import time

import cv2
import numpy as np
import scipy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SEED = 0

W, H = 1280, 960
F = 1000.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


# ---------------------------------------------------------------- the scene

def rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * (Kx @ Kx)


def scene(n, rng, depth=(4.0, 8.0), spread=1.6):
    """Points in front of the first camera, at known positions."""
    z = rng.uniform(*depth, size=n)
    x = rng.uniform(-spread, spread, size=n) * z / 6.0
    y = rng.uniform(-spread, spread, size=n) * z / 6.0
    return np.stack([x, y, z], axis=1)


def cameras(baseline, verge_deg=None, depth_mid=6.0):
    """Two cameras: the first at the origin, the second translated along +x.

    By default the second camera is rotated to look at the middle of the scene (a
    verged rig), which is what a real stereo head does; `verge_deg` overrides it.
    """
    R1, t1 = np.eye(3), np.zeros(3)
    if verge_deg is None:
        verge_deg = np.degrees(np.arctan2(baseline, depth_mid))
    R2 = rot([0, 1, 0], -verge_deg)
    C2 = np.array([baseline, 0.0, 0.0])
    t2 = -R2 @ C2
    P1 = K @ np.hstack([R1, t1[:, None]])
    P2 = K @ np.hstack([R2, t2[:, None]])
    return (R1, t1, P1), (R2, t2, P2), C2


def project(P, X):
    h = (P @ np.hstack([X, np.ones((len(X), 1))]).T).T
    return h[:, :2] / h[:, 2:3]


# ---------------------------------------------------------------- triangulators

def tri_dlt(P1, P2, x1, x2, **kw):
    Xh = cv2.triangulatePoints(P1, P2, x1.T.astype(np.float64),
                               x2.T.astype(np.float64))
    return (Xh[:3] / Xh[3]).T


def tri_midpoint(P1, P2, x1, x2, cams=None, **kw):
    """The point minimising the sum of squared distances to the two rays.

    Written out rather than called, because it is the estimator that ignores the image
    plane entirely: it is the one whose failure at a short baseline is purely geometric.
    """
    (R1, t1, _), (R2, t2, _) = cams
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    d1 = np.linalg.inv(K) @ np.hstack([x1, np.ones((len(x1), 1))]).T
    d2 = np.linalg.inv(K) @ np.hstack([x2, np.ones((len(x2), 1))]).T
    d1 = (R1.T @ d1).T
    d2 = (R2.T @ d2).T
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    # closest approach of L1(s) = C1 + s*d1 and L2(t) = C2 + t*d2, with w = C1 - C2
    w = C1 - C2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = np.einsum("ij,j->i", d1, w)
    e = np.einsum("ij,j->i", d2, w)
    den = a * c - b * b
    ok = np.abs(den) > 1e-14
    s = np.where(ok, (b * e - c * d) / np.where(ok, den, 1), 0.0)
    t = np.where(ok, (a * e - b * d) / np.where(ok, den, 1), 0.0)
    p1 = C1 + s[:, None] * d1
    p2 = C2 + t[:, None] * d2
    return 0.5 * (p1 + p2)


def tri_optimal(P1, P2, x1, x2, Fm=None, **kw):
    a, b = cv2.correctMatches(Fm, x1.reshape(1, -1, 2).astype(np.float64),
                              x2.reshape(1, -1, 2).astype(np.float64))
    return tri_dlt(P1, P2, a.reshape(-1, 2), b.reshape(-1, 2))


def tri_iterative(P1, P2, x1, x2, iters=8, **kw):
    """Gauss-Newton on the reprojection error, started from the DLT.

    This is the maximum-likelihood estimate under isotropic Gaussian pixel noise, so if
    anything is going to be the best available it is this one - which makes it the right
    thing to compare the cheap estimators against.
    """
    X = tri_dlt(P1, P2, x1, x2)
    for _ in range(iters):
        Xh = np.hstack([X, np.ones((len(X), 1))])
        J = np.zeros((len(X), 4, 3))
        r = np.zeros((len(X), 4))
        for k, (P, x) in enumerate(((P1, x1), (P2, x2))):
            p = (P @ Xh.T).T
            u, v, w = p[:, 0], p[:, 1], p[:, 2]
            r[:, 2 * k] = u / w - x[:, 0]
            r[:, 2 * k + 1] = v / w - x[:, 1]
            for j in range(3):
                J[:, 2 * k, j] = (P[0, j] * w - u * P[2, j]) / w ** 2
                J[:, 2 * k + 1, j] = (P[1, j] * w - v * P[2, j]) / w ** 2
        JtJ = np.einsum("nij,nik->njk", J, J)
        Jtr = np.einsum("nij,ni->nj", J, r)
        JtJ += 1e-12 * np.eye(3)
        X = X - np.linalg.solve(JtJ, Jtr[..., None])[..., 0]
    return X


TRIANGULATORS = {"dlt": tri_dlt, "midpoint": tri_midpoint,
                 "optimal": tri_optimal, "iterative_lm": tri_iterative}


# ---------------------------------------------------------------- measurement

def fundamental(cam1, cam2):
    (R1, t1, _), (R2, t2, _) = cam1, cam2
    R = R2 @ R1.T
    t = t2 - R @ t1
    Tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = Tx @ R
    Ki = np.linalg.inv(K)
    return Ki.T @ E @ Ki


def decompose_error(Xe, Xt, view_dir):
    """Split the 3-D error into a component along the view ray and one across it."""
    d = Xe - Xt
    u = view_dir / np.linalg.norm(view_dir, axis=1, keepdims=True)
    along = np.einsum("ij,ij->i", d, u)
    lateral = np.linalg.norm(d - along[:, None] * u, axis=1)
    return np.abs(along), lateral


def run_case(n_pts, noise_px, baseline, method, seed=SEED, verge_deg=None):
    rng = np.random.default_rng(seed)
    X = scene(n_pts, rng)
    cam1, cam2, C2 = cameras(baseline, verge_deg=verge_deg)
    P1, P2 = cam1[2], cam2[2]
    x1, x2 = project(P1, X), project(P2, X)
    if noise_px > 0:
        x1 = x1 + rng.normal(scale=noise_px, size=x1.shape)
        x2 = x2 + rng.normal(scale=noise_px, size=x2.shape)
    Fm = fundamental(cam1, cam2)
    t0 = time.perf_counter()
    Xe = TRIANGULATORS[method](P1, P2, x1, x2, cams=(cam1, cam2), Fm=Fm)
    ms = (time.perf_counter() - t0) * 1000
    err = np.linalg.norm(Xe - X, axis=1)
    along, lateral = decompose_error(Xe, X, X)       # the ray from camera 1 is X itself
    # reprojection error of the ESTIMATE against the noisy measurements
    rp = 0.5 * (np.linalg.norm(project(P1, Xe) - x1, axis=1).mean() +
                np.linalg.norm(project(P2, Xe) - x2, axis=1).mean())
    mean_depth = float(X[:, 2].mean())
    # A triangulated point can land behind the camera or at an absurd depth when the
    # two rays are nearly parallel. Those are the cases a mean hides and a p95 shows.
    behind = float(np.mean(Xe[:, 2] <= 0))
    gross = float(np.mean(err > 0.5 * mean_depth))
    return {
        "frac_behind_camera": behind, "frac_gross_error": gross,
        "method": method, "n_pts": int(n_pts), "noise_px": noise_px,
        "baseline": baseline,
        "verge_deg": float(np.degrees(np.arctan2(baseline, 6.0))
                           if verge_deg is None else verge_deg),
        "baseline_over_depth": baseline / mean_depth,
        "err_mean": float(err.mean()), "err_median": float(np.median(err)),
        "err_p95": float(np.percentile(err, 95)),
        "err_depth_mean": float(along.mean()), "err_lateral_mean": float(lateral.mean()),
        "anisotropy": float(along.mean() / max(lateral.mean(), 1e-12)),
        "reproj_px": float(rp),
        "err_per_reproj_px": float(err.mean() / max(rp, 1e-12)),
        "mean_depth": mean_depth, "ms": round(ms, 3),
    }


def fit_exponent(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def main():
    N = 2000
    NOISES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    BASES = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []

    def log(r):
        rows.append(r)
        print(f"  {r['method']:13} noise={r['noise_px']:<5} base={r['baseline']:<5} "
              f"err {r['err_mean']:9.5f}  depth {r['err_depth_mean']:9.5f}  "
              f"lat {r['err_lateral_mean']:8.5f}  aniso {r['anisotropy']:7.2f}  "
              f"reproj {r['reproj_px']:6.3f} px  {r['ms']:7.2f} ms")

    print("1) pixel-noise sweep at a 0.5 m baseline")
    for m in TRIANGULATORS:
        for ns in NOISES:
            r = run_case(N, ns, 0.5, m)
            r["experiment"] = "noise"
            log(r)

    print("\n2) baseline sweep at 1 px noise")
    for m in TRIANGULATORS:
        for b in BASES:
            r = run_case(N, 1.0, b, m)
            r["experiment"] = "baseline"
            log(r)

    print("\n3) the full grid, for the interaction")
    for m in ("dlt", "iterative_lm"):
        for b in (0.05, 0.25, 1.0, 4.0):
            for ns in (0.1, 0.5, 2.0):
                r = run_case(N, ns, b, m)
                r["experiment"] = "grid"
                log(r)

    # ---------------------------------------------------------------- the two laws
    # The two laws hold while the estimate is still well conditioned. Once the error is
    # a sizeable fraction of the depth, a point can be pushed behind the camera and the
    # error saturates instead of growing, so fitting across that boundary would measure
    # the saturation. The fits below are restricted to the well-conditioned rows and the
    # cut is stated rather than hidden.
    expo = {}
    for m in TRIANGULATORS:
        s = sorted([r for r in rows if r["experiment"] == "noise" and r["method"] == m
                    and 0 < r["noise_px"] <= 4.0], key=lambda r: r["noise_px"])
        expo[f"error_vs_noise/{m}"] = fit_exponent([r["noise_px"] for r in s],
                                                   [r["err_median"] for r in s])
        s = sorted([r for r in rows if r["experiment"] == "baseline"
                    and r["method"] == m and r["baseline"] >= 0.1],
                   key=lambda r: r["baseline"])
        expo[f"error_vs_baseline/{m}"] = fit_exponent([r["baseline"] for r in s],
                                                      [r["err_median"] for r in s])
        expo[f"depth_error_vs_baseline/{m}"] = fit_exponent(
            [r["baseline"] for r in s], [r["err_depth_mean"] for r in s])
        expo[f"lateral_error_vs_baseline/{m}"] = fit_exponent(
            [r["baseline"] for r in s], [r["err_lateral_mean"] for r in s])

    # the noise-free case must be exact for every method
    exact = {m: max(r["err_mean"] for r in rows if r["experiment"] == "noise"
                    and r["method"] == m and r["noise_px"] == 0.0)
             for m in TRIANGULATORS}

    # reprojection error is not 3-D error: how many scene units per pixel of residual
    per_px = {}
    for b in BASES:
        per_px[str(b)] = {m: next(r for r in rows if r["experiment"] == "baseline"
                                  and r["method"] == m
                                  and r["baseline"] == b)["err_per_reproj_px"]
                          for m in TRIANGULATORS}

    aniso = {}
    for b in BASES:
        aniso[str(b)] = {m: next(r for r in rows if r["experiment"] == "baseline"
                                 and r["method"] == m
                                 and r["baseline"] == b)["anisotropy"]
                         for m in TRIANGULATORS}

    # how much better is the maximum-likelihood estimate than the cheap ones
    gain = {}
    for exp_name, key, vals in (("noise", "noise_px", NOISES),
                                ("baseline", "baseline", BASES)):
        d = {}
        for v in vals:
            base = next(r for r in rows if r["experiment"] == exp_name
                        and r["method"] == "iterative_lm" and r[key] == v)
            d[str(v)] = {m: round(next(r for r in rows if r["experiment"] == exp_name
                                       and r["method"] == m
                                       and r[key] == v)["err_mean"] /
                                  max(base["err_mean"], 1e-15), 3)
                         for m in TRIANGULATORS}
        gain[exp_name] = d

    # Do the four triangulators actually differ? Typical case against tail case.
    spread = {}
    for exp_name, key, vals in (("noise", "noise_px", [v for v in NOISES if v > 0]),
                                ("baseline", "baseline", BASES)):
        d = {}
        for v in vals:
            sel = [r for r in rows if r["experiment"] == exp_name and r[key] == v]
            med = [r["err_median"] for r in sel]
            p95 = [r["err_p95"] for r in sel]
            d[str(v)] = {"median_max_over_min": round(max(med) / min(med), 3),
                         "p95_max_over_min": round(max(p95) / min(p95), 3),
                         "best_median": min(sel, key=lambda r: r["err_median"])["method"],
                         "best_p95": min(sel, key=lambda r: r["err_p95"])["method"]}
        spread[exp_name] = d

    failures = {}
    for b in BASES:
        failures[str(b)] = {m: next(r for r in rows if r["experiment"] == "baseline"
                                    and r["method"] == m
                                    and r["baseline"] == b)["frac_gross_error"]
                            for m in TRIANGULATORS}

    res = {
        "project": "63_two_view_triangulation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "opencv": cv2.__version__},
        "seed": SEED, "n_points": N,
        "camera": {"width": W, "height": H, "focal_px": F,
                   "principal_point": [W / 2, H / 2]},
        "scene": {"depth_range": [4.0, 8.0], "mean_depth_nominal": 6.0},
        "ground_truth": "the 3-D points and both camera matrices are written down; "
                        "only the pixel measurements are perturbed, so the 3-D error "
                        "is exact",
        "methods": list(TRIANGULATORS),
        "noise_values_px": NOISES, "baselines": BASES,
        "exponents": expo,
        "noise_free_max_error": exact,
        "scene_units_of_error_per_pixel_of_reprojection": per_px,
        "depth_over_lateral_error": aniso,
        "error_relative_to_iterative_lm": gain,
        "spread_across_methods": spread,
        "gross_error_fraction_by_baseline": failures,
        "exponent_fit_window": {"noise_px": "0 < sigma <= 4", "baseline": ">= 0.1",
                                "statistic": "median error"},
        "rows": rows,
    }
    json.dump(res, io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                           newline="\n"), indent=1)

    print("\nfitted exponents (predicted +1 against noise, -1 against baseline):")
    for k, v in expo.items():
        print(f"  {k:36} {v:+.3f}" if v is not None else f"  {k:36} none")
    print("\nnoise-free error (must be zero to machine precision):")
    for m, v in exact.items():
        print(f"  {m:13} {v:.3e}")
    print("\ndepth error / lateral error, by baseline:")
    for b, d in aniso.items():
        print(f"  b={b:<5} " + "  ".join(f"{m}={v:.1f}" for m, v in d.items()))
    return res


if __name__ == "__main__":
    main()
