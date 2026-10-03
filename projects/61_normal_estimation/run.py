"""Surface normals by PCA over k neighbours, against normals that are known in closed form.

The normal of a point cloud sample is usually taken as the eigenvector of the smallest
eigenvalue of the covariance of its k nearest neighbours, with k picked by habit. Here
the points are sampled from analytic surfaces, so the true normal at every point is the
normalised gradient of the signed distance function and the error is an angle in degrees
with no reference method in it.

That turns the usual hand-waving into three power laws whose exponents are set by the
geometry rather than asserted, so each one is a prediction the run either reproduces or
contradicts:

  * On a PLANE with no noise the error must be exactly zero. There is no curvature to
    bias the fit and no noise to perturb it, so anything above floating-point dust means
    the estimator or the ground truth is wrong.
  * On a plane WITH noise, the error is the tilt of a least-squares plane through k
    points scattered by sigma over a patch of radius r. That tilt has standard deviation
    proportional to sigma / (sqrt(k) * r); at a fixed sample density the k-nearest
    patch grows as r ~ sqrt(k), so the two factors compound and the error should fall
    as k^-1 - not the k^-0.5 that "averaging k samples" suggests. It should also be
    linear in sigma.
  * On a CURVED surface with no noise the error does not fall with k at all. The patch
    centroid is off-centre by about r/sqrt(k), which tilts the fit by roughly
    r / (sqrt(k) * R); with r ~ sqrt(k/n) the k cancels and only the density is left,
    so the clean error should sit on a floor proportional to 1/(R * sqrt(n)) and fall
    as n^-0.5 when points are added.

The three together say where the optimal k is and how it moves with noise. A quadratic
(jet) fit is included as the classical cure for the curvature term, so the cost of
removing it is measured too.

No neural network, no training, no GPU.
"""
from __future__ import annotations

import io
import json
import os
import platform
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SEED = 0


# ---------------------------------------------------------------- analytic surfaces

class Plane:
    """Zero curvature: the noise-free error must vanish, which is a test in itself."""
    name = "plane"
    curvature = 0.0

    def __init__(self, side=2.0):
        self.side = side

    def sample(self, n, rng):
        xy = rng.uniform(-self.side / 2, self.side / 2, size=(n, 2))
        return np.concatenate([xy, np.zeros((n, 1))], axis=1)

    def normal(self, p):
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(p), 1))

    @property
    def area(self):
        return self.side ** 2


class Sphere:
    name = "sphere"

    def __init__(self, r=1.0):
        self.r = r
        self.curvature = 1.0 / r

    def sample(self, n, rng):
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True) * self.r

    def normal(self, p):
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    @property
    def area(self):
        return 4 * np.pi * self.r ** 2


class Torus:
    name = "torus"

    def __init__(self, R=1.0, a=0.35):
        self.R, self.a = R, a
        self.curvature = 1.0 / a           # the tube's principal curvature

    def sample(self, n, rng):
        out = []
        while sum(len(o) for o in out) < n:
            m = int((n + 64) * 1.6)
            u = rng.uniform(0, 2 * np.pi, m)
            v = rng.uniform(0, 2 * np.pi, m)
            acc = rng.uniform(0, 1, m) < (self.R + self.a * np.cos(v)) / (self.R + self.a)
            u, v = u[acc], v[acc]
            out.append(np.stack([(self.R + self.a * np.cos(v)) * np.cos(u),
                                 (self.R + self.a * np.cos(v)) * np.sin(u),
                                 self.a * np.sin(v)], axis=1))
        return np.concatenate(out)[:n]

    def normal(self, p):
        xy = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        c = np.stack([self.R * p[:, 0] / xy, self.R * p[:, 1] / xy,
                      np.zeros(len(p))], axis=1)      # nearest point on the centre circle
        g = p - c
        return g / np.linalg.norm(g, axis=1, keepdims=True)

    @property
    def area(self):
        return 4 * np.pi ** 2 * self.R * self.a


SHAPES = {s.name: s for s in (Plane(2.0), Sphere(1.0), Torus(1.0, 0.35))}


# ---------------------------------------------------------------- the estimators

def pca_normals(pts, query, k):
    """Plane fit: the eigenvector of the smallest eigenvalue of the local covariance."""
    _, idx = cKDTree(pts).query(query, k=k)
    nb = pts[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1]
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0], w


def pca_normals_radius(pts, query, radius, min_k=4):
    """The same fit over a fixed-radius ball instead of a fixed count.

    On a non-uniform sample the two are different estimators, which is the point of
    including it: k neighbours is a patch whose SIZE changes with density.
    """
    tree = cKDTree(pts)
    out = np.zeros((len(query), 3))
    used = np.zeros(len(query), int)
    for i, q in enumerate(query):
        j = tree.query_ball_point(q, radius)
        if len(j) < min_k:
            _, j = tree.query(q, k=min_k)
            j = np.atleast_1d(j)
        nb = pts[np.asarray(j)]
        used[i] = len(nb)
        nb = nb - nb.mean(axis=0)
        w, v = np.linalg.eigh(nb.T @ nb / len(nb))
        out[i] = v[:, 0]
    return out, used


def jet_normals(pts, query, k):
    """Quadratic (second-order jet) fit, the classical cure for the curvature bias.

    A plane fit through a curved patch tilts; fitting z = a x^2 + b xy + c y^2 + d x +
    e y + f in the PCA frame and reading the normal at the centre removes the leading
    term of that tilt. It costs a 6-parameter least squares per point.
    """
    n0, _ = pca_normals(pts, query, k)
    _, idx = cKDTree(pts).query(query, k=k)
    nb = pts[idx] - query[:, None, :]
    out = np.zeros_like(n0)
    for i in range(len(query)):
        n = n0[i]
        # an orthonormal frame with n as the third axis
        t = np.array([1.0, 0.0, 0.0])
        if abs(n[0]) > 0.9:
            t = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, t)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        loc = nb[i] @ np.stack([u, v, n], axis=1)
        x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
        A = np.stack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)], axis=1)
        try:
            c, *_ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            out[i] = n
            continue
        # gradient of the fitted height field at the centre is (d, e)
        g = np.array([-c[3], -c[4], 1.0])
        g /= np.linalg.norm(g)
        out[i] = np.stack([u, v, n], axis=1) @ g
    return out


ESTIMATORS = {"pca_knn": "k nearest neighbours",
              "pca_radius": "fixed-radius ball",
              "jet_knn": "quadratic jet, k nearest"}


# ---------------------------------------------------------------- measurement

def angle_error_deg(est, true):
    """Unsigned angle between the estimated and the true normal, in degrees.

    PCA returns an unoriented direction, so the sign is meaningless and |cos| is the
    honest comparison. Using the signed cosine would report ~180 degrees for half the
    points on a correct estimate.
    """
    c = np.abs(np.einsum("ij,ij->i", est, true))
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


def run_case(shape, n_pts, k, noise=0.0, estimator="pca_knn", seed=SEED, radius=None,
             n_query=4000):
    rng = np.random.default_rng(seed)
    clean = shape.sample(n_pts, rng)
    pts = clean + rng.normal(scale=noise, size=clean.shape) if noise > 0 else clean
    # The query points are the clean positions: the question is the normal OF THE
    # SURFACE at a known place, not of wherever the noise moved the sample to.
    # A fixed random subset is queried so that the largest k stays affordable; the
    # neighbourhood still comes from all n_pts points.
    q = clean if n_query >= n_pts else clean[
        np.random.default_rng(seed + 7).choice(n_pts, n_query, replace=False)]
    t0 = time.perf_counter()
    if estimator == "pca_knn":
        est, _ = pca_normals(pts, q, k)
        used = float(k)
    elif estimator == "pca_radius":
        est, u = pca_normals_radius(pts, q, radius)
        used = float(np.mean(u))
    else:
        est = jet_normals(pts, q, k)
        used = float(k)
    ms = (time.perf_counter() - t0) * 1000
    e = angle_error_deg(est, shape.normal(q))
    return {"shape": shape.name, "n_pts": int(n_pts), "k": int(k), "noise": noise,
            "n_query": int(len(q)),
            "estimator": estimator, "radius": radius,
            "mean_deg": float(e.mean()), "median_deg": float(np.median(e)),
            "p95_deg": float(np.percentile(e, 95)), "max_deg": float(e.max()),
            "frac_over_10deg": float(np.mean(e > 10.0)),
            "neighbours_used": used, "ms": round(ms, 1),
            "patch_radius": float(np.sqrt(k * shape.area / (np.pi * n_pts)))}


def fit_exponent(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def main():
    N = 20000
    KS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048]
    NOISES = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    rows = []

    def log(r):
        rows.append(r)
        print(f"  {r['shape']:6} {r['estimator']:10} k={r['k']:<4} "
              f"noise={r['noise']:<7} mean {r['mean_deg']:7.3f}deg  "
              f"median {r['median_deg']:7.3f}  p95 {r['p95_deg']:7.3f}  {r['ms']:7.1f} ms")

    print("1) k sweep x noise sweep, PCA over k nearest neighbours")
    for shape in SHAPES.values():
        for noise in NOISES:
            for k in KS:
                r = run_case(shape, N, k, noise=noise)
                r["experiment"] = "k_noise"
                log(r)

    print("\n2) the quadratic jet fit, same settings on the curved shapes")
    for shape in (SHAPES["sphere"], SHAPES["torus"]):
        for noise in (0.0, 0.001, 0.005, 0.02):
            for k in (8, 16, 32, 64, 128, 256, 512, 1024):
                r = run_case(shape, N, k, noise=noise, estimator="jet_knn")
                r["experiment"] = "jet"
                log(r)

    print("\n3) fixed-radius ball instead of a fixed count")
    for shape in (SHAPES["sphere"], SHAPES["torus"]):
        for noise in (0.0, 0.005):
            for radius in (0.02, 0.03, 0.05, 0.08, 0.12, 0.2):
                r = run_case(shape, N, 0, noise=noise, estimator="pca_radius",
                             radius=radius)
                r["experiment"] = "radius"
                log(r)

    print("\n4) density sweep at a fixed k, to separate k from patch size")
    for shape in (SHAPES["sphere"], SHAPES["torus"]):
        for n in (2500, 5000, 10000, 20000, 40000):
            r = run_case(shape, n, 32, noise=0.0)
            r["experiment"] = "density"
            log(r)

    # ---------------------------------------------------------------- fitted exponents
    def sel(**kw):
        return sorted([r for r in rows if all(r.get(a) == b for a, b in kw.items())],
                      key=lambda r: r["k"])

    expo = {}
    # prediction 2: on a plane, error ~ sigma * k^-1
    for ns in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
        rs = [r for r in sel(experiment="k_noise", shape="plane", noise=ns)
              if r["k"] >= 8]
        expo[f"noise_vs_k/plane/sigma={ns}"] = fit_exponent(
            [r["k"] for r in rs], [r["mean_deg"] for r in rs])
    # and linear in sigma at fixed k
    for k in (16, 64, 256):
        rs = [r for r in rows if r["experiment"] == "k_noise" and r["shape"] == "plane"
              and r["k"] == k and r["noise"] > 0]
        rs.sort(key=lambda r: r["noise"])
        expo[f"noise_vs_sigma/plane/k={k}"] = fit_exponent(
            [r["noise"] for r in rs], [r["mean_deg"] for r in rs])
    # the same k-law on the curved shapes once noise dominates the curvature floor
    for s in ("sphere", "torus"):
        rs = [r for r in sel(experiment="k_noise", shape=s, noise=0.02) if r["k"] >= 8]
        expo[f"noise_vs_k/{s}/sigma=0.02"] = fit_exponent(
            [r["k"] for r in rs], [r["mean_deg"] for r in rs])

    # prediction 1: the noise-free plane must be exactly zero, not merely small
    flat = sel(experiment="k_noise", shape="plane", noise=0.0)
    plane_zero = float(max(r["max_deg"] for r in flat))

    # prediction 3: the clean curved error is set by density, not by k
    for s in ("sphere", "torus"):
        rs = sorted([r for r in rows if r["experiment"] == "density" and r["shape"] == s],
                    key=lambda r: r["n_pts"])
        expo[f"clean_vs_density/{s}"] = fit_exponent([r["n_pts"] for r in rs],
                                                     [r["mean_deg"] for r in rs])
        mid = [r for r in sel(experiment="k_noise", shape=s, noise=0.0)
               if 32 <= r["k"] <= 256]
        expo[f"clean_vs_k_32_to_256/{s}"] = fit_exponent(
            [r["k"] for r in mid], [r["mean_deg"] for r in mid])
        big = [r for r in sel(experiment="k_noise", shape=s, noise=0.0) if r["k"] >= 512]
        expo[f"clean_vs_k_512_and_up/{s}"] = fit_exponent(
            [r["k"] for r in big], [r["mean_deg"] for r in big])

    # ---------------------------------------------------------------- optimal k
    best_k = {}
    for s in SHAPES:
        d = {}
        for ns in NOISES:
            rs = sel(experiment="k_noise", shape=s, noise=ns)
            b = min(rs, key=lambda r: r["mean_deg"])
            d[str(ns)] = {"k": b["k"], "mean_deg": b["mean_deg"],
                          "patch_radius": b["patch_radius"]}
        best_k[s] = d

    # ---------------------------------------------------------------- jet vs plane
    jet_gain = {}
    for s in ("sphere", "torus"):
        d = {}
        for ns in (0.0, 0.001, 0.005, 0.02):
            for k in (8, 16, 32, 64, 128, 256, 512, 1024):
                a = next(r for r in rows if r["experiment"] == "k_noise"
                         and r["shape"] == s and r["noise"] == ns and r["k"] == k)
                b = next(r for r in rows if r["experiment"] == "jet"
                         and r["shape"] == s and r["noise"] == ns and r["k"] == k)
                d[f"noise={ns},k={k}"] = {
                    "pca_deg": a["mean_deg"], "jet_deg": b["mean_deg"],
                    "ratio": round(a["mean_deg"] / b["mean_deg"], 2),
                    "time_ratio": round(b["ms"] / max(a["ms"], 1e-9), 1)}
        jet_gain[s] = d
    jet_best = {}
    for s in ("sphere", "torus"):
        for ns in (0.0, 0.001, 0.005, 0.02):
            rs = [r for r in rows if r["experiment"] == "jet" and r["shape"] == s
                  and r["noise"] == ns]
            pcas = [r for r in rows if r["experiment"] == "k_noise" and r["shape"] == s
                    and r["noise"] == ns and r["k"] in (8, 16, 32, 64, 128, 256, 512, 1024)]
            bj, bp = min(rs, key=lambda r: r["mean_deg"]), min(pcas, key=lambda r: r["mean_deg"])
            jet_best[f"{s}/noise={ns}"] = {
                "jet_best_k": bj["k"], "jet_best_deg": bj["mean_deg"],
                "pca_best_k": bp["k"], "pca_best_deg": bp["mean_deg"],
                "ratio": round(bp["mean_deg"] / bj["mean_deg"], 2)}

    res = {
        "project": "61_normal_estimation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "seed": SEED, "n_points": N, "k_values": KS, "noise_values": NOISES,
        "ground_truth": "analytic surface normals in closed form; error is the unsigned "
                        "angle in degrees between the estimated and the true normal",
        "shapes": {k: {"area": v.area, "curvature": v.curvature}
                   for k, v in SHAPES.items()},
        "estimators": ESTIMATORS,
        "exponents": expo,
        "noise_free_plane_max_error_deg": plane_zero,
        "best_k": best_k,
        "best_k_growth_torus": {
            "k_at_sigma_0": best_k["torus"]["0.0"]["k"],
            "k_at_sigma_0.02": best_k["torus"]["0.02"]["k"],
            "factor": round(best_k["torus"]["0.02"]["k"] /
                            best_k["torus"]["0.0"]["k"], 2)},
        "jet_vs_pca": jet_gain,
        "jet_vs_pca_best": jet_best,
        "rows": rows,
    }
    json.dump(res, io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                           newline="\n"), indent=1)

    print("\nfitted exponents (predicted: -1 vs k under noise, +1 vs sigma, "
          "-0.5 vs density when clean, 0 vs k when clean):")
    for k2, v in expo.items():
        print(f"  {k2:28} {v:+.3f}" if v is not None else f"  {k2:28} none")
    print(f"\nnoise-free plane, worst mean error over all k: {plane_zero:.3e} deg")
    print("\nbest k by noise level:")
    for s, d in best_k.items():
        print(f"  {s:7} " + "  ".join(f"s={ns}:k={v['k']}" for ns, v in d.items()))
    return res


if __name__ == "__main__":
    main()
