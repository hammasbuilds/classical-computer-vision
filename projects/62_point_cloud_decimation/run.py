"""Point-cloud decimation: voxel grid against farthest-point against random.

Three ways to throw away points, measured on four things that pull in different
directions:

  * accuracy - the distance from the kept points to the TRUE surface, exact from the
    analytic signed distance function, which catches a method that moves points off the
    surface rather than merely removing some;
  * coverage - the covering radius, the largest distance from any original point to its
    nearest kept point. This is the quantity farthest-point sampling greedily minimises
    and it is the one that decides whether a downsampled cloud still describes the
    shape;
  * density bias - a patch of the surface is sampled far more densely than the rest,
    and the patch's share of the AREA is known in closed form, so "did the decimation
    reproduce the shape or the scanner's dwell time" is a ratio against a planted
    answer rather than a judgement about a picture;
  * wall-clock time, because the whole point of decimation is to spend less of it later.

Chamfer distance back to the original cloud is reported too, since that is the number
usually quoted, and it turns out to be the one that hides the difference.

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


# ---------------------------------------------------------------- the surface

class Torus:
    """A torus carrying a deliberately OVER-SAMPLED patch.

    The patch is a box in the (u, v) parameter domain, so its area has a closed form:

        area = a * (2*wu) * (R * 2*wv + a * (sin(v0+wv) - sin(v0-wv)))

    That makes the ground truth for the feature question exact and independent of any
    algorithm. A decimation that normalises density should end up keeping the patch in
    proportion to its AREA; one that merely thins the cloud keeps it in proportion to
    how many points the scanner happened to put there, which is a property of the
    capture and not of the shape.
    """
    name = "torus_with_dense_patch"

    def __init__(self, R=1.0, a=0.35, u0=0.0, v0=0.0, wu=0.18, wv=0.25):
        self.R, self.a = R, a
        self.u0, self.v0, self.wu, self.wv = u0, v0, wu, wv

    def sdf(self, p):
        q = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2) - self.R
        return np.sqrt(q ** 2 + p[..., 2] ** 2) - self.a

    def xyz(self, u, v):
        return np.stack([(self.R + self.a * np.cos(v)) * np.cos(u),
                         (self.R + self.a * np.cos(v)) * np.sin(u),
                         self.a * np.sin(v)], axis=1)

    @property
    def area(self):
        return 4 * np.pi ** 2 * self.R * self.a

    @property
    def patch_area(self):
        """Closed form: the integral of the area element over the parameter box."""
        return self.a * (2 * self.wu) * (self.R * 2 * self.wv + self.a *
                                         (np.sin(self.v0 + self.wv) -
                                          np.sin(self.v0 - self.wv)))

    @property
    def patch_area_share(self):
        return self.patch_area / self.area

    def _sample_box(self, n, rng, u_lo, u_hi, v_lo, v_hi):
        """Uniform by AREA inside a parameter box, by rejection on (R + a cos v)."""
        out = [np.zeros((0, 3))]
        got = 0
        while got < n:
            m = int((n - got + 64) * 2.2)
            u = rng.uniform(u_lo, u_hi, m)
            v = rng.uniform(v_lo, v_hi, m)
            acc = rng.uniform(0, 1, m) < (self.R + self.a * np.cos(v)) /                 (self.R + self.a)
            u, v = u[acc], v[acc]
            out.append(self.xyz(u, v))
            got += len(u)
        return np.concatenate(out)[:n]

    def in_patch(self, p):
        """Membership by the planted parameter box, recovered from the coordinates."""
        u = np.arctan2(p[:, 1], p[:, 0])
        v = np.arctan2(p[:, 2], np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - self.R)
        du = np.abs((u - self.u0 + np.pi) % (2 * np.pi) - np.pi)
        dv = np.abs((v - self.v0 + np.pi) % (2 * np.pi) - np.pi)
        return (du <= self.wu) & (dv <= self.wv)

    def sample(self, n, rng, n_patch=2000):
        """n points uniform over the whole torus plus n_patch extra inside the patch."""
        base = self._sample_box(n, rng, 0, 2 * np.pi, 0, 2 * np.pi)
        extra = self._sample_box(n_patch, rng, self.u0 - self.wu, self.u0 + self.wu,
                                 self.v0 - self.wv, self.v0 + self.wv)
        pts = np.concatenate([base, extra])
        return pts, self.in_patch(pts)


# ---------------------------------------------------------------- the three methods

def random_sample(pts, target, rng, **kw):
    idx = rng.choice(len(pts), size=min(target, len(pts)), replace=False)
    return np.sort(idx)


def voxel_grid(pts, voxel, mode="centroid", **kw):
    """One point per occupied voxel.

    `mode` matters and is swept: "centroid" averages the points in a voxel, which moves
    the kept point OFF the surface wherever the surface is curved; "nearest" returns the
    original point closest to that centroid, which cannot.
    """
    key = np.floor((pts - pts.min(0)) / voxel).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    if mode == "first":
        return np.sort(first), None
    n_cells = inv.max() + 1
    sums = np.zeros((n_cells, 3))
    np.add.at(sums, inv, pts)
    cnt = np.bincount(inv, minlength=n_cells).astype(float)
    cent = sums / cnt[:, None]
    if mode == "centroid":
        return None, cent
    # "nearest": the original point of each cell that is closest to the cell centroid
    d = np.linalg.norm(pts - cent[inv], axis=1)
    best = np.full(n_cells, -1, np.int64)
    order = np.argsort(d, kind="stable")[::-1]
    best[inv[order]] = order
    return np.sort(best), None


def farthest_point(pts, target, seed=0, **kw):
    """Greedy farthest-point sampling: each new point is the one furthest from the set.

    This is the 2-approximation to the k-centre problem, so the covering radius it
    achieves is within a factor of two of the best possible for that many points -
    which is why it is the right baseline for the coverage question.
    """
    n = len(pts)
    target = min(target, n)
    idx = np.empty(target, np.int64)
    idx[0] = seed % n
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for i in range(1, target):
        j = int(np.argmax(d))
        idx[i] = j
        np.minimum(d, np.linalg.norm(pts - pts[j], axis=1), out=d)
    return np.sort(idx)


# ---------------------------------------------------------------- measurement

def measure(pts, in_patch, kept_idx, kept_pts, shape):
    """kept_idx may be None for the voxel centroid variant, which invents new points."""
    if kept_pts is None:
        kept_pts = pts[kept_idx]
    tree = cKDTree(kept_pts)
    d_orig, _ = tree.query(pts)                      # original -> kept
    d_back, _ = cKDTree(pts).query(kept_pts)         # kept -> original
    sdf = np.abs(shape.sdf(kept_pts))
    kept_in_patch = shape.in_patch(kept_pts)
    row = {
        "n_kept": int(len(kept_pts)),
        "ratio": float(len(kept_pts) / len(pts)),
        "surface_dist_mean": float(sdf.mean()),
        "surface_dist_p95": float(np.percentile(sdf, 95)),
        "surface_dist_max": float(sdf.max()),
        "covering_radius": float(d_orig.max()),
        "covering_p95": float(np.percentile(d_orig, 95)),
        "covering_radius_in_patch": float(d_orig[in_patch].max()),
        "covering_radius_outside_patch": float(d_orig[~in_patch].max()),
        "chamfer_to_original": float(0.5 * (d_orig.mean() + d_back.mean())),
        "patch_kept": int(kept_in_patch.sum()),
        "patch_share_kept": float(kept_in_patch.mean()),
        "patch_share_input": float(in_patch.mean()),
        "patch_share_area": float(shape.patch_area_share),
    }
    # 1.0 means the kept cloud represents the patch in proportion to its AREA, which is
    # what an unbiased sample of the surface would do; the input is at 5.4 by design
    row["patch_bias"] = row["patch_share_kept"] / row["patch_share_area"]
    return row


def run_one(pts, in_patch, shape, method, target=None, voxel=None, mode="nearest",
            seed=SEED):
    def once():
        rng = np.random.default_rng(seed)
        if method == "random":
            return random_sample(pts, target, rng), None
        if method == "farthest_point":
            return farthest_point(pts, target, seed=seed), None
        return voxel_grid(pts, voxel, mode=mode)

    # Repeat until at least 50 ms has elapsed. A single random selection takes under a
    # tenth of a millisecond, and timing that once gave a cost exponent of 0.64 against
    # input size - a measurement of the clock, not of the method.
    reps, elapsed = 0, 0.0
    t0 = time.perf_counter()
    while elapsed < 0.05 and reps < 200:
        idx, cent = once()
        reps += 1
        elapsed = time.perf_counter() - t0
    ms = elapsed / reps * 1000
    row = measure(pts, in_patch, idx, cent, shape)
    row.update({"method": method if method != "voxel" else f"voxel_{mode}",
                "family": method, "mode": mode if method == "voxel" else None,
                "voxel": voxel, "target": target, "ms": round(ms, 2),
                "n_input": int(len(pts))})
    return row


def voxel_for_target(pts, target, lo=1e-3, hi=1.0, iters=30):
    """Bisect the voxel size so the grid returns about `target` points.

    Without this the three methods cannot be compared: a voxel grid takes a size, not a
    count, and plotting them against different retained fractions would be comparing
    nothing.
    """
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        idx, _ = voxel_grid(pts, mid, mode="nearest")
        n = len(idx)
        if n > target:
            lo = mid
        else:
            hi = mid
        if abs(n - target) <= max(2, 0.01 * target):
            return mid, n
    return mid, n


def fit_exponent(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def main():
    shape = Torus()
    rng = np.random.default_rng(SEED)
    N = 40000
    pts, in_patch = shape.sample(N, rng, n_patch=2000)
    targets = [20000, 10000, 5000, 2500, 1250, 600, 300]
    rows = []

    def log(r):
        rows.append(r)
        print(f"  {r['method']:16} kept {r['n_kept']:6} ({r['ratio']:6.3%})  "
              f"cover {r['covering_radius']:.5f}  surf {r['surface_dist_mean']:.6f}  "
              f"chamfer {r['chamfer_to_original']:.5f}  "
              f"patch bias {r['patch_bias']:5.2f}  {r['ms']:8.1f} ms")

    print(f"1) matched-count comparison on {len(pts)} points "
          f"({int(in_patch.sum())} of them inside the dense patch, whose "
          f"area share is {shape.patch_area_share:.4%})")
    voxels = {}
    for t in targets:
        v, got = voxel_for_target(pts, t)
        voxels[t] = v
        for method, kw in (("random", {"target": t}),
                           ("farthest_point", {"target": t}),
                           ("voxel", {"voxel": v, "mode": "nearest"}),
                           ("voxel", {"voxel": v, "mode": "centroid"})):
            r = run_one(pts, in_patch, shape, method, **kw)
            r["experiment"] = "matched"
            r["target"] = t
            log(r)

    print("\n2) the voxel size sweep on its own terms")
    for v in (0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12):
        for mode in ("nearest", "centroid", "first"):
            r = run_one(pts, in_patch, shape, "voxel", voxel=v, mode=mode)
            r["experiment"] = "voxel_size"
            log(r)

    print("\n3) how the cost of each method grows with the input size")
    for n in (5000, 10000, 20000, 40000, 80000):
        p2, b2 = shape.sample(n, np.random.default_rng(SEED), n_patch=n // 20)
        t = max(n // 8, 100)
        v, _ = voxel_for_target(p2, t)
        for method, kw in (("random", {"target": t}),
                           ("farthest_point", {"target": t}),
                           ("voxel", {"voxel": v, "mode": "nearest"})):
            r = run_one(p2, b2, shape, method, **kw)
            r["experiment"] = "scaling"
            log(r)

    # ---------------------------------------------------------------- summary numbers
    def sel(exp, method):
        return sorted([r for r in rows if r["experiment"] == exp
                       and r["method"] == method], key=lambda r: r["n_kept"])

    methods = ["random", "farthest_point", "voxel_nearest", "voxel_centroid"]
    expo = {}
    for m in methods:
        s = sel("matched", m)
        expo[m] = {
            "covering_radius_vs_kept": fit_exponent([r["n_kept"] for r in s],
                                                    [r["covering_radius"] for r in s]),
            "chamfer_vs_kept": fit_exponent([r["n_kept"] for r in s],
                                            [r["chamfer_to_original"] for r in s]),
        }
    time_expo = {m: fit_exponent([r["n_input"] for r in sel("scaling", m)],
                                 [max(r["ms"], 1e-6) for r in sel("scaling", m)])
                 for m in ("random", "farthest_point", "voxel_nearest")}
    # Random selection costs a twentieth of a millisecond at the small end, which is
    # fixed overhead rather than work, so the full-range fit understates its slope.
    # Both fits are stored and the README says which one it is quoting.
    time_expo_large = {m: fit_exponent(
        [r["n_input"] for r in sel("scaling", m) if r["n_input"] >= 21000],
        [max(r["ms"], 1e-6) for r in sel("scaling", m) if r["n_input"] >= 21000])
        for m in ("random", "farthest_point", "voxel_nearest")}

    # coverage advantage of farthest-point over the others, at each matched count
    cover_ratio = {}
    for t in targets:
        base = next(r for r in rows if r["experiment"] == "matched"
                    and r["method"] == "farthest_point" and r["target"] == t)
        cover_ratio[str(t)] = {
            m: round(next(r for r in rows if r["experiment"] == "matched"
                          and r["method"] == m and r["target"] == t)["covering_radius"]
                     / base["covering_radius"], 2)
            for m in methods}

    # does the usual metric see any of this?
    chamfer_spread = {}
    for t in targets:
        v = [next(r for r in rows if r["experiment"] == "matched"
                  and r["method"] == m and r["target"] == t)["chamfer_to_original"]
             for m in methods]
        chamfer_spread[str(t)] = {"min": min(v), "max": max(v),
                                  "max_over_min": round(max(v) / min(v), 2)}

    patch_bias = {}
    for t in targets:
        patch_bias[str(t)] = {m: next(r for r in rows if r["experiment"] == "matched"
                                      and r["method"] == m
                                      and r["target"] == t)["patch_bias"]
                              for m in methods}

    # the centroid variant moves points off the surface; the offset should go as the
    # square of the voxel size, which is a planted exponent rather than an observation
    # Fitted only where a voxel holds enough points for its centroid to mean anything.
    # Below that the cells hold one or two points, whose centroid is on the surface by
    # construction, and the fit measures the occupancy instead of the curvature.
    cen = sorted([r for r in rows if r["experiment"] == "voxel_size"
                  and r["method"] == "voxel_centroid" and r["voxel"] >= 0.03],
                 key=lambda r: r["voxel"])
    cen_all = sorted([r for r in rows if r["experiment"] == "voxel_size"
                      and r["method"] == "voxel_centroid"], key=lambda r: r["voxel"])
    centroid_offset_exponent = fit_exponent([r["voxel"] for r in cen],
                                            [r["surface_dist_mean"] for r in cen])
    centroid_offset_exponent_all = fit_exponent(
        [r["voxel"] for r in cen_all], [r["surface_dist_mean"] for r in cen_all])
    centroid_points_per_cell = {str(r["voxel"]): round(len(pts) / r["n_kept"], 2)
                                for r in cen_all}
    on_surface = {m: max(r["surface_dist_max"] for r in rows if r["method"] == m)
                  for m in methods}

    res = {
        "project": "62_point_cloud_decimation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "seed": SEED, "n_input": int(len(pts)),
        "patch": {"n_points_inside": int(in_patch.sum()),
                  "share_of_cloud": float(in_patch.mean()),
                  "area": shape.patch_area,
                  "area_share": shape.patch_area_share,
                  "input_bias": float(in_patch.mean() / shape.patch_area_share),
                  "box": {"u0": shape.u0, "v0": shape.v0,
                          "wu": shape.wu, "wv": shape.wv}},
        "ground_truth": "distance to the analytic surface from its closed-form sdf; "
                        "the dense patch's area share is a closed-form integral",
        "targets": targets, "methods": methods,
        "voxel_sizes_for_target": {str(k): v for k, v in voxels.items()},
        "exponents": expo,
        "time_exponents": time_expo,
        "time_exponents_n_at_least_21000": time_expo_large,
        "covering_radius_over_farthest_point": cover_ratio,
        "chamfer_spread_across_methods": chamfer_spread,
        "patch_bias_by_target": patch_bias,
        "centroid_offset_vs_voxel_exponent": centroid_offset_exponent,
        "centroid_offset_vs_voxel_exponent_all_sizes": centroid_offset_exponent_all,
        "centroid_points_per_cell": centroid_points_per_cell,
        "max_distance_off_surface": on_surface,
        "rows": rows,
    }
    json.dump(res, io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                           newline="\n"), indent=1)

    print("\ncovering radius relative to farthest-point sampling:")
    for t, d in cover_ratio.items():
        print(f"  {t:>6} kept: " + "  ".join(f"{m}={v}" for m, v in d.items()))
    print("\nChamfer-to-original spread across the four methods "
          "(the metric usually quoted):")
    for t, d in chamfer_spread.items():
        print(f"  {t:>6} kept: max/min = {d['max_over_min']}")
    print("\ntime ~ n^k:", {m: round(v, 2) for m, v in time_expo.items()})
    return res


if __name__ == "__main__":
    main()
