"""ICP point-cloud registration, against transforms that are known exactly.

A rigid transform is applied to a point cloud on purpose, so the answer ICP should find
is known to machine precision. That makes three things measurable that a qualitative
"the clouds line up now" picture cannot show:

  * rotation and translation error in degrees and units, not just a residual;
  * the convergence basin - how far wrong the initial guess can be before ICP locks
    onto the wrong alignment, which is the failure mode that matters in practice;
  * what partial overlap and noise do to both.

Point-to-point and point-to-plane are both implemented here from the SVD/least-squares
primitives rather than called from a library, because the difference between them is
the thing being measured.

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


# ---------------------------------------------------------------- shapes and transforms

def sample_sphere(n, rng, r=1.0):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * r


def sample_torus(n, rng, R=1.0, a=0.35):
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    return np.stack([(R + a * np.cos(v)) * np.cos(u),
                     (R + a * np.cos(v)) * np.sin(u),
                     a * np.sin(v)], axis=1)


def sample_bunny_like(n, rng):
    """Three fused blobs: an asymmetric shape, so rotation is unambiguous.

    A sphere is rotationally symmetric and ICP can 'succeed' on it at any rotation,
    which would make the rotation error meaningless.
    """
    parts = [(np.array([0.0, 0.0, 0.0]), 0.6),
             (np.array([0.55, 0.25, 0.1]), 0.32),
             (np.array([-0.2, 0.5, -0.3]), 0.22)]
    out = []
    for c, r in parts:
        k = int(n * r ** 2 / sum(p[1] ** 2 for p in parts))
        out.append(c + sample_sphere(max(k, 8), rng, r))
    return np.concatenate(out)[:n]


def normals_of(pts, k=16):
    """Surface normals by PCA over each point's k nearest neighbours."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    nb = pts[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1]
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]                      # eigenvector of the smallest eigenvalue


def rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def rotation_angle(Ra, Rb):
    """Geodesic angle between two rotations, in degrees."""
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ---------------------------------------------------------------- the two ICP variants

def icp(src, dst, max_iter=60, tol=1e-9, mode="point", dst_normals=None,
        trim=1.0):
    """Returns (R, t, n_iter, rmse, ms). `trim` keeps the closest fraction of pairs."""
    t0 = time.perf_counter()
    tree = cKDTree(dst)
    R = np.eye(3)
    t = np.zeros(3)
    cur = src.copy()
    prev = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        d, j = tree.query(cur)
        keep = np.ones(len(cur), bool)
        if trim < 1.0:
            thresh = np.quantile(d, trim)
            keep = d <= thresh
        P, Q = cur[keep], dst[j[keep]]
        if mode == "point":
            pc, qc = P.mean(0), Q.mean(0)
            H = (P - pc).T @ (Q - qc)
            U, _, Vt = np.linalg.svd(H)
            dR = Vt.T @ U.T
            if np.linalg.det(dR) < 0:      # reflection guard
                Vt[-1] *= -1
                dR = Vt.T @ U.T
            dt = qc - dR @ pc
        else:
            # point-to-plane: linearise the rotation, solve 6x6 least squares
            N = dst_normals[j[keep]]
            A = np.hstack([np.cross(P, N), N])
            b = np.einsum("ij,ij->i", (Q - P), N)
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            dR = rot([1, 0, 0], 0)
            a, bb, c = x[:3]
            dR = np.array([[1, -c, bb], [c, 1, -a], [-bb, a, 1]])
            U, _, Vt = np.linalg.svd(dR)   # project back onto SO(3)
            dR = U @ Vt
            dt = x[3:]
        cur = (dR @ cur.T).T + dt
        R = dR @ R
        t = dR @ t + dt
        rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return R, t, it, prev, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- experiments

def run_case(pts, angle, axis, shift, noise, overlap, mode, rng, trim=1.0):
    R_true = rot(axis, angle)
    t_true = np.asarray(shift, float)
    dst = pts
    src_full = (R_true.T @ (pts - t_true).T).T  # so that R_true @ src + t_true == dst
    src = src_full
    if overlap < 1.0:
        # Crop BOTH clouds, on opposite sides, leaving a shared band of width `overlap`.
        #
        # Two earlier versions of this measured nothing and reported 0.000 degrees at
        # every setting. The first removed random points, which is decimation: the
        # thinner cloud still covers the whole shape. The second cropped only the
        # source, so the source became a SUBSET of the destination and every point
        # still had a correct match. Partial overlap means each cloud holds geometry
        # the other does not - which is what a second scan from another viewpoint gives.
        lo = np.quantile(pts[:, 0], 1.0 - overlap)
        hi = np.quantile(pts[:, 0], overlap)
        keep_src = pts[:, 0] >= lo                 # chosen in the SHARED frame,
        keep_dst = pts[:, 0] <= hi                 # then applied to each cloud
        src = src_full[keep_src]
        dst = pts[keep_dst]
    if noise > 0:
        src = src + rng.normal(scale=noise, size=src.shape)
    dn = normals_of(dst) if mode == "plane" else None
    R, t, it, rmse, ms = icp(src, dst, mode=mode, dst_normals=dn, trim=trim)
    return {"rot_err_deg": rotation_angle(R, R_true),
            "trans_err": float(np.linalg.norm(t - t_true)),
            "iters": it, "rmse": rmse, "ms": round(ms, 2)}


def main():
    rng = np.random.default_rng(SEED)
    pts = sample_bunny_like(2000, rng)
    axis = [0.3, 0.8, 0.5]
    rows = []

    # 1) convergence basin: how far wrong can the initial rotation be?
    for mode in ("point", "plane"):
        for angle in (2, 5, 10, 15, 20, 30, 45, 60, 90):
            r = run_case(pts, angle, axis, [0.05, -0.03, 0.02], 0.0, 1.0, mode,
                         np.random.default_rng(SEED))
            r.update({"experiment": "basin", "mode": mode, "angle": angle,
                      "noise": 0.0, "overlap": 1.0})
            rows.append(r)
            print(f"  basin   {mode:6} {angle:3}deg -> rot err {r['rot_err_deg']:7.3f}deg  "
                  f"trans {r['trans_err']:.4f}  {r['iters']:3} iters")

    # 2) noise
    for mode in ("point", "plane"):
        for noise in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
            r = run_case(pts, 10, axis, [0.05, -0.03, 0.02], noise, 1.0, mode,
                         np.random.default_rng(SEED))
            r.update({"experiment": "noise", "mode": mode, "angle": 10,
                      "noise": noise, "overlap": 1.0})
            rows.append(r)
            print(f"  noise   {mode:6} sd={noise:<6} -> rot err {r['rot_err_deg']:7.3f}deg  "
                  f"trans {r['trans_err']:.4f}")

    # 3) partial overlap, with and without trimming
    for trim in (1.0, 0.8):
        for overlap in (1.0, 0.9, 0.75, 0.6, 0.5):
            r = run_case(pts, 10, axis, [0.05, -0.03, 0.02], 0.0, overlap, "point",
                         np.random.default_rng(SEED), trim=trim)
            r.update({"experiment": "overlap", "mode": "point", "angle": 10,
                      "noise": 0.0, "overlap": overlap, "trim": trim})
            rows.append(r)
            print(f"  overlap trim={trim} ov={overlap:<5} -> rot err "
                  f"{r['rot_err_deg']:7.3f}deg  trans {r['trans_err']:.4f}")

    def basin_limit(mode):
        """Largest tested misalignment still recovered to under 1 degree."""
        ok = [r["angle"] for r in rows
              if r["experiment"] == "basin" and r["mode"] == mode
              and r["rot_err_deg"] < 1.0]
        return max(ok) if ok else None

    res = {"project": "59_icp_registration",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
           "seed": SEED, "n_points": int(len(pts)),
           "ground_truth": "a known rigid transform is applied on purpose; errors are "
                           "geodesic rotation angle and translation norm against it",
           "basin_limit_deg": {m: basin_limit(m) for m in ("point", "plane")},
           "rows": rows}
    json.dump(res, io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                           newline="\n"), indent=1)
    print("\nconvergence basin (largest misalignment recovered to <1 deg):")
    for m, v in res["basin_limit_deg"].items():
        print(f"  {m:6} {v} deg")
    return res


if __name__ == "__main__":
    main()
