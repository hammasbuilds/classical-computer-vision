"""Marching cubes against surfaces whose volume and area are known exactly.

Every number here is checked against a closed form, not against another algorithm's
output. A sphere of radius r has volume 4/3 pi r^3 and area 4 pi r^2; a torus with radii
(R, a) has volume 2 pi^2 R a^2 and area 4 pi^2 R a. So the error of an extracted mesh is
exact, and how that error shrinks as the grid is refined can be fitted rather than
asserted.

No neural network, no training, no GPU.
"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
from skimage import measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- analytic surfaces


class Sphere:
    name = "sphere"

    def __init__(self, r=1.0):
        self.r = r

    def sdf(self, x, y, z):
        return np.sqrt(x ** 2 + y ** 2 + z ** 2) - self.r

    @property
    def volume(self):
        return 4.0 / 3.0 * np.pi * self.r ** 3

    @property
    def area(self):
        return 4.0 * np.pi * self.r ** 2

    @property
    def extent(self):
        return self.r * 1.3


class Torus:
    name = "torus"

    def __init__(self, R=1.0, a=0.35):
        self.R, self.a = R, a

    def sdf(self, x, y, z):
        q = np.sqrt(x ** 2 + y ** 2) - self.R
        return np.sqrt(q ** 2 + z ** 2) - self.a

    @property
    def volume(self):
        return 2.0 * np.pi ** 2 * self.R * self.a ** 2

    @property
    def area(self):
        return 4.0 * np.pi ** 2 * self.R * self.a

    @property
    def extent(self):
        return (self.R + self.a) * 1.25


# ---------------------------------------------------------------- mesh measures


def mesh_volume(verts, faces):
    """Signed volume by the divergence theorem, summed over triangles."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    return float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def mesh_area(verts, faces):
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def surface_point_error(verts, shape):
    """Distance from each extracted vertex to the true surface, via the SDF.

    For these shapes the signed distance function IS the distance to the surface, so
    this is exact - no nearest-neighbour search against a sampled reference, which
    would measure the sampling as much as the method.
    """
    d = np.abs(shape.sdf(verts[:, 0], verts[:, 1], verts[:, 2]))
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)),
            "max": float(d.max())}


def grid(shape, n):
    e = shape.extent
    ax = np.linspace(-e, e, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return shape.sdf(X, Y, Z), 2 * e / (n - 1), -e


def extract(shape, n, method="lewiner"):
    vol, h, origin = grid(shape, n)
    t0 = time.perf_counter()
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(h, h, h),
                                                method=method)
    ms = (time.perf_counter() - t0) * 1000
    verts = verts + origin
    return verts, faces, h, ms


def fit_order(hs, errs):
    """Slope of log(error) against log(h): the observed order of convergence."""
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    ok = errs > 0
    if ok.sum() < 2:
        return None
    k, _ = np.polyfit(np.log(hs[ok]), np.log(errs[ok]), 1)
    return float(k)


def main():
    shapes = [Sphere(1.0), Torus(1.0, 0.35)]
    sizes = [16, 24, 32, 48, 64, 96, 128]
    methods = ["lewiner", "lorensen"]

    rows = []
    for shape in shapes:
        for method in methods:
            for n in sizes:
                verts, faces, h, ms = extract(shape, n, method)
                v, a = mesh_volume(verts, faces), mesh_area(verts, faces)
                err = surface_point_error(verts, shape)
                rows.append({
                    "shape": shape.name, "method": method, "n": n, "h": h,
                    "ms": round(ms, 2), "n_verts": int(len(verts)),
                    "n_faces": int(len(faces)),
                    "volume": v, "volume_true": shape.volume,
                    "volume_rel_err": abs(v - shape.volume) / shape.volume,
                    "area": a, "area_true": shape.area,
                    "area_rel_err": abs(a - shape.area) / shape.area,
                    "surface_dist_mean": err["mean"],
                    "surface_dist_p95": err["p95"],
                    "surface_dist_max": err["max"],
                })
                print(f"  {shape.name:7} {method:9} n={n:<4} "
                      f"vol err {rows[-1]['volume_rel_err'] * 100:7.4f}%  "
                      f"area err {rows[-1]['area_rel_err'] * 100:7.4f}%  "
                      f"surf {err['mean']:.5f}  {ms:6.1f} ms")

    # observed convergence order per (shape, method, measure)
    orders = {}
    for shape in shapes:
        for method in methods:
            sel = [r for r in rows if r["shape"] == shape.name and r["method"] == method]
            hs = [r["h"] for r in sel]
            orders[f"{shape.name}/{method}"] = {
                "volume": fit_order(hs, [r["volume_rel_err"] for r in sel]),
                "area": fit_order(hs, [r["area_rel_err"] for r in sel]),
                "surface_distance": fit_order(hs, [r["surface_dist_mean"] for r in sel]),
            }

    import platform
    import scipy
    import skimage
    res = {
        "project": "58_marching_cubes",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_image": skimage.__version__},
        "ground_truth": "closed form: sphere 4/3 pi r^3 and 4 pi r^2; "
                        "torus 2 pi^2 R a^2 and 4 pi^2 R a",
        "shapes": [{"name": s.name, "volume": s.volume, "area": s.area} for s in shapes],
        "grid_sizes": sizes, "methods": methods,
        "convergence_order": orders,
        "rows": rows,
    }
    json.dump(res, io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                           newline="\n"), indent=1)

    print("\nobserved convergence order (error ~ h^k, higher is faster):")
    for k, v in orders.items():
        print(f"  {k:18} volume {v['volume']:.2f}   area {v['area']:.2f}   "
              f"surface {v['surface_distance']:.2f}")
    return res


if __name__ == "__main__":
    main()
