"""Surface reconstruction from a point cloud, scored against the surface itself.

The usual way to show a reconstruction is a render of the mesh, which proves nothing: a
mesh that looks like a torus can have the wrong size, the wrong topology, or both. Here
the points are sampled from an analytic surface whose signed distance function is known
in closed form, so

  * accuracy - how far the reconstructed surface is from the true one - is EXACT. Every
    point sampled on a reconstructed triangle is scored by |sdf|, with no
    nearest-neighbour search against a reference cloud, which would measure the
    reference's sampling as much as the method.
  * topology is exact too. A sphere has Euler characteristic 2 and a torus has 0, so
    V - E + F on the output either matches or it does not.

Four classical methods, no neural network, no training, no GPU:

  convex_hull   scipy.spatial.ConvexHull - the baseline that cannot represent a hole
  alpha_shape   3-D Delaunay, tetrahedra kept by circumradius, boundary faces extracted
  hoppe         signed distance to the tangent plane of the nearest oriented point,
                sampled on a grid, then marching cubes
  poisson_fft   Poisson surface reconstruction in its spectral form: splat the oriented
                normals into a grid vector field, solve the Laplacian for the indicator
                function by FFT, extract its average level at the input points

The two implicit methods get *estimated* normals - PCA over k neighbours, oriented by
propagation over the k-nearest-neighbour graph - not the analytic ones, because handing
them the true normals would be handing them half the problem. The fraction of normals
that ends up correctly oriented is reported against the analytic normals, so the input
quality is measured rather than assumed.
"""
from __future__ import annotations

import io
import json
import os
import platform
import time

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import skimage
from skimage import measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SEED = 0

# ---------------------------------------------------------------- analytic surfaces


class Sphere:
    name = "sphere"
    euler = 2                      # V - E + F for a closed genus-0 surface

    def __init__(self, r=1.0):
        self.r = r

    def sdf(self, p):
        return np.linalg.norm(p, axis=-1) - self.r

    def normal(self, p):
        return p / np.linalg.norm(p, axis=-1, keepdims=True)

    def sample(self, n, rng):
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True) * self.r

    @property
    def area(self):
        return 4.0 * np.pi * self.r ** 2

    @property
    def extent(self):
        return self.r * 1.35


class Torus:
    name = "torus"
    euler = 0                      # genus 1

    def __init__(self, R=1.0, a=0.35):
        self.R, self.a = R, a

    def sdf(self, p):
        q = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2) - self.R
        return np.sqrt(q ** 2 + p[..., 2] ** 2) - self.a

    def normal(self, p):
        """Analytic outward normal: the gradient of the sdf, which has unit length."""
        xy = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
        q = xy - self.R
        d = np.sqrt(q ** 2 + p[..., 2] ** 2)
        s = q / np.maximum(d * np.maximum(xy, 1e-12), 1e-12)
        g = np.stack([p[..., 0] * s, p[..., 1] * s, p[..., 2] / np.maximum(d, 1e-12)],
                     axis=-1)
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def sample(self, n, rng):
        """Uniform by AREA, not by parameter: the Jacobian is (R + a cos v)."""
        out = []
        while sum(len(o) for o in out) < n:
            m = int((n + 64) * 1.6)
            u = rng.uniform(0, 2 * np.pi, m)
            v = rng.uniform(0, 2 * np.pi, m)
            acc = rng.uniform(0, 1, m) < (self.R + self.a * np.cos(v)) / \
                (self.R + self.a)
            u, v = u[acc], v[acc]
            out.append(np.stack([(self.R + self.a * np.cos(v)) * np.cos(u),
                                 (self.R + self.a * np.cos(v)) * np.sin(u),
                                 self.a * np.sin(v)], axis=1))
        return np.concatenate(out)[:n]

    @property
    def area(self):
        return 4.0 * np.pi ** 2 * self.R * self.a

    @property
    def extent(self):
        return (self.R + self.a) * 1.3


SHAPES = {"sphere": Sphere(1.0), "torus": Torus(1.0, 0.35)}


# ---------------------------------------------------------------- normals from points

def pca_normals(pts, k=18):
    """Unoriented normals: the smallest-eigenvalue direction of the local covariance."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    nb = pts[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1]
    _, vec = np.linalg.eigh(cov)
    return vec[:, :, 0]


def orient_normals(pts, nrm, k=12):
    """Hoppe's orientation step: propagate sign along a maximum-agreement spanning tree.

    Edge weight 1 - |n_i . n_j| is small where two normals are nearly parallel, so the
    minimum spanning tree prefers to propagate across flat neighbourhoods and crosses
    high-curvature regions as rarely as possible. The global sign is then fixed by the
    extreme point in x, whose outward normal must point in +x for any shape contained
    in a half-space on that side - a geometric fact about the sample, not ground truth.
    """
    n = len(pts)
    k = min(k + 1, n)
    d, idx = cKDTree(pts).query(pts, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    agree = np.abs(np.einsum("ij,ij->i", nrm[rows], nrm[cols]))
    w = 1.0 - agree + 1e-6
    g = coo_matrix((w, (rows, cols)), shape=(n, n))
    g = g.minimum(g.T) + (g - g.minimum(g.T)).maximum(0)   # symmetrise
    mst = minimum_spanning_tree(g)
    mst = mst + mst.T
    seed = int(np.argmax(pts[:, 0]))
    order, pred = depth_first_order(mst, seed, directed=False)
    out = nrm.copy()
    if out[seed, 0] < 0:
        out[seed] *= -1
    for i in order[1:]:
        p = pred[i]
        if p >= 0 and np.dot(out[i], out[p]) < 0:
            out[i] *= -1
    return out


def orientation_accuracy(pts, nrm, shape):
    """Fraction of normals pointing the same way as the analytic outward normal."""
    t = shape.normal(pts)
    return float(np.mean(np.einsum("ij,ij->i", nrm, t) > 0))


# ---------------------------------------------------------------- mesh measures

def euler_characteristic(faces):
    f = np.asarray(faces)
    v = np.unique(f).size
    e = np.unique(np.sort(np.concatenate(
        [f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), axis=1), axis=0).shape[0]
    return int(v - e + len(f))


def mesh_area(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def sample_mesh(verts, faces, n, rng):
    """Area-weighted uniform samples on the triangles - what the surface IS, not its
    vertices. Scoring vertices only would flatter a mesh with few, well-placed ones."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = ar.sum()
    if tot <= 0:
        return verts[:1]
    j = rng.choice(len(faces), size=n, p=ar / tot)
    u = rng.uniform(size=(n, 1))
    v = rng.uniform(size=(n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[j] + u * (b[j] - a[j]) + v * (c[j] - a[j])


def point_triangle_distance(q, a, b, c):
    """Exact distance from each point q[i] to each triangle (a,b,c)[i].

    Needed because completeness measured as "nearest of a dense point sample of the
    mesh" has a floor: the mean nearest-neighbour distance to m uniform samples of an
    area-A surface is about 0.5*sqrt(A/m), which at m = 120,000 on the unit sphere is
    0.005 - larger than the error of every method here, so it would hide all of them
    behind the sampler. Clamped barycentric coordinates give the true distance.
    """
    ab, ac, aq = b - a, c - a, q - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", aq, ab)
    d21 = np.einsum("ij,ij->i", aq, ac)
    den = d00 * d11 - d01 * d01
    safe = np.abs(den) > 1e-20
    u = np.zeros_like(den)
    v = np.zeros_like(den)
    u[safe] = (d11[safe] * d20[safe] - d01[safe] * d21[safe]) / den[safe]
    v[safe] = (d00[safe] * d21[safe] - d01[safe] * d20[safe]) / den[safe]
    inside = safe & (u >= 0) & (v >= 0) & (u + v <= 1)
    proj = a + u[:, None] * ab + v[:, None] * ac
    best = np.where(inside, np.linalg.norm(q - proj, axis=1), np.inf)
    for p0, p1 in ((a, b), (a, c), (b, c)):
        e = p1 - p0
        t = np.clip(np.einsum("ij,ij->i", q - p0, e) /
                    np.maximum(np.einsum("ij,ij->i", e, e), 1e-20), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(q - (p0 + t[:, None] * e), axis=1))
    return best


def distance_to_mesh(q, verts, faces, cand=10):
    """Point-to-surface distance, exactly, via the `cand` nearest triangle centroids."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cen = (a + b + c) / 3.0
    k = min(cand, len(faces))
    _, idx = cKDTree(cen).query(q, k=k)
    idx = np.atleast_2d(idx.T).T if k > 1 else idx.reshape(-1, 1)
    qq = np.repeat(q, k, axis=0)
    j = idx.ravel()
    d = point_triangle_distance(qq, a[j], b[j], c[j]).reshape(len(q), k)
    return d.min(axis=1)


def score(verts, faces, shape, rng, n_acc=40000, n_comp=20000):
    """accuracy from the closed form; completeness from a dense analytic sample."""
    if faces is None or len(faces) == 0:
        return {"accuracy_mean": float("nan"), "accuracy_p95": float("nan"),
                "completeness_mean": float("nan"), "chamfer": float("nan"),
                "euler": None, "euler_ok": False, "area": 0.0,
                "area_rel_err": float("nan"), "n_verts": 0, "n_faces": 0}
    s = sample_mesh(verts, faces, n_acc, rng)
    acc = np.abs(shape.sdf(s))
    gt = shape.sample(n_comp, rng)
    comp = distance_to_mesh(gt, verts, faces)
    eu = euler_characteristic(faces)
    a = mesh_area(verts, faces)
    return {"accuracy_mean": float(acc.mean()), "accuracy_p95": float(np.percentile(acc, 95)),
            "completeness_mean": float(comp.mean()),
            "chamfer": float(0.5 * (acc.mean() + comp.mean())),
            "euler": eu, "euler_ok": bool(eu == shape.euler),
            "area": a, "area_rel_err": float(abs(a - shape.area) / shape.area),
            "n_verts": int(np.unique(faces).size), "n_faces": int(len(faces))}


# ---------------------------------------------------------------- the four methods

def m_convex_hull(pts, shape, **kw):
    h = ConvexHull(pts)
    return pts, h.simplices


def _circumradii(pts, tets):
    """Circumradius of every tetrahedron, solved as a 3x3 system per tet."""
    p0 = pts[tets[:, 0]]
    A = 2.0 * np.stack([pts[tets[:, i]] - p0 for i in (1, 2, 3)], axis=1)
    sq = (pts[tets] ** 2).sum(-1)
    b = np.stack([sq[:, i] - sq[:, 0] for i in (1, 2, 3)], axis=1)
    det = np.linalg.det(A)
    r = np.full(len(tets), np.inf)
    ok = np.abs(det) > 1e-14
    if ok.any():
        cc = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
        r[ok] = np.linalg.norm(cc - p0[ok], axis=1)
    return r


def m_alpha_shape(pts, shape, alpha=0.25, **kw):
    tri = Delaunay(pts)
    tets = tri.simplices
    r = _circumradii(pts, tets)
    keep = tets[r < alpha]
    if len(keep) == 0:
        return pts, np.zeros((0, 3), int)
    faces = np.concatenate([keep[:, [0, 1, 2]], keep[:, [0, 1, 3]],
                            keep[:, [0, 2, 3]], keep[:, [1, 2, 3]]])
    srt = np.sort(faces, axis=1)
    uniq, inv, cnt = np.unique(srt, axis=0, return_inverse=True, return_counts=True)
    # a face on the boundary of the kept complex belongs to exactly one kept tetrahedron
    return pts, uniq[cnt == 1]


def _grid(shape, n):
    e = shape.extent
    ax = np.linspace(-e, e, n)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
    return g, 2 * e / (n - 1), -e


def m_hoppe(pts, shape, nrm=None, n_grid=64, **kw):
    g, h, o = _grid(shape, n_grid)
    flat = g.reshape(-1, 3)
    _, j = cKDTree(pts).query(flat)
    f = np.einsum("ij,ij->i", flat - pts[j], nrm[j]).reshape(g.shape[:3])
    v, fa, _, _ = measure.marching_cubes(f, level=0.0, spacing=(h, h, h))
    return v + o, fa


def m_poisson_fft(pts, shape, nrm=None, n_grid=64, smooth=1.4, **kw):
    """Poisson reconstruction, spectral form.

    Splat the oriented normals into a grid vector field V, then solve the Laplacian
    for the indicator function chi with div V as the right-hand side. In Fourier space
    that is one division, so no solver is needed:  chi = -i (k . V) / |k|^2, smoothed by
    a Gaussian so the splatting spikes do not survive. The iso-level is the mean of chi
    at the input points, which is exactly how Kazhdan's method chooses it: the points
    are supposed to lie ON the level set.
    """
    e = shape.extent
    h = 2 * e / (n_grid - 1)
    V = np.zeros((3, n_grid, n_grid, n_grid))
    gi = (pts + e) / h
    i0 = np.floor(gi).astype(int)
    fr = gi - i0
    for dx in (0, 1):                         # trilinear splat
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, fr[:, 0], 1 - fr[:, 0]) *
                     np.where(dy, fr[:, 1], 1 - fr[:, 1]) *
                     np.where(dz, fr[:, 2], 1 - fr[:, 2]))
                ix = np.clip(i0[:, 0] + dx, 0, n_grid - 1)
                iy = np.clip(i0[:, 1] + dy, 0, n_grid - 1)
                iz = np.clip(i0[:, 2] + dz, 0, n_grid - 1)
                for c in range(3):
                    np.add.at(V[c], (ix, iy, iz), w * nrm[:, c])
    k = 2 * np.pi * np.fft.fftfreq(n_grid, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    Vh = [np.fft.fftn(V[c]) for c in range(3)]
    num = 1j * (KX * Vh[0] + KY * Vh[1] + KZ * Vh[2])
    den = -K2
    chi_h = np.zeros_like(num)
    nz = K2 > 0
    chi_h[nz] = num[nz] / den[nz]
    chi_h *= np.exp(-0.5 * K2 * (smooth * h) ** 2)
    chi = np.real(np.fft.ifftn(chi_h))
    # chi at the input points, by trilinear interpolation, gives the iso-level
    vals = np.zeros(len(pts))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, fr[:, 0], 1 - fr[:, 0]) *
                     np.where(dy, fr[:, 1], 1 - fr[:, 1]) *
                     np.where(dz, fr[:, 2], 1 - fr[:, 2]))
                vals += w * chi[np.clip(i0[:, 0] + dx, 0, n_grid - 1),
                                np.clip(i0[:, 1] + dy, 0, n_grid - 1),
                                np.clip(i0[:, 2] + dz, 0, n_grid - 1)]
    lvl = float(np.mean(vals))
    if not (chi.min() < lvl < chi.max()):
        return pts, np.zeros((0, 3), int)
    v, fa, _, _ = measure.marching_cubes(chi, level=lvl, spacing=(h, h, h))
    return v - e, fa


METHODS = {"convex_hull": m_convex_hull, "alpha_shape": m_alpha_shape,
           "hoppe": m_hoppe, "poisson_fft": m_poisson_fft}
NEEDS_NORMALS = {"hoppe", "poisson_fft"}


# ---------------------------------------------------------------- one experiment

def reconstruct(shape, n_pts, method, noise=0.0, seed=SEED, k=18, **kw):
    rng = np.random.default_rng(seed)
    clean = shape.sample(n_pts, rng)
    pts = clean + rng.normal(scale=noise, size=clean.shape) if noise > 0 else clean
    extra = {}
    if method in NEEDS_NORMALS:
        nrm = orient_normals(pts, pca_normals(pts, k=k))
        extra["nrm"] = nrm
        extra_out = {"normal_orientation_acc": orientation_accuracy(pts, nrm, shape),
                     "normal_angle_deg": float(np.degrees(np.mean(np.arccos(np.clip(
                         np.abs(np.einsum("ij,ij->i", nrm, shape.normal(clean))),
                         0, 1)))))}
    else:
        extra_out = {}
    t0 = time.perf_counter()
    verts, faces = METHODS[method](pts, shape, **extra, **kw)
    ms = (time.perf_counter() - t0) * 1000
    row = score(verts, faces, shape, np.random.default_rng(seed + 1))
    row.update({"shape": shape.name, "method": method, "n_pts": int(n_pts),
                "noise": noise, "ms": round(ms, 1),
                "spacing": float(np.sqrt(shape.area / n_pts))})
    row.update(extra_out)
    row.update({k2: v for k2, v in kw.items()})
    return row


def fit_exponent(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


FLOOR = 0.001        # 0.1% of the sphere radius - see degeneracy note below
ALPHA_MULT = 4.0     # alpha as a multiple of the mean sample spacing


def delaunay_circumradius_stats(shape, n_pts, noise, seed=SEED):
    """Why the density sweep cannot be run on an exactly noise-free sphere.

    Every Delaunay tetrahedron of a sample lying exactly on a sphere is inscribed in
    that same sphere, so its circumradius is the sphere's radius and the alpha test
    `circumradius < alpha` keeps either nothing or everything. The degeneracy is a
    property of the input, not of the implementation, and it disappears under any
    noise at all - which is what this records.
    """
    rng = np.random.default_rng(seed)
    pts = shape.sample(n_pts, rng)
    if noise > 0:
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    r = _circumradii(pts, Delaunay(pts).simplices)
    f = r[np.isfinite(r)]
    return {"shape": shape.name, "n_pts": int(n_pts), "noise": noise,
            "circumradius_min": float(f.min()), "circumradius_p05": float(np.percentile(f, 5)),
            "circumradius_median": float(np.median(f)),
            "n_tets": int(len(r)),
            "frac_below_half_radius": float(np.mean(f < 0.5))}


def main():
    rows = []
    counts = [500, 1000, 2000, 4000, 8000]

    def log(r):
        rows.append(r)
        ch = r["chamfer"]
        print(f"  {r['shape']:6} {r['method']:12} n={r['n_pts']:<5} "
              f"noise={r['noise']:<6} acc {r['accuracy_mean']:.5f}  "
              f"comp {r['completeness_mean']:.5f}  chamfer {ch:.5f}  "
              f"euler {str(r['euler']):>6}{'  ' if r['euler_ok'] else ' X'}"
              f"  {r['ms']:7.1f} ms")

    print(f"1) density sweep at the {FLOOR} noise floor")
    for sh in SHAPES.values():
        for n in counts:
            # alpha and the grid are scaled to the sample spacing, because a fixed
            # setting would be comparing a well-tuned method against a starved one
            sp = np.sqrt(sh.area / n)
            for method, kw in (("convex_hull", {}),
                               ("alpha_shape", {"alpha": ALPHA_MULT * sp}),
                               ("hoppe", {"n_grid": 64}),
                               ("poisson_fft", {"n_grid": 64})):
                r = reconstruct(sh, n, method, noise=FLOOR, **kw)
                r["experiment"] = "density"
                log(r)

    print("\n2) noise sweep at 4000 points")
    for sh in SHAPES.values():
        sp = np.sqrt(sh.area / 4000)
        for noise in (0.0, 0.002, 0.005, 0.01, 0.02, 0.04):
            for method, kw in (("alpha_shape", {"alpha": ALPHA_MULT * sp}),
                               ("hoppe", {"n_grid": 64}),
                               ("poisson_fft", {"n_grid": 64})):
                r = reconstruct(sh, 4000, method, noise=noise, **kw)
                r["experiment"] = "noise"
                log(r)

    print("\n3) alpha sweep at 4000 points")
    for sh in SHAPES.values():
        sp = np.sqrt(sh.area / 4000)
        for mult in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0):
            r = reconstruct(sh, 4000, "alpha_shape", noise=FLOOR, alpha=mult * sp)
            r["experiment"] = "alpha"
            r["alpha_over_spacing"] = mult
            log(r)

    print("\n4) grid sweep for the implicit methods, 4000 points")
    for sh in SHAPES.values():
        for n_grid in (24, 32, 48, 64, 96):
            for method in ("hoppe", "poisson_fft"):
                r = reconstruct(sh, 4000, method, noise=FLOOR, n_grid=n_grid)
                r["experiment"] = "grid"
                log(r)

    print("\n5) the Delaunay degeneracy of an exactly spherical sample")
    degen = [delaunay_circumradius_stats(SHAPES["sphere"], 2000, ns)
             for ns in (0.0, FLOOR)]
    for d in degen:
        print(f"  noise={d['noise']:<6} min circumradius {d['circumradius_min']:.4f}  "
              f"median {d['circumradius_median']:.4f}  "
              f"below r/2: {d['frac_below_half_radius']:.1%}")

    # ---- fitted power laws and summary numbers
    expo = {}
    for sh in SHAPES:
        for method in METHODS:
            sel = [r for r in rows if r["experiment"] == "density"
                   and r["shape"] == sh and r["method"] == method]
            sel.sort(key=lambda r: r["n_pts"])
            expo[f"{sh}/{method}"] = {
                "chamfer_vs_n_points": fit_exponent([r["n_pts"] for r in sel],
                                                    [r["chamfer"] for r in sel]),
                "accuracy_vs_n_points": fit_exponent([r["n_pts"] for r in sel],
                                                     [r["accuracy_mean"] for r in sel]),
            }
    topo = {}
    for method in METHODS:
        sel = [r for r in rows if r["experiment"] == "density" and r["method"] == method]
        topo[method] = {sh: all(r["euler_ok"] for r in sel if r["shape"] == sh)
                        for sh in SHAPES}

    def noise_breaks(method, shape, limit=0.02):
        """Largest noise sigma at which chamfer stays under `limit` scene units."""
        ok = [r["noise"] for r in rows if r["experiment"] == "noise"
              and r["method"] == method and r["shape"] == shape
              and r["chamfer"] < limit]
        return max(ok) if ok else None

    best_alpha = {}
    for sh in SHAPES:
        sel = [r for r in rows if r["experiment"] == "alpha" and r["shape"] == sh
               and np.isfinite(r["chamfer"])]
        b = min(sel, key=lambda r: r["chamfer"])
        best_alpha[sh] = {"alpha_over_spacing": b["alpha_over_spacing"],
                          "chamfer": b["chamfer"], "euler": b["euler"],
                          "euler_ok": b["euler_ok"]}

    # what the FFT solve costs and buys at a matched grid resolution
    grid_tradeoff = {}
    for sh in SHAPES:
        g = {}
        for n_grid in (24, 32, 48, 64, 96):
            a = next(r for r in rows if r["experiment"] == "grid" and r["shape"] == sh
                     and r["method"] == "hoppe" and r["n_grid"] == n_grid)
            b = next(r for r in rows if r["experiment"] == "grid" and r["shape"] == sh
                     and r["method"] == "poisson_fft" and r["n_grid"] == n_grid)
            g[str(n_grid)] = {"hoppe_ms": a["ms"], "poisson_ms": b["ms"],
                              "hoppe_chamfer": a["chamfer"],
                              "poisson_chamfer": b["chamfer"],
                              "speedup": round(a["ms"] / b["ms"], 2),
                              "accuracy_ratio": round(b["chamfer"] / a["chamfer"], 2)}
        grid_tradeoff[sh] = g

    # how much better Poisson is than Hoppe at each noise level, stored rather than
    # worked out by hand later - every ratio quoted in the README comes from here
    noise_ratio = {}
    for sh in SHAPES:
        d = {}
        for ns in (0.0, 0.002, 0.005, 0.01, 0.02, 0.04):
            a = next(r for r in rows if r["experiment"] == "noise" and r["shape"] == sh
                     and r["method"] == "hoppe" and r["noise"] == ns)
            b = next(r for r in rows if r["experiment"] == "noise" and r["shape"] == sh
                     and r["method"] == "poisson_fft" and r["noise"] == ns)
            d[str(ns)] = round(a["chamfer"] / b["chamfer"], 2)
        noise_ratio[sh] = d

    hull_torus = sorted([r for r in rows if r["experiment"] == "density"
                         and r["shape"] == "torus" and r["method"] == "convex_hull"],
                        key=lambda r: r["n_pts"])
    hull_torus_change = round(100 * (hull_torus[-1]["chamfer"] /
                                     hull_torus[0]["chamfer"] - 1), 1)

    n_alpha_ok = sum(1 for r in rows if r["experiment"] == "alpha" and r["euler_ok"])
    n_alpha = sum(1 for r in rows if r["experiment"] == "alpha")

    res = {
        "project": "60_surface_reconstruction",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_image": skimage.__version__},
        "seed": SEED,
        "noise_floor": FLOOR,
        "alpha_over_spacing_default": ALPHA_MULT,
        "delaunay_degeneracy": degen,
        "ground_truth": "points sampled from an analytic surface; accuracy is |sdf| at "
                        "area-weighted samples of the reconstructed triangles (exact), "
                        "topology is V-E+F against the known Euler characteristic "
                        "(sphere 2, torus 0)",
        "shapes": {k: {"area": v.area, "euler": v.euler} for k, v in SHAPES.items()},
        "methods": list(METHODS),
        "point_counts": counts,
        "exponents": expo,
        "topology_correct_everywhere": topo,
        "noise_limit_chamfer_under_0.02": {
            m: {s: noise_breaks(m, s) for s in SHAPES}
            for m in ("alpha_shape", "hoppe", "poisson_fft")},
        "best_alpha": best_alpha,
        "alpha_settings_with_correct_topology": {"correct": n_alpha_ok,
                                                 "tested": n_alpha},
        "grid_tradeoff": grid_tradeoff,
        "hoppe_over_poisson_chamfer": noise_ratio,
        "convex_hull_torus_chamfer_change_pct_500_to_8000": hull_torus_change,
        "rows": rows,
    }
    json.dump(res, io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                           newline="\n"), indent=1)

    print("\nfitted chamfer exponent (chamfer ~ n_points^k):")
    for k2, v in expo.items():
        print(f"  {k2:26} {v['chamfer_vs_n_points']}")
    print("\ntopology correct at every density:")
    for m, v in topo.items():
        print(f"  {m:12} {v}")
    return res


if __name__ == "__main__":
    main()
