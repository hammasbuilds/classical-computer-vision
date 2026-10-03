"""Tests for 60_surface_reconstruction.

Properties, not stored numbers. The two that matter most are the ones that would catch
the experiment measuring nothing: `test_the_error_metric_is_not_floored_by_its_own_
sampler`, which plants a known displacement and checks the metric sees it, and
`test_more_points_actually_change_the_answer`, because a sweep whose rows are identical
is a broken sweep rather than a stable method.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Every project keeps its code in its own run.py, so a plain `import run`
# resolves to whichever project pytest collected first and the rest of the
# suite silently tests the wrong module. Load it by path under a unique name.
import importlib.util                                         # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("run_surface_reconstruction",
                                               os.path.join(_HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)


def test_samples_lie_on_the_analytic_surface_and_normals_match_the_gradient():
    rng = np.random.default_rng(0)
    for shape in R.SHAPES.values():
        p = shape.sample(3000, rng)
        assert np.abs(shape.sdf(p)).max() < 1e-12
        eps = 1e-6
        g = np.stack([(shape.sdf(p + eps * np.eye(3)[i]) -
                       shape.sdf(p - eps * np.eye(3)[i])) / (2 * eps)
                      for i in range(3)], axis=1)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        assert np.einsum("ij,ij->i", g, shape.normal(p)).min() > 1 - 1e-6


def test_euler_characteristic_of_known_meshes():
    """A tetrahedron is a sphere (2); a torus built as a 6x6 grid of quads is 0."""
    tet = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    assert R.euler_characteristic(tet) == 2
    n = 6
    idx = lambda i, j: (i % n) * n + (j % n)                  # noqa: E731
    faces = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [[a, b, c], [a, c, d]]
    assert R.euler_characteristic(np.array(faces)) == 0


def test_point_triangle_distance_against_cases_solved_by_hand():
    q = np.array([[0, 0, 3.0],          # above the interior -> the plane distance
                  [-1, -1, 0.0],        # off the corner at the origin
                  [0.5, -2.0, 0.0]])    # off the middle of an edge
    a = np.tile(np.array([0.0, 0, 0]), (3, 1))
    b = np.tile(np.array([1.0, 0, 0]), (3, 1))
    c = np.tile(np.array([0.0, 1, 0]), (3, 1))
    d = R.point_triangle_distance(q, a, b, c)
    assert d[0] == pytest.approx(3.0)
    assert d[1] == pytest.approx(np.sqrt(2))
    assert d[2] == pytest.approx(2.0)


def test_the_error_metric_is_not_floored_by_its_own_sampler():
    """The guard against an experiment that cannot see what it claims to measure.

    An earlier version estimated completeness as the nearest neighbour in a dense point
    sample of the mesh. That has a floor of about 0.5*sqrt(area/m) - 0.005 on the unit
    sphere - which was larger than the error of every method, so all four scored the
    same and the comparison measured the sampler. Here a sphere mesh is scaled by a
    known factor and the metric must report that exact displacement.
    """
    sh = R.Sphere(1.0)
    rng = np.random.default_rng(0)
    pts = sh.sample(4000, rng)
    verts, faces = R.m_hoppe(pts, sh, nrm=sh.normal(pts), n_grid=96)
    base = R.score(verts, faces, sh, np.random.default_rng(1))
    assert base["accuracy_mean"] < 1e-3 and base["completeness_mean"] < 1e-3

    # a displacement far below the sampler's old floor must still be visible
    small = R.score(verts * 1.0005, faces, sh, np.random.default_rng(1))
    assert small["accuracy_mean"] > base["accuracy_mean"] * 1.2
    assert small["completeness_mean"] > base["completeness_mean"] * 1.2

    # and once the displacement dominates the mesh's own error it IS the measurement
    for planted in (0.002, 0.01, 0.05):
        s = R.score(verts * (1.0 + planted), faces, sh, np.random.default_rng(1))
        assert s["accuracy_mean"] == pytest.approx(planted, rel=0.25), \
            f"planted {planted}, measured accuracy {s['accuracy_mean']}"
        assert s["completeness_mean"] == pytest.approx(planted, rel=0.25), \
            f"planted {planted}, measured completeness {s['completeness_mean']}"


def test_more_points_actually_change_the_answer():
    """A sweep whose rows are identical is broken, not stable."""
    sh = R.Torus(1.0, 0.35)
    ch = [R.reconstruct(sh, n, "hoppe", noise=R.FLOOR, n_grid=64)["chamfer"]
          for n in (500, 2000, 8000)]
    assert len(set(np.round(ch, 8))) == 3, f"identical rows across the sweep: {ch}"
    assert ch[0] > ch[1] > ch[2], f"error did not fall with density: {ch}"


def test_the_convex_hull_cannot_represent_a_hole():
    """Topology is checked, not eyeballed: the hull of a torus is a genus-0 surface."""
    sh = R.Torus(1.0, 0.35)
    r = R.reconstruct(sh, 3000, "convex_hull", noise=R.FLOOR)
    assert r["euler"] == 2 and not r["euler_ok"]
    # and it does not improve with more points, which is the point of the finding
    big = R.reconstruct(sh, 12000, "convex_hull", noise=R.FLOOR)
    assert big["chamfer"] > 0.5 * r["chamfer"]


def test_implicit_methods_recover_the_right_topology_on_both_shapes():
    for name, shape in R.SHAPES.items():
        for method in ("hoppe", "poisson_fft"):
            r = R.reconstruct(shape, 4000, method, noise=R.FLOOR, n_grid=64)
            assert r["euler_ok"], f"{name}/{method} gave V-E+F = {r['euler']}"


def test_an_exactly_spherical_sample_is_delaunay_degenerate():
    """The reason the density sweep runs at a small noise floor.

    Four points on a sphere are cospherical, so every tetrahedron's circumsphere IS
    the sphere. Without this the alpha test has nothing to separate.
    """
    d0 = R.delaunay_circumradius_stats(R.Sphere(1.0), 1200, 0.0)
    assert d0["circumradius_min"] == pytest.approx(1.0, abs=1e-6)
    assert d0["frac_below_half_radius"] == 0.0
    d1 = R.delaunay_circumradius_stats(R.Sphere(1.0), 1200, 1e-3)
    assert d1["frac_below_half_radius"] > 0.1


def test_normal_orientation_uses_no_ground_truth_and_still_gets_it_right():
    rng = np.random.default_rng(0)
    for shape in R.SHAPES.values():
        pts = shape.sample(3000, rng)
        n = R.orient_normals(pts, R.pca_normals(pts, 18))
        assert R.orientation_accuracy(pts, n, shape) > 0.98


def test_poisson_beats_hoppe_once_the_noise_is_large():
    """The headline finding, as an inequality rather than a quoted number."""
    sh = R.Sphere(1.0)
    kw = dict(n_grid=64, noise=0.04)
    a = R.reconstruct(sh, 4000, "hoppe", **kw)
    b = R.reconstruct(sh, 4000, "poisson_fft", **kw)
    assert b["chamfer"] < a["chamfer"] / 2
    assert b["euler_ok"] and not a["euler_ok"]


def test_fit_exponent_recovers_a_planted_power_law():
    x = np.array([500.0, 1000, 2000, 4000, 8000])
    for k in (-1.0, -0.5, -2.0):
        assert R.fit_exponent(x, 3.3 * x ** k) == pytest.approx(k, abs=1e-9)
