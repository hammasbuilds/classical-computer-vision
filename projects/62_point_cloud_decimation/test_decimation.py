"""Tests for 62_point_cloud_decimation.

Properties, not stored numbers. The guards against measuring nothing are
`test_the_patch_area_share_matches_an_independent_monte_carlo`, which checks the
planted ground truth itself rather than trusting the formula, and
`test_the_three_methods_do_not_return_the_same_cloud`, because a decimation comparison
in which the methods agree is a comparison of one method with itself.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R                                              # noqa: E402


def _cloud(n=8000, n_patch=400, seed=0):
    shape = R.Torus()
    pts, in_patch = shape.sample(n, np.random.default_rng(seed), n_patch=n_patch)
    return shape, pts, in_patch


def test_every_sampled_point_lies_exactly_on_the_torus():
    shape, pts, _ = _cloud()
    assert np.abs(shape.sdf(pts)).max() < 1e-12


def test_the_patch_area_share_matches_an_independent_monte_carlo():
    """The planted ground truth, checked against a method that shares no code with it.

    `patch_area` is a closed-form integral of the area element over a parameter box.
    `in_patch` recovers membership from xyz coordinates. If either were wrong the
    density-bias numbers would be wrong, and the two are checked against each other.
    """
    shape = R.Torus()
    q = shape._sample_box(400000, np.random.default_rng(3), 0, 2 * np.pi, 0, 2 * np.pi)
    mc = float(shape.in_patch(q).mean())
    closed = shape.patch_area_share
    se = np.sqrt(closed * (1 - closed) / len(q))
    assert abs(mc - closed) < 4 * se, f"closed form {closed:.6f}, Monte Carlo {mc:.6f}"


def test_index_based_methods_keep_original_points_and_the_centroid_does_not():
    shape, pts, in_patch = _cloud()
    v, _ = R.voxel_for_target(pts, 800)
    for method, kw in (("random", {"target": 800}),
                       ("farthest_point", {"target": 800}),
                       ("voxel", {"voxel": v, "mode": "nearest"})):
        r = R.run_one(pts, in_patch, shape, method, **kw)
        assert r["surface_dist_max"] < 1e-12, (method, r["surface_dist_max"])
    c = R.run_one(pts, in_patch, shape, "voxel", voxel=v, mode="centroid")
    assert c["surface_dist_max"] > 1e-4, c["surface_dist_max"]


def test_farthest_point_gives_the_smallest_covering_radius():
    """It greedily minimises exactly this, so anything else winning means a bug."""
    shape, pts, in_patch = _cloud()
    for target in (400, 1600):
        v, _ = R.voxel_for_target(pts, target)
        fps = R.run_one(pts, in_patch, shape, "farthest_point", target=target)
        others = [R.run_one(pts, in_patch, shape, "random", target=target),
                  R.run_one(pts, in_patch, shape, "voxel", voxel=v, mode="nearest")]
        for o in others:
            assert fps["covering_radius"] < o["covering_radius"], \
                f"target {target}: {o['method']} covered better than farthest point"


def test_random_preserves_the_input_density_bias_and_the_others_remove_it():
    """The headline finding, as an inequality."""
    shape, pts, in_patch = _cloud(n=8000, n_patch=800)
    input_bias = in_patch.mean() / shape.patch_area_share
    assert input_bias > 4, input_bias
    v, _ = R.voxel_for_target(pts, 1000)
    rnd = R.run_one(pts, in_patch, shape, "random", target=1000)
    fps = R.run_one(pts, in_patch, shape, "farthest_point", target=1000)
    vox = R.run_one(pts, in_patch, shape, "voxel", voxel=v, mode="nearest")
    assert rnd["patch_bias"] > 0.6 * input_bias, rnd["patch_bias"]
    assert fps["patch_bias"] < 0.4 * input_bias, fps["patch_bias"]
    assert vox["patch_bias"] < 0.4 * input_bias, vox["patch_bias"]


def test_chamfer_to_the_original_hides_what_coverage_shows():
    """The point of the project: the usual metric separates them far less."""
    shape, pts, in_patch = _cloud(n=12000, n_patch=1200)
    target = 1500
    v, _ = R.voxel_for_target(pts, target)
    rs = [R.run_one(pts, in_patch, shape, "random", target=target),
          R.run_one(pts, in_patch, shape, "farthest_point", target=target),
          R.run_one(pts, in_patch, shape, "voxel", voxel=v, mode="nearest")]
    cham = [r["chamfer_to_original"] for r in rs]
    cover = [r["covering_radius"] for r in rs]
    assert max(cover) / min(cover) > 1.5 * (max(cham) / min(cham)), \
        f"chamfer spread {max(cham) / min(cham):.2f}, coverage spread " \
        f"{max(cover) / min(cover):.2f}"


def test_the_three_methods_do_not_return_the_same_cloud():
    """A comparison whose arms agree is not a comparison."""
    shape, pts, in_patch = _cloud()
    target = 900
    v, _ = R.voxel_for_target(pts, target)
    sets = [set(R.random_sample(pts, target, np.random.default_rng(0)).tolist()),
            set(R.farthest_point(pts, target, seed=0).tolist()),
            set(R.voxel_grid(pts, v, mode="nearest")[0].tolist())]
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = len(sets[i] & sets[j]) / min(len(sets[i]), len(sets[j]))
            assert overlap < 0.6, f"methods {i} and {j} agree on {overlap:.0%} of points"


def test_voxel_bisection_hits_the_requested_count():
    """Without this the matched-count comparison is comparing different budgets."""
    shape, pts, _ = _cloud()
    for target in (300, 1200, 4000):
        v, got = R.voxel_for_target(pts, target)
        assert abs(got - target) <= max(2, 0.03 * target), (target, got)


def test_the_covering_radius_falls_with_roughly_the_square_root_of_the_count():
    """A surface is two-dimensional, so n points cover it to about n^-0.5."""
    shape, pts, in_patch = _cloud(n=20000, n_patch=0)
    ns = [250, 500, 1000, 2000]
    cov = [R.run_one(pts, in_patch, shape, "farthest_point",
                     target=n)["covering_radius"] for n in ns]
    assert R.fit_exponent(ns, cov) == pytest.approx(-0.5, abs=0.12), cov


def test_fit_exponent_recovers_a_planted_power_law():
    x = np.array([250.0, 500, 1000, 2000, 4000])
    for k in (-0.5, -1.0, 2.0):
        assert R.fit_exponent(x, 0.4 * x ** k) == pytest.approx(k, abs=1e-9)
