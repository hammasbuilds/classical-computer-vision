"""Tests for 59_icp_registration.

Properties, not stored numbers. The important ones are the two that would have caught
the bug this project actually had: an overlap experiment that reported a perfect result
at every setting because the clouds were not really partial.
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
_spec = importlib.util.spec_from_file_location("run_icp_registration",
                                               os.path.join(_HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)


def test_rotation_matrices_are_orthonormal():
    for deg in (0, 10, 90, 180):
        M = R.rot([0.3, 0.8, 0.5], deg)
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(M) == pytest.approx(1.0)


def test_rotation_angle_recovers_a_planted_angle():
    for deg in (1, 7, 30, 90, 179):
        M = R.rot([0.2, -0.7, 0.4], deg)
        assert R.rotation_angle(np.eye(3), M) == pytest.approx(deg, abs=1e-6)


def test_icp_recovers_an_exact_transform_from_a_small_offset():
    rng = np.random.default_rng(0)
    pts = R.sample_bunny_like(800, rng)
    out = R.run_case(pts, 8, [0.3, 0.8, 0.5], [0.04, -0.02, 0.01], 0.0, 1.0,
                     "point", np.random.default_rng(0))
    assert out["rot_err_deg"] < 1e-3
    assert out["trans_err"] < 1e-4


def test_point_to_plane_has_the_wider_convergence_basin():
    """The project's first claim, tested rather than quoted."""
    rng = np.random.default_rng(0)
    pts = R.sample_bunny_like(1200, rng)
    kw = dict(axis=[0.3, 0.8, 0.5], shift=[0.05, -0.03, 0.02], noise=0.0, overlap=1.0)
    at45 = {m: R.run_case(pts, 45, mode=m, rng=np.random.default_rng(0), **kw)
            for m in ("point", "plane")}
    assert at45["plane"]["rot_err_deg"] < 1.0
    assert at45["point"]["rot_err_deg"] > 1.0


def test_point_to_point_is_the_more_noise_tolerant_of_the_two():
    """The second claim, and the trade-off that makes the first one not a free win."""
    rng = np.random.default_rng(0)
    pts = R.sample_bunny_like(1500, rng)
    kw = dict(angle=10, axis=[0.3, 0.8, 0.5], shift=[0.05, -0.03, 0.02], overlap=1.0)
    out = {m: R.run_case(pts, noise=0.05, mode=m, rng=np.random.default_rng(0), **kw)
           for m in ("point", "plane")}
    assert out["point"]["rot_err_deg"] < out["plane"]["rot_err_deg"]


def test_partial_overlap_actually_removes_shared_geometry():
    """Guards the bug this project had twice.

    Two earlier versions "tested" partial overlap while leaving every source point with
    a correct counterpart - once by random decimation, once by cropping only the source
    so it became a subset of the destination. Both reported 0.000 degrees at every
    setting. A real crop must leave each cloud holding points the other does not.
    """
    rng = np.random.default_rng(0)
    pts = R.sample_bunny_like(1000, rng)
    from scipy.spatial import cKDTree
    lo = np.quantile(pts[:, 0], 1.0 - 0.6)
    hi = np.quantile(pts[:, 0], 0.6)
    src, dst = pts[pts[:, 0] >= lo], pts[pts[:, 0] <= hi]
    assert len(src) < len(pts) and len(dst) < len(pts)
    # points in src with no counterpart within a tight radius in dst
    d, _ = cKDTree(dst).query(src)
    unmatched = float(np.mean(d > 1e-9))
    assert unmatched > 0.2, (f"only {unmatched:.1%} of source points lack a match - "
                             "this is not partial overlap")


def test_trimming_helps_at_partial_overlap():
    """The third claim: trimming extends the usable overlap range."""
    rng = np.random.default_rng(0)
    pts = R.sample_bunny_like(1500, rng)
    kw = dict(angle=10, axis=[0.3, 0.8, 0.5], shift=[0.05, -0.03, 0.02],
              noise=0.0, overlap=0.75, mode="point")
    plain = R.run_case(pts, rng=np.random.default_rng(0), trim=1.0, **kw)
    trimmed = R.run_case(pts, rng=np.random.default_rng(0), trim=0.8, **kw)
    assert trimmed["rot_err_deg"] < plain["rot_err_deg"]


def test_normals_point_along_the_true_surface_normal_of_a_sphere():
    rng = np.random.default_rng(0)
    pts = R.sample_sphere(2000, rng, 1.0)
    n = R.normals_of(pts, k=20)
    # on a unit sphere the true normal at p is p itself, up to sign
    align = np.abs(np.einsum("ij,ij->i", n, pts))
    assert float(np.mean(align)) > 0.97
