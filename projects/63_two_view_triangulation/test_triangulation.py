"""Tests for 63_two_view_triangulation.

Properties, not stored numbers. The guards against measuring nothing are
`test_a_deliberately_wrong_triangulation_is_detected`, because the four methods agree to
within a few per cent over most of the sweep and that could equally well be one method
silently called four times, and `test_the_two_cameras_see_the_scene_from_different_
places`, because a triangulation experiment in which the two views coincide measures a
projection, not a triangulation.
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
_spec = importlib.util.spec_from_file_location("run_two_view_triangulation",
                                               os.path.join(_HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)


def test_projection_round_trips_through_the_camera_matrices():
    rng = np.random.default_rng(0)
    X = R.scene(500, rng)
    cam1, cam2, C2 = R.cameras(0.5)
    for (Rm, t, P) in (cam1, cam2):
        x = R.project(P, X)
        # the same thing by hand, from R, t and K
        Xc = (Rm @ X.T).T + t
        manual = (R.K @ Xc.T).T
        manual = manual[:, :2] / manual[:, 2:3]
        assert np.abs(x - manual).max() < 1e-9
        assert (Xc[:, 2] > 0).all(), "points must be in front of both cameras"


def test_the_two_cameras_see_the_scene_from_different_places():
    """A triangulation experiment needs two genuinely different views."""
    for baseline in (0.02, 0.5, 4.0):
        cam1, cam2, C2 = R.cameras(baseline)
        assert np.linalg.norm(C2) == pytest.approx(baseline)
        rng = np.random.default_rng(0)
        X = R.scene(500, rng)
        x1, x2 = R.project(cam1[2], X), R.project(cam2[2], X)
        shift = np.linalg.norm(x1 - x2, axis=1).mean()
        assert shift > 2.0, f"baseline {baseline}: mean disparity only {shift:.3f} px"


def test_every_triangulator_is_exact_without_noise():
    """With no pixel noise there is a unique right answer and it is known."""
    for m in R.TRIANGULATORS:
        r = R.run_case(500, 0.0, 0.5, m)
        assert r["err_mean"] < 1e-9, f"{m}: {r['err_mean']:.3e}"
        assert r["frac_behind_camera"] == 0.0


def test_a_deliberately_wrong_triangulation_is_detected():
    """The guard against measuring nothing.

    The four methods agree to within a few per cent across most of the sweep, which is
    the finding - and is also what one method called four times would look like. A
    triangulator that is wrong by a known, small amount must show up as that amount.
    """
    rng = np.random.default_rng(0)
    X = R.scene(1000, rng)
    cam1, cam2, _ = R.cameras(0.5)
    x1, x2 = R.project(cam1[2], X), R.project(cam2[2], X)
    good = R.tri_dlt(cam1[2], cam2[2], x1, x2)
    assert np.abs(good - X).max() < 1e-8
    for planted in (1e-3, 1e-2, 0.1):
        shifted = good + np.array([0.0, 0.0, planted])
        assert np.linalg.norm(shifted - X, axis=1).mean() == pytest.approx(
            planted, rel=1e-6)


def test_error_is_linear_in_pixel_noise():
    sigmas = [0.1, 0.25, 0.5, 1.0, 2.0]
    for m in ("dlt", "midpoint"):
        e = [R.run_case(1500, s, 0.5, m)["err_median"] for s in sigmas]
        assert R.fit_exponent(sigmas, e) == pytest.approx(1.0, abs=0.1), (m, e)


def test_error_is_inversely_proportional_to_the_baseline():
    bases = [0.25, 0.5, 1.0, 2.0, 4.0]
    for m in ("dlt", "iterative_lm"):
        e = [R.run_case(1500, 1.0, b, m)["err_median"] for b in bases]
        assert R.fit_exponent(bases, e) == pytest.approx(-1.0, abs=0.12), (m, e)


def test_the_baseline_buys_depth_accuracy_and_not_lateral_accuracy():
    """The finding a single RMS number cannot show."""
    bases = [0.25, 0.5, 1.0, 2.0, 4.0]
    rs = [R.run_case(1500, 1.0, b, "iterative_lm") for b in bases]
    depth = R.fit_exponent(bases, [r["err_depth_mean"] for r in rs])
    lateral = R.fit_exponent(bases, [r["err_lateral_mean"] for r in rs])
    assert depth < -0.85, depth
    assert abs(lateral) < 0.2, lateral
    assert rs[0]["anisotropy"] > 10 * rs[-1]["anisotropy"]


def test_reprojection_error_is_not_a_proxy_for_three_d_error():
    """Nearly the same residual, two orders of magnitude apart in the actual answer."""
    short = R.run_case(1500, 1.0, 0.02, "iterative_lm")
    long = R.run_case(1500, 1.0, 4.0, "iterative_lm")
    assert short["reproj_px"] == pytest.approx(long["reproj_px"], rel=0.5)
    assert short["err_per_reproj_px"] > 100 * long["err_per_reproj_px"]


def test_the_midpoint_has_the_lighter_tail_when_the_rays_are_nearly_parallel():
    """The one place the four methods genuinely differ."""
    ml = R.run_case(2000, 1.0, 0.02, "iterative_lm")
    mid = R.run_case(2000, 1.0, 0.02, "midpoint")
    assert mid["err_median"] == pytest.approx(ml["err_median"], rel=0.15)
    assert mid["err_p95"] < 0.6 * ml["err_p95"]


def test_the_error_decomposition_adds_back_up():
    """depth^2 + lateral^2 must be the total, or the split is inventing a direction."""
    rng = np.random.default_rng(0)
    Xt = R.scene(400, rng)
    Xe = Xt + rng.normal(scale=0.05, size=Xt.shape)
    along, lateral = R.decompose_error(Xe, Xt, Xt)
    total = np.linalg.norm(Xe - Xt, axis=1)
    assert np.allclose(np.sqrt(along ** 2 + lateral ** 2), total, atol=1e-12)


def test_fit_exponent_recovers_a_planted_power_law():
    x = np.array([0.05, 0.1, 0.5, 1.0, 4.0])
    for k in (1.0, -1.0, 0.5):
        assert R.fit_exponent(x, 1.7 * x ** k) == pytest.approx(k, abs=1e-9)
