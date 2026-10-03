"""Tests for 61_normal_estimation.

Properties, not stored numbers. Two of them exist to catch the experiment measuring
nothing: the noise-free plane reports exactly 0.000 degrees, which would also be what a
broken estimator returned, so one test proves the same code path reports a non-zero
error on a curved surface and another plants a known rotation and requires the metric to
report that exact angle.
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
_spec = importlib.util.spec_from_file_location("run_normal_estimation",
                                               os.path.join(_HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

SMALL = dict(n_pts=6000, n_query=1500)


def _sdf_gradient(f, p, eps=1e-6):
    g = np.stack([(f(p + eps * np.eye(3)[i]) - f(p - eps * np.eye(3)[i])) / (2 * eps)
                  for i in range(3)], axis=1)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def test_analytic_normals_are_unit_and_are_the_sdf_gradient():
    rng = np.random.default_rng(0)
    sdfs = {
        "plane": lambda p: p[:, 2],
        "sphere": lambda p: np.linalg.norm(p, axis=1) - 1.0,
        "torus": lambda p: np.sqrt((np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1.0) ** 2
                                   + p[:, 2] ** 2) - 0.35,
    }
    for name, shape in R.SHAPES.items():
        p = shape.sample(2000, rng)
        n = shape.normal(p)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
        assert np.abs(sdfs[name](p)).max() < 1e-12
        agree = np.einsum("ij,ij->i", _sdf_gradient(sdfs[name], p), n)
        assert agree.min() > 1 - 1e-6, f"{name}: worst agreement {agree.min()}"


def test_the_angle_metric_reports_a_planted_rotation_exactly():
    """If this is wrong, every degree in the project is wrong."""
    rng = np.random.default_rng(0)
    true = rng.normal(size=(500, 3))
    true /= np.linalg.norm(true, axis=1, keepdims=True)
    for planted in (0.5, 3.0, 17.0, 80.0):
        # rotate each normal by exactly `planted` degrees about a perpendicular axis
        t = rng.normal(size=(500, 3))
        perp = t - true * np.einsum("ij,ij->i", t, true)[:, None]
        perp /= np.linalg.norm(perp, axis=1, keepdims=True)
        a = np.deg2rad(planted)
        est = np.cos(a) * true + np.sin(a) * perp
        assert R.angle_error_deg(est, true) == pytest.approx(planted, abs=1e-9)
        # PCA returns an unoriented direction, so a flip must not change the answer
        assert R.angle_error_deg(-est, true) == pytest.approx(planted, abs=1e-9)


def test_a_noise_free_plane_gives_exactly_zero_at_every_k():
    for k in (4, 16, 64, 256):
        r = R.run_case(R.SHAPES["plane"], k=k, noise=0.0, **SMALL)
        assert r["max_deg"] == 0.0, f"k={k}: worst error {r['max_deg']}"


def test_zero_on_the_plane_is_not_the_estimator_always_returning_zero():
    """The guard against the experiment measuring nothing.

    Exactly 0.000 degrees is the right answer on a plane and is also what a broken
    estimator - or a metric comparing something with itself - would print. The same
    code path must produce a non-zero, ordered set of errors elsewhere.
    """
    out = {name: R.run_case(shape, k=32, noise=0.0, **SMALL)["mean_deg"]
           for name, shape in R.SHAPES.items()}
    assert out["plane"] == 0.0
    assert out["sphere"] > 0.1
    # the torus tube is tighter than the sphere, so its curvature term must be larger
    assert out["torus"] > out["sphere"], out


def test_noisy_error_falls_as_k_to_the_minus_one_not_minus_a_half():
    """The counter-intuitive prediction: the k-nearest patch grows as it is filled."""
    ks = [16, 32, 64, 128, 256]
    for noise in (0.001, 0.005):
        e = [R.run_case(R.SHAPES["plane"], k=k, noise=noise, **SMALL)["mean_deg"]
             for k in ks]
        k_exp = R.fit_exponent(ks, e)
        assert -1.3 < k_exp < -0.85, f"noise {noise}: fitted {k_exp:.3f}, expected ~-1"


def test_error_is_linear_in_the_noise_level():
    sigmas = [0.001, 0.002, 0.005, 0.01]
    e = [R.run_case(R.SHAPES["plane"], k=64, noise=s, **SMALL)["mean_deg"]
         for s in sigmas]
    assert R.fit_exponent(sigmas, e) == pytest.approx(1.0, abs=0.2)


def test_the_clean_error_on_a_curved_surface_is_set_by_density_not_by_k():
    sh = R.SHAPES["sphere"]
    flat = [R.run_case(sh, k=k, noise=0.0, **SMALL)["mean_deg"]
            for k in (32, 64, 128, 256)]
    assert abs(R.fit_exponent([32, 64, 128, 256], flat)) < 0.2, flat
    ns = [2500, 5000, 10000, 20000]
    dens = [R.run_case(sh, n_pts=n, k=32, noise=0.0, n_query=1500)["mean_deg"]
            for n in ns]
    assert R.fit_exponent(ns, dens) == pytest.approx(-0.5, abs=0.12), dens


def test_a_sweep_that_produces_identical_rows_is_broken():
    e = [R.run_case(R.SHAPES["torus"], k=k, noise=0.005, **SMALL)["mean_deg"]
         for k in (8, 32, 128)]
    assert len(set(np.round(e, 9))) == 3, f"identical rows across the sweep: {e}"


def test_the_quadratic_fit_removes_the_curvature_term_on_clean_points():
    for name in ("sphere", "torus"):
        a = R.run_case(R.SHAPES[name], k=16, noise=0.0, **SMALL)["mean_deg"]
        b = R.run_case(R.SHAPES[name], k=16, noise=0.0, estimator="jet_knn",
                       **SMALL)["mean_deg"]
        assert b < a / 10, f"{name}: plane fit {a:.4f}, jet {b:.4f}"


def test_the_quadratic_fit_loses_its_advantage_under_noise():
    """The trade-off that makes the previous test not a recommendation."""
    sh = R.SHAPES["torus"]
    clean = (R.run_case(sh, k=32, noise=0.0, **SMALL)["mean_deg"] /
             R.run_case(sh, k=32, noise=0.0, estimator="jet_knn", **SMALL)["mean_deg"])
    noisy = (R.run_case(sh, k=32, noise=0.02, **SMALL)["mean_deg"] /
             R.run_case(sh, k=32, noise=0.02, estimator="jet_knn", **SMALL)["mean_deg"])
    assert clean > 5 * noisy, f"clean gain {clean:.2f}, noisy gain {noisy:.2f}"


def test_fit_exponent_recovers_a_planted_power_law():
    x = np.array([8.0, 16, 32, 64, 128, 256])
    for k in (-1.0, -0.5, 1.0):
        assert R.fit_exponent(x, 2.5 * x ** k) == pytest.approx(k, abs=1e-9)
