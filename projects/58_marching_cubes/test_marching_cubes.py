"""Tests for 58_marching_cubes.

These assert properties, not stored numbers. A test pinned to "volume error is 0.0249%"
would break on a different scikit-image build without anything being wrong, and would
pass if the whole experiment silently degraded to a coarser grid. The properties below
fail only if the method or the measurement is actually broken.
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
_spec = importlib.util.spec_from_file_location("run_marching_cubes",
                                               os.path.join(_HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)


def test_analytic_volume_and_area_are_the_known_closed_forms():
    s = R.Sphere(2.0)
    assert s.volume == pytest.approx(4 / 3 * np.pi * 8)
    assert s.area == pytest.approx(4 * np.pi * 4)
    t = R.Torus(1.0, 0.25)
    assert t.volume == pytest.approx(2 * np.pi ** 2 * 1.0 * 0.0625)
    assert t.area == pytest.approx(4 * np.pi ** 2 * 1.0 * 0.25)


def test_sdf_is_zero_on_the_surface():
    s = R.Sphere(1.0)
    assert s.sdf(1.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    t = R.Torus(1.0, 0.35)
    assert t.sdf(1.35, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_mesh_volume_of_a_unit_cube_is_one():
    """The divergence-theorem volume, checked on a shape with no discretisation error."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    assert R.mesh_volume(v, f) == pytest.approx(1.0)
    assert R.mesh_area(v, f) == pytest.approx(6.0)


def test_refining_the_grid_reduces_the_error():
    s = R.Sphere(1.0)
    errs = []
    for n in (16, 32, 64):
        verts, faces, _, _ = R.extract(s, n)
        errs.append(abs(R.mesh_volume(verts, faces) - s.volume) / s.volume)
    assert errs[0] > errs[1] > errs[2], f"error did not fall monotonically: {errs}"


def test_observed_convergence_is_second_order():
    """The headline claim. Second order means halving h quarters the error."""
    s = R.Sphere(1.0)
    hs, errs = [], []
    for n in (24, 32, 48, 64, 96):
        verts, faces, h, _ = R.extract(s, n)
        hs.append(h)
        errs.append(abs(R.mesh_volume(verts, faces) - s.volume) / s.volume)
    k = R.fit_order(hs, errs)
    assert 1.8 <= k <= 2.2, f"expected order near 2, measured {k:.3f}"


def test_fit_order_recovers_a_planted_exponent():
    """The estimator itself, on data whose exponent is known by construction."""
    h = np.array([0.1, 0.05, 0.025, 0.0125])
    for planted in (1.0, 2.0, 3.0):
        err = 0.7 * h ** planted
        assert R.fit_order(h, err) == pytest.approx(planted, abs=1e-9)


def test_extracted_vertices_lie_on_the_true_surface():
    s = R.Sphere(1.0)
    verts, _, h, _ = R.extract(s, 64)
    d = R.surface_point_error(verts, s)
    # Every vertex sits on a grid edge, so it cannot be further off than the spacing.
    assert d["max"] < h, f"max surface distance {d['max']:.5f} exceeds spacing {h:.5f}"
    assert d["mean"] < 0.02 * h


def test_the_two_variants_agree_once_the_grid_resolves_the_shape():
    """They differ only where the grid undersamples - which is the project's finding."""
    t = R.Torus(1.0, 0.35)
    for n in (32, 64):
        va, fa, _, _ = R.extract(t, n, "lewiner")
        vb, fb, _, _ = R.extract(t, n, "lorensen")
        rel = abs(R.mesh_volume(va, fa) - R.mesh_volume(vb, fb)) / t.volume
        assert rel < 1e-3, f"n={n}: variants differ by {rel:.2e}"
