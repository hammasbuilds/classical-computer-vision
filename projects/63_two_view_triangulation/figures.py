"""Figures for 63_two_view_triangulation.

Three figures. The first draws the two predicted power laws with reference slopes, so
"+1 in noise, -1 in baseline" can be checked by eye. The second separates depth from
lateral error, which is the whole reason a single RMS is misleading. The third is the
one that matters in practice: how little the reprojection residual says about the 3-D
error, and where the four triangulators actually differ.
"""
from __future__ import annotations

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

COL = {"dlt": "#2a78d6", "midpoint": "#59a14f", "optimal": "#eb6834",
       "iterative_lm": "#b07aa1"}
LAB = {"dlt": "DLT (cv2)", "midpoint": "midpoint of rays",
       "optimal": "optimal correction + DLT", "iterative_lm": "Gauss-Newton (ML)"}
MUTED, GRID, CRIT = "#7a7873", "#e8e7e2", "#d03b3b"


def style(ax):
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def pick(rows, exp, m, key):
    return sorted([r for r in rows if r["experiment"] == exp and r["method"] == m],
                  key=lambda r: r[key])


def laws_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=140)

    ax = axes[0]
    for m in COL:
        s = [r for r in pick(rows, "noise", m, "noise_px") if r["noise_px"] > 0]
        e = res["exponents"][f"error_vs_noise/{m}"]
        ax.loglog([r["noise_px"] for r in s], [r["err_median"] for r in s], "o-",
                  color=COL[m], lw=2, ms=5.5, markeredgecolor="white",
                  markeredgewidth=1.0, label=f"{LAB[m]}   {e:+.3f}")
    ax.set_xlabel("pixel noise σ (px)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("median 3-D error (scene units)", fontsize=9.5, color=MUTED)
    ax.set_title("Error is linear in pixel noise\n(predicted exponent +1)",
                 fontsize=10.5)
    ax.set_xticks([0.05, 0.25, 1.0, 4.0, 10.0])
    ax.set_xticklabels(["0.05", "0.25", "1", "4", "10"], fontsize=9)
    ax.minorticks_off()
    style(ax)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")

    ax = axes[1]
    for m in COL:
        s = pick(rows, "baseline", m, "baseline")
        e = res["exponents"][f"error_vs_baseline/{m}"]
        ax.loglog([r["baseline"] for r in s], [r["err_median"] for r in s], "o-",
                  color=COL[m], lw=2, ms=5.5, markeredgecolor="white",
                  markeredgewidth=1.0, label=f"{LAB[m]}   {e:+.3f}")
    s = pick(rows, "baseline", "dlt", "baseline")
    b = np.array([r["baseline"] for r in s])
    y0 = s[-1]["err_median"]
    ax.loglog(b, y0 * 0.35 * (b / b[-1]) ** -1.0, "--", color=MUTED, lw=1.3,
              label="slope −1")
    ax.axvspan(0.015, 0.1, color="#f6efe6", zorder=0)
    ax.annotate("the fit excludes this band:\nthe error saturates once it\napproaches "
                "the scene depth",
                xy=(0.021, 0.97), xycoords=("data", "axes fraction"), color=MUTED,
                fontsize=8, ha="left", va="top")
    ax.set_xlabel("baseline (scene units; mean depth is 6)", fontsize=9.5, color=MUTED)
    ax.set_title("and inversely proportional to the baseline\n(predicted exponent −1)",
                 fontsize=10.5)
    ax.set_xticks([0.02, 0.1, 0.5, 2.0, 4.0])
    ax.set_xticklabels(["0.02", "0.1", "0.5", "2", "4"], fontsize=9)
    ax.minorticks_off()
    style(ax)
    ax.legend(frameon=False, fontsize=8.3, loc="lower left")

    fig.suptitle("Two predictions from the geometry, both reproduced to within 1.2%",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "laws.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def anisotropy_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=140)

    ax = axes[0]
    s = pick(rows, "baseline", "iterative_lm", "baseline")
    b = np.array([r["baseline"] for r in s])
    ed = res["exponents"]["depth_error_vs_baseline/iterative_lm"]
    el = res["exponents"]["lateral_error_vs_baseline/iterative_lm"]
    ax.loglog(b, [r["err_depth_mean"] for r in s], "o-", color="#d03b3b", lw=2, ms=6,
              markeredgecolor="white", markeredgewidth=1.1,
              label=f"along the view ray   {ed:+.3f}")
    ax.loglog(b, [r["err_lateral_mean"] for r in s], "s-", color="#2a78d6", lw=2, ms=6,
              markeredgecolor="white", markeredgewidth=1.1,
              label=f"across it   {el:+.3f}")
    ax.set_xlabel("baseline", fontsize=9.5, color=MUTED)
    ax.set_ylabel("mean error (scene units)", fontsize=9.5, color=MUTED)
    ax.set_title("A longer baseline buys depth accuracy\nand almost no lateral accuracy",
                 fontsize=10.5)
    ax.set_xticks([0.02, 0.1, 0.5, 2.0, 4.0])
    ax.set_xticklabels(["0.02", "0.1", "0.5", "2", "4"], fontsize=9)
    ax.minorticks_off()
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[1]
    for m in COL:
        s = pick(rows, "baseline", m, "baseline")
        ax.loglog([r["baseline"] for r in s], [r["anisotropy"] for r in s], "o-",
                  color=COL[m], lw=2, ms=5.5, markeredgecolor="white",
                  markeredgewidth=1.0, label=LAB[m])
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (5, 4)))
    ax.annotate("isotropic", xy=(0.03, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", color=MUTED, fontsize=8.5,
                ha="left", va="bottom")
    ax.set_xlabel("baseline", fontsize=9.5, color=MUTED)
    ax.set_ylabel("depth error ÷ lateral error", fontsize=9.5, color=MUTED)
    ax.set_title("At a 2 cm baseline depth is 513× worse\nthan lateral position",
                 fontsize=10.5)
    ax.set_xticks([0.02, 0.1, 0.5, 2.0, 4.0])
    ax.set_xticklabels(["0.02", "0.1", "0.5", "2", "4"], fontsize=9)
    ax.set_yticks([1, 10, 100, 500])
    ax.set_yticklabels(["1", "10", "100", "500"], fontsize=9)
    ax.minorticks_off()
    style(ax)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")

    fig.suptitle("A single RMS number hides which direction the error is in",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "anisotropy.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def reprojection_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), dpi=140)

    ax = axes[0]
    for m in COL:
        s = pick(rows, "baseline", m, "baseline")
        ax.loglog([r["baseline"] for r in s], [r["err_per_reproj_px"] for r in s],
                  "o-", color=COL[m], lw=2, ms=5.5, markeredgecolor="white",
                  markeredgewidth=1.0, label=LAB[m])
    ax.set_xlabel("baseline", fontsize=9.5, color=MUTED)
    ax.set_ylabel("scene units of 3-D error per pixel of residual", fontsize=9.5,
                  color=MUTED)
    s = pick(rows, "baseline", "iterative_lm", "baseline")
    ratio = s[0]["err_per_reproj_px"] / s[-1]["err_per_reproj_px"]
    ax.set_title(f"The same residual means {ratio:.0f} times\nmore error at a short "
                 "baseline", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")

    ax = axes[1]
    for m in COL:
        s = pick(rows, "baseline", m, "baseline")
        ax.loglog([r["baseline"] for r in s], [r["err_median"] for r in s], "o-",
                  color=COL[m], lw=2, ms=5.5, markeredgecolor="white",
                  markeredgewidth=1.0, label=f"{LAB[m]} — median")
        ax.loglog([r["baseline"] for r in s], [r["err_p95"] for r in s], "s--",
                  color=COL[m], lw=1.4, ms=4, alpha=0.8)
    ax.set_xlabel("baseline", fontsize=9.5, color=MUTED)
    ax.set_ylabel("3-D error (solid: median, dashed: p95)", fontsize=9.5, color=MUTED)
    ax.set_title("The methods agree in the median and\nseparate in the tail",
                 fontsize=10.5)

    ax = axes[2]
    for m in COL:
        s = pick(rows, "baseline", m, "baseline")
        ax.semilogx([r["baseline"] for r in s],
                    [100 * r["frac_gross_error"] for r in s], "o-", color=COL[m],
                    lw=2, ms=5.5, markeredgecolor="white", markeredgewidth=1.0,
                    label=LAB[m])
    ax.set_xlabel("baseline", fontsize=9.5, color=MUTED)
    ax.set_ylabel("points off by more than half the scene depth (%)", fontsize=9.5,
                  color=MUTED)
    ax.set_title("and in how often they fail outright", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")

    for ax in axes:
        ax.set_xticks([0.02, 0.1, 0.5, 2.0, 4.0])
        ax.set_xticklabels(["0.02", "0.1", "0.5", "2", "4"], fontsize=9)
        ax.minorticks_off()
        style(ax)

    fig.suptitle("A one-pixel residual is not a one-unit error, and the gap is the "
                 "baseline", fontsize=12.5, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "reprojection.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def main():
    res = json.load(io.open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    for p in (laws_figure(res), anisotropy_figure(res), reprojection_figure(res)):
        print("wrote", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
