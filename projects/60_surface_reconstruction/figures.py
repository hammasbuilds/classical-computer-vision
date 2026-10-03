"""Figures for 60_surface_reconstruction.

Four panels of numbers and one of meshes. Every error plot marks the settings where the
output has the WRONG TOPOLOGY with a red cross, because a low Chamfer distance on a
surface that is not the right kind of surface is not a good reconstruction, and a plain
error curve hides that completely.
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

COL = {"convex_hull": "#7a7873", "alpha_shape": "#59a14f",
       "hoppe": "#2a78d6", "poisson_fft": "#eb6834"}
LAB = {"convex_hull": "convex hull", "alpha_shape": "alpha shape",
       "hoppe": "Hoppe tangent plane", "poisson_fft": "Poisson (FFT)"}
MUTED, GRID, CRIT = "#7a7873", "#e8e7e2", "#d03b3b"


def style(ax):
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def mark_bad_topology(ax, xs, ys, oks):
    bad = [(x, y) for x, y, ok in zip(xs, ys, oks) if not ok and np.isfinite(y)]
    if bad:
        ax.plot([b[0] for b in bad], [b[1] for b in bad], "x", color=CRIT,
                ms=9, mew=2.0, zorder=6)


def pick(rows, **kw):
    out = [r for r in rows
           if all(r.get(k) == v for k, v in kw.items())]
    return out


def density_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=140)
    for ax, shape in zip(axes, ("sphere", "torus")):
        for m in ("convex_hull", "alpha_shape", "hoppe", "poisson_fft"):
            sel = sorted(pick(rows, experiment="density", shape=shape, method=m),
                         key=lambda r: r["n_pts"])
            x = [r["n_pts"] for r in sel]
            y = [r["chamfer"] for r in sel]
            ax.loglog(x, y, "o-", color=COL[m], lw=2, ms=6, markeredgecolor="white",
                      markeredgewidth=1.2, label=LAB[m], zorder=4)
            mark_bad_topology(ax, x, y, [r["euler_ok"] for r in sel])
        k = res["exponents"][f"{shape}/hoppe"]["chamfer_vs_n_points"]
        ax.set_title(f"{shape}", fontsize=11.5, pad=8)
        ax.set_xlabel("input points", fontsize=9.5, color=MUTED)
        style(ax)
        ax.set_xticks([500, 1000, 2000, 4000, 8000])
        ax.set_xticklabels(["500", "1k", "2k", "4k", "8k"], fontsize=9)
        ax.minorticks_off()
        del k
    axes[0].set_ylabel("Chamfer distance to the true surface", fontsize=9.5, color=MUTED)
    # One legend for both panels, placed under the figure so nothing sits on the data.
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=CRIT, marker="x", ls="", ms=9, mew=2.0))
    l.append("wrong topology")
    fig.legend(h, l, frameon=False, fontsize=9.5, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("More points help three methods and do nothing at all for the fourth",
                 fontsize=12.5, y=1.0)
    plt.tight_layout()
    p = os.path.join(OUT, "density.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def noise_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=140)
    for ax, shape in zip(axes, ("sphere", "torus")):
        for m in ("alpha_shape", "hoppe", "poisson_fft"):
            sel = sorted(pick(rows, experiment="noise", shape=shape, method=m),
                         key=lambda r: r["noise"])
            # 0 cannot be drawn on a log axis; the noise-free case is plotted at the
            # left edge and labelled, rather than silently dropped.
            x = [r["noise"] if r["noise"] > 0 else 8e-4 for r in sel]
            y = [r["chamfer"] for r in sel]
            ax.loglog(x, y, "o-", color=COL[m], lw=2, ms=6, markeredgecolor="white",
                      markeredgewidth=1.2, label=LAB[m], zorder=4)
            mark_bad_topology(ax, x, y, [r["euler_ok"] for r in sel])
        ax.axvline(8e-4, color=MUTED, lw=1, ls=":", zorder=1)
        ax.annotate("noise-free", xy=(8e-4, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -11), textcoords="offset points", rotation=90,
                    color=MUTED, fontsize=8, va="top")
        # A missing point would otherwise look like a plotting slip. On an exactly
        # cospherical sample the alpha test is degenerate and returns no surface.
        gap = [r for r in pick(rows, experiment="noise", shape=shape,
                               method="alpha_shape") if not np.isfinite(r["chamfer"])]
        if gap:
            ax.annotate("at zero noise the alpha shape returns nothing:\n"
                        "every Delaunay circumradius is exactly the sphere radius",
                        xy=(0.98, 0.03), xycoords="axes fraction",
                        color=COL["alpha_shape"], fontsize=8.2, va="bottom", ha="right")
        ax.set_title(shape, fontsize=11.5, pad=8)
        ax.set_xlabel("noise standard deviation (scene units)", fontsize=9.5,
                      color=MUTED)
        style(ax)
        ax.set_xticks([8e-4, 0.002, 0.005, 0.01, 0.02, 0.04])
        ax.set_xticklabels(["0", "0.002", "0.005", "0.01", "0.02", "0.04"], fontsize=9)
        ax.minorticks_off()
    axes[0].set_ylabel("Chamfer distance to the true surface", fontsize=9.5, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=CRIT, marker="x", ls="", ms=9, mew=2.0))
    l.append("wrong topology")
    fig.legend(h, l, frameon=False, fontsize=9.5, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("The global solve is what survives noise, not the better clean score",
                 fontsize=12.5, y=1.0)
    plt.tight_layout()
    p = os.path.join(OUT, "noise.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def alpha_and_grid_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=140)

    ax = axes[0]
    for shape, col in (("sphere", "#59a14f"), ("torus", "#b07aa1")):
        sel = sorted(pick(rows, experiment="alpha", shape=shape),
                     key=lambda r: r["alpha_over_spacing"])
        x = [r["alpha_over_spacing"] for r in sel]
        y = [r["chamfer"] for r in sel]
        ax.loglog(x, y, "o-", color=col, lw=2, ms=6, markeredgecolor="white",
                  markeredgewidth=1.2, label=shape, zorder=4)
        mark_bad_topology(ax, x, y, [r["euler_ok"] for r in sel])
    ax.set_xlabel("alpha / mean sample spacing", fontsize=9.5, color=MUTED)
    ax.set_ylabel("Chamfer distance to the true surface", fontsize=9.5, color=MUTED)
    ax.set_title("Alpha shapes: three settings out of sixteen\nget the topology right",
                 fontsize=11)
    ax.set_xticks([1.5, 2, 3, 4, 6, 8, 12, 20])
    ax.set_xticklabels(["1.5", "2", "3", "4", "6", "8", "12", "20"], fontsize=9)
    ax.minorticks_off()
    style(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")

    ax = axes[1]
    for m in ("hoppe", "poisson_fft"):
        for shape, ls in (("sphere", "-"), ("torus", "--")):
            sel = sorted(pick(rows, experiment="grid", shape=shape, method=m),
                         key=lambda r: r["ms"])
            ax.loglog([r["ms"] for r in sel], [r["chamfer"] for r in sel],
                      "o" + ls, color=COL[m], lw=2, ms=6, markeredgecolor="white",
                      markeredgewidth=1.2, label=f"{LAB[m]} — {shape}", zorder=4)
    ax.set_xlabel("reconstruction time (ms)", fontsize=9.5, color=MUTED)
    ax.set_title("At a matched grid the FFT solve is 3-7x faster\n"
                 "and 1.2-3.6x less accurate", fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    plt.tight_layout()
    p = os.path.join(OUT, "alpha_and_grid.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def mesh_figure():
    """The torus as each method reconstructs it from the same 2,000 points."""
    import run as R
    sh = R.SHAPES["torus"]
    sp = float(np.sqrt(sh.area / 2000))
    specs = [("convex_hull", {}), ("alpha_shape", {"alpha": 4.0 * sp}),
             ("hoppe", {"n_grid": 64}), ("poisson_fft", {"n_grid": 64})]
    fig = plt.figure(figsize=(12.5, 3.5), dpi=140)
    rng = np.random.default_rng(R.SEED)
    pts = sh.sample(2000, rng) + rng.normal(scale=R.FLOOR, size=(2000, 3))
    for i, (m, kw) in enumerate(specs):
        extra = {}
        if m in R.NEEDS_NORMALS:
            extra["nrm"] = R.orient_normals(pts, R.pca_normals(pts, 18))
        v, f = R.METHODS[m](pts, sh, **extra, **kw)
        sc = R.score(v, f, sh, np.random.default_rng(1))
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        if len(f):
            ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], color="#9ec5f4",
                            edgecolor="#2a78d6", linewidth=0.06, shade=True)
        ok = "V-E+F = 0" if sc["euler_ok"] else f"V-E+F = {sc['euler']}"
        ax.set_title(f"{LAB[m]}\nChamfer {sc['chamfer']:.4f}   {ok}", fontsize=9.5,
                     color="#222" if sc["euler_ok"] else CRIT)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=32, azim=-62)
        ax.set_box_aspect((1, 1, 0.5))
    fig.suptitle("The same 2,000 points, four classical reconstructions",
                 fontsize=12.5, y=1.04)
    plt.tight_layout()
    p = os.path.join(OUT, "meshes.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def main():
    res = json.load(io.open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    for p in (density_figure(res), noise_figure(res), alpha_and_grid_figure(res),
              mesh_figure()):
        print("wrote", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
