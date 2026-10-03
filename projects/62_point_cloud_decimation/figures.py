"""Figures for 62_point_cloud_decimation.

The first figure is the project's argument in one picture: the metric everyone quotes
(Chamfer back to the original cloud) separates the four methods by less than 2x, while
the covering radius separates them by more than 3x, and the density bias by nearly 10x.
The second is what each decimation costs. The third shows where the kept points actually
land.
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

COL = {"random": "#7a7873", "farthest_point": "#2a78d6",
       "voxel_nearest": "#eb6834", "voxel_centroid": "#59a14f"}
LAB = {"random": "random", "farthest_point": "farthest point",
       "voxel_nearest": "voxel (nearest)", "voxel_centroid": "voxel (centroid)"}
MUTED, GRID, CRIT = "#7a7873", "#e8e7e2", "#d03b3b"


def style(ax):
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def matched(rows, m):
    return sorted([r for r in rows if r["experiment"] == "matched"
                   and r["method"] == m], key=lambda r: r["n_kept"])


def three_metrics_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4), dpi=140)

    for key, ax, title, ylab in (
            ("chamfer_to_original", axes[0],
             "Chamfer back to the original cloud\n— spread 1.24x to 1.87x",
             "Chamfer distance"),
            ("covering_radius", axes[1],
             "Covering radius: the worst-served\npoint — spread up to 3.28x",
             "covering radius"),
            ("patch_bias", axes[2],
             "Density bias on the over-sampled patch\n(1.0 = kept in proportion to area)",
             "kept share ÷ area share")):
        for m in COL:
            s = matched(rows, m)
            ax.loglog([r["n_kept"] for r in s], [r[key] for r in s], "o-",
                      color=COL[m], lw=2, ms=5.5, markeredgecolor="white",
                      markeredgewidth=1.0, label=LAB[m], zorder=4)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("points kept", fontsize=9.5, color=MUTED)
        ax.set_ylabel(ylab, fontsize=9.5, color=MUTED)
        ax.set_xticks([300, 1000, 3000, 10000, 20000])
        ax.set_xticklabels(["300", "1k", "3k", "10k", "20k"], fontsize=9)
        ax.minorticks_off()
        style(ax)
    axes[2].axhline(1.0, color=CRIT, lw=1.4, ls=(0, (5, 4)), zorder=1)
    axes[2].axhline(res["patch"]["input_bias"], color="#9467bd", lw=1.3, ls=":",
                    zorder=1)
    axes[2].annotate(f"the input itself is at {res['patch']['input_bias']:.2f}",
                     xy=(0.03, res["patch"]["input_bias"]),
                     xycoords=("axes fraction", "data"), xytext=(0, -13),
                     textcoords="offset points", color="#9467bd", fontsize=8.5,
                     ha="left", va="top")
    axes[2].annotate("unbiased", xy=(0.97, 1.0), xycoords=("axes fraction", "data"),
                     xytext=(0, -12), textcoords="offset points", color=CRIT,
                     fontsize=8.5, ha="right", va="top")
    axes[2].set_yticks([1, 2, 5, 10])
    axes[2].set_yticklabels(["1", "2", "5", "10"], fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("The metric usually quoted is the one that cannot tell these apart",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "metrics.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def cost_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=140)

    ax = axes[0]
    for m in ("random", "voxel_nearest", "farthest_point"):
        s = sorted([r for r in rows if r["experiment"] == "scaling"
                    and r["method"] == m], key=lambda r: r["n_input"])
        e = res["time_exponents_n_at_least_21000"][m]
        ax.loglog([r["n_input"] for r in s], [r["ms"] for r in s], "o-", color=COL[m],
                  lw=2, ms=5.5, markeredgecolor="white", markeredgewidth=1.0,
                  label=f"{LAB[m]}   n^{e:.2f}")
    ax.set_xlabel("input points (one eighth kept)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("time (ms)", fontsize=9.5, color=MUTED)
    slow = max(r["ms"] for r in rows if r["experiment"] == "scaling"
               and r["method"] == "farthest_point")
    ax.set_title(f"Farthest point costs n², and it shows:\n"
                 f"{slow / 1000:.0f} seconds at 84,000 points", fontsize=10.5)
    ax.set_xticks([5250, 10500, 21000, 42000, 84000])
    ax.set_xticklabels(["5k", "10k", "21k", "42k", "84k"], fontsize=9)
    ax.minorticks_off()
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[1]
    s = sorted([r for r in rows if r["experiment"] == "voxel_size"
                and r["method"] == "voxel_centroid"], key=lambda r: r["voxel"])
    v = np.array([r["voxel"] for r in s])
    y = np.array([r["surface_dist_mean"] for r in s])
    ax.loglog(v, y, "o-", color=COL["voxel_centroid"], lw=2, ms=5.5,
              markeredgecolor="white", markeredgewidth=1.0, label="mean")
    ax.loglog(v, [r["surface_dist_max"] for r in s], "s--",
              color=COL["voxel_centroid"], lw=1.5, ms=4.5, alpha=0.75, label="worst")
    ref = y[-1] * 0.25 * (v / v[-1]) ** 2
    ax.loglog(v, ref, "--", color=MUTED, lw=1.3,
              label="slope 2 — curvature alone")
    # The other three sit at 3.9e-16, which cannot share a log axis with this without
    # emptying it, so the number is stated instead of plotted.
    off = max(res["max_distance_off_surface"][m]
              for m in ("random", "farthest_point", "voxel_nearest"))
    ax.annotate(f"the other three keep original points, so they land\n"
                f"{off:.1e} off the surface — floating-point dust",
                xy=(0.97, 0.18), xycoords="axes fraction", color=CRIT, fontsize=8.5,
                ha="right", va="bottom")
    ax.set_xlabel("voxel size", fontsize=9.5, color=MUTED)
    ax.set_ylabel("distance off the true surface", fontsize=9.5, color=MUTED)
    ax.set_title("A voxel centroid is not on the surface,\nand the gap grows with the "
                 "voxel", fontsize=10.5)
    ax.set_ylim(3e-7, 4e-2)
    ax.minorticks_off()
    ax.set_xticks([0.01, 0.02, 0.05, 0.12])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.12"], fontsize=9)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plt.tight_layout()
    p = os.path.join(OUT, "cost.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def scatter_figure(res):
    """Where the kept points land: the same 1250 from the same 42,000."""
    import run as R
    shape = R.Torus()
    pts, in_patch = shape.sample(40000, np.random.default_rng(R.SEED), n_patch=2000)
    target = 1250
    voxel = res["voxel_sizes_for_target"][str(target)]
    specs = [("random", dict(target=target)),
             ("farthest_point", dict(target=target)),
             ("voxel", dict(voxel=voxel, mode="nearest"))]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), dpi=140)
    for ax, (m, kw) in zip(axes, specs):
        r = R.run_one(pts, in_patch, shape, m, **kw)
        rng = np.random.default_rng(R.SEED)
        if m == "random":
            idx = R.random_sample(pts, target, rng)
        elif m == "farthest_point":
            idx = R.farthest_point(pts, target, seed=R.SEED)
        else:
            idx, _ = R.voxel_grid(pts, voxel, mode="nearest")
        kept = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.6, color="#dedcd6", linewidths=0,
                   zorder=1)
        inp = shape.in_patch(kept)
        ax.scatter(kept[~inp, 0], kept[~inp, 1], s=5.0, color=COL[r["method"]],
                   linewidths=0, zorder=3)
        ax.scatter(kept[inp, 0], kept[inp, 1], s=12.0, color=CRIT, linewidths=0,
                   zorder=4)
        ax.set_title(f"{LAB[r['method']]}\ncovering radius {r['covering_radius']:.4f}"
                     f"   patch bias {r['patch_bias']:.2f}", fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("The same 1,250 points kept from the same 42,000, seen from above — "
                 "red marks the over-sampled patch", fontsize=12, y=1.0)
    plt.tight_layout()
    p = os.path.join(OUT, "scatter.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def main():
    res = json.load(io.open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    for p in (three_metrics_figure(res), cost_figure(res), scatter_figure(res)):
        print("wrote", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
