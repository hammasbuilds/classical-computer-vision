"""Figures for 58_marching_cubes: convergence on log-log axes, and the meshes.

The convergence panel is the point of the project: on log-log axes a power law is a
straight line, and its slope is the order. A reference line of slope 2 is drawn so the
claim can be checked by eye rather than taken on trust.
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

BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#7a7873", "#e8e7e2"


def convergence_figure(res):
    shapes = sorted({r["shape"] for r in res["rows"]})
    fig, axes = plt.subplots(1, len(shapes), figsize=(10.5, 4.3), dpi=140)
    if len(shapes) == 1:
        axes = [axes]
    for ax, shape in zip(axes, shapes):
        sel = [r for r in res["rows"] if r["shape"] == shape and r["method"] == "lewiner"]
        sel.sort(key=lambda r: r["h"])
        h = np.array([r["h"] for r in sel])
        # Reference first and visibly offset: drawn ON the data it coincides with, it
        # vanished under the orange line, so the one line the reader is meant to compare
        # against was invisible.
        base = np.array([r["area_rel_err"] for r in sel])
        ref = base[0] * 0.42 * (h / h[0]) ** 2
        ax.loglog(h, ref, "--", color=MUTED, lw=1.4, zorder=1)
        ax.annotate("slope 2, for comparison", xy=(h[-2], ref[-2]), xytext=(0, -15),
                    textcoords="offset points", color=MUTED, fontsize=8.5, ha="center")
        for key, col, lab in (("volume_rel_err", BLUE, "volume"),
                              ("area_rel_err", ORANGE, "surface area")):
            e = np.array([r[key] for r in sel])
            ax.loglog(h, e, "o-", color=col, lw=2, ms=6,
                      markeredgecolor="white", markeredgewidth=1.2, label=lab)
            # Label inside the axes, left of the last point: labels hung off the right
            # edge and were clipped by the figure border.
            ax.annotate(lab, xy=(h[-1], e[-1]), xytext=(-8, 9),
                        textcoords="offset points", color=col, fontsize=9.5,
                        ha="right", va="bottom", fontweight="bold")
        o = res["convergence_order"][f"{shape}/lewiner"]
        ax.set_title(f"{shape}  —  fitted order {o['volume']:.2f} / {o['area']:.2f}",
                     fontsize=10.5)
        ax.set_xlabel("grid spacing h", fontsize=9.5, color=MUTED)
        ax.grid(True, which="both", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # Default log minor ticks printed 2x10-2 3x10-2 4x10-2 on top of each other.
        # Label the grid sizes that were actually run instead.
        ax.set_xticks(h)
        ax.set_xticklabels([f"{v:.3f}" for v in h], fontsize=8.5, rotation=45)
        ax.minorticks_off()
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("relative error", fontsize=9.5, color=MUTED)
    fig.suptitle("Error falls as the square of the grid spacing", fontsize=12.5, y=0.99)
    plt.tight_layout()
    p = os.path.join(OUT, "convergence.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def cost_figure(res):
    """What the accuracy costs in time - the other half of any resolution choice."""
    fig, ax = plt.subplots(figsize=(6.6, 4.1), dpi=140)
    for shape, col in (("sphere", BLUE), ("torus", ORANGE)):
        sel = [r for r in res["rows"] if r["shape"] == shape and r["method"] == "lewiner"]
        sel.sort(key=lambda r: r["ms"])
        ax.plot([r["ms"] for r in sel], [r["volume_rel_err"] * 100 for r in sel],
                "o-", color=col, lw=2, ms=6, markeredgecolor="white",
                markeredgewidth=1.2, label=shape)
        last = sel[-1]
        ax.annotate(shape, xy=(last["ms"], last["volume_rel_err"] * 100),
                    xytext=(6, 0), textcoords="offset points", color=col,
                    fontsize=9.5, va="center", fontweight="600")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("extraction time (ms)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("volume error (%)", fontsize=9.5, color=MUTED)
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Ten times the time buys about three times the accuracy", fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT, "cost.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def mesh_figure():
    """The extracted surfaces at three resolutions, so the numbers have a picture."""
    import run as R
    fig = plt.figure(figsize=(10.5, 3.6), dpi=140)
    shape = R.Torus(1.0, 0.35)
    for i, n in enumerate((16, 32, 96)):
        verts, faces, h, _ = R.extract(shape, n)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        color="#9ec5f4", edgecolor="#2a78d6", linewidth=0.08,
                        antialiased=True, shade=True)
        err = abs(R.mesh_volume(verts, faces) - shape.volume) / shape.volume
        ax.set_title(f"n={n}   volume error {err * 100:.2f}%", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=28, azim=-60)
        ax.set_box_aspect((1, 1, 0.55))
    plt.tight_layout()
    p = os.path.join(OUT, "meshes.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def main():
    res = json.load(io.open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    for p in (convergence_figure(res), cost_figure(res), mesh_figure()):
        print("wrote", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
