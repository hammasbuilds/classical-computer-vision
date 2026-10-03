"""Figures for 59_icp_registration.

Three panels, one per claim: the convergence basin, the noise response, and what
trimming does to partial overlap. Each plots the measured error against the thing that
was varied, with the 1-degree success threshold drawn so "worked" and "did not work"
are visible rather than asserted.
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
BLUE, ORANGE, MUTED, GRID, CRIT = "#2a78d6", "#eb6834", "#7a7873", "#e8e7e2", "#d03b3b"


def style(ax):
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def threshold(ax, label_x, label=False):
    ax.axhline(1.0, color=CRIT, lw=1.4, ls=(0, (5, 4)), zorder=1)
    if label:
        ax.annotate("1° — treated as success", xy=(label_x, 1.0), xytext=(0, 7),
                    textcoords="offset points", color=CRIT, fontsize=8.5, ha="right")


def main():
    res = json.load(io.open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    rows = res["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=140)

    # --- 1. convergence basin ---------------------------------------------------
    ax = axes[0]
    for mode, col, lab in (("point", BLUE, "point-to-point"),
                           ("plane", ORANGE, "point-to-plane")):
        sel = sorted([r for r in rows if r["experiment"] == "basin"
                      and r["mode"] == mode], key=lambda r: r["angle"])
        x = [r["angle"] for r in sel]
        y = [max(r["rot_err_deg"], 1e-4) for r in sel]
        ax.semilogy(x, y, "o-", color=col, lw=2, ms=6, markeredgecolor="white",
                    markeredgewidth=1.2, label=lab)
    threshold(ax, 90, label=True)
    ax.set_xlabel("initial misalignment (degrees)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("recovered rotation error (degrees)", fontsize=9.5, color=MUTED)
    ax.set_title("Point-to-plane tolerates twice the misalignment", fontsize=10.5)
    style(ax)

    # --- 2. noise ---------------------------------------------------------------
    ax = axes[1]
    for mode, col, lab in (("point", BLUE, "point-to-point"),
                           ("plane", ORANGE, "point-to-plane")):
        sel = sorted([r for r in rows if r["experiment"] == "noise"
                      and r["mode"] == mode], key=lambda r: r["noise"])
        x = [r["noise"] for r in sel]
        y = [max(r["rot_err_deg"], 1e-4) for r in sel]
        ax.semilogy(x, y, "o-", color=col, lw=2, ms=6, markeredgecolor="white",
                    markeredgewidth=1.2, label=lab)
    threshold(ax, 0.05)
    ax.set_xlabel("noise standard deviation (scene units)", fontsize=9.5, color=MUTED)
    ax.set_title("and pays for it under noise", fontsize=10.5)
    style(ax)

    # --- 3. overlap, trimmed and not --------------------------------------------
    ax = axes[2]
    for trim, col, lab in ((1.0, BLUE, "all pairs"), (0.8, ORANGE, "closest 80% kept")):
        sel = sorted([r for r in rows if r["experiment"] == "overlap"
                      and r.get("trim") == trim], key=lambda r: -r["overlap"])
        x = [r["overlap"] * 100 for r in sel]
        y = [max(r["rot_err_deg"], 1e-4) for r in sel]
        ax.semilogy(x, y, "o-", color=col, lw=2, ms=6, markeredgecolor="white",
                    markeredgewidth=1.2, label=lab)
    threshold(ax, 100)
    ax.invert_xaxis()
    ax.set_xlabel("overlap between the two clouds (%)", fontsize=9.5, color=MUTED)
    ax.set_title("Trimming buys 15 points of overlap", fontsize=10.5)
    style(ax)

    fig.suptitle("ICP: the same algorithm fails three different ways",
                 fontsize=12.5, y=1.02)
    for ax in axes:
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    plt.tight_layout()
    p = os.path.join(OUT, "icp.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print("wrote", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
