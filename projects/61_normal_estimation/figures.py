"""Figures for 61_normal_estimation.

Three figures, one per claim. The first shows the two regimes - a k^-1 fall under noise
and a flat floor when clean - with a reference slope drawn so the exponent can be
checked by eye. The second shows the three fitted power laws on their own axes. The
third is the cost of the classical cure for curvature bias.
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
MUTED, GRID, CRIT = "#7a7873", "#e8e7e2", "#d03b3b"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax):
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def noise_colours(noises):
    cm = plt.get_cmap("viridis")
    return {ns: cm(0.08 + 0.82 * i / max(len(noises) - 1, 1))
            for i, ns in enumerate(noises)}


def pick(rows, **kw):
    return sorted([r for r in rows if all(r.get(a) == b for a, b in kw.items())],
                  key=lambda r: r["k"])


def k_sweep_figure(res):
    rows = res["rows"]
    noises = res["noise_values"]
    col = noise_colours(noises)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6), dpi=140)
    for ax, shape in zip(axes, ("plane", "sphere", "torus")):
        for ns in noises:
            sel = pick(rows, experiment="k_noise", shape=shape, noise=ns)
            k = np.array([r["k"] for r in sel], float)
            y = np.array([r["mean_deg"] for r in sel], float)
            if shape == "plane" and ns == 0.0:
                continue                     # identically zero; stated in the caption
            ax.loglog(k, y, "o-", color=col[ns], lw=1.9, ms=4.5,
                      markeredgecolor="white", markeredgewidth=0.8,
                      label=f"σ = {ns:g}", zorder=4)
        if shape != "plane":
            sel = pick(rows, experiment="k_noise", shape=shape, noise=0.0)
            b = res["best_k"][shape]
            for ns in noises:
                r = next(x for x in pick(rows, experiment="k_noise", shape=shape,
                                         noise=ns) if x["k"] == b[str(ns)]["k"])
                ax.plot([r["k"]], [r["mean_deg"]], "v", color=CRIT, ms=7,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=7)
            del sel
        # reference slope, drawn clear of the data so it is visible on its own
        kk = np.array([16.0, 2048.0])
        base = next(r for r in pick(rows, experiment="k_noise", shape=shape,
                                    noise=0.02) if r["k"] == 16)["mean_deg"]
        ax.loglog(kk, base * 0.35 * (kk / 16.0) ** -1.0, "--", color=MUTED, lw=1.3,
                  zorder=2)
        ax.annotate("slope −1", xy=(kk[0], base * 0.35), xytext=(9, 13),
                    textcoords="offset points", color=MUTED, fontsize=8.5,
                    ha="left", va="bottom")
        ax.set_title(shape, fontsize=11.5, pad=8)
        ax.set_xlabel("k (nearest neighbours)", fontsize=9.5, color=MUTED)
        ax.set_xticks([4, 16, 64, 256, 1024])
        ax.set_xticklabels(["4", "16", "64", "256", "1024"], fontsize=9)
        ax.minorticks_off()
        style(ax)
    axes[0].set_ylabel("mean angular error (degrees)", fontsize=9.5, color=MUTED)
    axes[0].annotate("σ = 0 is exactly 0.000°\nat every k, so it cannot\nbe drawn on a "
                     "log axis", xy=(0.03, 0.04), xycoords="axes fraction",
                     fontsize=8.5, color=MUTED, va="bottom")
    # handles come from the torus panel: it is the one that draws sigma = 0
    h, l = axes[2].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=CRIT, marker="v", ls="", ms=7))
    l.append("best k")
    fig.legend(h, l, frameon=False, fontsize=9, ncol=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Under noise the error falls as k⁻¹; with clean points it does not "
                 "fall at all", fontsize=12.5, y=1.0)
    plt.tight_layout()
    p = os.path.join(OUT, "k_sweep.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def laws_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), dpi=140)

    # --- error against sigma on the plane: predicted exponent +1
    ax = axes[0]
    for k, c in ((16, BLUE), (64, "#59a14f"), (256, ORANGE)):
        sel = sorted([r for r in rows if r["experiment"] == "k_noise"
                      and r["shape"] == "plane" and r["k"] == k and r["noise"] > 0],
                     key=lambda r: r["noise"])
        e = res["exponents"][f"noise_vs_sigma/plane/k={k}"]
        ax.loglog([r["noise"] for r in sel], [r["mean_deg"] for r in sel], "o-",
                  color=c, lw=2, ms=5.5, markeredgecolor="white", markeredgewidth=1.0,
                  label=f"k = {k}   fitted {e:+.2f}")
    ax.set_xlabel("noise standard deviation σ", fontsize=9.5, color=MUTED)
    ax.set_ylabel("mean angular error (degrees)", fontsize=9.5, color=MUTED)
    ax.set_title("Plane: error is linear in σ\n(predicted exponent +1)", fontsize=10.5)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # --- clean error against point count: predicted exponent -0.5
    ax = axes[1]
    for shape, c in (("sphere", BLUE), ("torus", ORANGE)):
        sel = sorted([r for r in rows if r["experiment"] == "density"
                      and r["shape"] == shape], key=lambda r: r["n_pts"])
        e = res["exponents"][f"clean_vs_density/{shape}"]
        ax.loglog([r["n_pts"] for r in sel], [r["mean_deg"] for r in sel], "o-",
                  color=c, lw=2, ms=5.5, markeredgecolor="white", markeredgewidth=1.0,
                  label=f"{shape}   fitted {e:+.2f}")
    ax.set_xlabel("points on the surface (k fixed at 32)", fontsize=9.5, color=MUTED)
    ax.set_title("Clean: the floor is set by density,\nnot by k (predicted −0.5)",
                 fontsize=10.5)
    ax.set_xticks([2500, 5000, 10000, 20000, 40000])
    ax.set_xticklabels(["2.5k", "5k", "10k", "20k", "40k"], fontsize=9)
    ax.set_yticks([0.05, 0.1, 0.3, 1.0, 3.0])
    ax.set_yticklabels(["0.05", "0.1", "0.3", "1.0", "3.0"], fontsize=9)
    ax.minorticks_off()
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # --- the optimum moves with noise
    ax = axes[2]
    for shape, c in (("sphere", BLUE), ("torus", ORANGE)):
        d = res["best_k"][shape]
        xs = [float(s) for s in d]
        ys = [d[s]["k"] for s in d]
        ax.semilogy([x if x > 0 else 2e-4 for x in xs], ys, "o-", color=c, lw=2,
                    ms=5.5, markeredgecolor="white", markeredgewidth=1.0, label=shape)
    ax.set_xscale("log")
    ax.set_xticks([2e-4, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02])
    ax.set_xticklabels(["0", "0.0005", "0.001", "0.002", "0.005", "0.01", "0.02"],
                       fontsize=8, rotation=38, ha="right")
    ax.set_yticks([96, 128, 192, 256, 512, 1024, 2048])
    ax.set_yticklabels(["96", "128", "192", "256", "512", "1024", "2048"], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("noise standard deviation σ", fontsize=9.5, color=MUTED)
    ax.set_title("The best k grows with noise\n(torus: 96 → 256)", fontsize=10.5)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle("Three predictions from the geometry, three fitted exponents",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "laws.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def jet_and_radius_figure(res):
    rows = res["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), dpi=140)
    for ax, shape in zip(axes[:2], ("sphere", "torus")):
        for ns, c in ((0.0, "#59a14f"), (0.005, BLUE), (0.02, ORANGE)):
            a = [r for r in pick(rows, experiment="k_noise", shape=shape, noise=ns)
                 if r["k"] in (8, 16, 32, 64, 128, 256, 512, 1024)]
            b = pick(rows, experiment="jet", shape=shape, noise=ns)
            ax.loglog([r["k"] for r in a], [r["mean_deg"] for r in a], "o-", color=c,
                      lw=2, ms=5, markeredgecolor="white", markeredgewidth=1.0,
                      label=f"plane fit, σ = {ns:g}")
            ax.loglog([r["k"] for r in b], [r["mean_deg"] for r in b], "s--", color=c,
                      lw=1.6, ms=4.5, markeredgecolor="white", markeredgewidth=1.0,
                      label=f"quadratic jet, σ = {ns:g}")
        ax.set_title(shape, fontsize=11.5, pad=8)
        ax.set_xlabel("k (nearest neighbours)", fontsize=9.5, color=MUTED)
        ax.set_xticks([8, 32, 128, 512])
        ax.set_xticklabels(["8", "32", "128", "512"], fontsize=9)
        ax.minorticks_off()
        style(ax)
    axes[0].set_ylabel("mean angular error (degrees)", fontsize=9.5, color=MUTED)
    axes[0].legend(frameon=False, fontsize=7.6, loc="upper right", ncol=1)

    ax = axes[2]
    for shape, c in (("sphere", BLUE), ("torus", ORANGE)):
        for ns, ls, mk in ((0.0, "-", "o"), (0.005, "--", "s")):
            sel = sorted([r for r in rows if r["experiment"] == "radius"
                          and r["shape"] == shape and r["noise"] == ns],
                         key=lambda r: r["neighbours_used"])
            ax.loglog([r["neighbours_used"] for r in sel],
                      [r["mean_deg"] for r in sel], mk + ls, color=c, lw=1.8, ms=5,
                      markeredgecolor="white", markeredgewidth=1.0,
                      label=f"{shape}, σ = {ns:g}")
    ax.set_xlabel("mean neighbours inside the ball", fontsize=9.5, color=MUTED)
    ax.set_title("Fixed-radius ball: the same curve,\nbut the radius must be chosen",
                 fontsize=10.5)
    ax.minorticks_off()
    ax.set_xticks([4, 10, 30, 100, 200])
    ax.set_xticklabels(["4", "10", "30", "100", "200"], fontsize=9)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.suptitle("The quadratic fit is 2090× better on clean points and 1.9× better "
                 "at σ = 0.02", fontsize=12.5, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "jet_and_radius.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def main():
    res = json.load(io.open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    for p in (k_sweep_figure(res), laws_figure(res), jet_and_radius_figure(res)):
        print("wrote", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
