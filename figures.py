"""
Generate the figures for the paper as vector PDFs in ./figures/.

    python3 figures.py

Radii come from the full recursion (all j < k) for every N used here.
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SQ2 = math.sqrt(2)
CAP = SQ2 - 1
A_VAL = 3 - 2 * SQ2
B_VAL = SQ2 - 1

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#111111", "#52514e", "#9a9993", "#e6e5e0"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ recursion

def radii(N):
    """Full recursion: r_0 .. r_N (r_N is the interior ball)."""
    r = np.empty(N + 1)
    r[0] = 0.5
    for k in range(1, N):
        p = N - k
        rj = r[:k]
        s = k - np.arange(k)
        if p >= 2:
            d = 4 * p * rj * rj - (p - 1) * s * (1 - rj) ** 2
            b = np.where(d >= 0,
                         ((p + 1) * rj - np.sqrt(np.maximum(d, 0))) / (p - 1),
                         np.inf)
        else:
            b = s * (1 - rj) ** 2 / (4 * rj)
        r[k] = min(CAP, b.min())
    j = np.arange(N)
    r[N] = (np.sqrt(N - j) * (1 - r[:N]) - r[:N]).min()
    return r


def xstar(p):
    return math.sqrt(p - 1) / (2 * math.sqrt(p) + math.sqrt(p - 1))


# ------------------------------------------------------------------ figure 1

def fig_construction():
    fig = plt.figure(figsize=(6.4, 3.0))

    # (a) N = 2
    ax = fig.add_subplot(1, 2, 1)
    r = radii(2)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec=INK, lw=0.9))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(plt.Circle((sx * (1 - r[0]), sy * (1 - r[0])), r[0],
                                    fc=BLUE, alpha=0.30, ec=BLUE, lw=0.8))
    for (cx, cy) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ax.add_patch(plt.Circle((cx * (1 - r[1]), cy * (1 - r[1])), r[1],
                                fc=ORANGE, alpha=0.35, ec=ORANGE, lw=0.8))
    ax.add_patch(plt.Circle((0, 0), r[2], fc=AQUA, alpha=0.35, ec=AQUA, lw=0.8))
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"(a) $N=2$:  $r_0=\frac{1}{2}$,  $r_1=\frac{1}{8}$,  $r_2=\frac{\sqrt{2}-1}{2}$")

    # (b) N = 3
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    r = radii(3)
    u, v = np.mgrid[0:2 * np.pi:28j, 0:np.pi:16j]
    sphere = (np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v))
    colors = [BLUE, ORANGE, "#8a6fd1", AQUA]
    import itertools
    for k in range(4):
        for free in itertools.combinations(range(3), k):
            fixed = [i for i in range(3) if i not in free]
            for signs in itertools.product((-1, 1), repeat=len(fixed)):
                c = [0.0, 0.0, 0.0]
                for i, sg in zip(fixed, signs):
                    c[i] = sg * (1 - r[k])
                ax.plot_surface(c[0] + r[k] * sphere[0], c[1] + r[k] * sphere[1],
                                c[2] + r[k] * sphere[2], color=colors[k],
                                alpha=0.55 if k else 0.18, linewidth=0,
                                shade=True, antialiased=True)
    # cube edges
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            ax.plot([-1, 1], [s1, s1], [s2, s2], color=MUTED, lw=0.6)
            ax.plot([s1, s1], [-1, 1], [s2, s2], color=MUTED, lw=0.6)
            ax.plot([s1, s1], [s2, s2], [-1, 1], color=MUTED, lw=0.6)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=35)
    ax.set_axis_off()
    ax.set_title(r"(b) $N=3$: corners (faint), edges, faces, interior", pad=-4)
    fig.savefig(f"{OUT}/fig1_construction.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2

def fig_interior(nmax=1000):
    Ns = np.arange(2, nmax + 1)
    vals = np.array([radii(int(N))[N] for N in Ns])
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    corner = (np.sqrt(Ns) - 1) / 2
    m = corner <= 0.62
    ax.plot(Ns[m], corner[m], ls=(0, (4, 3)), color=MUTED, lw=1.1)
    ax.annotate(r"corners only: $(\sqrt{N}-1)/2$", xy=(7, (math.sqrt(7) - 1) / 2),
                xytext=(40, 0.585), color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    isA = np.isclose(vals, A_VAL, atol=1e-9)
    isB = np.isclose(vals, B_VAL, atol=1e-9)
    other = ~(isA | isB)
    ax.plot(Ns[isB], vals[isB], "|", ms=7, mew=1.3, color=ORANGE,
            label=r"$r_N=\sqrt{2}-1$")
    ax.plot(Ns[isA], vals[isA], "|", ms=7, mew=1.3, color=BLUE,
            label=r"$r_N=3-2\sqrt{2}$")
    ax.plot(Ns[other], vals[other], "o", ms=3.5, color=INK, label="other values")
    ax.axvline(597.5, color=INK2, lw=0.7, ls=":")
    ax.text(560, 0.29, r"$N=598$", color=INK2, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1.8, nmax * 1.05)
    ax.set_ylim(0.12, 0.63)
    ax.set_xlabel(r"dimension $N$ (log scale)")
    ax.set_ylabel(r"interior radius $r_N$")
    ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.6, 1.02), ncol=3)
    fig.savefig(f"{OUT}/fig2_interior.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3

def fig_top(Ns=(597, 598, 2000), P=45):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(CAP, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    styles = [(BLUE, 1.3, "-"), (ORANGE, 2.6, "-"), (AQUA, 1.1, "-")]
    for (N, (c, lw, ls)) in zip(Ns, styles):
        r = radii(N)
        p = np.arange(1, P + 1)
        ax.plot(p, r[N - p], color=c, lw=lw, ls=ls, label=fr"$N={N}$",
                alpha=0.9 if N != 598 else 0.55)
    ax.invert_xaxis()
    ax.set_xlabel(r"$p=N-k$ (distance below the interior ball)")
    ax.set_ylabel(r"radius $r_k$")
    ax.set_title(r"(a) top of the stack; $N=598$ and $N=2000$ coincide")
    ax.legend(frameon=False, loc="lower left", ncol=3)

    ax = axes[1]
    N = 2000
    r = radii(N)
    p = np.arange(300, 361)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(1 / 3, color=INK2, lw=0.8, ls=":")
    ax.plot(p, [xstar(int(q)) for q in p], color=ORANGE, lw=0.9, ls="--")
    ax.plot(p, r[N - p], "-o", color=BLUE, lw=1.0, ms=2.6)
    ax.invert_xaxis()
    ax.set_ylim(0.315, 0.42)
    ax.set_xlabel(r"$p=N-k$")
    ax.set_title(r"(b) cascades, $N=2000$")
    ax.text(0.97, 0.97, "dotted: fixed point $1/3$\ndashed: threshold $x^*(p)$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig3_top.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figure 4

def fig_caps(nmin=10, nmax=800, P=120):
    Ns = np.arange(nmin, nmax + 1)
    grid = np.full((len(Ns), P), np.nan)
    interior = np.zeros(len(Ns))
    for i, N in enumerate(Ns):
        r = radii(int(N))
        top = r[:N][::-1][:P]                 # p = 1, 2, ...
        grid[i, :len(top)] = np.isclose(top, CAP, atol=1e-13)
        v = r[N]
        interior[i] = 1 if abs(v - A_VAL) < 1e-9 else (2 if abs(v - B_VAL) < 1e-9 else 0)
    fig = plt.figure(figsize=(6.4, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[36, 1], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharey=ax)
    cmap = ListedColormap(["#eef3fb", BLUE])
    cmap.set_bad("white")
    ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest", rasterized=True,
              extent=[0.5, P + 0.5, nmin - 0.5, nmax + 0.5])
    ax.set_xlim(P + 0.5, 0.5)
    ax.set_xlabel(r"$p=N-k$ (top of the stack at right)")
    ax.set_ylabel(r"dimension $N$")
    axs.imshow(interior[:, None], aspect="auto", origin="lower",
               cmap=ListedColormap(["#c9c8c2", BLUE, ORANGE]), vmin=0, vmax=2,
               interpolation="nearest", rasterized=True,
               extent=[0, 1, nmin - 0.5, nmax + 0.5])
    axs.set_xticks([])
    axs.tick_params(labelleft=False)
    for s in ("left", "bottom"):
        axs.spines[s].set_visible(False)
    axs.set_title(r"$r_N$", fontsize=8)
    fig.savefig(f"{OUT}/fig4_caps.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_construction(); print("fig1")
    fig_interior(); print("fig2")
    fig_top(); print("fig3")
    fig_caps(); print("fig4")
