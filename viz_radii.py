"""
Visualize the greedy face-ball radii in the hypercube [-1, 1]^N.

Two views:

  1) trajectory  - r_k against k for one or a few fixed N
       python3 viz_radii.py trajectory 597 598 2000
       python3 viz_radii.py trajectory 597 598 2000 --from-top 60
           (x-axis becomes p = N - k, so different N line up at the top)

  2) heatmap     - N on the vertical axis, layer on the horizontal axis
       python3 viz_radii.py heatmap --nmin 10 --nmax 800
       python3 viz_radii.py heatmap --nmin 10 --nmax 800 --from-top 80
       python3 viz_radii.py heatmap --nmin 10 --nmax 800 --show caps

     --show radius  colors each cell by r_k (single-hue scale)
     --show caps    colors only the capped layers (the sawtooth skeleton)
     A strip at the right edge marks each N's interior radius:
     a = 3 - 2sqrt2, B = sqrt2 - 1, gray = anything else.

Add --out FILE.png to save instead of opening a window.

Radii come from the full recursion (every j < k) when N <= --exact-max
(default 1500), and from the reduced one-variable map above that:
    r_k = min(sqrt2 - 1, f_{N-k}(r_{k-1}))
which matches the full recursion exactly for 10 <= N <= 600 (checked).
"""

import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SQ2 = math.sqrt(2)
CAP = SQ2 - 1
A_VAL = 3 - 2 * SQ2
B_VAL = SQ2 - 1

# Colors (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # up to three overlaid N
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
OTHER = "#c9c8c2"


# ---------------------------------------------------------------- recursion

def solve_full(N):
    """Full recursion: every lower layer j < k is checked."""
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


def _f(p, x):
    if p == 1:
        return (1 - x) ** 2 / (4 * x)
    d = 4 * p * x * x - (p - 1) * (1 - x) ** 2
    return math.inf if d < 0 else ((p + 1) * x - math.sqrt(d)) / (p - 1)


def solve_1d(N):
    """Reduced map: only the cap or the layer directly below binds."""
    r = np.empty(N + 1)
    r[0] = 0.5
    for k in range(1, N):
        r[k] = min(CAP, _f(N - k, r[k - 1]))
    r[N] = min(math.sqrt(N - j) * (1 - r[j]) - r[j] for j in (N - 1, N - 2))
    return r


def radii(N, exact_max=1500):
    if N < 10 or N <= exact_max:
        return solve_full(N)
    return solve_1d(N)


def interior_label(x):
    if abs(x - A_VAL) < 1e-9:
        return "a"
    if abs(x - B_VAL) < 1e-9:
        return "B"
    return ""


# ---------------------------------------------------------------- styling

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def finish(fig, out):
    fig.patch.set_facecolor(SURFACE)
    if out:
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        print(f"saved {out}")
    else:
        plt.show()


# ---------------------------------------------------------------- view 1

def plot_trajectory(Ns, from_top=None, exact_max=1500, out=None):
    if len(Ns) > 3:
        raise SystemExit("Overlay at most 3 values of N; run again for more.")
    fig, ax = plt.subplots(figsize=(11, 5))
    style_axes(ax)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.axhline(CAP, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.0, CAP, "  cap  sqrt2 - 1", transform=ax.get_yaxis_transform(),
            color=INK_2, fontsize=9, va="center", ha="left", clip_on=False)

    for i, N in enumerate(Ns):
        r = radii(N, exact_max)
        k = np.arange(N + 1)
        x = (N - k) if from_top else k
        keep = (x <= from_top) if from_top else np.ones_like(k, bool)
        layers = keep & (k < N)
        color = SERIES[i]
        lab = interior_label(r[N])
        name = f"N = {N}" + (f"   (interior {lab})" if lab else f"   (interior {r[N]:.4f})")

        ax.plot(x[layers], r[layers], color=color, linewidth=2 if len(Ns) == 1 else 1.6,
                alpha=0.95, label=name, zorder=3)
        capped = layers & np.isclose(r, CAP, atol=1e-13)
        ax.plot(x[capped], r[capped], "o", ms=5, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        # interior ball as its own marker at k = N (p = 0)
        ax.plot([x[N]], [r[N]], marker="D", ms=9, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5,
                linestyle="none")

    if from_top:
        ax.set_xlabel("p = N - k   (distance from the top; p = 0 is the interior ball)")
        ax.invert_xaxis()
    else:
        ax.set_xlabel("layer k   (number of free coordinates; k = N is the interior ball)")
    ax.set_ylabel("radius")
    ttl = "Radii by layer" + (f", top {from_top} layers" if from_top else "")
    ax.set_title(ttl + "   (dots = capped layers, diamond = interior ball)",
                 fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK)
    finish(fig, out)


# ---------------------------------------------------------------- view 2

def plot_heatmap(nmin, nmax, from_top=None, show="radius", exact_max=1500, out=None):
    Ns = np.arange(nmin, nmax + 1)
    width = from_top + 1 if from_top else nmax + 1
    grid = np.full((len(Ns), width), np.nan)
    interior = np.zeros(len(Ns))       # 0 = other, 1 = a, 2 = B

    for row, N in enumerate(Ns):
        r = radii(int(N), exact_max)
        lab = interior_label(r[N])
        interior[row] = {"a": 1, "B": 2}.get(lab, 0)
        vals = r[:N]                    # layers only; interior shown in the strip
        if from_top:
            seg = vals[::-1][:from_top]          # layer N-1 first, i.e. p = 1
            grid[row, 1:1 + len(seg)] = seg      # column p = 1..from_top
        else:
            grid[row, :N] = vals
        if row % 100 == 0:
            print(f"  N = {N}", flush=True)

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1], sharey=ax)
    style_axes(ax)
    style_axes(ax_s)

    extent = [-0.5, width - 0.5, nmin - 0.5, nmax + 0.5]
    if show == "caps":
        capped = np.where(np.isnan(grid), np.nan,
                          np.isclose(grid, CAP, atol=1e-13).astype(float))
        cmap = ListedColormap(["#eef3fb", SERIES[0]])
        cmap.set_bad(SURFACE)
        ax.imshow(capped, aspect="auto", origin="lower", extent=extent,
                  cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("Capped layers (blue) vs. layers pinned by the one below (pale)",
                     fontsize=11, loc="left")
    else:
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad(SURFACE)
        lo = np.nanpercentile(grid, 2)   # clip the few tiny top radii so the bulk has contrast
        im = ax.imshow(grid, aspect="auto", origin="lower", extent=extent,
                       cmap=cmap, vmin=lo, vmax=CAP, interpolation="nearest")
        cb = fig.colorbar(im, ax=[ax, ax_s], fraction=0.025, pad=0.06)
        cb.set_label("radius  (darkest = cap sqrt2 - 1)", color=INK_2)
        cb.ax.tick_params(colors=INK_2, labelsize=8)
        cb.outline.set_visible(False)
        ax.set_title("Layer radii across dimensions", fontsize=11, loc="left")

    if from_top:
        ax.set_xlabel("p = N - k   (distance from the top)")
        ax.set_xlim(width - 0.5, 0.5)       # top of the stack on the right
    else:
        ax.set_xlabel("layer k")
    ax.set_ylabel("dimension N")

    strip_cmap = ListedColormap([OTHER, SERIES[0], SERIES[1]])
    ax_s.imshow(interior[:, None], aspect="auto", origin="lower",
                extent=[0, 1, nmin - 0.5, nmax + 0.5], cmap=strip_cmap,
                vmin=0, vmax=2, interpolation="nearest")
    ax_s.set_xticks([])
    ax_s.tick_params(labelleft=False)
    ax_s.set_title("interior", fontsize=9, color=INK_2)
    for side in ("left", "bottom"):
        ax_s.spines[side].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(color=SERIES[0], label="interior a = 3 - 2sqrt2"),
               Patch(color=SERIES[1], label="interior B = sqrt2 - 1"),
               Patch(color=OTHER, label="interior other")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=9, bbox_to_anchor=(0.45, -0.01))
    for t in leg.get_texts():
        t.set_color(INK)
    finish(fig, out)


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trajectory", help="r_k vs k for fixed N")
    t.add_argument("N", type=int, nargs="+")
    t.add_argument("--from-top", type=int, default=None,
                   help="show only the top P layers, x-axis p = N - k")

    h = sub.add_parser("heatmap", help="N vs layer, colored by radius")
    h.add_argument("--nmin", type=int, default=10)
    h.add_argument("--nmax", type=int, default=600)
    h.add_argument("--from-top", type=int, default=None,
                   help="align rows at the top of the stack, show top P layers")
    h.add_argument("--show", choices=["radius", "caps"], default="radius")

    for p in (t, h):
        p.add_argument("--exact-max", type=int, default=1500,
                       help="use the full recursion up to this N (slower, exact)")
        p.add_argument("--out", default=None, help="save to this PNG instead of showing")

    a = ap.parse_args()
    if a.cmd == "trajectory":
        plot_trajectory(a.N, a.from_top, a.exact_max, a.out)
    else:
        plot_heatmap(a.nmin, a.nmax, a.from_top, a.show, a.exact_max, a.out)


if __name__ == "__main__":
    main()
