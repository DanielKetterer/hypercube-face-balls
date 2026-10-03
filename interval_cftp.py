"""
Set-valued "coupling from the past" for the reduced radius map.

Reduced map (only the cap or the layer directly below binds):
    g_p(x) = sqrt2 - 1                       if x < x*(p)   (lower ball too small to constrain)
           = min(sqrt2 - 1, f_p(x))          otherwise
    f_p(x) = ((p+1) x - sqrt(d)) / (p-1),  d = 4 p x^2 - (p-1)(1-x)^2   (p >= 2)
    f_1(x) = (1-x)^2 / (4x)
    x*(p)  = sqrt(p-1) / (2 sqrt(p) + sqrt(p-1))      (the root of d = 0)

Idea: instead of one radius, push the WHOLE SET of radii a layer could have
down the stack. Start at depth p = D with every value in (0, 1/2]. That set
contains the true layer radius for every dimension N > D, whatever happened
below. If the set collapses to a single point before reaching the top, then
every N > D has the same top structure, and the same interior radius.

The image of an interval under f_p is computed exactly: endpoints plus any
interior critical points (f_p' = 0 reduces to a quadratic). Every image is
padded outward by PAD to absorb floating-point rounding.

Caveat: this assumes Lemma 1 (only the cap or the layer directly below ever
binds). That is verified numerically for 10 <= N <= 2000 but not yet proven.

Usage:
    python3 interval_cftp.py 600                 # exact set-valued run from depth 600
    python3 interval_cftp.py 600 --trace         # print the set every step
    python3 interval_cftp.py 2000 --basin 20000  # sample 20000 starting radii instead
"""

import argparse
import math

SQ2 = math.sqrt(2)
CAP = SQ2 - 1
PAD = 1e-12


def xstar(p):
    return math.sqrt(p - 1) / (2 * math.sqrt(p) + math.sqrt(p - 1))


def f(p, x):
    if p == 1:
        return (1 - x) ** 2 / (4 * x)
    d = 4 * p * x * x - (p - 1) * (1 - x) ** 2
    d = max(d, 0.0)  # only called with x >= x*(p); clamp rounding noise
    return ((p + 1) * x - math.sqrt(d)) / (p - 1)


def critical_points(p):
    """Solutions of f_p'(x) = 0, i.e. 2 (p+1) sqrt(d) = d'(x),
    squared: 4 (p+1)^2 d(x) = d'(x)^2 with d' = 8 p x + 2 (p-1)(1-x)."""
    # d(x)  = (4p - (p-1)) x^2 + 2(p-1) x - (p-1)
    a_d, b_d, c_d = 4 * p - (p - 1), 2 * (p - 1), -(p - 1)
    # d'(x) = (8p - 2(p-1)) x + 2(p-1)
    m, c = 8 * p - 2 * (p - 1), 2 * (p - 1)
    K = 4 * (p + 1) ** 2
    A = K * a_d - m * m
    B = K * b_d - 2 * m * c
    C = K * c_d - c * c
    pts = []
    if abs(A) < 1e-300:
        if B != 0:
            pts.append(-C / B)
    else:
        disc = B * B - 4 * A * C
        if disc >= 0:
            s = math.sqrt(disc)
            pts += [(-B - s) / (2 * A), (-B + s) / (2 * A)]
    # keep only genuine roots of the unsquared equation (d' must be >= 0)
    out = []
    for x in pts:
        dprime = m * x + c
        if dprime >= 0:
            out.append(x)
    return out


def image_of_interval(p, lo, hi):
    """Image of [lo, hi] under g_p, as a list of intervals (cap point included)."""
    pieces = []
    if p == 1:
        a, b = f(1, hi), f(1, lo)  # f_1 is decreasing
        return [(min(a, CAP) - PAD, min(b, CAP) + PAD)]
    xs = xstar(p)
    if lo < xs:
        pieces.append((CAP, CAP))  # the part below threshold fires to the cap
    a, b = max(lo, xs), hi
    if a <= b:
        cand = [a, b] + [x for x in critical_points(p) if a < x < b]
        vals = [min(CAP, f(p, x)) for x in cand]
        pieces.append((min(vals) - PAD, min(max(vals) + PAD, CAP)))
    return pieces


def merge(intervals):
    intervals = sorted(intervals)
    out = []
    for lo, hi in intervals:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def run(D, trace=False):
    S = [(1e-9, 0.5)]           # every possible radius at depth D
    history = {}
    for p in range(D - 1, 0, -1):
        new = []
        for lo, hi in S:
            new += image_of_interval(p, lo, hi)
        S = merge(new)
        history[p] = S
        width = sum(h - l for l, h in S)
        if trace or p <= 30 or len(S) > 1 and p % 50 == 0:
            if trace or p <= 30:
                print(f"p={p:5d}  pieces={len(S):3d}  total width={width:.3e}  "
                      + ("  ".join(f"[{l:.6f},{h:.6f}]" for l, h in S[:4]))
                      + (" ..." if len(S) > 4 else ""))
    return history


def g(p, x):
    if p == 1:
        return min(CAP, f(1, x))
    return CAP if x < xstar(p) else min(CAP, f(p, x))


def outcome(D, x):
    """Push a single starting radius x at depth D to the top; return a, B or other."""
    prev = None
    for p in range(D - 1, 0, -1):
        prev, x = x, g(p, x)
    r1, r2 = x, prev
    v = min((1 - r1) - r1, SQ2 * (1 - r2) - r2)
    if abs(v - (3 - 2 * SQ2)) < 1e-9:
        return "a"
    if abs(v - (SQ2 - 1)) < 1e-9:
        return "B"
    return "other"


def basin(D, samples):
    """Fraction of evenly spaced starting radii in (0, 1/2] that end in each outcome."""
    from collections import Counter
    xs = [0.5 * (i + 0.5) / samples for i in range(samples)]
    res = [(x, outcome(D, x)) for x in xs]
    c = Counter(o for _, o in res)
    print(f"depth D = {D}, {samples} starting radii:")
    for k in ("a", "B", "other"):
        if c[k]:
            print(f"  {k}: {c[k] / samples:.4%}")
    bad = [x for x, o in res if o != "a"]
    if bad:
        print(f"  non-a starts lie in [{min(bad):.5f}, {max(bad):.5f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("D", type=int)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--basin", type=int, default=0,
                    help="instead of the interval run, sample this many starting radii")
    a = ap.parse_args()
    if a.basin:
        basin(a.D, a.basin)
        return
    hist = run(a.D, a.trace)

    # first p (counting down) from which the set is a single point forever after
    collapsed_from = None
    for p in range(1, a.D):
        S = hist[p]
        if len(S) == 1 and S[0][1] - S[0][0] < 1e-9:
            collapsed_from = p
        else:
            break
    top = hist[1][0]
    if collapsed_from:
        # set has collapsed for all p <= collapsed_from; find where it first did
        first = max(p for p in range(1, a.D)
                    if all(len(hist[q]) == 1 and hist[q][0][1] - hist[q][0][0] < 1e-9
                           for q in range(1, p + 1)))
        print(f"\nCOLLAPSED: from p = {first} up to the top, every N > {a.D} "
              f"has one and the same radius at each layer.")
        r1 = hist[1][0][0]
        r2 = hist[2][0][0]
        interior = min(1 * (1 - r1) - r1, SQ2 * (1 - r2) - r2)
        print(f"facet layer r = {r1:.12f} ({'cap' if abs(r1 - CAP) < 1e-9 else 'pinned'}), "
              f"interior = {interior:.12f}  (3 - 2sqrt2 = {3 - 2 * SQ2:.12f})")
    else:
        print(f"\nNOT collapsed: at the facet layer the possible set is {hist[1]}")


if __name__ == "__main__":
    main()
