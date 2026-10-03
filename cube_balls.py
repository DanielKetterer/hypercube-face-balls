"""
Greedy face-ball packing in the hypercube [-1, 1]^N.

Implements Daniel's recursion:
  r_0 = 1/2
  r_k = min( sqrt(2) - 1, min_{j<k} beta(k, j) )      for 1 <= k <= N-1
  r_N = min_{j<N} [ sqrt(N - j) * (1 - r_j) - r_j ]

with p = N - k, s = k - j and
  beta(k, j) = ((p+1) r_j - sqrt(4 p r_j^2 - (p-1) s (1-r_j)^2)) / (p-1)   p >= 2, disc >= 0
             = +inf                                                       p >= 2, disc < 0
             = s (1 - r_j)^2 / (4 r_j)                                    p = 1

Outputs:
  summary.csv  one row per N: interior radius, which layer binds it,
               and the top two layers with their binding constraints
  layers.csv   one row per (N, k): every layer radius and what binds it

Optional: --verify brute-forces every pair of balls for small N
(no overlaps, and every layer touches something).

Usage:
  python3 cube_balls.py              # N = 2..200, writes both CSVs
  python3 cube_balls.py --nmax 60
  python3 cube_balls.py --verify 8   # brute-force check N = 2..8
"""

import argparse
import csv
import itertools
import math

SQ2 = math.sqrt(2)

# Closed forms the values tend to land on. Extend this list as you find more.
KNOWN = {
    "1/2": 0.5,
    "sqrt2-1": SQ2 - 1,
    "(sqrt2-1)/2": (SQ2 - 1) / 2,
    "3-2sqrt2": 3 - 2 * SQ2,
    "1/3": 1 / 3,
    "1/4": 0.25,
    "(sqrtN-1)/2": None,  # filled per N below (corner-only answer)
}


def label(x, N=None, tol=1e-12):
    for name, v in KNOWN.items():
        if name == "(sqrtN-1)/2":
            if N is not None and abs(x - (math.sqrt(N) - 1) / 2) < tol:
                return name
            continue
        if abs(x - v) < tol:
            return name
    return ""


def beta(N, k, j, rj):
    p, s = N - k, k - j
    if p >= 2:
        disc = 4 * p * rj * rj - (p - 1) * s * (1 - rj) ** 2
        if disc < 0:
            return math.inf
        return ((p + 1) * rj - math.sqrt(disc)) / (p - 1)
    if p == 1:
        return s * (1 - rj) ** 2 / (4 * rj)
    raise ValueError("p = 0 is the interior ball; use g(j)")


def solve(N):
    """Return (radii r_0..r_N, binders) where binders[k] is 'same', 'base', or j."""
    r = [0.5]
    binders = ["base"]
    for k in range(1, N):
        best, who = SQ2 - 1, "same"
        for j in range(k):
            b = beta(N, k, j, r[j])
            if b < best - 1e-15:
                best, who = b, j
        r.append(best)
        binders.append(who)
    g = [math.sqrt(N - j) * (1 - r[j]) - r[j] for j in range(N)]
    rN = min(g)
    r.append(rN)
    binders.append(g.index(rN))
    return r, binders


def brute_verify(N):
    """Place every ball explicitly; return (worst gap, does every layer touch)."""
    r, _ = solve(N)
    balls = []
    for k in range(N + 1):
        for free in itertools.combinations(range(N), k):
            fixed = [i for i in range(N) if i not in free]
            for signs in itertools.product([-1, 1], repeat=len(fixed)):
                c = [0.0] * N
                for i, sg in zip(fixed, signs):
                    c[i] = sg * (1 - r[k])
                balls.append((k, c))
    worst = math.inf
    tight = {}
    for a in range(len(balls)):
        for b in range(a):
            ka, ca = balls[a]
            kb, cb = balls[b]
            gap = math.dist(ca, cb) - r[ka] - r[kb]
            worst = min(worst, gap)
            top = max(ka, kb)
            tight[top] = min(tight.get(top, math.inf), gap)
    return worst, all(abs(v) < 1e-9 for v in tight.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument("--verify", type=int, default=0,
                    help="brute-force check N = 2..VERIFY (keep it <= 9)")
    args = ap.parse_args()

    if args.verify:
        for N in range(2, args.verify + 1):
            worst, touching = brute_verify(N)
            print(f"N={N:2d}  worst gap={worst:+.2e}  every layer touches: {touching}")
        return

    with open("summary.csv", "w", newline="") as f_sum, \
         open("layers.csv", "w", newline="") as f_lay:
        ws = csv.writer(f_sum)
        wl = csv.writer(f_lay)
        ws.writerow(["N", "r_N", "r_N_closed_form", "r_N_bound_by_layer_j",
                     "r_{N-1}", "r_{N-1}_closed_form", "r_{N-1}_bound_by",
                     "r_{N-2}", "r_{N-2}_closed_form", "r_{N-2}_bound_by",
                     "corner_only_answer"])
        wl.writerow(["N", "k", "r_k", "closed_form", "bound_by"])

        for N in range(2, args.nmax + 1):
            r, who = solve(N)
            row = [N, f"{r[N]:.10f}", label(r[N], N), who[N]]
            for k in (N - 1, N - 2):
                if k >= 0:
                    row += [f"{r[k]:.10f}", label(r[k], N), who[k]]
                else:
                    row += ["", "", ""]
            row.append(f"{(math.sqrt(N) - 1) / 2:.10f}")
            ws.writerow(row)
            for k in range(N + 1):
                wl.writerow([N, k, f"{r[k]:.10f}", label(r[k], N), who[k]])

    print(f"Wrote summary.csv and layers.csv for N = 2..{args.nmax}")


if __name__ == "__main__":
    main()
