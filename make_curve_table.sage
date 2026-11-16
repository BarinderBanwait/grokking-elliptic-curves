"""
Build the point table for E(F_113), where E is the reduction of Cremona 61a1 mod 113.

61a1 has a-invariants [1, 0, 0, -2, 1], good reduction at 113 and a_113 = 1, so
#E(F_113) = 113 and E(F_113) is cyclic of prime order.

The script changes to a short Weierstrass model y^2 = x^3 + A x + B over F_113,
fixes a generator G (smallest affine x, then smallest y), and writes

    data/e61a1_f113_points.csv

with header log,x,y and one row per point, sorted by log = log_G(P) in 0..112.
The point at infinity has log 0 and x = y = inf.

It checks that log(P + Q) = log P + log Q mod 113 for all ordered pairs, using
Sage's point addition, and checks the x-coordinate structure of the table.

Run from the repository root:  sage make_curve_table.sage
"""

import csv
import os

p = 113
F = GF(p)

# Reduction of 61a1 mod 113, and a short Weierstrass model.
E0 = EllipticCurve(F, [1, 0, 0, -2, 1])
E = E0.short_weierstrass_model()
a1, a2, a3, a4, a6 = E.a_invariants()
assert a1 == 0 and a2 == 0 and a3 == 0
A, B = a4, a6

iso = E0.isomorphism_to(E)
assert E0.is_isomorphic(E)
assert E0.j_invariant() == E.j_invariant()
assert E.order() == p
assert E0.order() == p

print("Reduced 61a1 model:", E0)
print("Short model:       ", E)
print("A =", A, " B =", B)
print("Isomorphism from the reduced 61a1 model to the short model:")
print(iso)
u, r, s, t = iso.tuple()
print("(u, r, s, t) =", (u, r, s, t))
print("  x = u^2 x' + r,  y = u^3 y' + s u^2 x' + t,  (x, y) on 61a1, (x', y') on the short model")
# Check the stated formulas against the map on every affine point.
for P0 in E0.points():
    if P0.is_zero():
        assert iso(P0).is_zero()
        continue
    P1 = iso(P0)
    assert P0[0] == u^2 * P1[0] + r
    assert P0[1] == u^3 * P1[1] + s * u^2 * P1[0] + t
print("#E(F_113) =", E.order())

# Generator: smallest x among affine points, then smallest y.
affine = [P for P in E.points() if P[2] != 0]
G = min(affine, key=lambda P: (int(P[0]), int(P[1])))
assert G.order() == p
print("G =", (int(G[0]), int(G[1])))

# Discrete logs, by listing k*G for k = 0, ..., 112.
log = {}
Q = E(0)
for k in range(p):
    log[Q] = k
    Q = Q + G
assert Q == E(0)
assert len(log) == p
assert set(log.keys()) == set(E.points())


def coords(P):
    if P.is_zero():
        return ("inf", "inf")
    return (int(P[0]), int(P[1]))


rows = sorted(((log[P],) + coords(P) for P in E.points()), key=lambda row: row[0])

script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
out_path = os.path.join(script_dir, "data", "e61a1_f113_points.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["log", "x", "y"])
    w.writerows(rows)
print("Wrote", out_path)

# Homomorphism check: log(P + Q) = log P + log Q mod 113, using point addition.
pts = list(E.points())
for P in pts:
    for Q in pts:
        assert log[P + Q] == (log[P] + log[Q]) % p
print("Checked log(P + Q) = log P + log Q mod 113 for all", len(pts) ** 2, "ordered pairs.")

# x-coordinate structure.
xs = {}
for P in affine:
    xs.setdefault(int(P[0]), []).append(P)
assert len(xs) == 56
for x, Ps in xs.items():
    assert len(Ps) == 2
    P1, P2 = Ps
    assert P2 == -P1
    assert P2[1] == -P1[1]
    assert log[P2] == (p - log[P1]) % p
    assert log[P1] != 0 and log[P2] == p - log[P1]
assert all(P[1] != 0 for P in affine)
print("Distinct affine x-values:", len(xs))
print("Each x-value occurs for exactly two points P and -P, with log(-P) = 113 - log P.")
print("No affine point has y = 0.")

print("First 10 rows (log, x, y):")
for row in rows[:10]:
    print(row)
