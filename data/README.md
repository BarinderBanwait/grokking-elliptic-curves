# e61a1_f113_points.csv

The curve is Cremona 61a1, with a-invariants [1, 0, 0, -2, 1], reduced mod 113.
It has good reduction at 113 and a_113 = 1, so E(F_113) is cyclic of order 113.

The table uses the short Weierstrass model y^2 = x^3 + 93x + 20 over F_113.
In this model -(x, y) = (x, -y).
The isomorphism from the reduced 61a1 model is (u, r, s, t) = (94, 47, 56, 33):
x = u^2 x' + r and y = u^3 y' + s u^2 x' + t, where (x, y) is on 61a1 and (x', y') is on the short model.

The generator is G = (1, 1). It is the affine point with the smallest x, then the smallest y.

The CSV has header `log,x,y` and 113 rows sorted by log.
`log` is the discrete logarithm n in 0..112 with P = nG.
`x` and `y` are integers in 0..112.
The point at infinity is the row with log 0, and its x and y are the string `inf`.

To regenerate, run from the repository root:

    sage make_curve_table.sage
