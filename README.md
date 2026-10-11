# linalg-from-scratch

Solving Ax = b and finding the four subspaces of a matrix, written by hand while following MIT OCW 18.06 (Gilbert Strang):

Ax = b -> split A into PA = LU (L lower triangular, U upper triangular, P the row swaps) -> forward substitution solves Ly = Pb -> back substitution solves Ux = y -> x

numpy is only used to hold the arrays and to check the answers, never as the solver.

## Why

To understand linear algebra better.

## How it works

- LU: elimination turns A into an upper triangular U, and the multipliers used along the way are stored in L.
- Row swaps (PLU): if a 0 lands on a pivot, elimination would divide by 0, so rows are swapped first. The code swaps in the biggest entry available, which also keeps rounding errors small.
- Substitution: a triangular system is easy to solve. The last row has only one unknown, so solve it, put it into the row above, and that row now has only one unknown too. Work up the rows (or down, for L).
- RREF: reduces A to its simplest row form. From it:
  - rank = the number of pivots
  - column space = every b that Ax can reach
  - null space = every x that A sends to 0
- QR (Gram-Schmidt): take each column, subtract the parts that point along the columns already done, then scale it to length 1. That gives perpendicular columns Q, and R records what was subtracted, so A = QR.

## Results

- solve(A, b) gives x = (-1, -1, 4) for the test system, the same as numpy, and matches numpy on 1000 out of 1000 random systems.
- PA = LU holds.
- For the 3x4 test matrix the pivots are in columns 1, 3 and 4, so the rank is 3, and the null space is spanned by (-2, 1, 0, 0). A times that vector gives 0.
- QR on the test matrix: Q transpose times Q is the identity, and Q times R gives back the original matrix.

## Known limitations

- lu_decompose (the version without row swaps) breaks if a 0 lands on a pivot, for example on [[0, 1], [1, 1]] it gives inf and nan. That is why plu_decompose exists, and solve uses that one.
- Gram-Schmidt loses accuracy when the columns are nearly dependent. On Hilbert matrices, the standard hard test, the columns of Q stop being exactly perpendicular: the error is about 6e-10 at 6x6 and 2e-7 at 8x8. At 10x10 the rank check wrongly decides the columns are dependent and refuses.
- Everything uses a fixed tolerance of 1e-9 to decide what counts as 0.

## Running it

Needs Python with numpy:

    pip install numpy
    python lu.py

It prints the QR, RREF, rank, column space, null space and LU results for the test matrices, with numpy's answer next to the solve result.
