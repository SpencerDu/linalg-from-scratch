import numpy as np


def lu_decompose(A):
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)

    for k in range(n):
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] = U[i, k:] - m * U[k, k:]

    return L, U

def plu_decompose(A):
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.zeros((n, n))
    P = np.eye(n)

    for k in range(n):

        pivot = np.argmax(np.abs(U[k:, k])) + k

        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k+1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] = U[i, k:] - m * U[k, k:]

    L = L + np.eye(n)

    return P, L, U

def forward_substitution(L, c) :
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n) :
        y[i] = c[i] - L[i, :i] @ y[:i]
    return y

def backward_substitution(U, y) :
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1) :
        x[i] = (y[i] - U[i, i+1: ] @ x[i+1:]) / U[i, i]
    return x

def solve(A, b):
    P, L, U = plu_decompose(A)
    c = P @ b
    y = forward_substitution(L, c)
    x = backward_substitution(U, y)
    return x

def rref(A, tol=1e-9):
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivot_columns = []
    pivot_row = 0

    for col in range(n):
        if pivot_row == m:
            break

        candidate = np.argmax(np.abs(R[pivot_row:, col])) + pivot_row

        if abs(R[candidate, col]) < tol:
            continue

        if candidate != pivot_row:
            R[[pivot_row, candidate]] = R[[candidate, pivot_row]]

        R[pivot_row] = R[pivot_row] / R[pivot_row, col]

        for i in range(m):
            if i != pivot_row:
                R[i] = R[i] - R[i, col] * R[pivot_row]

        pivot_columns.append(col)
        pivot_row += 1

    return R, pivot_columns

def matrix_rank(A, tol=1e-9):
    R, pivot_columns = rref(A, tol)
    return len(pivot_columns)

def column_space(A, tol=1e-9):
    A = np.array(A, dtype=float)
    R, pivot_columns = rref(A, tol)
    return A[:, pivot_columns]

def null_space(A, tol=1e-9):
    A = np.array(A, dtype=float)
    R, pivot_columns = rref(A, tol)
    m, n = R.shape

    free_columns = [c for c in range (n) if c not in pivot_columns]

    special_solutions = []
    for f in free_columns:
        s = np.zeros(n)
        s[f] = 1
        for i, p in enumerate(pivot_columns):
            s[p] = -R[i, f]
        special_solutions.append(s)

    return special_solutions

def qr_decompose(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    R_check, pivot_columns = rref(A)
    if len(pivot_columns) != n:
        raise ValueError("columns of A are dependent — no orthonormal basis exists")

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for k in range(n):
        v = A[:, k].copy()

        for j in range(k):
            R[j, k] = Q[:, j] @ v
            v = v - R[j, k] * Q[:, j]

        R[k, k] = np.sqrt(v @ v)
        Q[:, k] = v / R[k, k]

    return Q, R

if __name__ == "__main__":

    A = [
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2],
    ]

    b = np.array([1, 2, 3], dtype=float)

    M = [
        [1, 2, 1, 3],
        [2, 4, 2, 8],
        [1, 2, 2, 5],
    ]
    R_rref, pivots = rref(M)

    G = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]
    Q, R = qr_decompose(G)

    print("Q=")
    print(Q)
    print("R=")
    print(R)
    print("Q.T @ Q (should be I)=")
    print(Q.T @ Q)
    print("Q @ R (should be original)=")
    print(Q @ R)

    print("RREF=")
    print(R_rref)
    print("pivot_columns=", pivots)
    print("rank=", matrix_rank(M))
    print("column space basis=")
    print(column_space(M))
    print("nullspace basis=")
    for s in null_space(M):
        print(s)

    print("solve(A, b)=")
    print(solve(A, b))
    print("np.linalg.solve=")
    print(np.linalg.solve(A, b))

    L, U = lu_decompose(A)
    print("L=")
    print(L)
    print("U=")
    print(U)
    print("LxU=")
    print(L @ U)
