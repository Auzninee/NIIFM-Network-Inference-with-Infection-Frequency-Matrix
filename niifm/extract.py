import numpy as np


def extract_abc(S: np.ndarray, nod_1based: int):
    """
    Strict Python rewrite of MATLAB Extract_ABC(S, nod).

    Input
    -----
    S : (m, n) array-like, binary {0,1}
        Time-series state matrix (state_nodes).
    nod_1based : int
        Target node index in MATLAB style (1..n).

    Output
    ------
    A : (m2, n-1) ndarray
        State matrix of other nodes at times t when node i is 0.
    B : (m2,) ndarray
        State of node i at time t+1 corresponding to rows in A.
    C : (m2, nchoosek(n-1,2)) ndarray
        Pairwise products of columns of A (three-body candidates), in MATLAB column order.
    """
    S = np.asarray(S)
    m, n = S.shape
    nod = nod_1based - 1  # convert to 0-based

    # B1 = S(:, nod)
    B1 = S[:, nod]

    # A1 = S; A1(:, nod) = []
    A1 = np.delete(S, nod, axis=1)  # shape (m, n-1)

    # t = find(B1 == 0); remove t == m (MATLAB 1-based) -> remove index m-1 in 0-based
    t = np.flatnonzero(B1 == 0)
    t = t[t != (m - 1)]

    # A = A1(t, :); B = B1(t+1)
    A = A1[t, :]
    B = B1[t + 1]

    # dl = find(sum(A,2) == 0 | sum(A,2) >= n-1)
    row_sum = A.sum(axis=1)
    dl_mask = (row_sum == 0) | (row_sum >= (n - 1))

    # A(dl,:) = []; B(dl) = []
    A = A[~dl_mask, :]
    B = B[~dl_mask]

    # Build C: columns are A[:,i] .* A[:,j] for i<j in increasing order
    m2, n1 = A.shape  # n1 = n-1
    if n1 < 2:
        C = np.zeros((m2, 0), dtype=A.dtype)
        return A, B, C

    n2 = n1 * (n1 - 1) // 2
    C = np.zeros((m2, n2), dtype=A.dtype)

    col = 0
    for i in range(n1 - 1):
        for j in range(i + 1, n1):
            C[:, col] = A[:, i] * A[:, j]
            col += 1

    return A, B, C


def extract_x(S: np.ndarray, nod_1based: int) -> np.ndarray:
    """
    Strict Python rewrite of MATLAB Extract_X(S, nod):

        [A,B,C]=Extract_ABC(S,nod);
        X = sum(bsxfun(@times, A, B), 1);

    Returns
    -------
    X : (n-1,) float ndarray
        Statistical vector (NOT normalized probability), matching MATLAB behavior.
    """
    A, B, _ = extract_abc(S, nod_1based)

    A = A.astype(np.float64, copy=False)
    B = B.astype(np.float64, copy=False)

    # bsxfun(@times, A, B) <=> A * B[:,None], then sum over rows
    X = np.sum(A * B[:, None], axis=0)
    return X
