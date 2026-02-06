#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def contract(M):
    """Remove all empty rows and columns from a matrix."""
    mask_0 = ~np.all(M == 0, axis=0)
    M_prime = M[:, mask_0]
    mask_1 = ~np.all(M_prime == 0, axis=1)
    M_final = M_prime[mask_1, :]
    return M_final, mask_0, mask_1


def expand_1d_array(x, mask):
    """Re-expand a contracted 1D matrix."""
    y = np.zeros(mask.shape[0])
    ix = -1
    for i, nonemptyrow in enumerate(mask):
        if nonemptyrow:
            ix += 1
            y[i] = x[ix]
    return y


def expand_2d_array(x, mask_0, mask_1):
    """Re-expand a contracted 2D matrix."""
    y = np.zeros((mask_1.shape[0], mask_0.shape[0]))
    ix = -1
    for i, nonemptyrow in enumerate(mask_1):
        if nonemptyrow:
            ix += 1
            jx = -1
            for j, nonemptycol in enumerate(mask_0):
                if nonemptycol:
                    jx += 1
                    y[i, j] = x[ix, jx]
    return y


def sort(A, B, reverse_X=True, reverse_Y=True):
    """Sort array A based on the margins of identically sized array B."""
    margin0 = B.sum(axis=0)
    margin1 = B.sum(axis=1)
    if reverse_X:
        argsort0 = np.argsort(-margin0)
    else:
        argsort0 = np.argsort(margin0)
    if reverse_Y:
        argsort1 = np.argsort(margin1)
    else:
        argsort1 = np.argsort(-margin1)
    return A[argsort1, :][:, argsort0]

def contract_square(M: np.ndarray):
    """
    Contract a square matrix by removing nodes whose row and column are all zeros.
    Returns:
      M_c : contracted square matrix
      mask : boolean mask of kept nodes (length n)
    """
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("contract_square expects a square 2D matrix.")
    row_nonzero = ~np.all(M == 0, axis=1)
    col_nonzero = ~np.all(M == 0, axis=0)
    mask = row_nonzero & col_nonzero
    M_c = M[np.ix_(mask, mask)]
    return M_c, mask

def expand_square(X_c: np.ndarray, mask: np.ndarray):
    """
    Expand a contracted square matrix back to (n,n), filling missing entries with 0.
    """
    n = mask.shape[0]
    X = np.zeros((n, n), dtype=X_c.dtype)
    idx = np.flatnonzero(mask)
    X[np.ix_(idx, idx)] = X_c
    return X
