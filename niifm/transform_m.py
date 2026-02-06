import numpy as np
from pathlib import Path
from niifm.extract import extract_x  # 确保 extract.py 里有 extract_x(S, nod_1based=...)


def transform_M_for_w(state_nodes: np.ndarray, w: int) -> np.ndarray:
    """
    Transform state_nodes[:w, :] into the statistical matrix M (n x n), integer-valued.
    This matches your MATLAB logic for a fixed window length w.
    """
    S = np.asarray(state_nodes, dtype=int)
    if S.ndim != 2:
        raise ValueError("state_nodes must be a 2D array (T, n).")
    if w <= 1 or w > S.shape[0]:
        raise ValueError(f"w must be in [2, T]. Got w={w}, T={S.shape[0]}")

    S_w = S[:w, :]
    n = S_w.shape[1]
    M = np.zeros((n, n), dtype=np.int64)

    for nod in range(1, n + 1):  # 1-based node index
        X = extract_x(S_w, nod_1based=nod)      # length n-1
        i = nod - 1
        a = np.insert(X, i, 0).astype(np.int64) # length n
        M[i, :] = a

    return M


def save_M_txt(M: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, M.astype(np.int64), fmt="%d")


def save_M_npy(M: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, M.astype(np.int64))
