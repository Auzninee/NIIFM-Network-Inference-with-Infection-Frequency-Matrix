import numpy as np
from pathlib import Path


def run_sis(
    A: np.ndarray,
    beta: float,
    mu: float,
    T: int,
    rho0: float,
    seed: int = 0,
) -> np.ndarray:
    """
    SIS simulation matching your MATLAB SIS_RSC_2 logic, WITHOUT high-order terms.

    Parameters
    ----------
    A : (n,n) 0/1 adjacency matrix
    beta : infection probability (two-body)
    mu : recovery probability
    T : number of time steps
    rho0 : initial infected density
    seed : random seed

    Returns
    -------
    state_nodes : (T,n) int8 matrix
    """
    rng = np.random.default_rng(seed)

    A = (np.asarray(A) > 0).astype(np.int8)
    n = A.shape[0]
    state_nodes = np.zeros((T, n), dtype=np.int8)

    state = np.zeros(n, dtype=np.int8)
    k0 = int(np.ceil(rho0 * n))
    k0 = max(1, min(n, k0))
    init_idx = rng.permutation(n)[:k0]
    state[init_idx] = 1

    for t in range(T):
        infected_count = int(state.sum())
        susceptible_count = n - infected_count

        if infected_count > 0 and susceptible_count != 0:
            state0 = state.copy()
            I_node = np.flatnonzero(state0 == 1)

            # infection
            for i in I_node:
                neig = np.flatnonzero(A[i] == 1)
                if neig.size == 0:
                    continue
                mask = rng.random(neig.size) < beta
                state[neig[mask]] = 1

            # recovery (based on I_node from state0, matching MATLAB)
            if state.sum() < n:
                rec_mask = rng.random(I_node.size) < mu
                state[I_node[rec_mask]] = 0

        else:
            # reset when full infection or full recovery
            state[:] = 0
            init_idx = rng.permutation(n)[:k0]
            state[init_idx] = 1

        state_nodes[t, :] = state

    return state_nodes


def save_state_nodes_txt(state_nodes: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, state_nodes.astype(int), fmt="%d")


def load_adjacency_txt(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, dtype=int)
