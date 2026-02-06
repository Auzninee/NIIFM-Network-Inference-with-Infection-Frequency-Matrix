import numpy as np
import networkx as nx
from pathlib import Path


def generate_network(
    n: int,
    graph_type: str = "ER",
    k: int = 6,
    p: float | None = None,
    seed: int = 0,
    rewire_p: float = 0.1,   # for WS
) -> np.ndarray:
    """
    Generate an undirected network and return its adjacency matrix (0/1).

    graph_type: ER / BA / WS (case-insensitive)
    k: average degree (ER/BA/WS)
    p: ER edge prob, default p = k/(n-1)
    """
    graph_type = graph_type.upper()

    if n <= 1:
        raise ValueError("n must be > 1")

    if graph_type == "ER":
        if p is None:
            p = k / (n - 1)
        p = float(max(0.0, min(1.0, p)))
        G = nx.erdos_renyi_graph(n, p, seed=seed)

    elif graph_type == "BA":
        # BA uses parameter m = edges to attach for each new node
        m = max(1, k // 2)
        G = nx.barabasi_albert_graph(n, m, seed=seed)

    elif graph_type == "WS":
        if k >= n:
            raise ValueError("For WS, need k < n")
        if k % 2 != 0:
            raise ValueError("For WS, k must be even (watts_strogatz_graph requirement)")
        G = nx.watts_strogatz_graph(n, k, rewire_p, seed=seed)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    A = nx.to_numpy_array(G, dtype=int)
    np.fill_diagonal(A, 0)
    return A


def save_adjacency_txt(A: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, A.astype(int), fmt="%d")
