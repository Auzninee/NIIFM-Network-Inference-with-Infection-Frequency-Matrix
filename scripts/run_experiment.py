from pathlib import Path
import argparse
import numpy as np
from cmdstanpy import CmdStanModel

from niifm.generate_networks import generate_network, save_adjacency_txt
from niifm.sis import run_sis, save_state_nodes_txt
from niifm.transform_m import transform_M_for_w, save_M_txt, save_M_npy
from niifm.f1_score import F1Score_undirected_threshold, F1Score_undirected_best_threshold


def parse_args():
    p = argparse.ArgumentParser("Run full NIIFM pipeline: A -> SIS -> M -> Stan -> F1")

    # ---- Network ----
    p.add_argument("--graph", type=str, default="BA", choices=["ER", "BA", "WS"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # ---- SIS ----
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--rho0", type=float, default=0.2)
    p.add_argument("--T", type=int, default=10000)

    # ---- M window ----
    p.add_argument("--w", type=int, default=10000)

    # ---- Stan ----
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--stan_seed", type=int, default=42)
    p.add_argument("--max_treedepth", type=int, default=15)

    # ---- Evaluation ----
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--curve_points", type=int, default=101)

    return p.parse_args()


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Generate network A
    # -------------------------
    A = generate_network(n=args.n, graph_type=args.graph, k=args.k, seed=args.seed)
    save_adjacency_txt(A, out_dir / "A.txt")
    print("[1] Saved outputs/A.txt")

    # -------------------------
    # 2) SIS simulation
    # -------------------------
    state_nodes = run_sis(
    A=A,
    beta=args.beta,
    mu=args.mu,
    T=args.T,
    rho0=args.rho0,
    seed=args.seed,
    )
    save_state_nodes_txt(state_nodes, out_dir / "state_nodes.txt")
    print("[2] Saved outputs/state_nodes.txt")


    # -------------------------
    # 3) Transform to M (single w)
    # -------------------------
    if args.w > state_nodes.shape[0]:
        raise ValueError(f"w={args.w} > T={state_nodes.shape[0]} in state_nodes")
    M = transform_M_for_w(state_nodes, w=args.w)

    save_M_txt(M, out_dir / f"M_w{args.w}.txt")
    save_M_npy(M, out_dir / f"M_w{args.w}.npy")
    print(f"[3] Saved outputs/M_w{args.w}.txt and .npy")

    # -------------------------
    # 4) Stan inference -> Q_mean
    # -------------------------
    stan_path = ROOT / "stan" / "model.stan"
    model = CmdStanModel(stan_file=str(stan_path))

    fit = model.sample(
        data={"n": int(args.n), "M": M.astype(int), "C": int(args.w)},
        chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.samples,
        seed=args.stan_seed,
        max_treedepth=args.max_treedepth,
        show_progress=True,
    )

    Q_mean = fit.stan_variable("Q").mean(axis=0)
    q_path = out_dir / f"Q_mean_w{args.w}.txt"
    np.savetxt(q_path, Q_mean, fmt="%.6f")
    print(f"[4] Saved {q_path}")

    # -------------------------
    # 5) F1 evaluation
    # -------------------------
    f1_fixed = F1Score_undirected_threshold(A, Q_mean, threshold=args.thr)
    print(f"[5] F1-score @ thr={args.thr}: {f1_fixed}")

    # save thresholded adjacency
    A_hat = (Q_mean > args.thr).astype(int)
    np.fill_diagonal(A_hat, 0)
    a_hat_path = out_dir / f"A1_hat_w{args.w}_thr{args.thr}.txt"
    np.savetxt(a_hat_path, A_hat, fmt="%d")
    print(f"[5] Saved {a_hat_path}")

    # best-threshold sweep
    thresholds = np.linspace(0.0, 1.0, args.curve_points)
    res = F1Score_undirected_best_threshold(A, Q_mean, thresholds=thresholds)
    print(f"[5] Best thr={res['best_threshold']}  Best F1-score={res['best_f1']}")

    curve_path = out_dir / f"f1_curve_w{args.w}.txt"
    np.savetxt(
        curve_path,
        np.column_stack([res["thresholds"], res["f1_scores"]]),
        fmt="%.6f",
        header="threshold f1",
        comments="",
    )
    print(f"[5] Saved {curve_path}")


if __name__ == "__main__":
    main()
