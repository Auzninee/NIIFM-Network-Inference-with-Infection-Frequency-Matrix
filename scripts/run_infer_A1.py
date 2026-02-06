from pathlib import Path
import argparse
import numpy as np
from cmdstanpy import CmdStanModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=10000)
    p.add_argument("--base_w", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_treedepth", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    w = args.w
    slice_idx = w // args.base_w - 1
    C = int(w)

    M3 = np.load(out_dir / "M_3d.npy")
    M = M3[:, :, slice_idx].astype(int)
    n = M.shape[0]

    model = CmdStanModel(stan_file=str(ROOT / "stan" / "model.stan"))
    fit = model.sample(
        data={"n": int(n), "M": M, "C": int(C)},
        chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.samples,
        seed=args.seed,
        max_treedepth=args.max_treedepth,
        show_progress=True,
    )

    Q_mean = fit.stan_variable("Q").mean(axis=0)
    q_out = out_dir / f"Q_mean_w{w}.txt"
    np.savetxt(q_out, Q_mean, fmt="%.6f")
    print(f"Saved {q_out} with shape {Q_mean.shape}")


if __name__ == "__main__":
    main()
