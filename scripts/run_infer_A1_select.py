from pathlib import Path
import argparse
import numpy as np
from cmdstanpy import CmdStanModel


def parse_args():
    p = argparse.ArgumentParser(description="Run NIIFM A1 inference for selected w values.")
    p.add_argument("--w", nargs="*", type=int, default=[10000],
                   help="Window lengths w to infer (e.g., --w 1000 4000 7000 10000). Default: 10000")
    p.add_argument("--base_w", type=int, default=1000, help="Base step used to build M_3d slices. Default: 1000")
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

    # load M_3d
    m3_path = out_dir / "M_3d.npy"
    if not m3_path.exists():
        raise FileNotFoundError(f"Missing {m3_path}. Please build M_3d.npy first.")
    M3 = np.load(m3_path)  # (n, n, num_blocks)
    n, _, num_blocks = M3.shape

    # compile model once
    stan_path = ROOT / "stan" / "model.stan"
    model = CmdStanModel(stan_file=str(stan_path))

    # run selected w's
    for w in args.w:
        if w % args.base_w != 0:
            raise ValueError(f"w={w} must be a multiple of base_w={args.base_w} to index M_3d slices.")

        slice_idx = w // args.base_w - 1
        if slice_idx < 0 or slice_idx >= num_blocks:
            raise ValueError(
                f"w={w} maps to slice_idx={slice_idx}, but M_3d has num_blocks={num_blocks}. "
                f"(Allowed w: {args.base_w}..{args.base_w*num_blocks})"
            )

        M = M3[:, :, slice_idx].astype(int)
        C = int(w)

        print(f"\n=== Inference for w={w} (slice {slice_idx}) ===")
        fit = model.sample(
            data={"n": int(n), "M": M, "C": C},
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
        print(f"Saved {q_out}")

    print("\nAll requested w finished.")


if __name__ == "__main__":
    main()
