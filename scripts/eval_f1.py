from pathlib import Path
import argparse
import numpy as np
from niifm.f1_score import F1Score_undirected_threshold, F1Score_undirected_best_threshold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=10000)
    p.add_argument("--thr", type=float, default=0.1)
    p.add_argument("--curve_points", type=int, default=101)  # 0..1 等分点数
    return p.parse_args()


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    out_dir = ROOT / "outputs"

    A_true = np.loadtxt(out_dir / "A.txt", dtype=int)
    Q_mean = np.loadtxt(out_dir / f"Q_mean_w{args.w}.txt")

    f1_fixed = F1Score_undirected_threshold(A_true, Q_mean, threshold=args.thr)
    print(f"F1-score @ thr={args.thr}:", f1_fixed)

    thresholds = np.linspace(0.0, 1.0, args.curve_points)
    res = F1Score_undirected_best_threshold(A_true, Q_mean, thresholds=thresholds)
    print("Best threshold:", res["best_threshold"], "Best F1-score:", res["best_f1"])

    # 保存曲线
    curve_out = out_dir / f"f1_curve_w{args.w}.txt"
    np.savetxt(
        curve_out,
        np.column_stack([res["thresholds"], res["f1_scores"]]),
        fmt="%.6f",
        header="threshold f1",
        comments="",
    )
    print(f"Saved {curve_out}")


if __name__ == "__main__":
    main()
