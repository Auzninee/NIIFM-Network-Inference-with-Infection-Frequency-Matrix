from __future__ import annotations
from pathlib import Path
import numpy as np
from cmdstanpy import CmdStanModel, CmdStanMCMC


def get_project_root() -> Path:
    # niifm/stan_runner.py -> niifm -> project root
    return Path(__file__).resolve().parents[1]


def get_model(stan_file: str | Path) -> CmdStanModel:
    stan_file = Path(stan_file)
    if not stan_file.is_absolute():
        stan_file = get_project_root() / stan_file
    if not stan_file.exists():
        raise FileNotFoundError(f"Stan file not found: {stan_file}")

    # cmdstanpy 会自动编译并缓存
    return CmdStanModel(stan_file=str(stan_file))


def sample_model(
    model: CmdStanModel,
    data: dict,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = 42,
    max_treedepth: int = 15,
) -> CmdStanMCMC:
    fit = model.sample(
        data=data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        max_treedepth=max_treedepth,
        show_progress=True,
    )
    return fit


def save_fit_csv(fit: CmdStanMCMC, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # cmdstanpy 会把每条链保存为 csv
    fit.save_csvfiles(dir=str(out_dir))
    return out_dir


def load_fit_csv(model: CmdStanModel, csv_dir: str | Path) -> CmdStanMCMC:
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No csv files found in {csv_dir}")
    return CmdStanMCMC.from_csv(files=[str(p) for p in csv_files], model=model)


def estimate_network_Q_mean(fit: CmdStanMCMC, var_name: str = "Q") -> np.ndarray:
    """
    Return posterior mean of Q (edge probability matrix).
    Assumes Stan model defines parameter `Q` with shape (n, n).
    """
    Q = fit.stan_variable(var_name)   # shape: (draws, n, n) or (draws*chains, n, n)
    return np.mean(Q, axis=0)


def test_chains_by_lp(fit: CmdStanMCMC, tol: float = 0.1) -> bool:
    """
    Check whether any chain has markedly lower average lp__.
    Uses draws_pd() which includes chain column.
    """
    df = fit.draws_pd(vars=["lp__"])
    means = df.groupby("chain")["lp__"].mean()
    best = means.max()
    return bool(((means - (1 - tol) * best) > 0).all())
