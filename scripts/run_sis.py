from niifm.sis import load_adjacency_txt, simulate_sis_from_matlab_logic, save_states_txt


def main():
    A = load_adjacency_txt("outputs/A.txt")

    # 这些参数你按论文/实验改
    beta1 = 0.1
    mu = 1
    T = 10000
    rho0 = 0.2
    seed = 42

    state_nodes = simulate_sis_from_matlab_logic(
        A=A,
        beta1=beta1,
        mu=mu,
        T=T,
        rho0=rho0,
        seed=seed,
    )

    save_states_txt(state_nodes, "outputs/state_nodes.txt")
    print("Saved to outputs/state_nodes.txt, shape =", state_nodes.shape)


if __name__ == "__main__":
    main()
