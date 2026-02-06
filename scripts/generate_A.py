from niifm.generate_networks import generate_network, save_adjacency


if __name__ == "__main__":
    n = 100
    k = 10
    graph_type = "ER"
    seed = 42

    A = generate_network(
        n=n,
        graph_type=graph_type,
        k=k,
        seed=seed
    )

    save_adjacency(A, "outputs/demo/A.txt")

    print("Adjacency matrix saved to outputs/demo/A.txt")
