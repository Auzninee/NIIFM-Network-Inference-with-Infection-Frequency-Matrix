import numpy as np
from niifm.extract import extract_x


def main():
    state_nodes = np.loadtxt("outputs/state_nodes.txt", dtype=int)

    n = 100
    base_w = 1000
    num_blocks = 10

    # M shape: (nod, neighbor, j)
    M = np.zeros((n, n, num_blocks), dtype=np.int64)

    for j in range(1, num_blocks + 1):
        print("j =", j)
        S_w = state_nodes[: base_w * j, :]

        for nod in range(1, n + 1):
            X = extract_x(S_w, nod_1based=nod)   # length n-1, statistical vector

            # insert 0 at nod position -> length n
            i = nod - 1
            a = np.insert(X, i, 0).astype(np.int64)
            M[i, :, j - 1] = a

    # 保存：强烈建议存 npy（保留 3D，不丢信息）
    np.savetxt("outputs/M_w10000.txt", M[:,:,9], fmt="%d")
    np.save("outputs/M_3d.npy", M)

    # 如需和 MATLAB 一样每个 j 单独看，也可以保存 txt
    for k in range(num_blocks):
        w = base_w * (k + 1)
        np.savetxt(f"outputs/M_w{w}.txt", M[:, :, k], fmt="%.6f")

    print("Saved outputs/M_3d.npy and outputs/M_w*.txt, shape =", M.shape)


if __name__ == "__main__":
    main()
