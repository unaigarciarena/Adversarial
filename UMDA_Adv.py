import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tabulate
path = "AdvVsUMDA.npy"
df_path = "AdvVsUMDA.csv"
problems = [0, 1, 5, 6, 10, 11]
svrs = ["AdvUMDA", "UMDA"]


def load_data(force=True):

    rootpath = "Init_res/"
    total = np.zeros((4, 2, 6, 50000))


    if not os.path.isfile(path) or force:
        for seed in range(1, 5):
            print(seed)
            for svri, svr in enumerate([4, 5]):
                print("\t", svr)
                for fi, f in enumerate(problems):
                    minipath = rootpath + "Evals_" + str(seed) + "_10_" + str(svr) + "_1000_50_" + str(f) + "_1_0_1_1_1_1_0_0_0_0_0_0.npy"
                    if os.path.isfile(minipath):
                        total[seed-1, svri, fi] = np.load(minipath)[-50000:]
                    else:
                        print(minipath)
        np.save(path, total)
    else:
        total = np.load(path)

    if not os.path.isfile(df_path) or force:
        for ip, problem in enumerate(problems):
            total[:, :, ip, :] = total[:, :, ip, :] - np.min(total[:, :, ip, :])
            total[:, :, ip, :] = total[:, :, ip, :] + 0.01 * np.max(total[:, :, ip, :])
            total[:, :, ip, :] = total[:, :, ip, :] / np.max(total[:, :, ip, :])
        total = np.log(total)
        for ev in range(1, 50000):
            total[:, :, :, ev] = np.min((total[:, :, :, ev - 1], total[:, :, :, ev]), axis=0)
        df = pd.DataFrame(columns=["Function", "Seed", "Solver", "Evaluation", "Fitness"])
        for ip, problem in enumerate(problems):
            print(problem)
            for seed in range(4):
                print("\t" + str(seed))
                for svri, svr in enumerate([4, 5]):
                    print("\t\t" + str(svr))
                    # print(total[seed, ip, 0, n_evals-5:n_evals])
                    # print(total[seed, ip, 1, n_evals - 5:n_evals])
                    for evaluation in np.arange(0, 50000, 200):
                        df.loc[df.shape[0]] = ["F" + str(problem), seed, svrs[svri], evaluation, total[seed, svri, ip, evaluation]]

        df.to_csv(df_path)


def visualize():
    total = pd.read_csv(df_path)
    sns.lineplot(x="Evaluation", y="Fitness", hue="Function", style="Solver", size="Solver", data=total)
    plt.show()


def tables():
    total = np.load(path)
    #total = np.zeros((4, 2, 6, 50000))
    table = np.zeros((24, 10))
    for ip, problem in enumerate(problems):
        total[:, :, ip, :] = total[:, :, ip, :] - np.min(total[:, :, ip, :])

        total[:, :, ip, :] = total[:, :, ip, :] / np.max(total[:, :, ip, :])
    for ev in range(1, 50000):
        total[:, :, :, ev] = np.min((total[:, :, :, ev - 1], total[:, :, :, ev]), axis=0)
    for svri, svr in enumerate([4, 5]):
        for fi, f in enumerate(problems):
            for evaluation in range(0, 50000, 5000):
                table[fi * 4 + svri * 2 + 0, evaluation // 5000] = np.mean(total[:, svri, fi, evaluation])
                table[fi * 4 + svri * 2 + 1, evaluation // 5000] = np.log(np.var(total[:, svri, fi, evaluation]))
    print(tabulate.tabulate(table.T, tablefmt="latex"))


if __name__ == "__main__":
    #load_data()
    #visualize()
    tables()
