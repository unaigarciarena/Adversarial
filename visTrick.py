import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import seaborn as sns
import pandas as pd

tot_path = "tot_trick.npy"
df_path = "df_trick.csv"
problems = [0, 1, 5, 6, 10, 11]
problem_names = ["F1", "F2", "F6", "F8", "F13", "F14"]


def load_data(force=True):
    n_evals = 50000

    if not os.path.isfile(tot_path) or force:
        total = np.zeros((5, 6, 2, 50000))
        for ip, problem in enumerate(problems):
            for seed in range(5):
                for trick in range(2):
                    path = "Evals_" + str(seed) + "_10_2_1000_50_" + str(problem) + "_1_0_0_0_0_1_1_1_1_" + str(trick) + "_0_1.npy"
                    if os.path.isfile(path):
                        total[seed, ip, trick] = np.load(path)[-50000:]

                    else:
                        print(path)
        np.save(tot_path, total)
    else:
        total = np.load(tot_path)

    if not os.path.isfile(df_path) or force:
        for ip, problem in enumerate(problems):
            total[:, ip, :, :] = total[:, ip, :, :] - np.min(total[:, ip, :, :])
            total[:, ip, :, :] = total[:, ip, :, :] + 0.01*np.max(total[:, ip, :, :])
            total[:, ip, :, :] = total[:, ip, :, :] / np.max(total[:, ip, :, :])
        total = np.log(total)
        for ev in range(1, 50000):
            total[:, :, :, ev] = np.min((total[:, :, :, ev - 1], total[:, :, :, ev]), axis=0)
        df = pd.DataFrame(columns=["Function", "Seed", "Trick", "Evaluation", "Fitness"])
        for ip, problem in enumerate(problems):
            print(problem)
            for seed in range(5):
                print("\t" + str(seed))
                for trick in range(2):
                    print("\t\t" + str(trick))
                    #print(total[seed, ip, 0, n_evals-5:n_evals])
                    #print(total[seed, ip, 1, n_evals - 5:n_evals])
                    for evaluation in np.arange(0, n_evals, 20):
                        df.loc[df.shape[0]] = [problem_names[ip], seed, ["No tricking", "Tricking"][trick], evaluation, total[seed, ip, trick, evaluation]]

        df.to_csv(df_path)


def curves():
    df = pd.read_csv(df_path)

    sns.lineplot(x="Evaluation", y="Fitness", hue="Function", style="Trick", size="Trick", sizes=[0.8, 2], data=df, palette="viridis")
    plt.ylabel(r"$log$(scaled Fitness)")
    plt.legend(loc=6)
    plt.show()


if __name__ == "__main__":
    #load_data()
    curves()