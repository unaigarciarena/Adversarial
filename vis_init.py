import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
from matplotlib import cm
from matplotlib.lines import Line2D
import tabulate

total_path = "total_res.npy"
progress_path = "prog_ress.npy"
pd_path = "total_res.csv"
noisy_path = "noisy_res.csv"
root_path = "Init_res/"
problems = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
problem_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24"]
#methods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
methods = [1, 5, 6, 10, 15]
method_names = ["GradientAttack", "DecoupledDirectionNormL2Attack", "EADAttack", "GradientSignAttack", "AdamL1BasicIterativeAttack", "RandomPGD", "RandomStartProjectedGDAttack", "AdamL2BasicIterativeAttack", "AdditiveUniformNoiseAttack", "MomentumIterativeAttack", "DeepFoolAttack", "LBFGSAttack", "L2BasicIterativeAttack", "SparseFoolAttack", "AdditiveGaussianNoiseAttack", "BlendedUniformNoiseAttack"]
short_names = ["DDL2", "RPGD", "RSPGD", "DF", "BUN"]

def load_data(force=False):

    data = np.zeros((len(problems), len(methods), 5, 2, 2, 2, 50000))

    if not os.path.isfile(root_path + total_path) or force:
        for ip, problem in enumerate(problems):
            missing = 0
            for seed in range(1, 6):
                for im, method in enumerate(methods):
                    for tb in range(2):
                        for t1 in range(2):
                            for t0 in range(2):
                                path = root_path + "Evals_" + str(seed) + "_10_2_1000_50_" + str(problem) + "_1_0_1_1_1_" + str(method) + "_" + str(tb) + "_" + str(t1) + "_" + str(t0) + "_0_0_0.npy"
                                if os.path.isfile(path):
                                    a = np.load(path)
                                    data[ip, im, seed-1, tb, t1, t0, :] = a
                                else:
                                    missing += 1
                                    #print(path)
            print(problem, missing)

        np.save(root_path + total_path, data)
    else:
        data = np.load(root_path + total_path)
    if not os.path.isfile(root_path + progress_path) or force:
        progress = np.copy(data)
        for ev in range(1, 50000):
            progress[:, :, :, :, :, :, ev] = np.min((progress[:, :, :, :, :, :, ev-1], progress[:, :, :, :, :, :, ev]), axis=0)
        np.save(root_path + progress_path, progress)
    else:
        progress = np.load(root_path + progress_path)

    if not os.path.isfile(root_path + pd_path) or not os.path.isfile(root_path + noisy_path) or force:

        df = pd.DataFrame(columns=["Problem", "Method", "Seed", "Tb", "T1", "T0", "Fitness"])
        noisy_df = pd.DataFrame(columns=["Problem", "Method", "Seed", "Tb", "T1", "T0", "Fitness"])
        last = progress[:, :, :, :, :, :, -1]
        print(last.shape)
        noise = 0.02
        for ip, problem in enumerate(problems):
            last_p = last[ip]
            last_p[last_p==0] = np.max(last_p)
            last_p -= np.min(last_p)
            last_p += np.max(last_p)*0.01
            last_p /= np.max(last_p)
            for seed in range(1, 6):
                for im, method in enumerate(methods):
                    for tb in range(2):
                        for t1 in range(2):
                            for t0 in range(2):
                                df.loc[df.shape[0]] = [problem, method, seed, tb, t1, t0, np.log(last_p[im, seed-1, tb, t1, t0])]
                                if last_p[im, seed-1, tb, t1, t0] <= np.percentile(last_p, 0.5):
                                    print(ip, last_p[im, seed-1, tb, t1, t0])
                                    noisy_df.loc[noisy_df.shape[0]] = [problem, np.random.uniform(method-noise*16, method+noise*16), seed, np.random.uniform(tb-noise, tb+noise)*16, np.random.uniform(t1-noise, t1+noise)*16, np.random.uniform(t0-noise, t0+noise)*16, np.round(np.log(last_p[im, seed - 1, tb, t1, t0]), decimals=0)]
        df.to_csv(root_path + pd_path)
        noisy_df.to_csv(root_path + noisy_path)


def curves():
    progress = np.load(root_path + progress_path)
    for ip, problem in enumerate(problems):
        for im, method in enumerate(methods):
            if method == 2 or method == 12 or method == 0 or method == 14 or method == 11 or method == 3:
                continue
            to_plot = progress[ip, im, :, :, :, :, :]
            plt.plot(np.sum(to_plot, axis=(0, 1, 2, 3))/np.sum(to_plot[:, :, :, :, -1] != 0), label="Method " + str(method))
        plt.title("Problem " + str(problem))
        plt.legend()
        plt.show()#plt.savefig("Problem" + str(problem) + ".pdf")
        plt.clf()


def end_box():
    df = pd.read_csv(root_path + pd_path)
    #df = df[df["Method"] != 5]
    df = df[df["Method"] != 10]
    df = df[df["Method"] != 15]
    df = df[df["T0"] == 1]

    sns.boxplot("Problem", "Fitness", "Method", df)
    plt.show()


def heatmap():
    data = np.load(root_path + progress_path)[:-5, :, :, :, :, :, -1]
    for p in range(data.shape[0]):
        data[p] = data[p] - np.min(data[p])
        data[p] = data[p] + 0.01*np.max(data[p])
        data[p] = data[p] / np.max(data[p])
    # data[ip, im, seed - 1, tb, t1, t0, :]
    variances = np.var(data, axis=(2, 3, 4))
    means = np.mean(data, axis=(2, 3, 4))

    v = []
    m = []
    for i in range(variances.shape[0]):
        v += [[]]
        m += [[]]
        for j in range(variances.shape[1]):
            v[-1] += [variances[i, j, 0], variances[i, j, 1]]
            m[-1] += [means[i, j, 0], means[i, j, 1]]

    m = np.array(m)
    v = np.array(v)

    fig, ax = plt.subplots()
    ax_c = ax.twiny()
    pos = ax.imshow(m)
    for i in range(17):
        for j in range(10):
            ax.text(j, i, -np.int(np.floor(np.log10(v[i, j]))), ha="center", va="center", color="w" if m[i, j] < 0.5 else "b")
    color_bar = fig.colorbar(pos, ax=ax_c)
    color_bar.set_label("Mean scaled fitness", size="x-large")
    ax.set_yticks(np.arange(0, 17.5, 1), minor=False)
    ax.set_ylim((-0.5, 16.5))
    ax.set_yticklabels(problem_names, size="large")
    ax.set_xticks(np.arange(-0.5, 10.5, 2), minor=True)
    ax.xaxis.grid(True, which='minor', linewidth=2, color="w")

    ax.set_xticks(np.arange(0.5, 10, 2), minor=False)
    ax.set_xticklabels(short_names, size="large")
    # ax.xaxis.grid(True, which='minor', linewidth=2, color="w")
    # ax.set_xticks(np.arange(0.5, 6.5, 1), minor=True)
    #
    ax_c.set_xlim((0, 10))
    ax_c.set_xticks(np.arange(0.5, 10, 1), minor=False)
    ax_c.set_xticklabels(["M", "R"] * 5)

    ax.set_xlabel("Method", size="x-large")
    ax.set_ylabel("Problem", size="x-large")
    ax_c.set_xlabel("Variant", size="x-large")

    plt.show()

def par_coord():
    noisy_df = pd.read_csv(root_path + noisy_path, index_col=0, dtype={"Problem": np.int32, "Method": np.float32, "Seed": np.int32, "Tb": np.float32, "T1": np.float32, "T0": np.float32, "Fitness": np.float32})
    del noisy_df["Seed"]
    del noisy_df["Fitness"]
    noisy_df = noisy_df[noisy_df["Problem"] != 2]
    noisy_df = noisy_df[noisy_df["Problem"] !=3]
    noisy_df = noisy_df[noisy_df["Problem"] !=4]
    noisy_df = noisy_df[noisy_df["Problem"] !=7]
    noisy_df = noisy_df[noisy_df["Problem"] !=8]
    noisy_df = noisy_df[noisy_df["Problem"] !=9]
    noisy_df = noisy_df[noisy_df["Problem"] !=10]
    noisy_df = noisy_df[noisy_df["Problem"] !=11]
    noisy_df = noisy_df[noisy_df["Problem"] !=12]
    noisy_df = noisy_df[noisy_df["Problem"] !=13]
    noisy_df = noisy_df[noisy_df["Problem"] !=14]
    noisy_df = noisy_df[noisy_df["Problem"] !=15]
    noisy_df = noisy_df[noisy_df["Problem"] !=16]
    noisy_df = noisy_df[noisy_df["Problem"] !=17]
    noisy_df = noisy_df[noisy_df["Problem"] !=18]
    noisy_df = noisy_df[noisy_df["Problem"] !=19]
    noisy_df = noisy_df[noisy_df["Problem"] !=20]
    noisy_df = noisy_df[noisy_df["Problem"] != 21]
    noisy_df = noisy_df[noisy_df["Problem"] != 22]
    noisy_df = noisy_df[noisy_df["Problem"] != 23]
    noisy_df = noisy_df[noisy_df["Problem"] != 24]


    viridis = cm.get_cmap('viridis', len(problems))
    colors = [viridis(i) for i in range(len(problems))]
    parallel_coordinates(noisy_df, class_column="Problem", color=colors)
    plt.yticks(methods, np.array(method_names)[methods])
    plt.xticks([0, 1, 2, 3], ["Method", "Starting point", "Bad sols. in ret.", "Good sols. in ret."])
    custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(problems))]

    plt.legend(custom_lines, ["Function " + str(i+1) for i in problems], loc=(0.57, 0.35))
    plt.text(1, -0.5, "Worst solutions", horizontalalignment='center', verticalalignment='center')
    plt.text(1, 16.5, "Random solutions", horizontalalignment='center', verticalalignment='center')
    plt.text(2.5, -0.5, "Mutate and use as start", horizontalalignment='center', verticalalignment='center')
    plt.text(2.5, 16.5, "Ignore and select random", horizontalalignment='center', verticalalignment='center')

    plt.show()


if __name__ == "__main__":
    #heatmap()
    #load_data()
    #end_box()
    par_coord()
    #curves()
