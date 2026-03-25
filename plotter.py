import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.size'] = 18


def plot_single_trajectory(actions_executed, w_trajectory, horizon,
                           action_label='cooperate', w_label='w'):
    actions_executed = np.array(actions_executed)
    w_trajectory = np.array(w_trajectory)
    time = np.arange(horizon)
    plt.plot(w_trajectory, label=w_label)
    plt.scatter(time[actions_executed == 1],
                w_trajectory[:-1][actions_executed == 1],
                label=action_label)
    plt.xticks(np.arange(0, horizon+1, 5))
    plt.legend(fontsize=12)


def plot_w_policy(policy, w_grid, dw, horizon):
    f, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(policy,
                cmap=sns.color_palette('husl', 2), cbar=True, vmin=0, vmax=1)
    ax.set_yticks(np.arange(0, len(w_grid), int(len(w_grid)/5)))
    ax.set_yticklabels(np.arange(0, len(w_grid), int(len(w_grid)/5))*dw)
    ax.set_xticks(np.arange(0, horizon+1, 5))
    ax.set_xticklabels(np.arange(0, horizon+1, 5))
    ax.set_xlabel('time step')
    ax.set_ylabel('w')
    ax.invert_yaxis()
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels([0, 1])
    plt.show()
