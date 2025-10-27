import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(policy_state, cmap, ylabel='', xlabel='', title='',
                 colorbar_ticks=[0.25, 0.75],
                 colorbar_ticklabels=['SHIRK', 'WORK'], vmin=0, vmax=1):
    """
    heat map of full policy in a given state
    """

    f, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sns.heatmap(policy_state, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params()
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(colorbar_ticklabels)
    f.suptitle(title)


def plot_Q_value_diff(Q_diff, cmap, ylabel, xlabel, title='', vmin=-0.5,
                      vmax=0.5):
    """
    plot diff in Q-values between actions for a given state where there are two
    actions
    """

    f, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sns.heatmap(Q_diff, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params()
    f.suptitle(title)


def get_effective_policy(states, policy_full, horizon):
    effective_policy = []
    for state in states:
        effective_policy.append(np.array(
            [policy_full[horizon-1-i][state][i] for i in range(horizon)]))
    effective_policy = np.array(effective_policy, dtype=int)
    return effective_policy
