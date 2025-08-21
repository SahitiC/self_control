from scipy.stats import poisson
import seaborn as sns
import mdp_algms
import diff_discount_factors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['text.usetex'] = False

# %%


def get_reward_functions_precommit(
        states, reward_do, effort_do, reward_completed, cost_not_completed,
        effort_precommit):
    """
    reward and cost functions when there is a choice to precommit in state 0;
    this action leads to a 'precommited state' where the action to shirk is
    not available
    """

    reward_func = []
    cost_func = []
    # reward for actions (depends on current state and next state)
    # rewards for shirk, precommit & do in state 0
    reward_func.append([np.array([0, 0, 0]),
                        np.array([0, 0, 0]),
                        np.array([0, 0, reward_do])])
    # in precommitted state only work:
    reward_func.append([np.array([0, 0, reward_do])])
    # rewards for completed (state 2)
    reward_func.append([np.array([0, 0, 0])])

    # reward from final evaluation for the three states
    reward_func_last = np.array([0, 0, reward_completed])

    # effort for actions (depends on current state and next state)
    # assume no cost for precommitment
    cost_func.append([np.array([0, 0, 0]),
                      np.array([effort_precommit, effort_precommit,
                                effort_precommit]),
                      np.array([effort_do, effort_do, effort_do])])
    cost_func.append([np.array([effort_do, effort_do, effort_do])])
    cost_func.append([np.array([0, 0, 0])])

    # reward from final evaluation for the two states
    cost_func_last = np.array([cost_not_completed, cost_not_completed, 0])

    return reward_func, cost_func, reward_func_last, cost_func_last


def get_transition_prob(states, efficacy):

    T = np.full(len(states), np.nan, dtype=object)

    # for 3 states: base, precommited, completed
    # transitions for shirk, precommit, work in state 0
    T[0] = [np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1-efficacy, 0, efficacy])]
    T[1] = [np.array([0, 1-efficacy, efficacy])]  # transitions for precommited
    T[2] = [np.array([0, 0, 1])]  # transitions for completed

    return T

# %%


N_PRECOMMIT = 1  # 0 otherwise
# intermediate + initial and finished states (2)
STATES = np.arange(2 + N_PRECOMMIT)

# actions available in each state
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
# actions for all states but final:
ACTIONS[0] = ['shirk', 'precommit', 'work']
ACTIONS[1] = ['work']
ACTIONS[2] = ['shirk']  # actions for final state

HORIZON = 6  # deadline
DISCOUNT_FACTOR_REWARD = 0.8  # discounting factor for rewards
DISCOUNT_FACTOR_COST = 0.4  # discounting factor for costs
EFFICACY = 0.6  # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 2.0
EFFORT_DO = -1.0
EFFORT_PRECOMMIT = -0.0
# no delayed rewards:
REWARD_COMPLETED = 0.0
COST_NOT_COMPLETED = -0.0

# %%

reward_func, cost_func, reward_func_last, cost_func_last = (
    get_reward_functions_precommit(
        STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED, COST_NOT_COMPLETED,
        EFFORT_PRECOMMIT))

T = get_transition_prob(STATES, EFFICACY)

policy_state_0, effective_naive_policy, Q_values_full_naive, V_full_naive = (
    diff_discount_factors.get_naive_policy(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST,
        reward_func, cost_func, reward_func_last, cost_func_last, T))

level_no = HORIZON-1
Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels = (
    diff_discount_factors.get_policy_self_control_actions(
        level_no, Q_values_full_naive, effective_naive_policy, STATES, ACTIONS,
        HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, reward_func,
        cost_func, reward_func_last, cost_func_last, T))

diff_discount_factors.plot_heatmap(
    np.array(policy_levels_state_0), cmap=sns.color_palette('husl', 3),
    ylabel='level k effective policy', xlabel='agent at timestep',
    colorbar_ticks=[0.4, 1, 1.6],
    colorbar_ticklabels=['SHIRK', 'PRECOMMIT', 'WORK'], vmin=0, vmax=2)

# diff_discount_factors.plot_Q_value_diff(
#     np.array(Q_diff_levels_state_0), 'coolwarm',
#     ylabel='level k diff in Q-values \n (WORK-SHIRK)',
#     xlabel='agent at timestep', vmin=-0.65, vmax=0.65)
