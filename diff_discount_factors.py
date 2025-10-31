# %%
import task_structure
import helper
import mdp_algms
import self_control
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2

# %% procrastination with different discount factors

# states of markov chain
N_INTERMEDIATE_STATES = 0
# intermediate + initial and finished states (2)
STATES = np.arange(2 + N_INTERMEDIATE_STATES)

# actions available in each state
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
# actions for all states but final:
ACTIONS[:-1] = [['shirk', 'work']
                for i in range(len(STATES)-1)]
ACTIONS[-1] = ['shirk']  # actions for final state

HORIZON = 6  # deadline
DISCOUNT_FACTOR_REWARD = 0.7  # discounting factor for rewards
DISCOUNT_FACTOR_COST = 0.5  # discounting factor for costs
DISCOUNT_FACTOR_COMMON = 0.9  # common d iscount factor for both
EFFICACY = 0.6  # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 2.0
EFFORT_DO = -1.0
# no delayed rewards:
REWARD_COMPLETED = 0.0
COST_COMPLETED = -0.0

reward_func, cost_func, reward_func_last, cost_func_last = (
    task_structure.rewards_efforts_procrastination_separate(
        STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED))

T = task_structure.transitions_procrastination(STATES, EFFICACY)

# %% naive policy

V_full_naive, policy_full_naive, Q_values_full_naive = (
    mdp_algms.find_optimal_policy_diff_discount_factors(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST,
        reward_func, cost_func, reward_func_last, cost_func_last, T))

effective_naive_policy = helper.get_effective_policy(
    STATES, policy_full_naive, HORIZON)

state_to_get = 0
policy_state = np.array([policy_full_naive[i][state_to_get]
                         for i in range(HORIZON)])
Q_values = [Q_values_full_naive[i][state_to_get] for i in range(HORIZON)]
Q_diff_full = np.array([a[1]-a[0] for a in Q_values])

helper.plot_heatmap(policy_state, cmap=sns.color_palette('husl', 2),
                    ylabel='horizon', xlabel='timestep', vmin=0, vmax=1)
helper.plot_Q_value_diff(Q_diff_full, cmap='coolwarm', ylabel='horizon',
                         xlabel='timestep', vmin=-0.7, vmax=0.7)

# %% self control policy


level_no = HORIZON-1
Q_diff_levels_state, policy_levels_state, policy_full_levels = (
    self_control.get_all_levels_self_control(
        level_no, Q_values_full_naive, effective_naive_policy, STATES, ACTIONS,
        HORIZON, T, reward_func, reward_func_last, cost_func=cost_func,
        cost_func_last=cost_func_last,
        discount_factor_reward=DISCOUNT_FACTOR_REWARD,
        discount_factor_cost=DISCOUNT_FACTOR_COST, disc_func='diff_disc',
        state_to_get=state_to_get))

helper.plot_heatmap(np.array(policy_levels_state),
                    cmap=sns.color_palette('husl', 2),
                    ylabel='level k effective policy',
                    xlabel='agent at timestep')

helper.plot_Q_value_diff(np.array(Q_diff_levels_state), 'coolwarm',
                         ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                         xlabel='agent at timestep', vmin=-0.65, vmax=0.65)

# %% plan with stickiness
P_STICKY = 0.7
Q_diff_levels_state, policy_levels_state, policy_full_levels = (
    self_control.get_all_levels_self_control(
        level_no, Q_values_full_naive, effective_naive_policy, STATES, ACTIONS,
        HORIZON, T, reward_func, reward_func_last, cost_func=cost_func,
        cost_func_last=cost_func_last,
        discount_factor_reward=DISCOUNT_FACTOR_REWARD,
        discount_factor_cost=DISCOUNT_FACTOR_COST, disc_func='diff_disc',
        state_to_get=state_to_get, sticky=True, p_sticky=P_STICKY))

helper.plot_heatmap(np.array(policy_levels_state),
                    cmap=sns.color_palette('husl', 2),
                    ylabel='level k effective policy',
                    xlabel='agent at timestep')

helper.plot_Q_value_diff(np.array(Q_diff_levels_state), 'coolwarm',
                         ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                         xlabel='agent at timestep', vmin=-0.65, vmax=0.65)

# %%
