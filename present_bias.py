# %%
import task_structure
import helper
import mdp_algms
import self_control
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2

# %% health rewards one step later

exception = True

if exception:

    # exceptions

    # same situation as below but with additional exception states

    P_EXCEPTION = 0.1  # prob of exception happening
    # 4 STATES:
    # 0 (no health reward pending), 1 (health reward pending)
    # 2, 3 (exception happened, (no) health reward pending)
    STATES = np.arange(4)

    # actions available in each state
    ACTIONS = np.full(len(STATES), np.nan, dtype=object)
    ACTIONS = [['tempt', 'resist']
               for i in range(len(STATES))]

    HORIZON = 4  # deadline
    DISCOUNT_BETA = 0.7  # discounting factor for rewards
    DISCOUNT_DELTA = 0.8  # discounting factor for costs

    # utilities:
    EFFORT_RESIST = 0
    REWARD_TEMPT_NORMAL = 0.5
    REWARD_RESIST_NORMAL = 0.8
    REWARD_TEMPT_EXCEPTION = 0.65
    REWARD_RESIST_EXCEPTION = 0.8

    reward_func1, _ = task_structure.rewards_cake(
        STATES, REWARD_TEMPT_NORMAL, EFFORT_RESIST, REWARD_RESIST_NORMAL)

    reward_func2, _ = task_structure.rewards_cake(
        STATES, REWARD_TEMPT_EXCEPTION, EFFORT_RESIST, REWARD_RESIST_EXCEPTION)

    reward_func = reward_func1 + reward_func2

    reward_func_last = np.array([0.0, REWARD_RESIST_NORMAL,
                                 0.0, REWARD_RESIST_EXCEPTION])

    # transition function for all STATES:
    # with prob 1-p_expception, tempt -> state 0, if resist -> state 1
    # prob p_exception of exception, then tempt -> state 2, resist -> state 3
    T = []
    for state in range(len(STATES)):
        T.append([np.array([1-P_EXCEPTION, 0, P_EXCEPTION, 0]),
                  np.array([0, 1-P_EXCEPTION, 0, P_EXCEPTION])])

else:

    # STATES = 0 (no health reward pending), 1 (health reward pending)
    # if agent resists, state -> 1
    # if agent gets tempted, state -> 0
    STATES = np.arange(2)

    # actions available in each state
    ACTIONS = np.full(len(STATES), np.nan, dtype=object)
    ACTIONS = [['tempt', 'resist']
               for i in range(len(STATES))]

    HORIZON = 4  # deadline
    DISCOUNT_BETA = 0.7  # discounting factor for rewards
    DISCOUNT_DELTA = 0.8  # discounting factor for costs

    # utilities :
    EFFORT_RESIST = 0
    REWARD_TEMPT = 0.5
    REWARD_RESIST = 0.8

    reward_func, reward_func_last = task_structure.rewards_cake(
        STATES, REWARD_TEMPT, EFFORT_RESIST, REWARD_RESIST)

    # get transition function
    T = task_structure.transitions_cake()

# %% naive policy
V_full_naive, policy_full_naive, Q_values_full_naive = (
    mdp_algms.find_optimal_policy_beta_delta(
        STATES, ACTIONS, HORIZON, DISCOUNT_BETA, DISCOUNT_DELTA,
        reward_func, reward_func_last, T))

effective_naive_policy = helper.get_effective_policy(
    STATES, policy_full_naive, HORIZON)

state_to_get = 3
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
        HORIZON, T, reward_func, reward_func_last, discount_beta=DISCOUNT_BETA,
        discount_delta=DISCOUNT_DELTA, disc_func='beta_delta',
        state_to_get=state_to_get))

helper.plot_heatmap(np.array(policy_levels_state),
                    cmap=sns.color_palette('husl', 2),
                    ylabel='level k effective policy',
                    xlabel='agent at timestep',
                    colorbar_ticklabels=['TEMPT', 'RESIST'])

helper.plot_Q_value_diff(np.array(Q_diff_levels_state), 'coolwarm',
                         ylabel='level k diff in Q-values \n (RESIST-TEMPT)',
                         xlabel='agent at timestep',
                         vmin=-0.65, vmax=0.65)

# %% plan with one step stickiness
P_STICKY = 0.9
Q_diff_levels_state, policy_levels_state, policy_full_levels = (
    self_control.get_all_levels_self_control(
        level_no, Q_values_full_naive, effective_naive_policy, STATES, ACTIONS,
        HORIZON, T, reward_func, reward_func_last,
        discount_beta=DISCOUNT_BETA, discount_delta=DISCOUNT_DELTA,
        disc_func='beta_delta', state_to_get=state_to_get, sticky='one_step',
        p_sticky=P_STICKY))

helper.plot_heatmap(np.array(policy_levels_state),
                    cmap=sns.color_palette('husl', 2),
                    ylabel='level k effective policy',
                    xlabel='agent at timestep',
                    colorbar_ticklabels=['TEMPT', 'RESIST'])

helper.plot_Q_value_diff(np.array(Q_diff_levels_state), 'coolwarm',
                         ylabel='level k diff in Q-values \n (RESIST-TEMPT)',
                         xlabel='agent at timestep', vmin=-0.65, vmax=0.65)

# %%
p = 0.9
alpha = 0.99
dx = 0.01
Q_diff_levels_state, policy_levels_state, policy_full_levels = (
    self_control.get_all_levels_self_control(
        level_no, Q_values_full_naive, effective_naive_policy, STATES, ACTIONS,
        HORIZON, T, reward_func, reward_func_last,
        discount_beta=DISCOUNT_BETA, discount_delta=DISCOUNT_DELTA,
        disc_func='beta_delta', state_to_get=state_to_get, sticky='multi_step',
        p=p, alpha=alpha, dx=dx))

# simulate actions with level=3 policy
actions_executed, state_trajectory, x_trajectory = (
    self_control.simulate_behavior_with_habit(
        policy_full_levels[3], T, alpha, dx, STATES, ACTIONS, HORIZON,
        plot=True))

# %% vary params
alpha = 0.1
acts = []
for p in [0.0, 0.3, 0.6, 0.9]:

    Q_diff_levels_state, policy_levels_state, policy_full_levels = (
        self_control.get_all_levels_self_control(
            level_no, Q_values_full_naive, effective_naive_policy, STATES,
            ACTIONS, HORIZON, T, reward_func, reward_func_last,
            discount_beta=DISCOUNT_BETA, discount_delta=DISCOUNT_DELTA,
            disc_func='beta_delta', state_to_get=state_to_get,
            sticky='multi_step', p=p, alpha=alpha, dx=dx))

    # simulate actions with level=3 policy
    actions_executed, state_trajectory, x_trajectory = (
        self_control.simulate_behavior_with_habit(
            policy_full_levels[3], T, alpha, dx, STATES, ACTIONS, HORIZON,
            plot=False))

    actions_executed = np.array(actions_executed)
    acts.append(actions_executed)

f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(acts), linewidths=.5, cmap=sns.color_palette('husl', 2))
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['defect', 'cooperate'])
ax.set_xlabel('timestep')
ax.set_ylabel('prob of habit (p)')
ax.set_yticklabels([0.0, 0.3, 0.6, 0.9])

# %%
p = 0.9
acts = []
for alpha in [0.0, 0.5, 0.9]:

    Q_diff_levels_state, policy_levels_state, policy_full_levels = (
        self_control.get_all_levels_self_control(
            level_no, Q_values_full_naive, effective_naive_policy, STATES,
            ACTIONS, HORIZON, T, reward_func, reward_func_last,
            discount_beta=DISCOUNT_BETA, discount_delta=DISCOUNT_DELTA,
            disc_func='beta_delta', state_to_get=state_to_get,
            sticky='multi_step', p=p, alpha=alpha, dx=dx))

    # simulate actions with level=3 policy
    actions_executed, state_trajectory, x_trajectory = (
        self_control.simulate_behavior_with_habit(
            policy_full_levels[3], T, alpha, dx, STATES, ACTIONS, HORIZON,
            plot=False))

    actions_executed = np.array(actions_executed)
    acts.append(actions_executed)

f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(acts), linewidths=.5, cmap=sns.color_palette('husl', 2))
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['defect', 'cooperate'])
ax.set_xlabel('timestep')
ax.set_ylabel('memory strength (alpha)')
ax.set_yticklabels([0.0, 0.5, 0.9])

# %%
