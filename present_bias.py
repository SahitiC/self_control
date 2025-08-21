from scipy.stats import poisson
import seaborn as sns
import mdp_algms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['text.usetex'] = False

# %%


def self_control_with_actions(prev_level_effective_policy, states, actions,
                              horizon, discount_beta, discount_delta,
                              reward_func, reward_func_last, T):

    V_real_full = []
    Q_values_full = []

    # solve for optimal policy for i_iter-agent,
    # given real actions of future agents
    for i_iter in range(horizon-1, -1, -1):

        V_real = np.zeros((len(states), horizon+1))
        Q_values = np.zeros(len(states), dtype=object)

        for i_state, state in enumerate(states):

            # arrays to store Q-values for each action in each state
            Q_values[i_state] = np.full((len(actions[i_state]), horizon),
                                        np.nan)

            # "Q_values" for last time-step
            V_real[i_state, -1] = (
                (discount_beta*discount_delta**(horizon-i_iter))
                * reward_func_last[i_state])

        # backward induction to derive optimal policy starting from
        # timestep i_iter
        for i_timestep in range(horizon-1, i_iter-1, -1):

            for i_state, state in enumerate(states):

                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    if i_timestep == i_iter:
                        r = reward_func[i_state][i_action]
                    else:
                        r = ((discount_beta
                              * discount_delta**(i_timestep-i_iter))
                             * reward_func[i_state][i_action])

                    # q-value for each action (bellman equation)
                    Q[i_action] = (T[i_state][i_action] @ r.T
                                   + T[i_state][i_action]
                                   @ V_real[states, i_timestep+1])

                Q_values[i_state][:, i_timestep] = Q

                # what are the real V's? i.e. not the max Q value
                # but the Q-value of the best action of the level-1 agent
                V_real[i_state, i_timestep] = Q[
                    prev_level_effective_policy[i_state, i_timestep]]

        V_real_full.append(V_real)
        Q_values_full.append(Q_values)

    return V_real_full, Q_values_full

# def get_rewards_at_end(reward_tempt, reward_resist, effort_tempt)

# %%


states = np.arange(3)

# actions available in each state
actions = np.full(len(states), np.nan, dtype=object)
# actions for all states but final:
actions[:-1] = [['tempt', 'resist']
                for i in range(len(states)-1)]
actions[-1] = ['final']  # actions for final state

horizon = 4  # deadline
discount_beta = 0.3  # discounting factor for rewards
discount_delta = 0.8  # discounting factor for costs

# utilities :
effort_resist = 0
reward_tempt = 0.5
reward_resist = 2.0

# get reward matrix
reward_func = []
for i in range(len(states)-1):
    reward_func.append([np.full(len(states), reward_tempt),
                        np.full(len(states), effort_resist)])
reward_func.append([np.full(len(states), 0.0)])  # rewards for completed
# reward_func_last = [reward_resist*i for i in range(len(states)-1, -1, -1)]
reward_func_last = [4.7, 2.0, 0.0]

# get transition function
T = []
for state in range(len(states)-1):
    T_state = []
    for action in range(len(actions[state])):
        t = np.full(len(states), 0.0)
        t[1-action+state] = 1.0
        T_state.append(t)
    T.append(T_state)
# for final state:
t = np.full(len(states), 0.0)
t[-1] = 1.0
T.append([t])


# %%
Q_diffs_state_0 = []
policys_state_0 = []

V_opt_full, policy_full, Q_values_full = mdp_algms.find_optimal_policy_beta_delta(
    states, actions, horizon, discount_beta, discount_delta,
    reward_func, reward_func_last, T)

state_to_plot = 0

policy_state_0 = [policy_full[i][state_to_plot] for i in range(horizon)]
policy_state_0 = np.array(policy_state_0)
# actual policy followed by agent
effective_naive_policy = []
for state in states:
    effective_naive_policy.append(np.array(
        [policy_full[horizon-1-i][state][i] for i in range(horizon)]))
effective_naive_policy = np.array(effective_naive_policy, dtype=int)
policys_state_0.append(effective_naive_policy[state_to_plot])
Q_diff_naive = []
for t in range(horizon):
    Q_diff_naive.append(np.diff(
        Q_values_full[horizon-1-t][state_to_plot][:, t])[0])
Q_diffs_state_0.append(Q_diff_naive)

# calculate level 1 policy
effective_policy_prev_level = effective_naive_policy
V_level_1, Q_level_1 = self_control_with_actions(
    effective_policy_prev_level, states, actions, horizon, discount_beta,
    discount_delta, reward_func, reward_func_last, T)
# update effective policy, Q_diff
effective_policy_level_1 = np.full((len(states), horizon), 100)
Q_diff = []  # diff only for state=0
for t in range(horizon):
    Q_diff.append(np.diff(Q_level_1[horizon-1-t][state_to_plot][:, t])[0])
    for state in states:
        effective_policy_level_1[state, horizon-1-t] = np.argmax(
            Q_level_1[t][state][:, horizon-1-t])
Q_diffs_state_0.append(Q_diff)
policys_state_0.append(effective_policy_level_1[state_to_plot])

# calculate level 2 policy
effective_policy_prev_level = effective_policy_level_1
V_level_2, Q_level_2 = self_control_with_actions(
    effective_policy_prev_level, states, actions, horizon, discount_beta,
    discount_delta, reward_func, reward_func_last, T)
# update effective policy, Q_diff
effective_policy_level_2 = np.full((len(states), horizon), 100)
Q_diff = []  # diff only for state=0
for t in range(horizon):
    Q_diff.append(np.diff(Q_level_2[horizon-1-t][state_to_plot][:, t])[0])
    for state in states:
        effective_policy_level_2[state, horizon-1-t] = np.argmax(
            Q_level_2[t][state][:, horizon-1-t])
Q_diffs_state_0.append(Q_diff)
policys_state_0.append(effective_policy_level_2[state_to_plot])
# next level is exacly the same since the policy is the same


# calculate level 3 policy
effective_policy_prev_level = effective_policy_level_2
V_level_3, Q_level_3 = self_control_with_actions(
    effective_policy_prev_level, states, actions, horizon, discount_beta,
    discount_delta, reward_func, reward_func_last, T)
# update effective policy, Q_diff
effective_policy_level_3 = np.full((len(states), horizon), 100)
Q_diff = []  # diff only for state=0
for t in range(horizon):
    Q_diff.append(np.diff(Q_level_3[horizon-1-t][state_to_plot][:, t])[0])
    for state in states:
        effective_policy_level_3[state, horizon-1-t] = np.argmax(
            Q_level_3[t][state][:, horizon-1-t])
Q_diffs_state_0.append(Q_diff)
policys_state_0.append(effective_policy_level_3[state_to_plot])
# next level is exacly the same since the policy is the same

policy_state_0 = [policy_full[i][state_to_plot] for i in range(horizon)]
f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(policy_state_0), linewidths=.5,
            cmap=sns.color_palette('husl', 2), vmin=0, vmax=1)
ax.set_xlabel('time')
ax.set_ylabel('horizon')
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['DEFECT', 'RESIST'])

f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(Q_diffs_state_0), linewidths=.5, cmap='coolwarm',
            vmin=-0.6, vmax=0.6)
ax.set_xlabel('agent at timestep')
ax.set_ylabel('level k diff in Q-values \n (resist - defect)')
ax.tick_params()

f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(policys_state_0), linewidths=.5,
            cmap=sns.color_palette('husl', 2), vmin=0, vmax=1)
ax.set_xlabel('agent at timestep')
ax.set_ylabel('level k effective policy')
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['DEFECT', 'RESIST'])

# %% precommit

n_precommit = 1  # 0 otherwise
states = np.arange(2+n_precommit)

# actions available in each state
actions = np.full(len(states), np.nan, dtype=object)
# actions for all states but final:
actions[0] = ['tempt', 'precommit', 'resist']
actions[1] = ['resist']
actions[-1] = ['final']  # actions for final state

horizon = 3  # deadline
discount_beta = 0.3  # discounting factor for rewards
discount_delta = 0.8  # discounting factor for costs

# utilities :
effort_resist = 0.0
reward_tempt = 0.5
reward_resist = 2.0
effort_precommit = 0.0

# get reward matrix
reward_func = []
reward_func.append([np.full(len(states), reward_tempt),
                    np.full(len(states), effort_precommit),
                    np.full(len(states), effort_resist)])
reward_func.append([np.full(len(states), effort_resist)])
reward_func.append([np.full(len(states), 0.0)])  # rewards for completed
reward_func_last = [reward_resist, reward_resist, 0]

# get transition function
T = np.full(len(states), np.nan, dtype=object)

# for 3 states: base, precommited, final
# transitions for tempt, precommit, resist in state 0
T[0] = [np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0, 0])]
T[1] = [np.array([0, 1, 0])]  # transitions for precommited
T[2] = [np.array([0, 0, 1])]  # transitions for completed


# %%
Q_diffs_state_0 = []
policys_state_0 = []

V_opt_full, policy_full, Q_values_full = mdp_algms.find_optimal_policy_beta_delta(
    states, actions, horizon, discount_beta, discount_delta,
    reward_func, reward_func_last, T)

state_to_plot = 1

policy_state_0 = [policy_full[i][state_to_plot] for i in range(horizon)]
policy_state_0 = np.array(policy_state_0)
# actual policy followed by agent
effective_naive_policy = []
for state in states:
    effective_naive_policy.append(np.array(
        [policy_full[horizon-1-i][state][i] for i in range(horizon)]))
effective_naive_policy = np.array(effective_naive_policy, dtype=int)
policys_state_0.append(effective_naive_policy[state_to_plot])
Q_diff_naive = []
for t in range(horizon):
    Q_diff_naive.append(np.diff(
        Q_values_full[horizon-1-t][state_to_plot][:, t])[0])
Q_diffs_state_0.append(Q_diff_naive)

policy_state_0 = [policy_full[i][state_to_plot] for i in range(horizon)]
f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(policy_state_0), linewidths=.5,
            cmap=sns.color_palette('husl', 3), vmin=0, vmax=2)
ax.set_xlabel('time')
ax.set_ylabel('horizon')
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.4, 1, 1.6])
colorbar.set_ticklabels(['DEFECT', 'PRECOMMIT', 'RESIST'])

effective_policy_prev_level = effective_naive_policy
V_level_1, Q_level_1 = self_control_with_actions(
    effective_policy_prev_level, states, actions, horizon, discount_beta,
    discount_delta, reward_func, reward_func_last, T)
# update effective policy, Q_diff
effective_policy_level_1 = np.full((len(states), horizon), 100)
Q_diff = []  # diff only for state=0
for t in range(horizon):
    Q_diff.append(np.diff(Q_level_1[horizon-1-t][state_to_plot][:, t])[0])
    for state in states:
        effective_policy_level_1[state, horizon-1-t] = np.argmax(
            Q_level_1[t][state][:, horizon-1-t])
Q_diffs_state_0.append(Q_diff)
policys_state_0.append(effective_policy_level_1[state_to_plot])
# next level is exacly the same since the policy is the same

f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(policys_state_0), linewidths=.5,
            cmap=sns.color_palette('husl', 3), vmin=0, vmax=2)
ax.set_xlabel('agent at timestep')
ax.set_ylabel('level k effective policy')
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.4, 1, 1.6])
colorbar.set_ticklabels(['DEFECT', 'PRECOMMIT', 'RESIST'])
