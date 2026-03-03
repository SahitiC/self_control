# %%
import mdp_algms
import task_structure
import helper
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%


def bamdp_inference(states, actions, horizon, discount_factor,
                    T, dr, reward_func_low, reward_func_high, reward_func_true,
                    reward_func_last_low, reward_func_last_high,
                    reward_func_last_true, policy_low, policy_high,
                    eps=10**-6):

    # belief that r_tempt is r_high (instead of r_low)
    belief_r = np.arange(0, 1 + dr, dr)

    # arrays for optimal values, policy, Q-values
    V_opt = np.full((len(belief_r), len(states), horizon+1), np.nan)
    policy_opt = np.full((len(belief_r), len(states), horizon), 100)
    Q_values = np.full((len(belief_r), len(states)), np.nan, dtype=object)

    for i_state, state in enumerate(states):
        for i_b, b in enumerate(belief_r):
            # expected V_opt for last time-step
            V_opt[i_b, i_state, -1] = reward_func_last_high[i_state] * b + \
                reward_func_last_low[i_state] * (1 - b)
            # arrays to store Q-values for each action in each state
            Q_values[i_b, i_state] = np.full(
                (len(actions[i_state]), horizon), np.nan)

    # backward induction
    for i_timestep in range(horizon-1, -1, -1):
        for i_b, b in enumerate(belief_r):
            for i_state, _ in enumerate(states):
                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, _ in enumerate(actions[i_state]):

                    expected_r = reward_func_high[i_state][i_action] * b + \
                        reward_func_low[i_state][i_action] * (1 - b)

                    # belief update: if 'action' is taken, then what is the
                    # probability that r_tempt is r_high
                    # is action optimal according to policy_high or policy_low?
                    # or both ?
                    p_high = ((1-eps) * int(
                        policy_high[i_state, i_timestep] == i_action) + eps)
                    p_low = ((1-eps) * int(
                        policy_low[i_state, i_timestep] == i_action) + eps)
                    b_next = (p_high * b) / (p_high * b + p_low * (1 - b))
                    i_b_next = int(b_next/dr)

                    Q[i_action] = (T[i_state][i_action] @ expected_r.T
                                   + discount_factor
                                   * (T[i_state][i_action]
                                      @ V_opt[i_b_next, states, i_timestep+1]))

                # find optimal action (which gives max q-value)
                V_opt[i_b, i_state, i_timestep] = np.max(Q)
                policy_opt[i_b, i_state, i_timestep] = np.argmax(Q)
                Q_values[i_b, i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values


def simulate_behaviour_with_inference(
        initial_belief, initial_state, policy_opt_infer, T, states,
        policy_opt_low, policy_opt_high, dr, horizon, eps=10**-6, plot=False):

    state = initial_state
    b = initial_belief
    belief_trajectory = [b]
    actions_trajectory = []

    for t in range(horizon):
        i_b = int(b/dr)
        action = policy_opt_infer[i_b, state, t]
        actions_trajectory.append(action)

        # belief update
        p_high = ((1-eps) * int(policy_opt_high[state, t] == action) + eps)
        p_low = ((1-eps) * int(policy_opt_low[state, t] == action) + eps)
        b_next = (p_high * b) / (p_high * b + p_low * (1 - b))
        belief_trajectory.append(b_next)
        b = b_next

        # state update
        state_next = np.random.choice(len(states), p=T[state][action])
        state = state_next

    if plot:
        plt.figure(figsize=(5, 4))
        time = np.arange(horizon)
        plt.plot(belief_trajectory, label='belief')
        plt.scatter(time, actions_trajectory, color='red', marker='x',
                    label='actions')
        plt.xticks(np.arange(horizon+1))
        plt.xlabel('time step')
        plt.legend()

    return belief_trajectory, actions_trajectory
# %%


# STATES = 0 (no health reward pending), 1 (health reward pending)
# if agent resists, state -> 1
# if agent gets tempted, state -> 0
STATES = np.arange(2)

# actions available in each state
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
ACTIONS = [['tempt', 'resist']
           for i in range(len(STATES))]

HORIZON = 4  # deadline
DISCOUNT_FACTOR = 0.8

# utilities :
EFFORT_RESIST = 0
REWARD_RESIST = 0.8

# get transition function
T = task_structure.transitions_cake()

# say reward_tempt is unknown: and is one of two, high or low
dr = 0.01
r_tempts = [0.3, 0.7]  # r_low, r_high
# probability of r_tempt being r_high
belief_r = np.arange(0, 1 + dr, dr)
reward_func_low, reward_func_last_low = task_structure.rewards_cake(
    STATES, r_tempts[0], EFFORT_RESIST, REWARD_RESIST)
reward_func_high, reward_func_last_high = task_structure.rewards_cake(
    STATES, r_tempts[1], EFFORT_RESIST, REWARD_RESIST)

_, policy_opt_low, _ = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func_low,
    reward_func_last_low, T)
_, policy_opt_high, _ = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func_high,
    reward_func_last_high, T)

f, ax = plt.subplots(figsize=(5, 2))
sns.heatmap(policy_opt_low[0, :][np.newaxis, :], linewidths=0.5,
            cmap=sns.color_palette('husl', 2), cbar=False, vmin=0, vmax=1)
ax.set_yticks([])
ax.set_xlabel('time step')

f, ax = plt.subplots(figsize=(5, 2))
sns.heatmap(policy_opt_high[0, :][np.newaxis, :], linewidths=0.5,
            cmap=sns.color_palette('husl', 2), cbar=False, vmin=0, vmax=1)
ax.set_yticks([])
ax.set_xlabel('time step')

# %% inference
reward_func_true = reward_func_low
reward_func_last_true = reward_func_last_high
V_opt_infer, policy_opt_infer, Q_values_infer = bamdp_inference(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, T, dr, reward_func_low,
    reward_func_high, reward_func_true, reward_func_last_low,
    reward_func_last_high, reward_func_last_true,
    policy_opt_low, policy_opt_high)

# %% plot policy
f, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(policy_opt_infer[:, 0, :],
            cmap=sns.color_palette('husl', 2))
ax.set_yticks(np.arange(0, (1/dr)+1, int(0.1/dr)))
ax.set_yticklabels(np.round(np.arange(0, (1/dr)+1, int(0.1/dr))*dr, 1))
ax.set_ylabel('belief(r_tempt = r_high)')
ax.set_xlabel('time step')
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['DEFECT', 'COOPERATE'])

# %% sample trajectory

initial_belief = 0.5
initial_state = 0
beliefs, actions = simulate_behaviour_with_inference(
    initial_belief, initial_state, policy_opt_infer, T, STATES,
    policy_opt_low, policy_opt_high, dr, HORIZON, eps=10**-6, plot=True)

# %%
