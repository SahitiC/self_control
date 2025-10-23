"""
Demonstration of planning with habits without inconsistencies
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import mdp_algms
import seaborn as sns
import task_structure

# %%

# construct reward functions separately for rewards and costs


def get_reward_functions(states, reward_do, effort_do, reward_completed,
                         cost_not_completed):

    reward_func = []

    # reward for actions (depends on current state and next state)
    # rewards for don't and do
    reward_func.append([np.array([0, 0]),
                        np.array([effort_do, reward_do+effort_do])])
    reward_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    reward_func_last = np.array([cost_not_completed, reward_completed])

    return reward_func, reward_func_last


def plan_with_habits(p_sticky, states, actions, horizon, discount_factor,
                     reward_func, reward_func_last, T):
    """Derive optimal higher level policy taking stickiness into account."""

    # arrays for optimal values, policy, Q-values
    V_opt = np.full((len(states), horizon+1), np.nan)
    policy_opt = np.full((len(states), horizon), 100)
    Q_values = np.full(len(states), np.nan, dtype=object)

    for i_state, state in enumerate(states):

        # V_opt for last time-step
        V_opt[i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full((len(actions[i_state]), horizon), np.nan)

    # backward induction to derive optimal policy
    for i_timestep in range(horizon-1, -1, -1):

        for i_state, state in enumerate(states):

            Q = np.full(len(actions[i_state]), np.nan)

            for i_action, action in enumerate(actions[i_state]):

                # value of next state (if same as current state) in the next
                # timestep is: p * value of current action in next state +
                # (1-p) * value of optimal action in next state
                if i_timestep < horizon-1:
                    value_next = np.full(len(states), 0.0)
                    for next_state in range(len(states)):
                        if next_state == i_state:
                            opt_action = policy_opt[next_state, i_timestep+1]
                            value_next[next_state] = (
                                p_sticky *
                                Q_values[next_state][i_action, i_timestep+1]
                                + (1-p_sticky) *
                                Q_values[next_state][opt_action, i_timestep+1])
                        else:
                            value_next[next_state] = V_opt[next_state,
                                                           i_timestep+1]

                else:
                    value_next = V_opt[:, i_timestep+1]

                # q-value for each action (bellman equation)
                Q[i_action] = (T[i_state][i_action]
                               @ reward_func[i_state][i_action].T
                               + discount_factor * (T[i_state][i_action]
                                                    @ value_next))

            # find optimal action (which gives max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            Q_values[i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values

# %%


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

HORIZON = 3  # deadline
DISCOUNT_FACTOR = 0.9  # common d iscount factor for both
EFFICACY = 1.0  # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 0.0
EFFORT_DO = -1.0
# no delayed rewards:
REWARD_COMPLETED = 2.0
COST_NOT_COMPLETED = -0.0


reward_func, reward_func_last = task_structure.rewards_procrastination_common(
    STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED, COST_NOT_COMPLETED)

T = task_structure.transitions_procrastination(STATES, EFFICACY)

# %% policies

# level -1: play with environment
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func, reward_func_last,
    T)

# level 0: take habits into account
p_sticky = 0.6
V_opt_habit, policy_opt_habit, Q_values_habit = plan_with_habits(
    p_sticky, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
    reward_func_last, T)

# %% vary params

policies = []

discount = 0.9
T = task_structure.transitions_procrastination(STATES, EFFICACY)
# level -1
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, discount, reward_func, reward_func_last,
    T)
policies.append(policy_opt[0])

# level 0: take habits into account
for p_sticky in [0.1, 0.3, 0.6, 0.9]:
    V_opt_habit, policy_opt_habit, Q_values_habit = plan_with_habits(
        p_sticky, STATES, ACTIONS, HORIZON, discount, reward_func,
        reward_func_last, T)
    policies.append(policy_opt_habit[0])


f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(policies, linewidths=.5,
            cmap=sns.color_palette('husl', 2),
            vmin=0,
            vmax=1)
ax.set_xlabel('timestep')
ax.set_ylabel('p_sticky')
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['SHIRK', 'WORK'])
ax.set_yticklabels([0, 0.1, 0.3, 0.6, 0.9])
plt.show()

# %%
