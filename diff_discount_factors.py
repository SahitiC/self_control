import seaborn as sns
import mdp_algms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['text.usetex'] = False

# %%


# construct reward functions separately for rewards and costs
def get_reward_functions(states, reward_do, effort_do, reward_completed,
                         cost_completed):

    reward_func = []
    cost_func = []
    # reward for actions (dependis on current state and next state)
    reward_func.append([np.array([0, 0]),
                        np.array([0, reward_do])])  # rewards for don't and do
    reward_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    reward_func_last = np.array([0, reward_completed])

    # effort for actions (dependis on current state and next state)
    cost_func.append([np.array([0, 0]),  # rewards for don't and do
                      np.array([effort_do, effort_do])])
    cost_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    cost_func_last = np.array([cost_completed, 0])

    return reward_func, cost_func, reward_func_last, cost_func_last


# construct common reward functions
def get_reward_functions_common(states, reward_do, effort_do, reward_completed,
                                cost_completed):

    reward_func = []
    # reward for actions (dependis on current state and next state)
    reward_func.append([np.array([0, 0]),  # rewards for don't and do
                        np.array([effort_do, effort_do+reward_do])])
    reward_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    reward_func_last = np.array([cost_completed, reward_completed])

    return reward_func, reward_func_last


def get_transition_prob(states, efficacy):

    T = np.full(len(states), np.nan, dtype=object)

    # for 2 states:
    T[0] = [np.array([1, 0]),
            np.array([1-efficacy, efficacy])]  # transitions for shirk, work
    T[1] = [np.array([0, 1])]  # transitions for completed

    return T


def plot_policy(policy_full, state, cmap, vmin=0, vmax=1):
    """
    heat map of full policy in state = state
    """

    policy_state = [policy_full[i][state] for i in range(HORIZON)]
    policy_state = np.array(policy_state)
    f, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sns.heatmap(policy_state, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('timestep')
    ax.set_ylabel('horizon')
    ax.tick_params()
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['SHIRK', 'WORK'])


def plot_Q_value_diff(Q_values_full, cmap, vmin, vmax):
    """
    plot diff in Q-values between actions for state=0 where there are two
    actions
    """

    Q_values = [Q_values_full[i][0] for i in range(HORIZON)]
    Q_diff = [a[1]-a[0] for a in Q_values]
    Q_diff - np.array(Q_diff)
    f, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sns.heatmap(Q_diff, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('timestep')
    ax.set_ylabel('horizon')
    ax.tick_params()
    f.suptitle('diff in Q_values (WORK-SHIRK)')

    return Q_diff


def self_control(prev_level_values, level):

    current_level_values = np.full((len(STATES), HORIZON), np.nan)
    current_level_policy = np.full((len(STATES), HORIZON), np.nan)
    current_level_Q_values = []

    # no. of agents at level = k; for eg for t=3 problem, there are three
    # level -1 agents, two level 0 agents and one level 1 agent
    prev_level_agent_no = HORIZON - level

    for i_iter in range(HORIZON):

        # horizon of agent
        i_horizon = prev_level_agent_no-1-i_iter

        if i_horizon-1 < 0:
            break

        Q_values = []

        for i_state in range(len(STATES)):

            Q = np.full((len(ACTIONS[i_state])), np.nan)

            for i_action in range(len(ACTIONS[i_state])):

                r = ((DISCOUNT_FACTOR_REWARD**(0))
                     * reward_func[i_state][i_action]
                     + (DISCOUNT_FACTOR_COST**(0))
                     * cost_func[i_state][i_action])

                V_true = prev_level_values[:, i_iter+1]

                Q[i_action] = (T[i_state][i_action] @ r.T
                               + T[i_state][i_action] @ V_true)

            Q_values.append(Q)
            current_level_values[i_state, i_iter] = np.max(Q)
            current_level_policy[i_state, i_iter] = np.argmax(Q)

        current_level_Q_values.append(Q_values)

    return current_level_values, current_level_policy, current_level_Q_values


# %% set up MDP

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
DISCOUNT_FACTOR_REWARD = 0.9  # discounting factor for rewards
DISCOUNT_FACTOR_COST = 0.7  # discounting factor for costs
DISCOUNT_FACTOR_COMMON = 0.9  # common discount factor for both
EFFICACY = 0.6  # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 2.0
EFFORT_DO = -1.0
# no delayed rewards:
REWARD_COMPLETED = 0.0
COST_COMPLETED = -0.0

# %% inconsistent policy with different discounts
reward_func, cost_func, reward_func_last, cost_func_last = (
    get_reward_functions(STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED,
                         COST_COMPLETED))

T = get_transition_prob(STATES, EFFICACY)

V_full, policy_full, Q_values_full = (
    mdp_algms.find_optimal_policy_diff_discount_factors(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD,
        DISCOUNT_FACTOR_COST, reward_func, cost_func, reward_func_last,
        cost_func_last, T)
)

plot_policy(policy_full, state=0, cmap=sns.color_palette('husl', 2),
            vmin=0, vmax=1)

# actual policy followed by agent
effective_naive_policy = np.array([policy_full[HORIZON-1-i][0][i]
                                   for i in range(HORIZON)])

Q_diff_full = plot_Q_value_diff(Q_values_full, cmap='coolwarm',
                                vmin=-0.5, vmax=0.5)

# %% self control

# level -1 (all selves are naive)
naive_values = np.array([V_full[HORIZON-1-i][:, i]
                         for i in range(HORIZON)]).T

# level 0 (each self considers naive Q-values of future ones)
level = 0
level_0_values, level_0_policy, level_0_Q_values = self_control(
    naive_values, level)

# level 1 (each self considers level 0 Q-values of future ones)
level = 1
level_1_values, level_1_policy, level_1_Q_values = self_control(
    level_0_values, level)

# plot level k policy and Q-values for state = 0
policy_levels_state_0 = [effective_naive_policy,
                         level_0_policy[0],
                         level_1_policy[0]]
f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(np.array(policy_levels_state_0), linewidths=.5,
            cmap=sns.color_palette('husl', 2))
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['SHIRK', 'WORK'])
ax.set_xlabel('agent at timestep')
ax.set_ylabel('level k effective policy')

Q_diff_naive = [Q_diff_full[HORIZON-1-i][i] for i in range(HORIZON)]
Q_diff_level_0 = [a[0][1]-a[0][0] for a in level_0_Q_values]
Q_diff_level_1 = [a[0][1]-a[0][0] for a in level_1_Q_values]
Q_diff_levels_state_0 = [Q_diff_naive,
                         Q_diff_level_0,
                         Q_diff_level_1]
padded_data = np.array([row + [np.nan] * (HORIZON - len(row))
                        for row in Q_diff_levels_state_0])
f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(padded_data, linewidths=.5,
            cmap='coolwarm', vmin=-0.5, vmax=0.5)
ax.tick_params()
ax.set_xlabel('agent at timestep')
ax.set_ylabel('level k diff in Q-values \n (WORK-SHIRK)')

# %%
# solve for common discount case
reward_func, cost_func, reward_func_last, cost_func_last = (
    get_reward_functions(STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED,
                         COST_COMPLETED)
)
T = get_transition_prob(STATES, EFFICACY)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_COMMON, reward_func,
    reward_func_last, T)
