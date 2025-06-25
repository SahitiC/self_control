from scipy.stats import poisson
import seaborn as sns
import mdp_algms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['text.usetex'] = False

# %%


def normalized_poisson_pmf(mu, N):
    true_poisson = [poisson.pmf(h, mu) for h in range(N+1)]
    total = sum(true_poisson)
    norm_poisson = [f / total for f in true_poisson]
    assert np.isclose(sum(norm_poisson), 1.0, rtol=1e-9), 'sum of prob != 1'
    return norm_poisson


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


def plot_policy(policy_state, cmap, ylabel, xlabel, title='', vmin=0, vmax=1,):
    """
    heat map of full policy in state = state
    """

    f, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sns.heatmap(policy_state, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params()
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['SHIRK', 'WORK'])
    f.suptitle(title)


def plot_Q_value_diff(Q_diff, cmap, ylabel, xlabel, title='', vmin=-0.5,
                      vmax=0.5):
    """
    plot diff in Q-values between actions for state=0 where there are two
    actions
    """

    f, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sns.heatmap(Q_diff, linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params()
    f.suptitle(title)


def self_control_with_values(prev_level_values, level):

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


def self_control_with_actions(prev_level_effective_policy, states,
                              actions, horizon, discount_factor_reward,
                              discount_factor_cost, reward_func, cost_func,
                              reward_func_last, cost_func_last, T):

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
                (discount_factor_reward**(horizon-i_iter))
                * reward_func_last[i_state]
                + (discount_factor_cost**(horizon-i_iter))
                * cost_func_last[i_state])

        # backward induction to derive optimal policy starting from
        # timestep i_iter
        for i_timestep in range(horizon-1, i_iter-1, -1):

            for i_state, state in enumerate(states):

                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    r = ((discount_factor_reward**(i_timestep-i_iter))
                         * reward_func[i_state][i_action]
                         + (discount_factor_cost**(i_timestep-i_iter))
                         * cost_func[i_state][i_action])

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


def self_control_cognitive_hierarchy(policy_full_levels, level, mu, states,
                                     actions, horizon, discount_factor_reward,
                                     discount_factor_cost, reward_func,
                                     cost_func, reward_func_last,
                                     cost_func_last, T):

    # normalised poisson of agents from 0 to level-1
    poisson_prob = normalized_poisson_pmf(mu, level-1)

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
                (discount_factor_reward**(horizon-i_iter))
                * reward_func_last[i_state]
                + (discount_factor_cost**(horizon-i_iter))
                * cost_func_last[i_state])

        # backward induction to derive optimal policy starting from
        # timestep i_iter
        for i_timestep in range(horizon-1, i_iter-1, -1):

            for i_state, state in enumerate(states):

                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    r = ((discount_factor_reward**(i_timestep-i_iter))
                         * reward_func[i_state][i_action]
                         + (discount_factor_cost**(i_timestep-i_iter))
                         * cost_func[i_state][i_action])

                    # q-value for each action (bellman equation)
                    Q[i_action] = (T[i_state][i_action] @ r.T
                                   + T[i_state][i_action]
                                   @ V_real[states, i_timestep+1])

                Q_values[i_state][:, i_timestep] = Q

                # what are the real V's? i.e. not the max Q value
                # but the poisson-prob weighted Q-values of the best action of
                # each of the 0 to level-1 agents
                V_real[i_state, i_timestep] = sum(
                    [poisson_prob[h]
                     * Q[policy_full_levels[h, i_state, i_timestep]]
                     for h in range(level)])

        V_real_full.append(V_real)
        Q_values_full.append(Q_values)

    return V_real_full, Q_values_full


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

HORIZON = 6  # deadline
DISCOUNT_FACTOR_REWARD = 0.7  # discounting factor for rewards
DISCOUNT_FACTOR_COST = 0.5  # discounting factor for costs
DISCOUNT_FACTOR_COMMON = 0.9  # common discount factor for both
EFFICACY = 0.6  # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 2.0
EFFORT_DO = -1.0
# no delayed rewards:
REWARD_COMPLETED = 0.0
COST_COMPLETED = -0.0

MU = 1  # poisson hierarchy distribution mean

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

policy_state_0 = [policy_full[i][0] for i in range(HORIZON)]
policy_state_0 = np.array(policy_state_0)
plot_policy(policy_state_0, cmap=sns.color_palette('husl', 2),
            ylabel='horizon', xlabel='timestep',
            vmin=0, vmax=1)
plt.title('Naive policy')

# actual policy followed by agent
effective_naive_policy = []
for state in STATES:
    effective_naive_policy.append(np.array([policy_full[HORIZON-1-i][state][i]
                                            for i in range(HORIZON)]))
effective_naive_policy = np.array(effective_naive_policy, dtype=int)

Q_values = [Q_values_full[i][0] for i in range(HORIZON)]
Q_diff_full = [a[1]-a[0] for a in Q_values]
Q_diff_full = np.array(Q_diff_full)
plot_Q_value_diff(Q_diff_full, cmap='coolwarm',
                  ylabel='horizon', xlabel='timestep',
                  title='diff in Q_values (WORK-SHIRK)',
                  vmin=-0.7, vmax=0.7)

# %% self control with actions
Q_diff_levels_state_0 = []
policy_levels_state_0 = []
policy_full_levels = []

Q_diff_naive = []
for t in range(HORIZON):
    Q_diff_naive.append(np.diff(Q_values_full[HORIZON-1-t][0][:, t])[0])

Q_diff_levels_state_0.append(Q_diff_naive)
policy_levels_state_0.append(effective_naive_policy[0])

level_no = HORIZON-1
effective_policy_prev_level = effective_naive_policy
policy_full_levels.append(effective_policy_prev_level)

for _ in range(level_no):

    # calculate next level
    V_current_level, Q_current_level = self_control_with_actions(
        effective_policy_prev_level, STATES, ACTIONS, HORIZON,
        DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, reward_func, cost_func,
        reward_func_last, cost_func_last, T)

    # update effective policy, Q_diff
    effective_policy_prev_level = np.full((len(STATES), HORIZON), 100)
    Q_diff = []  # diff only for state=0
    for t in range(HORIZON):
        Q_diff.append(np.diff(Q_current_level[HORIZON-1-t][0][:, t])[0])
        for state in STATES:
            effective_policy_prev_level[state, HORIZON-1-t] = np.argmax(
                Q_current_level[t][state][:, HORIZON-1-t])

    Q_diff_levels_state_0.append(Q_diff)
    policy_levels_state_0.append(effective_policy_prev_level[0])
    policy_full_levels.append(effective_policy_prev_level)

plot_policy(np.array(policy_levels_state_0), cmap=sns.color_palette('husl', 2),
            ylabel='level k effective policy', xlabel='agent at timestep')

plot_Q_value_diff(np.array(Q_diff_levels_state_0), 'coolwarm',
                  ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                  xlabel='agent at timestep', vmin=-0.65, vmax=0.65)

# %% self control - cognitive heirarchy
# instead of assuming future agents are k-1 (i.e. exactly 1 level lower),
# have a probability distribution over 0 to k-1
Q_diff_levels_state_0 = []
policy_levels_state_0 = []
policy_full_levels = []

Q_diff_naive = []
for t in range(HORIZON):
    Q_diff_naive.append(np.diff(Q_values_full[HORIZON-1-t][0][:, t])[0])

Q_diff_levels_state_0.append(Q_diff_naive)
policy_levels_state_0.append(effective_naive_policy[0])

level_no = HORIZON-1
effective_policy_prev_level = effective_naive_policy
policy_full_levels.append(effective_policy_prev_level)

for level in range(1, level_no+1):

    # calculate next level
    V_current_level, Q_current_level = self_control_cognitive_hierarchy(
        np.array(policy_full_levels), level, MU, STATES, ACTIONS, HORIZON,
        DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, reward_func, cost_func,
        reward_func_last, cost_func_last, T)

    # update effective policy, Q_diff
    effective_policy_prev_level = np.full((len(STATES), HORIZON), 100)
    Q_diff = []  # diff only for state=0
    for t in range(HORIZON):
        Q_diff.append(np.diff(Q_current_level[HORIZON-1-t][0][:, t])[0])
        for state in STATES:
            effective_policy_prev_level[state, HORIZON-1-t] = np.argmax(
                Q_current_level[t][state][:, HORIZON-1-t])

    Q_diff_levels_state_0.append(Q_diff)
    policy_levels_state_0.append(effective_policy_prev_level[0])
    policy_full_levels.append(effective_policy_prev_level)

plot_policy(np.array(policy_levels_state_0), cmap=sns.color_palette('husl', 2),
            ylabel='level k effective policy', xlabel='agent at timestep')

plot_Q_value_diff(np.array(Q_diff_levels_state_0), 'coolwarm',
                  ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                  xlabel='agent at timestep', vmin=-0.65, vmax=0.65)


# %% self control with values: equivalent to removing discounting of future rewards

# level -1 (all selves are naive)
naive_values = np.array([V_full[HORIZON-1-i][:, i]
                         for i in range(HORIZON)]).T

# level 0 (each self considers naive Q-values of future ones)
level = 0
level_0_values, level_0_policy, level_0_Q_values = self_control_with_values(
    naive_values, level)

# level 1 (each self considers level 0 Q-values of future ones)
level = 1
level_1_values, level_1_policy, level_1_Q_values = self_control_with_values(
    level_0_values, level)

# plot level k policy and Q-values for state = 0
policy_levels_state_0 = [effective_naive_policy[0],
                         level_0_policy[0],
                         level_1_policy[0]]
plot_policy(np.array(policy_levels_state_0), cmap=sns.color_palette('husl', 2),
            ylabel='level k effective policy', xlabel='agent at timestep')

Q_diff_naive = [Q_diff_full[HORIZON-1-i][i] for i in range(HORIZON)]
Q_diff_level_0 = [a[0][1]-a[0][0] for a in level_0_Q_values]
Q_diff_level_1 = [a[0][1]-a[0][0] for a in level_1_Q_values]
Q_diff_levels_state_0 = [Q_diff_naive,
                         Q_diff_level_0,
                         Q_diff_level_1]
padded_data = np.array([row + [np.nan] * (HORIZON - len(row))
                        for row in Q_diff_levels_state_0])
plot_Q_value_diff(padded_data, 'coolwarm',
                  ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                  xlabel='agent at timestep', vmin=-0.5, vmax=0.5)

# %%

discount_factors_cost = np.linspace(0.4, 0.8, 5)
policies = np.full((HORIZON, len(discount_factors_cost)), 100, dtype=object)

reward_func, cost_func, reward_func_last, cost_func_last = (
    get_reward_functions(STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED,
                         COST_COMPLETED))
T = get_transition_prob(STATES, EFFICACY)

level_no = HORIZON-1

for i_d, disc_cost in enumerate(discount_factors_cost):

    # get naive policy
    V_full, policy_full, Q_values_full = (
        mdp_algms.find_optimal_policy_diff_discount_factors(
            STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD,
            disc_cost, reward_func, cost_func, reward_func_last,
            cost_func_last, T))

    # actual policy followed by agent
    effective_naive_policy = []
    for state in STATES:
        effective_naive_policy.append(
            np.array([policy_full[HORIZON-1-i][state][i]
                      for i in range(HORIZON)]))
    effective_naive_policy = np.array(effective_naive_policy, dtype=int)
    policies[0, i_d] = effective_naive_policy[0]

    effective_policy_prev_level = effective_naive_policy

    for level in range(level_no):

        # calculate next level
        V_current_level, Q_current_level = self_control_with_actions(
            effective_policy_prev_level, STATES, ACTIONS, HORIZON,
            DISCOUNT_FACTOR_REWARD, disc_cost, reward_func,
            cost_func, reward_func_last, cost_func_last, T)

        # update effective policy, Q_diff
        effective_policy_prev_level = np.full((len(STATES), HORIZON), 100)
        for t in range(HORIZON):
            for state in STATES:
                effective_policy_prev_level[state, HORIZON-1-t] = np.argmax(
                    Q_current_level[t][state][:, HORIZON-1-t])

        print(effective_policy_prev_level[0])
        policies[level+1, i_d] = effective_policy_prev_level[0]
