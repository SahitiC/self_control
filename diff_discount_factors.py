# %%
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
    # reward for actions (depends on current state and next state)
    reward_func.append([np.array([0, 0]),
                        np.array([0, reward_do])])  # rewards for don't and do
    reward_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    reward_func_last = np.array([0, reward_completed])

    # effort for actions (depends on current state and next state)
    cost_func.append([np.array([0, 0]),  # rewards for don't and do
                      np.array([effort_do, effort_do])])
    cost_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    cost_func_last = np.array([cost_completed, 0])

    return reward_func, cost_func, reward_func_last, cost_func_last


# construct reward functions separately for rewards and costs with exceptions
def get_reward_functions_exception(
        states, reward_do_normal, effort_do_normal, reward_do_exception,
        effort_do_exception, reward_completed, cost_completed):

    reward_func = []
    cost_func = []
    # reward for actions (depends on current state and next state)
    # for normal states
    reward_func.append([np.array([0, 0, 0, 0]),
                        np.array([0, reward_do_normal, 0, reward_do_normal])])
    reward_func.append([np.array([0, 0, 0, 0])])
    # for exception states
    reward_func.append([np.array([0, 0, 0, 0]),
                        np.array([0, reward_do_exception,
                                  0, reward_do_exception])])
    reward_func.append([np.array([0, 0, 0, 0])])

    # reward from final evaluation for the states
    reward_func_last = np.array([0, reward_completed, 0, reward_completed])

    # effort for actions (depends on current state and next state)
    # for normal states
    cost_func.append([np.array([0, 0, 0, 0]),
                      np.array([effort_do_normal, effort_do_normal,
                                effort_do_normal, effort_do_normal])])
    cost_func.append([np.array([0, 0, 0, 0])])
    # for exception states
    cost_func.append([np.array([0, 0, 0, 0]),
                      np.array([effort_do_exception, effort_do_exception,
                                effort_do_exception, effort_do_exception])])
    cost_func.append([np.array([0, 0, 0, 0])])

    # reward from final evaluation for the two states
    cost_func_last = np.array([cost_completed, 0, cost_completed, 0])

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


def get_transition_prob_exception(states, efficacy, p_exception):

    T = np.full(len(states), np.nan, dtype=object)

    T[0] = [np.array([1-p_exception, 0, p_exception, 0]),
            np.array([(1-efficacy)*(1-p_exception), efficacy*(1-p_exception),
                      (1-efficacy)*p_exception, efficacy*p_exception])]
    T[1] = [np.array([0, 1*(1-p_exception), 0, 1*p_exception])]
    # for exception states:
    T[2] = [np.array([1-p_exception, 0, p_exception, 0]),
            np.array([(1-efficacy)*(1-p_exception), efficacy*(1-p_exception),
                      (1-efficacy)*p_exception, efficacy*p_exception])]
    T[3] = [np.array([0, 1*(1-p_exception), 0, 1*p_exception])]
    return T


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


def find_optimal_policy_diff_discount_factors_habit(
        p_sticky, states, actions, horizon, discount_factor_reward,
        discount_factor_cost, reward_func, cost_func, reward_func_last,
        cost_func_last, T):

    V_opt_full = []
    policy_opt_full = []
    Q_values_full = []

    # solve for optimal policy at every time step
    for i_iter in range(horizon-1, -1, -1):

        V_opt = np.zeros((len(states), horizon+1))
        policy_opt = np.full((len(states), horizon), np.nan)
        Q_values = np.zeros(len(states), dtype=object)

        for i_state, state in enumerate(states):

            # V_opt for last time-step
            V_opt[i_state, -1] = ((discount_factor_reward**(horizon-i_iter))
                                  * reward_func_last[i_state]
                                  + (discount_factor_cost**(horizon-i_iter))
                                  * cost_func_last[i_state])
            # arrays to store Q-values for each action in each state
            Q_values[i_state] = np.full((len(actions[i_state]), horizon),
                                        np.nan)

        # backward induction to derive optimal policy starting from
        # timestep i_iter
        for i_timestep in range(horizon-1, i_iter-1, -1):

            for i_state, state in enumerate(states):

                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    # value of next state (if same as current state) in the
                    # next timestep is: p * value of current action in next
                    # state + (1-p) * value of optimal action in next state
                    if i_timestep < horizon-1:
                        value_next = np.full(len(states), 0.0)
                        for next_state in range(len(states)):
                            if actions[next_state] == actions[i_state]:
                                opt_action = policy_opt[next_state,
                                                        i_timestep+1]
                                value_next[next_state] = (
                                    p_sticky *
                                    Q_values[next_state][i_action,
                                                         i_timestep+1]
                                    + (1-p_sticky) *
                                    Q_values[next_state][int(opt_action),
                                                         i_timestep+1])
                            else:
                                value_next[next_state] = V_opt[next_state,
                                                               i_timestep+1]

                    else:
                        value_next = V_opt[:, i_timestep+1]

                    r = ((discount_factor_reward**(i_timestep-i_iter))
                         * reward_func[i_state][i_action]
                         + (discount_factor_cost**(i_timestep-i_iter))
                         * cost_func[i_state][i_action])

                    # q-value for each action (bellman equation)
                    Q[i_action] = (T[i_state][i_action] @ r.T
                                   + T[i_state][i_action]
                                   @ value_next)

                # find optimal action (which gives max q-value)
                V_opt[i_state, i_timestep] = np.max(Q)
                policy_opt[i_state, i_timestep] = np.argmax(Q)
                Q_values[i_state][:, i_timestep] = Q

        V_opt_full.append(V_opt)
        policy_opt_full.append(policy_opt)
        Q_values_full.append(Q_values)

    return V_opt_full, policy_opt_full, Q_values_full


def self_control_with_actions_habit(prev_level_effective_policy, p_sticky,
                                    states, actions, horizon,
                                    discount_factor_reward,
                                    discount_factor_cost, reward_func,
                                    cost_func, reward_func_last,
                                    cost_func_last, T):

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

                    # value of next state (if same as current state) in the
                    # next timestep is: p * value of current action in next
                    # state + (1-p) * value of optimal action in next state
                    if i_timestep < horizon-1:
                        value_next = np.full(len(states), 0.0)
                        for next_state in range(len(states)):
                            if actions[next_state] == actions[i_state]:
                                value_next[next_state] = (
                                    p_sticky *
                                    Q_values[next_state][i_action,
                                                         i_timestep+1]
                                    + (1-p_sticky) *
                                    V_real[next_state, i_timestep+1])

                            else:
                                value_next[next_state] = V_real[next_state,
                                                                i_timestep+1]
                    else:
                        value_next = V_real[:, i_timestep+1]

                    r = ((discount_factor_reward**(i_timestep-i_iter))
                         * reward_func[i_state][i_action]
                         + (discount_factor_cost**(i_timestep-i_iter))
                         * cost_func[i_state][i_action])

                    # q-value for each action (bellman equation)
                    Q[i_action] = (T[i_state][i_action] @ r.T
                                   + T[i_state][i_action]
                                   @ value_next)

                Q_values[i_state][:, i_timestep] = Q

                # what are the real V's? i.e. not the max Q value
                # but the Q-value of the best action of the level-1 agent
                V_real[i_state, i_timestep] = Q[
                    prev_level_effective_policy[i_state, i_timestep]]

        V_real_full.append(V_real)
        Q_values_full.append(Q_values)

    return V_real_full, Q_values_full


def self_control_cognitive_hierarchy(policy_full_levels, level, Lambda, states,
                                     actions, horizon, discount_factor_reward,
                                     discount_factor_cost, reward_func,
                                     cost_func, reward_func_last,
                                     cost_func_last, T):

    # normalised poisson of agents from 0 to level-1
    poisson_prob = normalized_poisson_pmf(Lambda, level-1)

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


# get policies

def get_naive_policy(states, actions, horizon, discount_factor_reward,
                     discount_factor_cost, reward_func, cost_func,
                     reward_func_last, cost_func_last, T, state_to_get=0,
                     sticky=False, p_sticky=None):

    if sticky:
        V_full, policy_full, Q_values_full = (
            find_optimal_policy_diff_discount_factors_habit(
                p_sticky, states, actions, horizon, discount_factor_reward,
                discount_factor_cost, reward_func, cost_func, reward_func_last,
                cost_func_last, T))
    else:
        V_full, policy_full, Q_values_full = (
            mdp_algms.find_optimal_policy_diff_discount_factors(
                states, actions, horizon, discount_factor_reward,
                discount_factor_cost, reward_func, cost_func, reward_func_last,
                cost_func_last, T))

    policy_state_0 = [policy_full[i][state_to_get] for i in range(horizon)]
    policy_state_0 = np.array(policy_state_0)
    # actual policy followed by agent
    effective_naive_policy = []
    for state in states:
        effective_naive_policy.append(np.array(
            [policy_full[horizon-1-i][state][i] for i in range(horizon)]))
    effective_naive_policy = np.array(effective_naive_policy, dtype=int)

    return policy_state_0, effective_naive_policy, Q_values_full, V_full


def get_policy_self_control_actions(
        level_no, Q_values_full_naive, effective_naive_policy, states, actions,
        horizon, discount_factor_reward, discount_factor_cost, reward_func,
        cost_func, reward_func_last, cost_func_last, T, state_to_get=0,
        sticky=False, p_sticky=None):

    Q_diff_levels_state_0 = []
    policy_levels_state_0 = []
    policy_full_levels = []

    Q_diff_naive = []
    for t in range(horizon):
        Q_diff_naive.append(np.diff(
            Q_values_full_naive[horizon-1-t][state_to_get][:, t])[0])

    Q_diff_levels_state_0.append(Q_diff_naive)
    policy_levels_state_0.append(effective_naive_policy[state_to_get])

    effective_policy_prev_level = effective_naive_policy
    policy_full_levels.append(effective_policy_prev_level)

    for _ in range(level_no):

        if sticky:
            # calculate next level
            V_current_level, Q_current_level = self_control_with_actions_habit(
                effective_policy_prev_level, p_sticky, states, actions,
                horizon, discount_factor_reward, discount_factor_cost,
                reward_func, cost_func, reward_func_last, cost_func_last, T)
        else:
            # calculate next level
            V_current_level, Q_current_level = self_control_with_actions(
                effective_policy_prev_level, states, actions, horizon,
                discount_factor_reward, discount_factor_cost, reward_func,
                cost_func, reward_func_last, cost_func_last, T)

        # update effective policy, Q_diff
        effective_policy_prev_level = np.full((len(states), horizon), 100)
        Q_diff = []  # diff only for state=0
        for t in range(horizon):
            Q_diff.append(np.diff(
                Q_current_level[horizon-1-t][state_to_get][:, t])[0])
            for state in states:
                effective_policy_prev_level[state, horizon-1-t] = np.argmax(
                    Q_current_level[t][state][:, horizon-1-t])

        Q_diff_levels_state_0.append(Q_diff)
        policy_levels_state_0.append(effective_policy_prev_level[state_to_get])
        policy_full_levels.append(effective_policy_prev_level)

    return Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels


def get_policy_self_control_cog_hierarchy(
        level_no, Q_values_full_naive, effective_naive_policy, states, actions,
        horizon, discount_factor_reward, discount_factor_cost, reward_func,
        cost_func, reward_func_last, cost_func_last, T, Lambda,
        state_to_get=0):

    Q_diff_levels_state_0 = []
    policy_levels_state_0 = []
    policy_full_levels = []

    Q_diff_naive = []
    for t in range(horizon):
        Q_diff_naive.append(np.diff(
            Q_values_full_naive[horizon-1-t][state_to_get][:, t])[0])

    Q_diff_levels_state_0.append(Q_diff_naive)
    policy_levels_state_0.append(effective_naive_policy[state_to_get])

    policy_full_levels.append(effective_naive_policy)

    for level in range(1, level_no+1):

        # calculate next level
        V_current_level, Q_current_level = self_control_cognitive_hierarchy(
            np.array(policy_full_levels), level, Lambda, states, actions,
            horizon, discount_factor_reward, discount_factor_cost, reward_func,
            cost_func, reward_func_last, cost_func_last, T)

        # update effective policy, Q_diff
        effective_policy = np.full((len(states), horizon), 100)
        Q_diff = []  # diff only for state=0
        for t in range(horizon):
            Q_diff.append(np.diff(
                Q_current_level[horizon-1-t][state_to_get][:, t])[0])
            for state in states:
                effective_policy[state, horizon-1-t] = np.argmax(
                    Q_current_level[t][state][:, horizon-1-t])

        Q_diff_levels_state_0.append(Q_diff)
        policy_levels_state_0.append(effective_policy[state_to_get])
        policy_full_levels.append(effective_policy)

    return Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels

# forward simulations


def forward_sampling_k(
        initial_state, Q_values_full_naive, effective_naive_policy, horizon,
        discount_factor_reward, discount_factor_cost, reward_func, cost_func,
        reward_func_last, cost_func_last, T, Lambda):
    """
    in the cognitive hierarchy case, each agent calculated its best course of
    action assuming that the future agents are drawn from a normalised Poisson
    distribution from levels 0 to k-1; so the agent calculated the expected
    Q_values from this distribution of future policies - and chose its best
    action based on this

    alternatively, we can say that the future agent is one particular level
    (k), and each agent samples from a poisson distribuion & then executes k+1
    action
    """

    # forward simulation:
    states_sim = np.full(horizon+1, 100)
    actions_sim = np.full(horizon, 100)
    levels_sim = np.full(horizon, 100)
    states_sim[0] = initial_state

    for t in range(horizon):

        # sample level of t+1 agent (this fixes levels of t+2, ..., N agents)
        level = np.random.poisson(lam=Lambda)
        levels_sim[t] = level

        # so the agent has to think from one level above, so caluclate policies
        # until level+1 (with strict k-1 assumption)
        level_no = level+1

        Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels = get_policy_self_control_actions(
            level_no, Q_values_full_naive, effective_naive_policy, horizon,
            discount_factor_reward, discount_factor_cost, reward_func,
            cost_func, reward_func_last, cost_func_last, T)

        # what action to take
        actions_sim[t] = policy_full_levels[level_no][states_sim[t]][t]

        # state transition based on action
        states_sim[t+1] = np.random.choice(
            len(STATES), p=T[states_sim[t]][actions_sim[t]])

    return states_sim, actions_sim, levels_sim


def time_to_finish(trajectories, states_no):
    """
    find when all units arre completed for each trajectory
    (of work) inputted; if threshold is never reached, returns NaN
    """

    times = []
    trajectories = np.array(trajectories)
    for trajectory in trajectories[:, 1:]:
        if trajectory[-1] == states_no-1:
            times.append(np.where(trajectory >= states_no-1)[0][0])
        else:
            times.append(np.nan)

    return times


def did_it_finish(trajectories, states_no):
    """
    find if all units have been completed for each trajectory inputted
    """

    completed = []
    trajectories = np.array(trajectories)
    for trajectory in trajectories[:, 1:]:
        if trajectory[-1] == states_no-1:
            completed.append(1)
        else:
            completed.append(0)

    return completed


# %%

if __name__ == 'main':

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
    DISCOUNT_FACTOR_COMMON = 0.9  # common d iscount factor for both
    EFFICACY = 0.6  # self-efficacy (probability of progress on working)

    # utilities :
    REWARD_DO = 2.0
    EFFORT_DO = -1.0
    # no delayed rewards:
    REWARD_COMPLETED = 0.0
    COST_COMPLETED = -0.0

    LAMBDA = 0.5  # poisson hierarchy distribution mean

    reward_func, cost_func, reward_func_last, cost_func_last = (
        get_reward_functions(STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED,
                             COST_COMPLETED))

    T = get_transition_prob(STATES, EFFICACY)

    # %% exceptions

    p_exception = 0.2  # probability of exception occurring
    # states of markov chain
    N_INTERMEDIATE_STATES = 0
    # intermediate + initial and finished states (2)
    STATES = np.arange(4 + N_INTERMEDIATE_STATES*2)

    # actions available in each state
    ACTIONS = np.full(len(STATES), np.nan, dtype=object)
    # actions for normal and exception states (0-1 units completed)
    ACTIONS[0] = ['shirk', 'work']
    ACTIONS[1] = ['shirk']
    ACTIONS[2] = ['shirk', 'work']
    ACTIONS[3] = ['shirk']

    HORIZON = 6  # deadline
    DISCOUNT_FACTOR_REWARD = 0.7  # discounting factor for rewards
    DISCOUNT_FACTOR_COST = 0.6  # discounting factor for costs
    DISCOUNT_FACTOR_COMMON = 0.8  # common d iscount factor for both
    EFFICACY = 0.8  # self-efficacy (probability of progress on working)

    # utilities :
    REWARD_DO_NORMAL = 2.0
    EFFORT_DO_NORMAL = -1.3
    REWARD_DO_EXCEPTION = 2.0
    EFFORT_DO_EXCEPTION = -1.7
    # no delayed rewards:
    REWARD_COMPLETED = 0.0
    COST_COMPLETED = -0.0

    reward_func, cost_func, reward_func_last, cost_func_last = (
        get_reward_functions_exception(
            STATES, REWARD_DO_NORMAL, EFFORT_DO_NORMAL, REWARD_DO_EXCEPTION,
            EFFORT_DO_EXCEPTION, REWARD_COMPLETED, COST_COMPLETED))

    T = get_transition_prob_exception(STATES, EFFICACY, p_exception)

    # %% inconsistent policy with different discounts

    state_to_get = 0

    policy_state_0, effective_naive_policy, Q_values_full_naive, V_full_naive = get_naive_policy(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST,
        reward_func, cost_func, reward_func_last, cost_func_last, T,
        state_to_get=state_to_get)

    plot_heatmap(policy_state_0, cmap=sns.color_palette('husl', 2),
                 ylabel='horizon', xlabel='timestep', vmin=0, vmax=1)
    plt.title('Naive policy')

    # Q-diff only for a given state
    Q_values = [Q_values_full_naive[i][state_to_get] for i in range(HORIZON)]
    Q_diff_full = [a[1]-a[0] for a in Q_values]
    Q_diff_full = np.array(Q_diff_full)
    plot_Q_value_diff(Q_diff_full, cmap='coolwarm',
                      ylabel='horizon', xlabel='timestep',
                      title='diff in Q_values (WORK-SHIRK)',
                      vmin=-0.7, vmax=0.7)

    # %% self control with actions
    level_no = HORIZON-1
    Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels = get_policy_self_control_actions(
        level_no, Q_values_full_naive, effective_naive_policy, STATES, ACTIONS,
        HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, reward_func,
        cost_func, reward_func_last, cost_func_last, T,
        state_to_get=state_to_get)

    plot_heatmap(np.array(policy_levels_state_0),
                 cmap=sns.color_palette('husl', 2),
                 ylabel='level k effective policy', xlabel='agent at timestep')

    plot_Q_value_diff(np.array(Q_diff_levels_state_0), 'coolwarm',
                      ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                      xlabel='agent at timestep', vmin=-0.65, vmax=0.65)

    # %% stickiness
    # what if we had 1 step stickiness?

    P_STICKY = 0.9
    state_to_get = 2

    policy_state_0_h, effective_naive_policy_h, Q_values_full_naive_h, V_full_naive_h = get_naive_policy(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST,
        reward_func, cost_func, reward_func_last, cost_func_last, T,
        state_to_get=state_to_get, sticky=True, p_sticky=P_STICKY)

    plot_heatmap(policy_state_0_h, cmap=sns.color_palette('husl', 2),
                 ylabel='horizon', xlabel='timestep', vmin=0, vmax=1)
    plt.title('Naive policy')

    # Q-diff only for state 0
    Q_values = [Q_values_full_naive_h[i][state_to_get] for i in range(HORIZON)]
    Q_diff_full = [a[1]-a[0] for a in Q_values]
    Q_diff_full = np.array(Q_diff_full)
    plot_Q_value_diff(Q_diff_full, cmap='coolwarm',
                      ylabel='horizon', xlabel='timestep',
                      title='diff in Q_values (WORK-SHIRK)',
                      vmin=-0.7, vmax=0.7)

    level_no = HORIZON-1
    Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels = get_policy_self_control_actions(
        level_no, Q_values_full_naive_h, effective_naive_policy_h, STATES, ACTIONS,
        HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, reward_func,
        cost_func, reward_func_last, cost_func_last, T,
        state_to_get=state_to_get, sticky=True, p_sticky=P_STICKY)

    plot_heatmap(np.array(policy_levels_state_0),
                 cmap=sns.color_palette('husl', 2),
                 ylabel='level k effective policy', xlabel='agent at timestep')

    plot_Q_value_diff(np.array(Q_diff_levels_state_0), 'coolwarm',
                      ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                      xlabel='agent at timestep', vmin=-0.65, vmax=0.65)

    # %% self control - cognitive heirarchy
    # instead of assuming future agents are k-1 (i.e. exactly 1 level lower),
    # have a poisson probability distribution over 0 to k-1
    level_no = HORIZON-1
    Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels = get_policy_self_control_cog_hierarchy(
        level_no, Q_values_full_naive, effective_naive_policy, STATES, ACTIONS,
        HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, reward_func,
        cost_func, reward_func_last, cost_func_last, T, LAMBDA)

    plot_heatmap(np.array(policy_levels_state_0),
                 cmap=sns.color_palette('husl', 2),
                 ylabel='level k effective policy', xlabel='agent at timestep')

    plot_Q_value_diff(np.array(Q_diff_levels_state_0), 'coolwarm',
                      ylabel='level k diff in Q-values \n (WORK-SHIRK)',
                      xlabel='agent at timestep', vmin=-0.65, vmax=0.65)

    # %% self control with values: equivalent to removing delay discounting

    # level -1 (all selves are naive)
    naive_values = np.array([V_full_naive[HORIZON-1-i][:, i]
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
    plot_heatmap(np.array(policy_levels_state_0),
                 cmap=sns.color_palette('husl', 2),
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

    # %% vary discounts

    discount_factors_reward = [0.6, 0.75, 0.9]
    discount_factors_cost = np.linspace(0.4, 0.8, 5)
    policies = np.full((len(discount_factors_reward), len(discount_factors_cost)),
                       np.nan, dtype=object)

    reward_func, cost_func, reward_func_last, cost_func_last = (
        get_reward_functions(STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED,
                             COST_COMPLETED))
    T = get_transition_prob(STATES, EFFICACY)

    level_no = HORIZON-1

    for i_r, disc_reward in enumerate(discount_factors_reward):

        for i_d, disc_cost in enumerate(discount_factors_cost):

            # get naive policy
            policy_state_0, effective_naive_policy, Q_values_full_naive, V_full_naive = get_naive_policy(
                STATES, ACTIONS, HORIZON, disc_reward, disc_cost, reward_func,
                cost_func, reward_func_last, cost_func_last, T)

            # get strict k-1 policy
            Q_diff_levels_state_0, policy_levels_state_0, policy_full_levels = get_policy_self_control_actions(
                level_no, Q_values_full_naive, effective_naive_policy,
                STATES, ACTIONS, HORIZON, disc_reward, disc_cost, reward_func,
                cost_func, reward_func_last, cost_func_last, T)

            policies[i_r, i_d] = policy_levels_state_0

    cmap = sns.color_palette('husl', 2)

    # naive policy
    for i_r, disc_reward in enumerate(discount_factors_reward):
        policy = []
        plt.figure(figsize=(6, 4))
        for i_c in range(len(discount_factors_cost)):
            policy.append(policies[i_r, i_c][0])
        plot_heatmap(policy, cmap, ylabel='discount factor cost',
                     xlabel='timestep', title=rf'$\gamma_r$={disc_reward}')
        plt.yticks(np.arange(len(discount_factors_cost))+0.5,
                   np.round(discount_factors_cost, 1))

    # level 1 policy: when it starts and ends
    for i_r, disc_reward in enumerate(discount_factors_reward):
        policy = []
        plt.figure(figsize=(6, 4))
        for i_c in range(len(discount_factors_cost)):
            policy.append(policies[i_r, i_c][1])
        plot_heatmap(policy, cmap, ylabel='discount factor cost',
                     xlabel='timestep', title=rf'$\gamma_r$={disc_reward}')
        plt.yticks(np.arange(len(discount_factors_cost))+0.5,
                   np.round(discount_factors_cost, 1))

    # level 2 policy: when it starts and ends
    for i_r, disc_reward in enumerate(discount_factors_reward):
        policy = []
        plt.figure(figsize=(6, 4))
        for i_c in range(len(discount_factors_cost)):
            policy.append(policies[i_r, i_c][2])
        plot_heatmap(policy, cmap, ylabel='discount factor cost',
                     xlabel='timestep', title=rf'$\gamma_r$={disc_reward}')
        plt.yticks(np.arange(len(discount_factors_cost))+0.5,
                   np.round(discount_factors_cost, 1))

    # %% sampling

    # in the cognitive hierarchy case, each agent calculated its best course of
    # action assuming that the future agents are drawn from normalised Poisson
    # distribution from levels 0 to k-1; so the agent calculated the expected
    # Q_values from this distribution of future policies - and chose its best
    # action based on this

    # alternatively, we can say the future agent is one particular level (k),
    # & each agent samples from poisson distribuion & then executes k+1 action

    discount_factors_reward = [0.6, 0.75, 0.9]
    discount_factors_cost = np.linspace(0.4, 0.8, 5)
    times_naive = np.full((len(discount_factors_reward),
                           len(discount_factors_cost)), np.nan)
    completed_naive = np.full((len(discount_factors_reward),
                               len(discount_factors_cost)), np.nan)
    times = np.full((len(discount_factors_reward),
                     len(discount_factors_cost)), np.nan)
    completed = np.full((len(discount_factors_reward),
                         len(discount_factors_cost)), np.nan)

    for i_r, disc_reward in enumerate(discount_factors_reward):

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        fig2, ax2 = plt.subplots(figsize=(5, 4))

        for i_d, disc_cost in enumerate(discount_factors_cost):

            reward_func, cost_func, reward_func_last, cost_func_last = (
                get_reward_functions(STATES, REWARD_DO, EFFORT_DO,
                                     REWARD_COMPLETED, COST_COMPLETED))

            T = get_transition_prob(STATES, EFFICACY)

            policy_state_0, effective_naive_policy, Q_values_full_naive, V_full_naive = get_naive_policy(
                STATES, ACTIONS, HORIZON, disc_reward, disc_cost, reward_func,
                cost_func, reward_func_last, cost_func_last, T)

            # forward simulation with naive policy:
            initial_state = 0
            N_trials = 500
            state_naive_list = []
            action_naive_list = []
            for _ in range(N_trials):
                s, a = mdp_algms.forward_runs(
                    effective_naive_policy, initial_state, HORIZON, STATES, T)
                state_naive_list.append(s)
                action_naive_list.append(a)

            state_naive_array = np.array(state_naive_list)
            action_naive_array = np.array(action_naive_list)
            # only consider actions associated with state 0
            valid = state_naive_array[:, :-1] == 0
            actions_naive_valid = np.where(valid, action_naive_array, np.nan)
            average_action_naive = np.nanmean(actions_naive_valid, axis=0)
            average_state_naive = np.nanmean(state_naive_array, axis=0)
            times_naive[i_r, i_d] = np.nanmean(
                time_to_finish(state_naive_list, len(STATES)))
            completed_naive[i_r, i_d] = np.nanmean(did_it_finish(
                state_naive_list, len(STATES)))
            ax1.plot(average_state_naive, marker='o', linestyle='dashed',
                     linewidth=3, label=f'{np.round(disc_cost, 1)}')

            # forward simulation with sampling k:
            initial_state = 0
            N_trials = 500
            state_list = []
            action_list = []
            level_list = []
            for _ in range(N_trials):
                s, a, le = forward_sampling_k(
                    initial_state, Q_values_full_naive, effective_naive_policy,
                    HORIZON, disc_reward, disc_cost,
                    reward_func, cost_func, reward_func_last, cost_func_last, T,
                    LAMBDA)
                state_list.append(s)
                action_list.append(a)
                level_list.append(le)

            state_array = np.array(state_list)
            action_array = np.array(action_list)
            # only consider actions associated with state 0
            valid = state_array[:, :-1] == 0
            actions_valid = np.where(valid, action_array, np.nan)
            average_action = np.nanmean(actions_valid, axis=0)
            average_state = np.nanmean(state_array, axis=0)
            times[i_r, i_d] = np.nanmean(
                time_to_finish(state_list, len(STATES)))
            completed[i_r, i_d] = np.nanmean(
                did_it_finish(state_list, len(STATES)))
            ax2.plot(average_state, marker='o', linestyle='dashed', linewidth=3,
                     label=f'{np.round(disc_cost, 1)}')

        ax1.set_title(rf'naive $\gamma_r$={disc_reward}')
        ax1.set_xlabel('time')
        ax1.set_ylabel('average progress')
        fig1.legend(title=r'$\gamma_c$', bbox_to_anchor=(1.15, 0.75))
        ax2.set_title(rf'$\gamma_r$={disc_reward}')
        ax2.set_xlabel('time')
        ax2.set_ylabel('average progress')
        fig2.legend(title=r'$\gamma_c$', bbox_to_anchor=(1.15, 0.75))

    # completion rates and times
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    for i_r, disc_reward in enumerate(discount_factors_reward):

        ax1.plot(times_naive[i_r, :], label=f'{np.round(disc_reward, 1)}',
                 marker='o', linestyle='--')
        ax2.plot(times[i_r, :], label=f'{np.round(disc_reward, 1)}',
                 marker='o', linestyle='--')
        ax3.plot(completed_naive[i_r, :], label=f'{np.round(disc_reward, 1)}',
                 marker='o', linestyle='--')
        ax4.plot(completed[i_r, :], label=f'{np.round(disc_reward, 1)}',
                 marker='o', linestyle='--')

    ax1.set_xlabel('discount factor cost')
    ax2.set_xlabel('discount factor cost')
    ax3.set_xlabel('discount factor cost')
    ax4.set_xlabel('discount factor cost')

    ax1.set_xticks(ticks=np.arange(len(discount_factors_cost)),
                   labels=np.round(discount_factors_cost, 1))
    ax2.set_xticks(np.arange(len(discount_factors_cost)),
                   labels=np.round(discount_factors_cost, 1))
    ax3.set_xticks(np.arange(len(discount_factors_cost)),
                   labels=np.round(discount_factors_cost, 1))
    ax4.set_xticks(np.arange(len(discount_factors_cost)),
                   labels=np.round(discount_factors_cost, 1))

    ax1.set_title('completion times naive')
    ax2.set_title('completion times')
    ax3.set_title('completion rates naive')
    ax4.set_title('completion rates')

    fig1.legend(title=r'$\gamma_r$', bbox_to_anchor=(1.15, 0.75))
    fig2.legend(title=r'$\gamma_r$', bbox_to_anchor=(1.15, 0.75))
    fig3.legend(title=r'$\gamma_r$', bbox_to_anchor=(1.15, 0.75))
    fig4.legend(title=r'$\gamma_r$', bbox_to_anchor=(1.15, 0.75))
    plt.show()
