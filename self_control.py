import numpy as np


def self_control_with_actions(
        prev_level_effective_policy, states, actions, horizon, T, reward_func,
        reward_func_last, cost_func=None, cost_func_last=None,
        discount_factor_reward=None, discount_factor_cost=None,
        discount_beta=None, discount_delta=None, disc_func='diff_disc'):
    """
    calculate higher level policies that account for true lower level policy
    for different discount functions
    """

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
            if disc_func == 'diff_disc':
                V_real[i_state, -1] = (
                    (discount_factor_reward**(horizon-i_iter))
                    * reward_func_last[i_state]
                    + (discount_factor_cost**(horizon-i_iter))
                    * cost_func_last[i_state])
            elif disc_func == 'beta_delta':
                V_real[i_state, -1] = (
                    (discount_beta*discount_delta**(horizon-i_iter))
                    * reward_func_last[i_state])

        # backward induction to derive optimal policy starting from
        # timestep i_iter
        for i_timestep in range(horizon-1, i_iter-1, -1):

            for i_state, state in enumerate(states):

                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    if disc_func == 'diff_disc':

                        r = ((discount_factor_reward**(i_timestep-i_iter))
                             * reward_func[i_state][i_action]
                             + (discount_factor_cost**(i_timestep-i_iter))
                             * cost_func[i_state][i_action])
                    elif disc_func == 'beta_delta':
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


def self_control_with_actions_one_step_habit(
        prev_level_effective_policy, level, p_sticky, states, actions, horizon,
        T, reward_func, reward_func_last, cost_func=None, cost_func_last=None,
        discount_factor_reward=None, discount_factor_cost=None,
        discount_beta=None, discount_delta=None, disc_func='diff_disc'):
    """
    calculate higher level policies taking into account the true lower level
    actions and stickiness  (stickiness is one step)
    """

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
            if disc_func == 'diff_disc':
                V_real[i_state, -1] = (
                    (discount_factor_reward**(horizon-i_iter))
                    * reward_func_last[i_state]
                    + (discount_factor_cost**(horizon-i_iter))
                    * cost_func_last[i_state])
            elif disc_func == 'beta_delta':
                V_real[i_state, -1] = (
                    (discount_beta*discount_delta**(horizon-i_iter))
                    * reward_func_last[i_state])

        # backward induction to derive optimal policy starting from
        # timestep i_iter
        for i_timestep in range(horizon-1, i_iter-1, -1):

            for i_state, state in enumerate(states):

                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    # value of next state is either the 'optimal' value or
                    # agent considers the effect of stickiness on future
                    # actions - whether the the future agents also consider
                    # this effect according to each i_iter agent depends on
                    # the level; at level horizon-1, the full recursion is
                    # calculated
                    # effect of stickiness is considered only if
                    # same actions are available in the next state
                    if (i_timestep < horizon-1
                            and i_timestep <= i_iter+level):
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

                    if disc_func == 'diff_disc':
                        r = ((discount_factor_reward**(i_timestep-i_iter))
                             * reward_func[i_state][i_action]
                             + (discount_factor_cost**(i_timestep-i_iter))
                             * cost_func[i_state][i_action])
                    elif disc_func == 'beta_delta':
                        if i_timestep == i_iter:
                            r = reward_func[i_state][i_action]
                        else:
                            r = ((discount_beta
                                  * discount_delta**(i_timestep-i_iter))
                                 * reward_func[i_state][i_action])

                    # q-value for each action (bellman equation)
                    Q[i_action] = (T[i_state][i_action] @ r.T
                                   + T[i_state][i_action]
                                   @ value_next)

                Q_values[i_state][:, i_timestep] = Q

                # what are the real V's? i.e. not the max Q value
                # but the Q-value of the best action of the level-1 agent
                # when levl = horizon-1, the full recursion is calculated
                V_real[i_state, i_timestep] = Q[
                    prev_level_effective_policy[i_state, i_timestep]]

        V_real_full.append(V_real)
        Q_values_full.append(Q_values)

    return V_real_full, Q_values_full


def get_all_levels_self_control(
        level_no, Q_values_full_naive, effective_naive_policy, states, actions,
        horizon, T, reward_func, reward_func_last, cost_func=None,
        cost_func_last=None, discount_factor_reward=None,
        discount_factor_cost=None,  discount_beta=None, discount_delta=None,
        disc_func='diff_disc', state_to_get=0, sticky=False, p_sticky=None):
    """
    Calculate the entire cascade of higher level policies and Q-diffs starting
    from the naive policy; specify for which state (state_to_get) we want
    Q-diffs
    """
    Q_diff_levels_state = []
    policy_levels_state = []
    policy_full_levels = []

    Q_diff_naive = []
    for t in range(horizon):
        Q_diff_naive.append(np.diff(
            Q_values_full_naive[horizon-1-t][state_to_get][:, t])[0])

    Q_diff_levels_state.append(Q_diff_naive)
    policy_levels_state.append(effective_naive_policy[state_to_get])

    effective_policy_prev_level = effective_naive_policy
    policy_full_levels.append(effective_policy_prev_level)

    for level in range(level_no):

        if sticky:
            # calculate next level with stickiness
            V_current_level, Q_current_level = (
                self_control_with_actions_one_step_habit(
                    effective_policy_prev_level, level, p_sticky, states,
                    actions, horizon, T, reward_func, reward_func_last,
                    cost_func, cost_func_last, discount_factor_reward,
                    discount_factor_cost, discount_beta, discount_delta,
                    disc_func))

        else:
            # calculate next level
            V_current_level, Q_current_level = self_control_with_actions(
                effective_policy_prev_level, states, actions, horizon, T,
                reward_func, reward_func_last, cost_func, cost_func_last,
                discount_factor_reward, discount_factor_cost, discount_beta,
                discount_delta, disc_func)

        # update effective policy, Q_diff
        effective_policy_prev_level = np.full((len(states), horizon), 100)
        Q_diff = []  # diff only for state=0
        for t in range(horizon):
            Q_diff.append(np.diff(
                Q_current_level[horizon-1-t][state_to_get][:, t])[0])
            for state in states:
                effective_policy_prev_level[state, horizon-1-t] = np.argmax(
                    Q_current_level[t][state][:, horizon-1-t])

        Q_diff_levels_state.append(Q_diff)
        policy_levels_state.append(effective_policy_prev_level[state_to_get])
        policy_full_levels.append(effective_policy_prev_level)

    return Q_diff_levels_state, policy_levels_state, policy_full_levels
