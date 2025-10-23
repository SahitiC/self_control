"""Functions for self control."""

import numpy as np


def self_control_with_actions(
        prev_level_effective_policy, states, actions, horizon, T, reward_func,
        reward_func_last, cost_func=None, cost_func_last=None,
        discount_factor_reward=None, discount_factor_cost=None,
        discount_beta=None, discount_delta=None, disc_func='diff_disc'):

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
