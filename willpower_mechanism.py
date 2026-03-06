# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mdp_algms
import task_structure
import helper
import self_control

# %%


def willpower_increase(prev_level_effective_policy, alpha, d_step,
                       states, actions, horizon, discount_beta, discount_delta,
                       reward_func, reward_func_last):

    # probability of success (willpower)
    willpower = np.arange(0, 1 + d_step, d_step)
    V_real_full = []
    Q_values_full = []

    # solve for optimal policy for i_iter-agent,
    # given real actions of future agents
    for i_iter in range(horizon-1, -1, -1):

        V_real = np.full((len(willpower), len(states), horizon+1), np.nan)
        Q_values = np.zeros((len(willpower), len(states)), dtype=object)

        for i_state, state in enumerate(states):

            for i in range(len(willpower)):
                # arrays to store Q-values for each action in each state
                Q_values[i, i_state] = np.full(
                    (len(actions[i_state]), horizon), np.nan)

                V_real[:, i_state, -1] = (
                    (discount_beta*discount_delta**(horizon-i_iter))
                    * reward_func_last[i_state])

        for i_timestep in range(horizon-1, i_iter-1, -1):
            for i_w, w in enumerate(willpower):
                T = task_structure.transitions_cake(p=w)
                for i_state, state in enumerate(states):

                    Q = np.full(len(actions[i_state]), np.nan)

                    for i_action, action in enumerate(actions[i_state]):

                        if i_timestep == i_iter:
                            r = reward_func[i_state][i_action]
                        else:
                            r = ((discount_beta
                                  * discount_delta**(i_timestep - i_iter))
                                 * reward_func[i_state][i_action])

                        # updates when even trying improves w
                        # update P_success (w) based on action:
                        if i_action == 0:
                            w_next = alpha * w + (1 - alpha) * 0
                        elif i_action == 1:
                            w_next = alpha * w + (1 - alpha) * 1
                        i_w_next = np.argmin(np.abs(willpower - w_next))
                        # Bellman update:
                        Q[i_action] = (
                            T[i_state][i_action] @ r.T
                            + T[i_state][i_action]
                            @ V_real[i_w_next, :, i_timestep+1])

                        # updates when only successes improve w
                        # # update p_success (w) if succcess:
                        # w_success = alpha * w + (1 - alpha) * 1
                        # i_w_success = np.argmin(np.abs(willpower - w_success))
                        # # if failed:
                        # w_failed = alpha * w + (1 - alpha) * 0
                        # i_w_failed = np.argmin(np.abs(willpower - w_failed))
                        # # Bellman update
                        # Q[i_action] = (
                        #     T[i_state][i_action] @ r.T
                        #     # value of next state if failure:
                        #     + T[i_state][i_action][0] *
                        #     V_real[i_w_failed, 0, i_timestep+1]
                        #     # if success:
                        #     + T[i_state][i_action][1] *
                        #     V_real[i_w_success, 1, i_timestep+1])

                    Q_values[i_w, i_state][:, i_timestep] = Q
                    # what are the real V's? i.e. not the max Q value
                    # but the Q-value of the best action of the level-1 agent
                    V_real[i_w, i_state, i_timestep] = Q[
                        prev_level_effective_policy[i_w, i_state, i_timestep]]

        V_real_full.append(V_real)
        Q_values_full.append(Q_values)

    return V_real_full, Q_values_full


# %%
STATES = np.arange(2)
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
ACTIONS = [['tempt', 'resist']
           for i in range(len(STATES))]
HORIZON = 4  # deadline
DISCOUNT_BETA = 0.7  # present bias
DISCOUNT_DELTA = 0.8  # standard discounting
# utilities :
REWARD_TEMPT = 0.5
EFFORT_RESIST = 0.0
REWARD_RESIST = 0.8
# probability of successfully resisting
P_SUCCESS = 0.5
state_to_get = 0  # state to plot the policies for

# %%
# the policy doesnt change with p_success because even if it is small,
# it is still worth trying; even if failed, agent gets atleast r_tempt
reward_func, reward_func_last = task_structure.rewards_cake(
    STATES, REWARD_TEMPT, EFFORT_RESIST, REWARD_RESIST)
T = task_structure.transitions_cake(p=P_SUCCESS)

# get and plot naive policy
V_full_naive, policy_full_naive, Q_values_full_naive = (
    mdp_algms.find_optimal_policy_beta_delta(
        STATES, ACTIONS, HORIZON, DISCOUNT_BETA, DISCOUNT_DELTA,
        reward_func, reward_func_last, T))
policy_state = np.array([policy_full_naive[i][state_to_get]
                         for i in range(HORIZON)])
helper.plot_heatmap(policy_state, cmap=sns.color_palette('husl', 2),
                    ylabel='horizon', xlabel='timestep', vmin=0, vmax=1)
effective_naive_policy = helper.get_effective_policy(
    STATES, policy_full_naive, HORIZON)

# get and plot mentalised policy
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

# %% increasing p_success on successful resist
d_step = 0.01
alpha = 1.0
willpower = np.arange(0, 1.0+d_step, d_step)

# naive agent doesn't consider stickiness: so no i_w dimension
# expand it to be able to compare with higher levels
e_repeat = np.repeat(
    effective_naive_policy[np.newaxis, :, :], len(willpower), axis=0)
prev_level_effective_policy = e_repeat
V_real_full, Q_values_full = willpower_increase(
    prev_level_effective_policy, alpha, d_step, STATES, ACTIONS, HORIZON,
    DISCOUNT_BETA, DISCOUNT_DELTA, reward_func, reward_func_last)

effective_policy_level_1 = np.full((len(willpower), len(STATES), HORIZON), 100)
for t in range(HORIZON):
    for i_w, _ in enumerate(willpower):
        for state in STATES:
            effective_policy_level_1[i_w, state, HORIZON-1-t] = np.argmax(
                Q_values_full[t][i_w, state][:, HORIZON-1-t])
