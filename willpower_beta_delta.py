# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mdp_algms
import task_structure
import helper
import self_control
import matplotlib as mpl
mpl.rcParams['font.size'] = 24

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

        for i_state, _ in enumerate(states):

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
                for i_state, _ in enumerate(states):

                    Q = np.full(len(actions[i_state]), np.nan)

                    for i_action, _ in enumerate(actions[i_state]):

                        if i_timestep == i_iter:
                            r = reward_func[i_state][i_action]
                        else:
                            r = ((discount_beta
                                  * discount_delta**(i_timestep - i_iter))
                                 * reward_func[i_state][i_action])

                        # # updates when even trying improves w
                        # # update P_success(w) based on action:
                        # if i_action == 0:
                        #     w_next = alpha * 0 + (1 - alpha) * w
                        # elif i_action == 1:
                        #     w_next = alpha * 1 + (1 - alpha) * w
                        # i_w_next = np.argmin(np.abs(willpower - w_next))
                        # # Bellman update:
                        # Q[i_action] = (
                        #     T[i_state][i_action] @ r.T
                        #     + T[i_state][i_action]
                        #     @ V_real[i_w_next, :, i_timestep+1])

                        # updates when only successes improve w
                        # update p_success (w) if succcess:
                        w_success = alpha * 1 + (1 - alpha) * w
                        i_w_success = np.argmin(np.abs(willpower - w_success))
                        # if failed:
                        w_failed = alpha * 0 + (1 - alpha) * w
                        i_w_failed = np.argmin(np.abs(willpower - w_failed))
                        # Bellman update
                        Q[i_action] = (
                            T[i_state][i_action] @ r.T
                            # value of next state if failure:
                            + T[i_state][i_action][0] *
                            V_real[i_w_failed, 0, i_timestep+1]
                            # if success:
                            + T[i_state][i_action][1] *
                            V_real[i_w_success, 1, i_timestep+1])

                    Q_values[i_w, i_state][:, i_timestep] = Q
                    # what are the real V's? i.e. not the max Q value
                    # but the Q-value of the best action of the level-1 agent
                    V_real[i_w, i_state, i_timestep] = Q[
                        prev_level_effective_policy[i_w, i_state, i_timestep]]

        V_real_full.append(V_real)
        Q_values_full.append(Q_values)

    return V_real_full, Q_values_full


def simulate_trajectory(effective_policy, w_init, alpha, d_step, states,
                        horizon, plot=False):

    s = 0  # initial
    w = w_init
    willpower = np.arange(0, 1+d_step, d_step)
    actions_executed = []
    s_trajectory = [s]
    w_trajectory = [w]

    for t in range(horizon):
        i_w = np.argmin(np.abs(willpower - w))
        action = effective_policy[i_w, s, t]
        actions_executed.append(action)
        T = task_structure.transitions_cake(p=w)
        s = np.random.choice(len(states), p=T[s][action])  # update state
        w = alpha * s + (1 - alpha) * w  # update w for success increasing w
        # w = alpha * action + (1 - alpha) * w  # update for trying increasing w
        s_trajectory.append(s)
        w_trajectory.append(w)

    if plot:
        actions_executed = np.array(actions_executed)
        w_trajectory = np.array(w_trajectory)
        time = np.arange(horizon)
        plt.plot(w_trajectory, label='w')
        plt.scatter(time[actions_executed == 1],
                    w_trajectory[:-1][actions_executed == 1],
                    label='action=cooperate')
        plt.xticks(np.arange(horizon+1))
        plt.legend()
        plt.show()

    return actions_executed, s_trajectory, w_trajectory


# what if w is uncertain; say w is fixed; naive policy
def uncertain_willpower_naive(
        states, actions, horizon, discount_beta, discount_delta, reward_func,
        reward_func_last, a0=1, b0=1):

    # a0, b0: initial alpha and beta

    V_opt_full = []
    policy_opt_full = []
    Q_values_full = []
    # beta prior for w
    alphas = np.arange(a0, horizon+a0+1, 1)
    betas = np.arange(b0, horizon+b0+1, 1)

    # solve MDP for each agent
    for i_iter in range(horizon-1, -1, -1):

        V_opt = np.full((len(states), len(alphas), len(betas), horizon+1),
                        np.nan)
        policy_opt = np.full((len(states), len(alphas), len(betas), horizon),
                             np.nan)
        Q_values = np.zeros(
            (len(states), len(alphas), len(betas)), dtype=object)

        # fill V for last timestep
        for i_a in range(len(alphas)):
            for i_b in range(len(betas)):
                for i_state in range(len(states)):
                    V_opt[i_state, i_a, i_b, -1] = (
                        (discount_beta*discount_delta**(horizon-i_iter))
                        * reward_func_last[i_state])
                    Q_values[i_state, i_a, i_b] = np.full(
                        (len(actions[i_state]), horizon), np.nan)

        # backward induction
        for i_timestep in range(horizon-1, i_iter-1, -1):
            for i_a, alpha in enumerate(alphas):
                for i_b, beta in enumerate(betas):

                    if (alpha - a0) + (beta - b0) > horizon:
                        continue
                    for i_state in range(len(states)):
                        Q = np.full(len(actions[i_state]), np.nan)

                        for i_action, _ in enumerate(actions[i_state]):

                            # reward with appropriate discounting
                            if i_timestep == i_iter:
                                r = reward_func[i_state][i_action]
                            else:
                                r = ((discount_beta
                                      * discount_delta**(i_timestep-i_iter))
                                     * reward_func[i_state][i_action])

                            # expected w and T
                            expected_w = alpha/(alpha+beta)
                            T = task_structure.transitions_cake(p=expected_w)

                            # belief update and Bellman update
                            if i_action == 0:
                                i_a_next = i_a
                                i_b_next = i_b
                                Q[i_action] = (
                                    T[i_state][i_action] @ r.T
                                    + T[i_state][i_action]
                                    @ V_opt[:, i_a_next, i_b_next,
                                            i_timestep+1])
                            elif i_action == 1:
                                i_a_success = min(i_a + 1, len(alphas) - 1)
                                i_b_success = i_b
                                i_a_fail = i_a
                                i_b_fail = min(i_b + 1, len(betas) - 1)
                                Q[i_action] = (
                                    T[i_state][i_action] @ r.T
                                    + T[i_state][i_action][0]
                                    * V_opt[0, i_a_fail, i_b_fail,
                                            i_timestep+1]
                                    + T[i_state][i_action][1]
                                    * V_opt[1, i_a_success, i_b_success,
                                            i_timestep+1])

                        # find optimal action (which gives max q-value)
                        V_opt[i_state, i_a, i_b, i_timestep] = np.max(Q)
                        policy_opt[i_state, i_a, i_b,
                                   i_timestep] = (np.nan if np.any(np.isnan(Q))
                                                  else np.argmax(Q))
                        Q_values[i_state, i_a, i_b][:, i_timestep] = Q

        V_opt_full.append(V_opt)
        policy_opt_full.append(policy_opt)
        Q_values_full.append(Q_values)

    return V_opt_full, policy_opt_full, Q_values_full


def uncertain_willpower_higher_level(
        prev_level_effective_policy, states, actions, horizon, discount_beta,
        discount_delta, reward_func, reward_func_last, a0=1, b0=1):

    V_real_full = []
    Q_values_full = []
    # beta prior for w
    alphas = np.arange(a0, horizon+a0+1, 1)
    betas = np.arange(b0, horizon+b0+1, 1)

    for i_iter in range(horizon-1, -1, -1):

        V_real = np.full((len(states), len(alphas), len(betas), horizon+1),
                         np.nan)
        Q_values = np.zeros(
            (len(states), len(alphas), len(betas)), dtype=object)

        # fill V for last timestep
        for i_a in range(len(alphas)):
            for i_b in range(len(betas)):
                for i_state in range(len(states)):
                    V_real[i_state, i_a, i_b, -1] = (
                        (discount_beta*discount_delta**(horizon-i_iter))
                        * reward_func_last[i_state])
                    Q_values[i_state, i_a, i_b] = np.full(
                        (len(actions[i_state]), horizon), np.nan)

        for i_timestep in range(horizon-1, i_iter-1, -1):
            for i_a, alpha in enumerate(alphas):
                for i_b, beta in enumerate(betas):

                    if (alpha - a0) + (beta - b0) > horizon:
                        continue
                    for i_state in range(len(states)):
                        Q = np.full(len(actions[i_state]), np.nan)

                        for i_action, _ in enumerate(actions[i_state]):

                            # reward with appropriate discounting
                            if i_timestep == i_iter:
                                r = reward_func[i_state][i_action]
                            else:
                                r = ((discount_beta
                                      * discount_delta**(i_timestep-i_iter))
                                     * reward_func[i_state][i_action])

                            # expected w and T
                            expected_w = alpha/(alpha+beta)
                            T = task_structure.transitions_cake(p=expected_w)

                            # belief update and Bellman update
                            if i_action == 0:
                                i_a_next = i_a
                                i_b_next = i_b
                                Q[i_action] = (
                                    T[i_state][i_action] @ r.T
                                    + T[i_state][i_action]
                                    @ V_real[:, i_a_next, i_b_next,
                                             i_timestep+1])
                            elif i_action == 1:
                                i_a_success = min(i_a + 1, len(alphas) - 1)
                                i_b_success = i_b
                                i_a_fail = i_a
                                i_b_fail = min(i_b + 1, len(betas) - 1)
                                Q[i_action] = (
                                    T[i_state][i_action] @ r.T
                                    + T[i_state][i_action][0]
                                    * V_real[0, i_a_fail, i_b_fail,
                                             i_timestep+1]
                                    + T[i_state][i_action][1]
                                    * V_real[1, i_a_success, i_b_success,
                                             i_timestep+1])

                        Q_values[i_state, i_a, i_b][:, i_timestep] = Q
                        # what are the real V's? i.e. not the max Q value
                        # but Q-value of the best action of the level-1 agent
                        a_real = prev_level_effective_policy[
                            i_state, i_a, i_b, i_timestep]
                        if ~np.isnan(a_real):
                            V_real[i_state, i_a, i_b, i_timestep] = Q[
                                int(a_real)]

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
REWARD_TEMPT = 0.3
EFFORT_RESIST = -0.1
REWARD_RESIST = 0.8
# probability of successfully resisting
P_SUCCESS = 0.7
state_to_get = 0  # state to plot the policies for

# %% policy with no uncertainty

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
                    ylabel='horizon', xlabel='timestep',
                    colorbar_ticklabels=['TEMPT', 'RESIST'])
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
alpha = 0.5
willpower = np.arange(0, 1.0+d_step, d_step)

# put together effective naive policy for different discretised values of w
effective_naive_policy = np.full((len(willpower), len(STATES), HORIZON), 100)
for i_w, w in enumerate(willpower):
    T = task_structure.transitions_cake(p=w)
    _, policy_full_naive, _ = mdp_algms.find_optimal_policy_beta_delta(
        STATES, ACTIONS, HORIZON, DISCOUNT_BETA, DISCOUNT_DELTA,
        reward_func, reward_func_last, T)
    naive_policy = helper.get_effective_policy(
        STATES, policy_full_naive, HORIZON)
    effective_naive_policy[i_w, :, :] = naive_policy

# higher level policys considering lower level policies
policy_full_levels = []
prev_level_effective_policy = effective_naive_policy
policy_full_levels.append(effective_naive_policy)
levels = HORIZON-1  # max level
for level in range(levels):
    V_real_full, Q_values_full = willpower_increase(
        prev_level_effective_policy, alpha, d_step, STATES, ACTIONS, HORIZON,
        DISCOUNT_BETA, DISCOUNT_DELTA, reward_func, reward_func_last)
    effective_policy_level = np.full(
        (len(willpower), len(STATES), HORIZON), 100)
    for t in range(HORIZON):
        for i_w, _ in enumerate(willpower):
            for state in STATES:
                effective_policy_level[i_w, state, HORIZON-1-t] = np.argmax(
                    Q_values_full[t][i_w, state][:, HORIZON-1-t])
    prev_level_effective_policy = effective_policy_level
    policy_full_levels.append(effective_policy_level)

w_init = 0.6
for _ in range(5):
    _, _, _ = simulate_trajectory(policy_full_levels[3], w_init, alpha, d_step,
                                  STATES, HORIZON, plot=True)


# %% uncertain w (but no learning)

# the policy doesnt change with p_success because even if it is small,
# it is still worth trying; even if failed, agent gets atleast r_tempt
reward_func, reward_func_last = task_structure.rewards_cake(
    STATES, REWARD_TEMPT, EFFORT_RESIST, REWARD_RESIST)
T = task_structure.transitions_cake(p=P_SUCCESS)
V_, p_n, Q_n = uncertain_willpower_naive(
    STATES, ACTIONS, HORIZON, DISCOUNT_BETA, DISCOUNT_DELTA, reward_func,
    reward_func_last)

a0 = 1
b0 = 1
alphas = np.arange(a0, HORIZON+a0+1, 1)
betas = np.arange(b0, HORIZON+b0+1, 1)
p_effective = np.full((len(STATES), len(alphas), len(betas), HORIZON), np.nan)
for state in STATES:
    for i_a in range(len(alphas)):
        for i_b in range(len(betas)):
            p_effective[state, i_a, i_b, :] = np.array(
                [p_n[HORIZON-1-i][state, i_a, i_b, i] for i in range(HORIZON)])

# higher levels
policy_full_levels = []
prev_level_effective_policy = p_effective
policy_full_levels.append(p_effective)
levels = HORIZON-1  # max level
for level in range(levels):
    V_level, Q_level = uncertain_willpower_higher_level(
        prev_level_effective_policy, STATES, ACTIONS, HORIZON, DISCOUNT_BETA,
        DISCOUNT_DELTA, reward_func, reward_func_last)
    p_effective_level = np.full(
        (len(STATES), len(alphas), len(betas), HORIZON), np.nan)
    for t in range(HORIZON):
        for state in STATES:
            for i_a in range(len(alphas)):
                for i_b in range(len(betas)):
                    Q = Q_level[t][state, i_a, i_b][:, HORIZON-1-t]
                    p_effective_level[state, i_a, i_b, HORIZON-1-t] = (
                        np.nan if np.any(np.isnan(Q)) else np.argmax(Q))
    prev_level_effective_policy = p_effective_level
    policy_full_levels.append(p_effective_level)


# %% plot policies
level=3 # 0=Naive and then 1, 2, ... HORIZON-1
for t in range(HORIZON):
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(policy_full_levels[level][0, :, :, t], 
                cmap=sns.color_palette('husl', 2),
                linewidths=.5, vmin=0, vmax=1, cbar=False)
    ax.set_xticklabels(betas)
    ax.set_yticklabels(alphas)
    ax.set_xlabel('beta')
    ax.set_ylabel('alpha')

# %%
