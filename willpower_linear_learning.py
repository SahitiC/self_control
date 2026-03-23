# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mdp_algms
import task_structure
import matplotlib as mpl
mpl.rcParams['font.size'] = 18

# %%

# willpower learning


def willpower_increase(
        eta, dw, states, actions, horizon, discount_factor, reward_func,
        reward_func_last):

    # probability of success (willpower)
    willpower = np.arange(0, 1 + dw, dw)
    # arrays to store values
    V_opt = np.full((len(willpower), len(states), horizon+1), np.nan)
    policy_opt = np.full((len(willpower), len(states), horizon), 100)
    Q_values = np.full((len(willpower), len(states)), np.nan, dtype=object)

    # rewards for final timestep
    for i_state, _ in enumerate(states):
        V_opt[:, i_state, -1] = reward_func_last[i_state]
        for i_w in range(len(willpower)):
            Q_values[i_w, i_state] = np.full(
                (len(actions[i_state]), horizon), np.nan)

    # backward induction
    for i_timestep in range(horizon-1, -1, -1):
        for i_w, w in enumerate(willpower):
            T = task_structure.transitions_cake(p=w)
            for i_state, _ in enumerate(states):
                Q = np.full(len(actions[i_state]), np.nan)
                for i_action, _ in enumerate(actions[i_state]):

                    if i_action == 0:
                        i_w_next = i_w
                        Q[i_action] = (
                            T[i_state][i_action] @ reward_func[i_state][i_action].T
                            + discount_factor * (
                                T[i_state][i_action]
                                @ V_opt[i_w_next, :, i_timestep+1]))

                    elif i_action == 1:
                        # updates when only successes improve w
                        # update p_success (w) if succcess:
                        w_success = np.min([w + eta, 1.0])
                        i_w_success = np.argmin(np.abs(willpower - w_success))
                        # if failed:
                        w_failed = np.max([w - eta, 0.0])
                        i_w_failed = np.argmin(np.abs(willpower - w_failed))
                        # if action=defection, then only w_failure is relevant
                        # Bellman update
                        Q[i_action] = (
                            T[i_state][i_action]
                            @ reward_func[i_state][i_action].T
                            # value of next state if failure/ defect:
                            + discount_factor * (
                                T[i_state][i_action][0]
                                * V_opt[i_w_failed, 0, i_timestep+1])
                            # if success:
                            + discount_factor * (
                                T[i_state][i_action][1]
                                * V_opt[i_w_success, 1, i_timestep+1]))
                    V_opt[i_w, i_state, i_timestep] = np.max(Q)
                    policy_opt[i_w, i_state, i_timestep] = np.argmax(Q)
                    Q_values[i_w, i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values


def simulate_trajectory(policy_opt, w_init, eta, d_step, states,
                        horizon, plot=False):

    s = 0  # initial
    w = w_init
    willpower = np.arange(0, 1+d_step, d_step)
    actions_executed = []
    s_trajectory = [s]
    w_trajectory = [w]

    for t in range(horizon):
        i_w = np.argmin(np.abs(willpower - w))
        action = policy_opt[i_w, s, t]
        actions_executed.append(action)
        T = task_structure.transitions_cake(p=w)
        s = np.random.choice(len(states), p=T[s][action])  # update state
        if action == 0:
            w = w
        elif action == 1:
            w = np.min([w+eta, 1.0]) if s == 1 else np.max([w-eta, 0.0])
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
        plt.xticks(np.arange(0, horizon+1, 5))
        plt.legend(fontsize=14)
        plt.show()

    return actions_executed, s_trajectory, w_trajectory


def uncertain_willpower(states, actions, horizon, discount_factor, reward_func,
                        reward_func_last, a0=1, b0=1):

    # belief vector over w_grid represented by alpha and beta
    alphas = np.arange(a0, horizon+a0+1, 1)
    betas = np.arange(b0, horizon+b0+1, 1)

    # arrays to store values
    V_opt = np.full((len(states), len(alphas), len(betas), horizon+1), np.nan)
    policy_opt = np.full((len(states), len(alphas), len(betas), horizon),
                         np.nan)
    Q_values = np.full((len(states), len(alphas), len(betas)), np.nan,
                       dtype=object)

    # rewards for final timestep
    for i_a in range(len(alphas)):
        for i_b in range(len(betas)):
            for i_state in range(len(states)):
                V_opt[i_state, i_a, i_b, -1] = reward_func_last[i_state]
                Q_values[i_state, i_a, i_b] = np.full(
                    (len(actions[i_state]), horizon), np.nan)

    # backward induction
    for i_t in range(horizon-1, -1, -1):
        for i_a, alpha in enumerate(alphas):
            for i_b, beta in enumerate(betas):
                if (alpha - a0) + (beta - b0) > horizon:
                    continue
                for i_state in range(len(states)):
                    Q = np.full(len(actions[i_state]), np.nan)
                    for i_action, _ in enumerate(actions[i_state]):
                        # expected w and T
                        expected_w = alpha/(alpha+beta)
                        T = task_structure.transitions_cake(p=expected_w)

                        # Belief update and Bellman equation
                        if i_action == 0:
                            i_a_next = i_a
                            i_b_next = i_b
                            Q[i_action] = (
                                T[i_state][i_action]
                                @ reward_func[i_state][i_action].T
                                + discount_factor * (
                                    T[i_state][i_action]
                                    @ V_opt[:, i_a_next, i_b_next, i_t+1]))
                        elif i_action == 1:
                            i_a_success = min(i_a + 1, len(alphas) - 1)
                            i_b_success = i_b
                            i_a_fail = i_a
                            i_b_fail = min(i_b + 1, len(betas) - 1)
                            Q[i_action] = (
                                T[i_state][i_action]
                                @ reward_func[i_state][i_action].T
                                + discount_factor * (
                                    T[i_state][i_action][0] *
                                    V_opt[0, i_a_fail, i_b_fail, i_t+1])
                                + discount_factor * (
                                    T[i_state][i_action][1] *
                                    V_opt[1, i_a_success, i_b_success, i_t+1]))

                    V_opt[i_state, i_a, i_b, i_t] = np.max(Q)
                    policy_opt[i_state, i_a, i_b, i_t] = (
                        np.nan if np.any(np.isnan(Q)) else np.argmax(Q))
                    Q_values[i_state, i_a, i_b][:, i_t] = Q

    return V_opt, policy_opt, Q_values


def simulate_trajectory_uncertainty(
        policy_opt_expl, a0, b0, w_true, states, horizon, plot=False):

    # initial s, alpha a, beta b
    s = 0
    a = a0
    b = b0
    # belief represented by alpha and beta:
    alphas = np.arange(a0, horizon+a0+1, 1)
    betas = np.arange(b0, horizon+b0+1, 1)

    actions_executed = []
    s_trajectory = [s]
    alpha_trajectory = [a0]
    beta_trajectory = [b0]

    for t in range(horizon):
        i_a = a - 1
        i_b = b - 1
        action = int(policy_opt_expl[s, i_a, i_b, t])
        actions_executed.append(action)
        T = task_structure.transitions_cake(p=w_true)  # transition by true w
        s = np.random.choice(len(states), p=T[s][action])  # update state
        if action == 0:
            a = a
            b = b
        elif action == 1:
            if s == 1:
                # success
                a = min(a + 1, len(alphas))
                b = b
            elif s == 0:
                # failure
                a = a
                b = min(b + 1, len(betas))
        s_trajectory.append(s)
        alpha_trajectory.append(a)
        beta_trajectory.append(b)

    if plot:
        actions_executed = np.array(actions_executed)
        alpha_trajectory = np.array(alpha_trajectory)
        beta_trajectory = np.array(beta_trajectory)
        expected_w = alpha_trajectory/(alpha_trajectory + beta_trajectory)
        time = np.arange(horizon)
        plt.plot(expected_w, label='expected w')
        plt.scatter(time[actions_executed == 1],
                    expected_w[:-1][actions_executed == 1],
                    label='action=cooperate')
        plt.xticks(np.arange(0, horizon+1, 5))
        plt.legend(fontsize=14)
        plt.show()

    return actions_executed, s_trajectory, alpha_trajectory, beta_trajectory


# case where real w improves on cooperation but there is also uncertainty
# about the real w
def compute_belief(belief_0, ns, nf, eta, w_grid):
    belief = belief_0.copy()
    for _ in range(ns):
        # Bayesian reweighting
        belief = belief * w_grid
        belief /= belief.sum()
        # shift w_grid by eta (success increases w)
        # move belief mass by that many steps in belief grid
        belief_new = np.interp(w_grid, w_grid + eta, belief, left=0, right=0)
        mass_right = belief[w_grid + eta > w_grid[-1]].sum()
        belief_new[-1] += mass_right
        belief /= belief.sum()

    for _ in range(nf):
        # Bayesian reweighting
        belief = belief * (1 - w_grid)
        belief /= belief.sum()
        # shift w_grid by -eta (failure decreases w)
        # move belief mass by that many steps in belief grid
        belief_new = np.interp(w_grid, w_grid - eta, belief, left=0, right=0)
        mass_left = belief[w_grid - eta < w_grid[0]].sum()
        belief_new[0] += mass_left
        belief /= belief.sum()

    return belief


def willpower_learning_uncertain(
        eta, dw, states, actions, horizon, discount_factor, reward_func,
        reward_func_last):

    # probability of success w (willpower)
    w_grid = np.arange(0, 1 + dw, dw)
    N = len(w_grid)
    # prior belief vector over w grid
    belief_0 = np.ones(N) / N

    belief_grid = {}
    for n_s in range(horizon+1):
        for n_f in range(horizon+1):
            if n_s + n_f > horizon:
                continue
            belief_grid[(n_s, n_f)] = compute_belief(
                belief_0, n_s, n_f, eta, w_grid)

    # arrays to store values; belief is determined by number of failures
    # and successes which will be at max n=horizon
    V_opt = np.full((len(states), horizon+1,
                     horizon+1, horizon+1), np.nan)
    policy_opt = np.full((len(states), horizon+1, horizon+1,
                          horizon), np.nan)
    Q_values = np.full((len(states), horizon+1, horizon+1), np.nan,
                       dtype=object)

    # rewards for final timestep
    for ns in range(horizon+1):
        for nf in range(horizon+1):
            for i_state in range(len(states)):
                V_opt[i_state, ns, nf, -1] = reward_func_last[i_state]
                Q_values[i_state, ns, nf] = np.full(
                    (len(actions[i_state]), horizon), np.nan)

    # backward induction
    for i_t in range(horizon-1, -1, -1):
        for ns in range(horizon+1):
            for nf in range(horizon+1):
                if ns + nf > i_t:
                    continue
                belief = belief_grid[ns, nf]
                expected_w = belief @ w_grid
                T = task_structure.transitions_cake(p=expected_w)
                for i_state in range(len(states)):
                    Q = np.full(len(actions[i_state]), np.nan)
                    for i_action, _ in enumerate(actions[i_state]):

                        if i_action == 0:
                            ns_next = ns
                            nf_next = nf
                            Q[i_action] = (
                                T[i_state][i_action]
                                @ reward_func[i_state][i_action].T
                                + discount_factor * (
                                    T[i_state][i_action]
                                    @ V_opt[:, ns_next, nf_next, i_t+1]))
                        elif i_action == 1:
                            ns_s = ns + 1
                            nf_s = nf
                            ns_f = ns
                            nf_f = nf + 1
                            Q[i_action] = (
                                T[i_state][i_action]
                                @ reward_func[i_state][i_action].T
                                + discount_factor * (
                                    T[i_state][i_action][0] *
                                    V_opt[0, ns_f, nf_f, i_t+1])
                                + discount_factor * (
                                    T[i_state][i_action][1] *
                                    V_opt[1, ns_s, nf_s, i_t+1]))

                    V_opt[i_state, ns, nf, i_t] = np.max(Q)
                    policy_opt[i_state, ns, nf, i_t] = (
                        np.nan if np.any(np.isnan(Q)) else np.argmax(Q))
                    Q_values[i_state, ns, nf][:, i_t] = Q

    return V_opt, policy_opt, Q_values


def simulate_trajectory_learning_uncertainty(
        policy_opt, w_init, dw, eta, states, horizon, plot=False):
    
    w_grid = np.arange(0, 1 + dw, dw)
    N = len(w_grid)
    # prior belief vector over w grid
    belief_0 = np.ones(N) / N
    belief_grid = {}
    for n_s in range(horizon+1):
        for n_f in range(horizon+1):
            if n_s + n_f > horizon:
                continue
            belief_grid[(n_s, n_f)] = compute_belief(
                belief_0, n_s, n_f, eta, w_grid)

    # initial s, w, belief
    w = w_init
    s = 0
    ns = 0
    nf = 0
    belief = belief_0

    actions_executed = []
    s_trajectory = [s]
    w_trajectory = [w]
    success_trajectory = [ns]
    failure_trajectory = [nf]
    expected_ws = [belief @ w_grid]

    for t in range(horizon):
        action = int(policy_opt[s, ns, nf, t])
        actions_executed.append(action)
        T = task_structure.transitions_cake(p=w)  # transition by current w
        s = np.random.choice(len(states), p=T[s][action])  # update state
        if action == 0:
            ns = ns
            nf = nf
            w = w
        elif action == 1:
            if s == 1:
                # success
                ns += 1
                nf = nf
                w = np.min([w+eta, 1.0])
            elif s == 0:
                # failure
                ns = ns
                nf += 1
                w = np.max([w-eta, 0.0])
        belief = belief_grid[ns, nf]
        expected_ws.append(belief @ w_grid)
        s_trajectory.append(s)
        w_trajectory.append(w)
        success_trajectory.append(ns)
        failure_trajectory.append(nf)

    if plot:
        actions_executed = np.array(actions_executed)
        expected_ws = np.array(expected_ws)
        time = np.arange(horizon)
        plt.plot(expected_ws, label='expected w')
        plt.scatter(time[actions_executed == 1],
                    expected_ws[:-1][actions_executed == 1],
                    label='action=cooperate')
        plt.xticks(np.arange(0, horizon+1, 5))
        plt.legend(fontsize=14)
        plt.show()

    return actions_executed, s_trajectory, w_trajectory, expected_ws, success_trajectory, failure_trajectory


# %%


STATES = np.arange(2)
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
ACTIONS = [['tempt', 'resist']
           for i in range(len(STATES))]
HORIZON = 20  # deadline
DISCOUNT_FACTOR = 1
# utilities :
REWARD_TEMPT = 0.5
EFFORT_RESIST = -0.1
REWARD_RESIST = 0.8
# probability of successfully resisting
P_SUCCESS = 0.32
state_to_get = 0  # state to plot the policies for

# %% policy without learning in w

reward_func, reward_func_last = task_structure.rewards_cake(
    STATES, REWARD_TEMPT, EFFORT_RESIST, REWARD_RESIST)
T = task_structure.transitions_cake(p=P_SUCCESS)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func, reward_func_last,
    T)

f, ax = plt.subplots(figsize=(5, 2))
sns.heatmap(policy_opt[state_to_get, :][np.newaxis, :], linewidths=0.5,
            cmap=sns.color_palette('husl', 2), cbar=False, vmin=0, vmax=1)
ax.set_yticks([])
ax.set_xlabel('time step')

# %% with learning
eta = 0.2
dw = 0.01
V_opt, policy_opt, Q_values = willpower_increase(
    eta, dw, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
    reward_func_last)

# simulate trajectories
w_init = 0.3
a, s, w = simulate_trajectory(policy_opt, w_init, eta, dw, STATES, HORIZON, plot=True)

# %% with uncertainty in willpower
a0 = 1
b0 = 1
V_opt_expl, policy_opt_expl, Q_values_expl = uncertain_willpower(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func, reward_func_last,
    a0=a0, b0=b0)

# simulated trajectories
w_real = 0.29
_, _, alpha_traj, beta_traj = simulate_trajectory_uncertainty(
    policy_opt_expl, a0, b0, w_real, STATES, HORIZON, plot=True)


# %% with learning and uncertainty in w
eta = 0.2
dw = 0.01
willpower = np.arange(0, 1.0+dw, dw)
V_opt, policy_opt, Q_values = willpower_learning_uncertain(
    eta, dw, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
    reward_func_last)
w_init = 0.3
actions_executed, s_trajectory, w_trajectory, expected_ws, success_trajectory, failure_trajectory = simulate_trajectory_learning_uncertainty(
        policy_opt, w_init, dw, eta, STATES, HORIZON, plot=True)
