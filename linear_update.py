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
