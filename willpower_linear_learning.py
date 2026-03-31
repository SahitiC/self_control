# %%
import plotter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mdp_algms
import task_structure
import bamdp_tree
import matplotlib as mpl
import matplotlib.colors as mcolors
from scipy.stats import beta
import pandas as pd
mpl.rcParams['font.size'] = 18
# %%

# willpower learning


def willpower_training(
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

                    # updates when only successes improve w
                    # update p_success (w) if succcess:
                    w_success = np.min([w + eta, 1.0])
                    i_w_success = np.argmin(np.abs(willpower - w_success))
                    # if failed:
                    # if action=defection, then only w_failure is relevant
                    w_failed = np.max([w - eta, 0.0])
                    i_w_failed = np.argmin(np.abs(willpower - w_failed))
                    # Bellman update
                    Q[i_action] = (
                        T[i_state][i_action] @ reward_func[i_state][i_action].T
                        # value of next state if failure / defect:
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
                        horizon, t_start=0, plot=False):

    s = 0  # initial
    w = w_init
    willpower = np.arange(0, 1+d_step, d_step)
    actions_executed = []
    s_trajectory = [s]
    w_trajectory = [w]

    for t in range(t_start, horizon):
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
        plotter.plot_single_trajectory(
            actions_executed, w_trajectory, horizon, t_start=t_start,
            action_label='cooperate', w_label='w')

    return actions_executed, s_trajectory, w_trajectory


def uncertain_willpower(states, actions, horizon, discount_factor, reward_func,
                        reward_func_last, a0=1, b0=1):

    # belief vector over w_grid represented by alpha and beta
    alphas = np.arange(1, horizon+a0+1, 1)
    betas = np.arange(1, horizon+b0+1, 1)

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
    policy_dicts = []
    for i_t in range(horizon-1, -1, -1):
        policy_dict = {}
        for i_a, alpha in enumerate(alphas):
            for i_b, Beta in enumerate(betas):
                if (alpha - a0) + (Beta - b0) > i_t:
                    continue
                for i_state in range(len(states)):
                    Q = np.full(len(actions[i_state]), np.nan)
                    for i_action, _ in enumerate(actions[i_state]):
                        # expected w and T
                        expected_w = alpha/(alpha+Beta)
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
                    max_action = (
                        np.nan if np.any(np.isnan(Q)) else np.argmax(Q))
                    policy_opt[i_state, i_a, i_b, i_t] = max_action
                    Q_values[i_state, i_a, i_b][:, i_t] = Q
                    policy_dict[alpha/(alpha+Beta)] = max_action
        policy_dicts.append(policy_dict)

    return V_opt, policy_opt, Q_values, policy_dicts


def simulate_trajectory_uncertainty(
        policy_opt_expl, a0, b0, policy_a0, policy_b0, w_true, states, horizon,
        t_start=0, plot=False):

    # initial s, alpha a, beta b
    s = 0
    a = a0
    b = b0
    # belief represented by alpha and beta:
    alpha_max = horizon-t_start+policy_a0+1
    beta_max = horizon-t_start+policy_b0+1

    actions_executed = []
    s_trajectory = [s]
    alpha_trajectory = [a0]
    beta_trajectory = [b0]

    for t in range(t_start, horizon):
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
                a = min(a + 1, alpha_max)
                b = b
            elif s == 0:
                # failure
                a = a
                b = min(b + 1, beta_max)
        s_trajectory.append(s)
        alpha_trajectory.append(a)
        beta_trajectory.append(b)

    if plot:
        alpha_trajectory = np.array(alpha_trajectory)
        beta_trajectory = np.array(beta_trajectory)
        expected_w = alpha_trajectory/(alpha_trajectory + beta_trajectory)
        plotter.plot_single_trajectory(
            actions_executed, expected_w, horizon, t_start=t_start,
            action_label='cooperate',
            w_label='expected w')

    return actions_executed, s_trajectory, alpha_trajectory, beta_trajectory


def forward_simulate_from_t(policy, initial_state_t, t, horizon, states, T):

    # arrays to store states, actions taken and values of actions in time
    states_forward = np.full(horizon+1-t, 100)
    actions_forward = np.full(horizon-t, 100)

    states_forward[0] = initial_state_t

    for i in range(t, horizon):

        # action at a state and timestep as given by policy
        actions_forward[i] = policy[states_forward[i], i]
        # sample next state from transition probabilities
        states_forward[i+1] = np.random.choice(
            len(states),
            p=T[states_forward[i]][actions_forward[i]])

    return states_forward, actions_forward


def beta_prior_on_grid(w_grid, a, b):
    w_grid = np.asarray(w_grid)
    N = len(w_grid)
    dw = np.zeros(N)  # compute bin widths
    dw[1:-1] = 0.5 * (w_grid[2:] - w_grid[:-2])
    dw[0] = 0.5 * (w_grid[1] - w_grid[0])
    dw[-1] = 0.5 * (w_grid[-1] - w_grid[-2])
    # convert to probability mass
    pdf_vals = beta.pdf(w_grid, a, b)
    p = pdf_vals * dw
    p /= p.sum()
    return p


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncates a colormap."""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


# %%
STATES = np.arange(2)
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
ACTIONS = [['tempt', 'resist']
           for i in range(len(STATES))]
HORIZON = 14  # deadline
DISCOUNT_FACTOR = 1
# utilities :
REWARD_TEMPT = 0.5
EFFORT_RESIST = -0.1
REWARD_RESIST = 0.8
# probability of successfully resisting
P_SUCCESS = 1/3
state_to_get = 0  # state to plot the policies for
SAVE_PLOTS = False
original_cmap = plt.get_cmap('GnBu')
# Use only the part of the original cmap
GnBu_trunc = truncate_colormap(original_cmap, 0.2, 1.0)
np.random.seed(0)

# %% policy without training in w

reward_func, reward_func_last = task_structure.rewards_cake(
    REWARD_TEMPT, EFFORT_RESIST, REWARD_RESIST)
T = task_structure.transitions_cake(p=P_SUCCESS)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func, reward_func_last,
    T)

# %% plot policy across w_grid (without training)
dw = 0.01
w_grid = np.arange(0, 1+dw, dw)
policy_w = []
for w in w_grid:
    T = task_structure.transitions_cake(p=w)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
        reward_func_last, T)
    policy_w.append(policy_opt[0, :])  # policy is the same for s= 0 or 1
policy_w = np.array(policy_w)
plotter.plot_w_policy(policy_w, w_grid, dw, HORIZON)
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/policy_no_training_no_exploration.svg',
        format='svg', dpi=300)

# %% plot avg cooperation in future
avg_cooperation = np.full((len(w_grid), HORIZON), np.nan)
for t in range(HORIZON):
    for i_w, w in enumerate(w_grid):
        all_actions = []
        for _ in range(100):
            p = np.repeat(policy_w[i_w, :][np.newaxis, :], 2, axis=0)
            s, a = mdp_algms.forward_runs(p, 0, HORIZON, STATES, T,
                                          t_start=t)
            all_actions.extend(a)
        avg_cooperation[i_w, t] = np.mean(all_actions)

f, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(avg_cooperation, cmap=GnBu_trunc, linewidths=0, rasterized=True)
ax.set_xticks(np.arange(0, HORIZON+1, 5))
ax.set_xticklabels(np.arange(0, HORIZON+1, 5))
ax.set_xlabel('time step')
ax.set_yticks(np.arange(0, len(w_grid), int(len(w_grid)/5)))
ax.set_yticklabels(np.arange(0, len(w_grid), int(len(w_grid)/5))*dw)
ax.set_ylabel('w')
ax.invert_yaxis()
ax.grid(False)
colorbar = ax.collections[0].colorbar
colorbar.set_label(f"Avg % future 'try' actions", rotation=270, labelpad=20)
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/avg_cooperation_no_training_no_exploration.svg',
        format='svg', dpi=300)

# %% get mean trajectories
w = 1/3
T = task_structure.transitions_cake(p=w)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
    reward_func_last, T)
acs = []
for _ in range(100):
    s, a = mdp_algms.forward_runs(policy_opt, 0, HORIZON, STATES, T)
    acs.append(a)
acs = np.array(acs)
avg_cooperation_no_training_no_uncertainty = np.mean(acs, axis=0)

# %% with w training
eta = 0.2
dw = 0.01
w_grid = np.arange(0, 1+dw, dw)
V_opt, policy_opt, Q_values = willpower_training(
    eta, dw, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
    reward_func_last)

# simulate example trajectory
w_init = 1/3
a, s, w = simulate_trajectory(
    policy_opt, w_init, eta, dw, STATES, HORIZON, t_start=0, plot=True)

# %% plot policy in w grid over time
plotter.plot_w_policy(policy_opt[:, 0, :], w_grid, dw, HORIZON)
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/policy_only_training.svg',
        format='svg', dpi=300)

# %% plot avg cooperation in future in w and time space
avg_cooperation = np.full((len(w_grid), HORIZON), np.nan)
for t in range(HORIZON):
    for i_w, w in enumerate(w_grid):
        all_actions = []
        for _ in range(100):
            a, _, _ = simulate_trajectory(
                policy_opt, w, eta, dw, STATES, HORIZON,
                t_start=t, plot=False)
            all_actions.extend(a)
        avg_cooperation[i_w, t] = np.mean(all_actions)

f, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(avg_cooperation, cmap=GnBu_trunc, linewidths=0, rasterized=True)
ax.set_xticks(np.arange(0, HORIZON+1, 5))
ax.set_xticklabels(np.arange(0, HORIZON+1, 5))
ax.set_xlabel('time step')
ax.set_yticks(np.arange(0, len(w_grid), int(len(w_grid)/5)))
ax.set_yticklabels(np.arange(0, len(w_grid), int(len(w_grid)/5))*dw)
ax.set_ylabel('w')
ax.invert_yaxis()
ax.grid(False)
colorbar = ax.collections[0].colorbar
colorbar.set_label(f"Avg % future 'try' actions", rotation=270, labelpad=20)
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/avg_cooperation_only_training.svg',
        format='svg', dpi=300)

# %% get mean trajectories
ws = []
acs = []
for i in range(100):
    a, s, w = simulate_trajectory(
        policy_opt, w_init, eta, dw, STATES, HORIZON, plot=False)
    ws.append(w)
    acs.append(a)
ws = np.array(ws)
acs = np.array(acs)
avg_cooperation_training_no_uncertainty = np.mean(acs, axis=0)

# %% plot example trajectories
plt.figure(figsize=(4, 4))
plotter.plot_single_trajectory(acs[11], ws[11], HORIZON, action_label='Try',
                               color='tab:orange', legend=False)
plotter.plot_single_trajectory(acs[15], ws[15], HORIZON, action_label='Try',
                               color='tab:orange', legend=False)
plotter.plot_single_trajectory(acs[16], ws[16], HORIZON, action_label='Try',
                               color='tab:orange')
plt.xlabel('trial')
sns.despine()
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/trajectories_only_training.svg',
        format='svg', dpi=300)

ac_only_training = acs[22]
w_only_training = ws[22]

# %% with uncertainty in willpower (no training)
a0 = 1
b0 = 2
V_opt_expl, policy_opt_expl, Q_values_expl, policy_dicts = uncertain_willpower(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func, reward_func_last,
    a0=a0, b0=b0)

# simulate example trajectories
w_real = 1/3
_, _, alpha_traj, beta_traj = simulate_trajectory_uncertainty(
    policy_opt_expl, a0, b0, a0, b0, w_real, STATES, HORIZON, plot=True)

# %% plot policy for expected w (a/a+b)
policy_dicts.reverse()
policy_df = pd.DataFrame(policy_dicts)
policy_df.columns = policy_df.columns.astype(float)
policy_df = policy_df.sort_index(axis=1)
f, ax = plt.subplots(figsize=(5, 4))
# policy_df = policy_df.interpolate(axis=1)
sns.heatmap(policy_df.T, cmap=sns.color_palette('husl', 2), cbar=True,
            vmin=0, vmax=1)
ax.set_yticks([])
ax.set_xticks(np.arange(0, HORIZON+1, 5))
ax.set_xticklabels(np.arange(0, HORIZON+1, 5))
ax.set_xlabel('time step')
ax.set_ylabel('E(w)')
ax.invert_yaxis()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels([0, 1])
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/policy_only_exploration.svg',
        format='svg', dpi=300)

# %% plot avg cooperation in future in w and time space
alphas = np.arange(1, HORIZON+a0+1, 1)
betas = np.arange(1, HORIZON+b0+1, 1)
avg_cooperation = {}
for t in range(HORIZON):
    for i_a, alpha in enumerate(alphas):
        for i_b, Beta in enumerate(betas):
            if (alpha - a0) + (Beta - b0) > t:
                continue
            all_actions = []
            for _ in range(100):
                a, _, _, _ = simulate_trajectory_uncertainty(
                    policy_opt_expl, alpha, Beta, a0, b0, w_real, STATES,
                    HORIZON, t_start=t, plot=False)
                all_actions.extend(a)
            avg_cooperation[(alpha/(alpha+Beta), t)] = np.mean(all_actions)
avg_cooperation = pd.DataFrame(
    [(w, t, p) for (w, t), p in avg_cooperation.items()],
    columns=["w", "time", "avg"])
heatmap_df = avg_cooperation.pivot_table(
    index="w",
    columns="time",
    values="avg")
f, ax = plt.subplots(figsize=(5, 4))
# heatmap_df = heatmap_df.interpolate(axis=0, limit_direction="both")
sns.heatmap(heatmap_df, vmin=0, vmax=1, cmap=GnBu_trunc, linewidths=0,
            rasterized=True)
ax.set_xticks(np.arange(0, HORIZON+1, 5))
ax.set_xticklabels(np.arange(0, HORIZON+1, 5))
ax.set_xlabel('trial')
ax.set_yticks([])
ax.set_ylabel('E(w)')
ax.invert_yaxis()
ax.grid(False)
colorbar = ax.collections[0].colorbar
colorbar.set_label(f"Avg % future 'try' actions", rotation=270, labelpad=20)
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/avg_cooperation_only_exploration.svg',
        format='svg', dpi=300)

# %% get avg trajectories
ws = []
acs = []
for i in range(100):
    a, _, alpha_traj, beta_traj = simulate_trajectory_uncertainty(
        policy_opt_expl, a0, b0, a0, b0, w_real, STATES, HORIZON, plot=False)
    w = np.array(alpha_traj)/(np.array(alpha_traj)+np.array(beta_traj))
    ws.append(w)
    acs.append(a)
ws = np.array(ws)
acs = np.array(acs)
avg_cooperation_no_training_uncertainty = np.mean(acs, axis=0)

# %% plot example trajectories
plt.figure(figsize=(4, 4))
plotter.plot_single_trajectory(acs[4], ws[4], HORIZON, action_label='try',
                               w_label='E(w)', color='tab:purple',
                               legend=False)
plotter.plot_single_trajectory(acs[8], ws[8], HORIZON, action_label='try',
                               w_label='E(w)', color='tab:purple',
                               legend=False)
plotter.plot_single_trajectory(acs[9], ws[9], HORIZON, action_label='try',
                               w_label='E(w)', color='tab:purple')
plt.xlabel('trial')
plt.ylim(0.1, 1.05)
sns.despine()
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/trajectories_only_exploration.svg',
        format='svg', dpi=300)
ac_only_exploration = acs[0]
w_only_exploration = ws[0]

# %% with training and uncertainty in w
HORIZON = 14
eta = 0.2
dw = 0.01
w_grid = np.arange(0, 1.0+dw, dw)
# set belief as a discretised beta prior (a, b)
a = 1
b = 2
belief_w = beta_prior_on_grid(w_grid, a, b)  # np.ones(len(w_grid))/len(w_grid)
belief_w = bamdp_tree.np_to_tuple(belief_w)
mdp = bamdp_tree.CakeMDP(states=[0, 1],
                         state_to_action_space=[[0, 1] for _ in range(2)],
                         transition_fn=bamdp_tree.transitions_cake,
                         reward_fn=bamdp_tree.rewards_cake,
                         reward_last_fn=bamdp_tree.rewards_last,
                         w_grid=w_grid,
                         eta=eta,
                         reward_tempt=REWARD_TEMPT,
                         effort_resist=EFFORT_RESIST,
                         reward_resist=REWARD_RESIST,
                         gamma=DISCOUNT_FACTOR)
h0 = bamdp_tree.BeliefState(s=0, belief_w=belief_w, t=0)
list_of_h_sets = bamdp_tree.forward_pass(h0, HORIZON, mdp)
V, Q, pi, pi_w = bamdp_tree.backward_pass(list_of_h_sets, mdp)

# %% plot policy for expected w
# policy_dicts.reverse()
policy_df = pd.DataFrame(
    [(t, s, w, p) for (t, s, w), p in pi_w.items()],
    columns=["time", "state", "w", "policy"])
policy_df = policy_df[policy_df["state"] == 0]
policy_df["w_bin"] = pd.cut(policy_df["w"], bins=100)
heatmap_df = policy_df.pivot_table(
    index="w_bin",
    columns="time",
    values="policy")
# policy_df = policy_df.sort_index(axis=1)
f, ax = plt.subplots(figsize=(5, 4))
# heatmap_df = heatmap_df.interpolate(axis=0)
sns.heatmap(heatmap_df, cmap=sns.color_palette('husl', 2), cbar=True,
            vmin=0, vmax=1, linewidths=0, rasterized=True)
ax.set_yticks([])
ax.set_xticks(np.arange(0, HORIZON+1, 5))
ax.set_xticklabels(np.arange(0, HORIZON+1, 5))
ax.set_xlabel('trial')
ax.set_ylabel('E(w)')
ax.invert_yaxis()
ax.grid(False)
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels([0, 1])
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/policy_training_exploration.svg',
        format='svg', dpi=300)

# %% simulate
w_true_init = 1/3
trajectory, rewards, actions, w_trues = bamdp_tree.online_simulation(
    pi, h0, w_true_init, w_grid, eta, HORIZON, mdp)
plot = True
if plot:
    w_trajectory = []
    for i in range(HORIZON + 1):
        w_expected = np.sum(np.array(trajectory[i].belief_w) * w_grid)
        w_trajectory.append(w_expected)
    plotter.plot_single_trajectory(
        actions, w_trajectory, HORIZON, action_label='cooperate',
        w_label='expected w')

# %% avg cooperation in future across w-time space

avg_cooperation = {}
for t in range(HORIZON):
    for h in list_of_h_sets[t]:
        all_actions = []
        for _ in range(10):
            _, _, a, _ = bamdp_tree.online_simulation(
                pi, h, w_true_init, w_grid, eta, HORIZON, mdp, t_start=t)
            all_actions.extend(a)
        avg_cooperation[(np.sum(h.belief_w*w_grid), t)] = np.mean(all_actions)
avg_cooperation = pd.DataFrame(
    [(w, t, p) for (w, t), p in avg_cooperation.items()],
    columns=["w", "time", "avg"])
avg_cooperation["w_bin"] = pd.cut(avg_cooperation["w"], bins=100)
heatmap_df = avg_cooperation.pivot_table(
    index="w_bin",
    columns="time",
    values="avg")
f, ax = plt.subplots(figsize=(5, 4))
# heatmap_df = heatmap_df.interpolate(axis=0)
sns.heatmap(heatmap_df, cmap=GnBu_trunc, linewidths=0, rasterized=True, vmin=0,
            vmax=1)
ax.set_xticks(np.arange(0, HORIZON+1, 5))
ax.set_xticklabels(np.arange(0, HORIZON+1, 5))
ax.set_xlabel('trial')
ax.set_yticks([])
ax.set_ylabel('E(w)')
ax.invert_yaxis()
ax.grid(False)
colorbar = ax.collections[0].colorbar
colorbar.set_label(f"Avg % future cooperation", rotation=270, labelpad=20)
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/avg_cooperation_training_exploration.svg',
        format='svg', dpi=300)

# %% get average trajectories
ws = []
acs = []
for i in range(100):
    trajectory, _, action, _ = bamdp_tree.online_simulation(
        pi, h0, w_true_init, w_grid, eta, HORIZON, mdp)
    w_trajectory = []
    for i in range(HORIZON + 1):
        w_expected = np.sum(np.array(trajectory[i].belief_w) * w_grid)
        w_trajectory.append(w_expected)
    acs.append(np.array(action))
    ws.append(np.array(w_trajectory))
ws = np.array(ws)
acs = np.array(acs)
avg_cooperation_training_uncertainty = np.mean(acs, axis=0)

# %% plot example trajectories + compare with prev two cases
# plotter.plot_single_trajectory(acs[6], ws[6], HORIZON, legend=False)
# plt.xlabel('time')
time = np.arange(0, HORIZON+1)
plt.figure(figsize=(4, 4))
plt.plot(time, w_only_training, label='only training', color='tab:orange',
         linestyle='--', linewidth=2)
plt.scatter(time[:-1][ac_only_training == 1],
            w_only_training[:-1][ac_only_training == 1], s=30,
            color=sns.color_palette('husl', 2)[1])
plt.plot(time, w_only_exploration, label='only exploration',
         color='tab:purple', linestyle='--', linewidth=2)
plt.scatter(time[:-1][ac_only_exploration == 1],
            w_only_exploration[:-1][ac_only_exploration == 1], s=30,
            color=sns.color_palette('husl', 2)[1])
plt.plot(time, ws[6], label='training + exploration', color='tab:red',
         linestyle='--', linewidth=2)
plt.scatter(time[:-1][acs[6] == 1],
            ws[6][:-1][acs[6] == 1], s=30,
            color=sns.color_palette('husl', 2)[1])
plt.xticks(np.arange(0, HORIZON+1, 5))
plt.xlabel('trial')
plt.legend(fontsize=12, frameon=False)
sns.despine()
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/trajectories_training_exploration.svg',
        format='svg', dpi=300)

# %% plot avg cooperation in all cases
time = np.arange(HORIZON)
plt.figure(figsize=(4, 4))
plt.plot(time, avg_cooperation_no_training_no_uncertainty,
         label='no training, no exploration', linewidth=2, color='tab:blue')
plt.plot(time, avg_cooperation_training_no_uncertainty,
         label='only training', linewidth=2, color='tab:orange')
plt.plot(time, avg_cooperation_no_training_uncertainty,
         label='only exploration', linewidth=2, color='tab:purple')
plt.plot(time, avg_cooperation_training_uncertainty,
         label='training + exploration', linewidth=2, color='tab:red')
plt.legend(bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
plt.xticks(np.arange(0, HORIZON, 5))
plt.xlabel('trial')
plt.title(f"Average % of 'try' actions")
if SAVE_PLOTS:
    plt.savefig(
        f'plots/vectors/avg_cooperation_all.svg',
        format='svg', dpi=300)

# %%
