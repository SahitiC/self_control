# %%
import numpy as np
import matplotlib.pyplot as plt
import mdp_algms
import seaborn as sns
import task_structure

# %%

# construct reward functions separately for rewards and costs


def get_reward_functions(states, reward_do, effort_do, reward_completed,
                         cost_not_completed):

    reward_func = []

    # reward for actions (depends on current state and next state)
    # rewards for don't and do
    reward_func.append([np.array([0, 0]),
                        np.array([effort_do, reward_do+effort_do])])
    reward_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    reward_func_last = np.array([cost_not_completed, reward_completed])

    return reward_func, reward_func_last


def plan_with_habits_one_step(
        p_sticky, states, actions, horizon, discount_factor, reward_func,
        reward_func_last, T):
    """Derive optimal higher level policy taking stickiness into account.
    Stickiness is one-step: current action affects next action with probability
    p_sticky."""

    # arrays for optimal values, policy, Q-values
    V_opt = np.full((len(states), horizon+1), np.nan)
    policy_opt = np.full((len(states), horizon), 100)
    Q_values = np.full(len(states), np.nan, dtype=object)

    for i_state, state in enumerate(states):

        # V_opt for last time-step
        V_opt[i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full((len(actions[i_state]), horizon), np.nan)

    # backward induction to derive optimal policy
    for i_timestep in range(horizon-1, -1, -1):

        for i_state, state in enumerate(states):

            Q = np.full(len(actions[i_state]), np.nan)

            for i_action, action in enumerate(actions[i_state]):

                # value of next state (if same as current state) in the next
                # timestep is: p * value of current action in next state +
                # (1-p) * value of optimal action in next state
                if i_timestep < horizon-1:
                    value_next = np.full(len(states), 0.0)
                    for next_state in range(len(states)):
                        if actions[next_state] == actions[i_state]:
                            opt_action = policy_opt[next_state, i_timestep+1]
                            value_next[next_state] = (
                                p_sticky *
                                Q_values[next_state][i_action, i_timestep+1]
                                + (1-p_sticky) *
                                Q_values[next_state][opt_action, i_timestep+1])
                        else:
                            value_next[next_state] = V_opt[next_state,
                                                           i_timestep+1]

                else:
                    value_next = V_opt[:, i_timestep+1]

                # q-value for each action (bellman equation)
                Q[i_action] = (T[i_state][i_action]
                               @ reward_func[i_state][i_action].T
                               + discount_factor * (T[i_state][i_action]
                                                    @ value_next))

            # find optimal action (which gives max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            Q_values[i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values


def update_memory_habit(alpha, x, W, action, action_num):
    if action_num > 1:
        W_next = alpha * W + 1
        # assuming action is 0/1 and 1 is "cooperate" (work or resist):
        x_next = (alpha * W * x + action) / W_next
    else:
        x_next = x
    return x_next


def plan_with_habits(
        p, alpha, d_step, states, actions, horizon, discount_factor,
        reward_func, reward_func_last, T):
    """Derive optimal higher level policy taking stickiness into account.
    Current actions affect all future actions but effect decays with time
    by exponential factor alpha; alpha=0 corresponds to one step stickiness"""

    # probability of cooperating
    X_norm = np.arange(0, 1+d_step, d_step)

    # arrays for optimal values, policy, Q-values
    V_opt = np.full((len(X_norm), len(states), horizon+1), np.nan)
    policy_opt = np.full((len(X_norm), len(states), horizon), 100)
    Q_values = np.full((len(X_norm), len(states)), np.nan, dtype=object)

    for i_state, state in enumerate(states):
        # V_opt for last time-step
        V_opt[:, i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        for i in range(len(X_norm)):
            Q_values[i, i_state] = np.full(
                (len(actions[i_state]), horizon), np.nan)

    # backward induction to derive optimal policy
    for i_timestep in range(horizon-1, -1, -1):
        # total weight is completely determined by time step
        W = (1 - alpha**i_timestep) / (1 - alpha)

        for i_x, x in enumerate(X_norm):

            for i_state, state in enumerate(states):
                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    if len(actions[i_state]) > 1:
                        # probability of [defect, cooperate]
                        p_exec = p * np.array([1 - x, x])
                        p_exec[i_action] += (1 - p)
                    else:  # here only one action
                        p_exec = [1.0]

                    q = 0
                    for a_exec in range(len(actions[i_state])):
                        x_next = update_memory_habit(
                            alpha, x, W, a_exec, len(actions[i_state]))
                        i_x_next = np.argmin(np.abs(X_norm - x_next))

                        q += p_exec[a_exec] * (
                            T[i_state][a_exec] @ reward_func[i_state][a_exec].T
                            + discount_factor * (
                                T[i_state][a_exec] @
                                V_opt[i_x_next, :, i_timestep+1]))

                    Q[i_action] = q

                V_opt[i_x, i_state, i_timestep] = np.max(Q)
                policy_opt[i_x, i_state, i_timestep] = np.argmax(Q)
                Q_values[i_x, i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values


def simulate_behavior_with_habit(
        policy_opt_habit, Q_values_habit, T, alpha, dx, states, actions,
        horizon, plot=False):

    # starting states:
    x = 0.0
    s = 0
    X_norm = np.arange(0, 1+dx, dx)
    actions_executed = []
    state_trajectory = [s]
    x_trajectory = [x]
    q_trajectory = []
    for t in range(horizon):
        i_x = np.argmin(np.abs(X_norm - x))
        action = policy_opt_habit[i_x, s, t]
        actions_executed.append(action)
        x = update_memory_habit(alpha, s, x, (1-alpha**t)/(1-alpha), action,
                                len(actions[s]))
        x_trajectory.append(x)
        # transition to next state
        s = np.random.choice(len(states), p=T[s][action])
        state_trajectory.append(s)
        q_trajectory.append(Q_values_habit[i_x][s][:, t])

    if plot:
        actions_executed = np.array(actions_executed)
        state_trajectory = np.array(state_trajectory)
        time = np.arange(horizon)
        plt.plot(state_trajectory, label='states')
        plt.scatter(time[actions_executed == 1],
                    state_trajectory[:-1][actions_executed == 1],
                    label='action=work',)
        plt.plot(x_trajectory, label='habit strength (work)')
        plt.xticks(np.arange(horizon+1))
        plt.legend()
        plt.show()

    return actions_executed, state_trajectory, x_trajectory, q_trajectory


# %%

# states of markov chain
N_INTERMEDIATE_STATES = 0
# intermediate + initial and finished states (2)
STATES = np.arange(2 + N_INTERMEDIATE_STATES)

# actions available in each state
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
# actions for all states but final:
ACTIONS[:-1] = [['shirk', 'work']
                for i in range(len(STATES)-1)]
ACTIONS[-1] = ['done']  # actions for final state

HORIZON = 5  # deadline
DISCOUNT_FACTOR = 0.9  # common d iscount factor for both
EFFICACY = 0.9  # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 0.0
EFFORT_DO = -0.8
# delayed rewards:
REWARD_COMPLETED = 2.0
COST_NOT_COMPLETED = -0.0

reward_func, reward_func_last = task_structure.rewards_procrastination_common(
    STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED, COST_NOT_COMPLETED)

T = task_structure.transitions_procrastination(STATES, EFFICACY)

# %% policies 1-step habit

# level -1: play with environment
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func, reward_func_last,
    T)

# level 0: take habits into account
p_sticky = 0.3
V_opt_habit_one, policy_opt_habit_one, Q_values_habit_one = (
    plan_with_habits_one_step(
        p_sticky, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
        reward_func_last, T))

# %% vary params

policies = []

discount = DISCOUNT_FACTOR
T = task_structure.transitions_procrastination(STATES, EFFICACY)
# level -1
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, discount, reward_func, reward_func_last,
    T)
policies.append(policy_opt[0])

# level 0: take habits into account
for p_sticky in [0.1, 0.3, 0.6, 0.9]:
    V_opt_habit_one, policy_opt_habit_one, Q_values_habit_one = (
        plan_with_habits_one_step(
            p_sticky, STATES, ACTIONS, HORIZON, discount, reward_func,
            reward_func_last, T))
    policies.append(policy_opt_habit_one[0])


f, ax = plt.subplots(figsize=(5, 4), dpi=100)
sns.heatmap(policies, linewidths=.5,
            cmap=sns.color_palette('husl', 2),
            vmin=0,
            vmax=1)
ax.set_xlabel('timestep')
ax.set_ylabel('p_sticky')
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['SHIRK', 'WORK'])
ax.set_yticklabels([0, 0.1, 0.3, 0.6, 0.9])
plt.show()

# %% policies exponential filtering habit
p = 0.3
alpha = 0.5
dx = 0.01
V_opt_habit, policy_opt_habit, Q_values_habit = plan_with_habits(
    p, alpha, dx, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
    reward_func_last, T)

# simulate behavior with habit
_, _, _, _ = (
    simulate_behavior_with_habit(policy_opt_habit, T, alpha, dx, STATES,
                                 ACTIONS, HORIZON, plot=True))

# %% vary params

for p in [0.0, 0.3, 0.6, 0.9]:
    V_opt_habit, policy_opt_habit, Q_values_habit = plan_with_habits(
        p, alpha, dx, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
        reward_func, reward_func_last, T)

    actions_executed, state_trajectory, x_trajectory, q_trajectory = (
        simulate_behavior_with_habit(policy_opt_habit, T, alpha, dx, STATES,
                                     ACTIONS, HORIZON, plot=False))

    actions_executed = np.array(actions_executed)
    state_trajectory = np.array(state_trajectory)
    time = np.arange(HORIZON)
    plt.plot(state_trajectory, label=f'p={p}')
    plt.scatter(time[actions_executed == 1],
                state_trajectory[:-1][actions_executed == 1])
    plt.xticks(np.arange(HORIZON+1))
    plt.legend()

# %%
p = 0.3
f1, ax1 = plt.subplots()
f2, ax2 = plt.subplots()
for alpha in [0.0, 0.5, 0.9]:
    V_opt_habit, policy_opt_habit, Q_values_habit = plan_with_habits(
        p, alpha, dx, STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
        reward_func, reward_func_last, T)

    actions_executed, state_trajectory, x_trajectory, q_trajectory = (
        simulate_behavior_with_habit(policy_opt_habit, Q_values_habit, T,
                                     alpha, dx, STATES, ACTIONS, HORIZON,
                                     plot=False))
    actions_executed = np.array(actions_executed)
    state_trajectory = np.array(state_trajectory)
    time = np.arange(HORIZON)
    ax1.plot(state_trajectory, label=f'alpha={alpha}')
    ax1.scatter(time[actions_executed == 1],
                state_trajectory[:-1][actions_executed == 1])
    ax1.set_xticks(np.arange(HORIZON+1))
    ax1.legend()

    q_diff = np.array([np.diff(a)[0] if len(a) > 1 else np.nan
                       for a in q_trajectory], dtype=float)
    print(x_trajectory)
    ax2.plot(time, q_diff, label=f'alpha={alpha}')
    ax2.set_xticks(np.arange(HORIZON))
    ax2.set_ylabel('Q(shirk) - Q(work)')
    ax2.legend()
