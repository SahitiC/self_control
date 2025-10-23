import numpy as np

# %% for procrastination case with immediate rewards and efforts


def rewards_procrastination_common(states, reward_do, effort_do,
                                   reward_completed, cost_not_completed):

    reward_func = []

    # reward for actions (depends on current state and next state)
    # rewards for don't and do
    reward_func.append([np.array([0, 0]),
                        np.array([effort_do, reward_do+effort_do])])
    reward_func.append([np.array([0, 0])])  # rewards for completed

    # reward from final evaluation for the two states
    reward_func_last = np.array([cost_not_completed, reward_completed])

    return reward_func, reward_func_last


def rewards_efforts_procrastination_separate(states, reward_do, effort_do,
                                             reward_completed, cost_completed):

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


def transitions_procrastination(states, efficacy):

    T = np.full(len(states), np.nan, dtype=object)

    # for 2 states:
    T[0] = [np.array([1, 0]),
            np.array([1-efficacy, efficacy])]  # transitions for shirk, work
    T[1] = [np.array([0, 1])]  # transitions for completed

    return T

# %%
