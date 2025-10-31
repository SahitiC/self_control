import numpy as np

# %% for procrastination case with immediate and delayed rewards and efforts


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

# %% cake-eating with immediate and delayed rewards


def rewards_cake(states, reward_tempt, effort_resist,
                 reward_resist):

    # get reward matrix
    reward_func = []
    # reward for state 0, for tempt, resist
    # if tempt -> r_tempt and if resist -> e_resist
    # (doesn't matter what the next state is ):
    reward_func.append([np.full(len(states), reward_tempt),
                        np.full(len(states), effort_resist)])
    # reward for state 1, for tempt, resist
    # if tempt -> r_tempt & if resist -> e_resist + r_resist from last timestep
    # in both cases (doesn't matter what the next state is  )
    reward_func.append([np.full(len(states), reward_resist + reward_tempt),
                        np.full(len(states), reward_resist + effort_resist)])
    reward_func_last = np.array([0.0, reward_resist])

    return reward_func, reward_func_last


def transitions_cake():

    # get transition function
    T = []
    # transition for state 0, if tempt -> state 0, if resist -> state 1
    T.append([np.array([1, 0]),
              np.array([0, 1])])
    # transition for state 1, if tempt -> state 0, if resist -> state 1
    T.append([np.array([1, 0]),
              np.array([0, 1])])

    return T
