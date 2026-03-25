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


def rewards_cake(reward_tempt, effort_resist,
                 reward_resist):

    # get reward matrix
    reward_func = []
    # reward for state 0, for tempt, resist
    # if tempt -> r_tempt and if resist -> e_resist only if resist works
    # and r_tempt if resist doesn't work
    reward_func.append([np.array([reward_tempt, 0]),
                        np.array([reward_tempt + effort_resist,
                                  effort_resist])])
    # reward for state 1, for tempt, resist
    # (if tempt -> r_tempt & if resist -> e_resist if resist works, r_tempt if
    # resist doesnt work) + r_resist from last step in both cases
    reward_func.append([np.array([reward_resist + reward_tempt, 0]),
                        np.array([reward_resist + reward_tempt + effort_resist,
                                  reward_resist + effort_resist])])
    reward_func_last = np.array([0.0, reward_resist])

    return reward_func, reward_func_last


def transitions_cake(p=1.0):

    # p is the probability that resisting will actually lead to resisting
    # (and hence transition to state 1), when it fails, remain in state 0

    # get transition function
    T = []
    # for state 0, if tempt -> state 0, if resist -> state 1 w prob p
    T.append([np.array([1, 0]),
              np.array([1-p, p])])
    # for state 1, if tempt -> state 0, if resist -> state 1 w prob p
    T.append([np.array([1, 0]),
              np.array([1-p, p])])

    return T
