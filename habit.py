import numpy as np
import matplotlib.pyplot as plt
import mdp_algms
import diff_discount_factors

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
ACTIONS[-1] = ['shirk']  # actions for final state

HORIZON = 3  # deadline
DISCOUNT_FACTOR = 0.9  # common d iscount factor for both
EFFICACY = 0.7  # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 0.0
EFFORT_DO = -1.0
# no delayed rewards:
REWARD_COMPLETED = 2.0
COST_NOT_COMPLETED = -0.0


reward_func, reward_func_last = get_reward_functions(
    STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED, COST_NOT_COMPLETED)

T = diff_discount_factors.get_transition_prob(STATES, EFFICACY)

# %% base policy
# play with environment

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func, reward_func_last,
    T)

# %%
# level 0: take habits into account
