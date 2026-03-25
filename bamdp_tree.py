import numpy as np
import task_structure
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BeliefState:
    s: int
    belief_w: tuple[float, ...]
    t: int


@dataclass
class CakeMDPSolution:
    Q: dict
    V: dict
    pi: dict


@dataclass
class CakeMDP:
    states: list
    state_to_action_space: list
    transition_fn: Callable[..., Any]
    reward_fn: Callable[..., Any]
    reward_last_fn: Callable[..., Any]
    w_grid: np.array
    eta: float
    reward_tempt: float
    effort_resist: float
    reward_resist: float
    gamma: float


def linear_training_update(w_grid, belief_w, eta, next_s):

    def shift_right(arr, k):
        result = np.zeros_like(arr)
        result[k:] = arr[:-k]
        return result

    def shift_left(arr, k):
        result = np.zeros_like(arr)
        result[:-k] = arr[k:]
        return result

    shift_size = eta/(w_grid[1]-w_grid[0])
    # double check eta is multiple of grid size
    assert np.isclose(shift_size, round(shift_size))
    shift_size = int(shift_size)

    if next_s == 0:
        # on defection or failure
        next_belief_w = shift_left(belief_w, shift_size)
        # next_belief_w = np.interp(
        #     w_grid, w_grid - eta, belief_w, left=0, right=0)
        mass_zero_bin = belief_w[w_grid - eta < w_grid[0]].sum()
        next_belief_w[0] += mass_zero_bin
        next_belief_w /= next_belief_w.sum()
    else:
        # on success
        next_belief_w = shift_right(belief_w, shift_size)
        # next_belief_w = np.interp(
        #     w_grid, w_grid + eta, belief_w, left=0, right=0)
        mass_one_bin = belief_w[w_grid + eta > w_grid[-1]].sum()
        next_belief_w[-1] += mass_one_bin
        next_belief_w /= next_belief_w.sum()

    return next_belief_w


def exponential_training_update():
    pass


def np_to_tuple(x):
    return tuple(x.tolist())


def rewards_cake(s, a, reward_tempt, effort_resist, reward_resist):
    if a == 0:
        r = reward_tempt
    else:
        r = np.array([reward_tempt, 0.0]) + effort_resist

    if s == 1:
        r += reward_resist
    return r


def rewards_last(reward_resist):
    return np.array([0, reward_resist])


def transitions_cake(h, a, w_grid, eta):

    # w is the willpower
    # update s
    # a=0 is defect, a=1 is cooperate, s=0 is last turn defected, s=1 is last turn cooperated
    belief_w = np.array(h.belief_w)
    w = np.sum(belief_w * w_grid)  # expected w

    if a == 0:
        next_s = [0]
        prob_next_s = [1.0]

        if belief_w[0] == 1.0:
            return next_s, [np_to_tuple(belief_w)], prob_next_s
        if belief_w[-1] == 1.0:
            return next_s, [np_to_tuple(belief_w)], prob_next_s

    else:
        next_s = [0, 1]
        prob_next_s = [1-w, w]  # fail, success
        prob_next_s = np.array(prob_next_s)

        if belief_w[0] == 1.0:
            return next_s, [np_to_tuple(belief_w), np_to_tuple(belief_w)], prob_next_s
        if belief_w[-1] == 1.0:
            return next_s, [np_to_tuple(belief_w), np_to_tuple(belief_w)], prob_next_s

    # posterior update to belief_w
    next_belief_w = []

    if a == 0:
        belief_w_defect = belief_w
    else:
        # failure
        belief_w_fail = belief_w * (1 - w_grid)
        belief_w_fail /= belief_w_fail.sum()

        # success
        belief_w_success = belief_w * w_grid
        belief_w_success /= belief_w_success.sum()

    # learning update to belief_w
    if a == 0:
        next_belief_w = [
            linear_training_update(w_grid, belief_w_defect, eta, next_s=0)]
    else:
        next_belief_w = [
            linear_training_update(w_grid, belief_w_fail, eta, next_s=0),
            linear_training_update(w_grid, belief_w_success, eta, next_s=1)]

    return next_s, [np_to_tuple(x) for x in next_belief_w], prob_next_s


def expand_node_action(h, a, h_set, mdp):

    t = h.t

    transition_fn = mdp.transition_fn
    # reward_fn = mdp.reward_fn
    w_grid = mdp.w_grid
    eta = mdp.eta
    # reward_tempt = mdp.reward_tempt
    # effort_tempt = mdp.effort_tempt
    # reward_resist = mdp.reward_resist

    next_s, next_belief_w, _ = transition_fn(h, a, w_grid, eta)
    # _, _ = reward_fn(reward_tempt, effort_resist, reward_resist)

    belief_states = [
        BeliefState(s=s, belief_w=belief_w, t=t+1)
        for s, belief_w in zip(next_s, next_belief_w)
    ]

    h_set.update(belief_states)


def forward_pass(h0, horizon, mdp):

    belief_state_sets = []
    belief_state_sets.append(set([h0]))

    for d in range(horizon):

        set_depth_d = set()

        for h in belief_state_sets[d]:
            action_space = mdp.state_to_action_space[h.s]
            for a in action_space:
                expand_node_action(h, a, set_depth_d, mdp)

        belief_state_sets.append(set_depth_d)

    return belief_state_sets


def solve_node(h, V, Q, pi, mdp):

    # Return the solved value if desired.
    if h in V:
        return V[h]

    action_space = mdp.state_to_action_space[h.s]
    transition_fn = mdp.transition_fn
    w_grid = mdp.w_grid
    eta = mdp.eta
    gamma = mdp.gamma
    reward_tempt = mdp.reward_tempt
    effort_resist = mdp.effort_resist
    reward_resist = mdp.reward_resist
    reward_fn = mdp.reward_fn

    Q_h = []

    t = h.t

    for a in action_space:

        next_s, next_belief_w, prob_next_s = transition_fn(h, a, w_grid, eta)
        r = reward_fn(h.s, a, reward_tempt, effort_resist, reward_resist)

        belief_states = [
            BeliefState(s=s, belief_w=belief_w, t=t+1)
            for s, belief_w in zip(next_s, next_belief_w)
        ]

        next_V = np.array([V[next_h] for next_h in belief_states])

        # apply bellman equation
        Q_h_a = np.sum(prob_next_s * (r + gamma * next_V))
        Q_h.append(Q_h_a)

    Q_h = np.array(Q_h)
    a_max = np.argmax(Q_h)
    Q_max = np.max(Q_h)

    Q[h] = Q_h
    V[h] = Q_max
    pi[h] = a_max


def backward_pass(list_of_h_sets, mdp):

    V = dict()
    Q = dict()
    pi = dict()

    reward_resist = mdp.reward_resist
    reward_last_fn = mdp.reward_last_fn
    r_last = reward_last_fn(reward_resist)

    for h in list_of_h_sets[-1]:
        V[h] = r_last[h.s]

    for hs in list_of_h_sets[-2::-1]:
        for h in hs:
            solve_node(h, V, Q, pi, mdp)

    return V, Q, pi


def online_simulation(pi, h0, w_true_init, eta, horizon, mdp):

    trajectory = [h0]
    rewards = []
    actions = []
    w_trues = [w_true_init]

    w_true = w_true_init
    h = h0

    def w_true_linear_update(w, success, eta):
        if success:
            w = np.min([w + eta, 1.0])
        else:
            w = np.max([w - eta, 0.0])
        return w

    def true_transition(h, a, w_true, eta, mdp):
        transition_fn = mdp.transition_fn
        reward_fn = mdp.reward_fn
        # sample success/fail using w
        next_s, next_belief_w, prob_next_s = transition_fn(h, a, w_grid, eta)
        r = reward_fn(h.s, a, reward_tempt, effort_resist, reward_resist)
        # sample according to w_true
        next_state = np.random.choice(len(next_s), p=prob_next_s)
        next_h = BeliefState(s=next_s[next_state],
                             belief_w=next_belief_w[next_state], t=t+1)
        # update true w
        w_true = w_true_linear_update(w_true, next_state == 1, eta)
        return next_h, r[next_state], w_true

    for t in range(horizon):
        a = pi[h]
        actions.append(a)
        next_h, r, w_true = true_transition(h, a, w_true, eta, mdp)
        trajectory.append(next_h)
        rewards.append(r)
        w_trues.append(w_true)

    return trajectory, rewards, actions, w_trues


if __name__ == "__main__":
    h = BeliefState(s=0, belief_w=(0.1, 0.3, 0.2, 0.1, 0.2, 0.1), t=0)
    # print(h)
    # print(hash(h))

    h2 = BeliefState(s=0, belief_w=(0.1, 0.3, 0.2, 0.1, 0.2, 0.1), t=0)
    # print(hash(h2))
    # print(hash(h) == hash(h2))

    dw = 0.2
    w_grid = np.arange(0, 1.0 + dw, dw)
    reward_tempt = 0.5
    effort_resist = -0.1
    reward_resist = 0.8

    mdp = CakeMDP(states=[0, 1],
                  state_to_action_space=[[0, 1] for _ in range(2)],
                  transition_fn=transitions_cake,
                  reward_fn=rewards_cake,
                  reward_last_fn=rewards_last,
                  w_grid=w_grid,
                  eta=0.2,
                  reward_tempt=0.5,
                  effort_resist=-0.1,
                  reward_resist=0.8,
                  gamma=1.0)

    # print(mdp.transition_fn(h, 1, mdp.w_grid, mdp.eta))
    # print(mdp.transition_fn(h, 0, mdp.w_grid, mdp.eta))
    # h = BeliefState(s=0, belief_w=(1.0, 0, 0, 0, 0, 0))
    # print(mdp.transition_fn(h, 1, mdp.w_grid, mdp.eta))
    # h = BeliefState(s=0, belief_w=(0.0, 0, 0, 0, 0, 1.0))
    # print(mdp.transition_fn(h, 1, mdp.w_grid, mdp.eta))

    h0 = BeliefState(s=0, belief_w=(0.2, 0.2, 0.2, 0.2, 0.2, 0.2), t=0)
    # h_set = set()
    # expand_node_action(h, 0, h_set, mdp)
    # expand_node_action(h, 1, h_set, mdp)
    # print(h_set)

    list_of_h_sets = forward_pass(h0, 1, mdp)
    # print(list_of_h_sets[0])
    # print(list_of_h_sets[1])
    # print([len(x) for x in list_of_h_sets])

    V, Q, pi = backward_pass(list_of_h_sets, mdp)
    # manipulate last layer's values
    # del V[h]
    # del Q[h]
    # pi = dict()
    # V[list(V.keys())[0]] = 1.0
    # V[list(V.keys())[1]] = 2.0
    # V[list(V.keys())[2]] = 3.0

    # solve_node(h, V, Q, pi, mdp)
    # print(Q)
    trajectory, rewards, actions, w_trues = online_simulation(
        pi, h0, 0.2, 0.2, 1, mdp)
    print(trajectory, rewards, actions, w_trues)
