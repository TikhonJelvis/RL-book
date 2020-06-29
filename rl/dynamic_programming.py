from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)
import numpy as np
from typing import Callable, Mapping, Iterator, TypeVar, List, Tuple, Dict
from rl.markov_decision_process import FinitePolicy
from rl.distribution import FiniteDistribution, Categorical
import operator

A = TypeVar('A')
S = TypeVar('S')
X = TypeVar('X')

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]

DEFAULT_TOLERANCE = 1e-5


# It would be more efficient if you iterated in place instead of
# returning a copy of the value each time, but the functional version
# of the code is a lot cleaner and easier to work with.
def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    '''Find the fixed point of a function f by applying it to its own
    result, yielding each intermediate value.

    That is, for a function f, iterate(f, x) will give us a generator
    producing:

    x, f(x), f(f(x)), f(f(f(x)))...

    '''
    state = start

    while True:
        yield state
        state = step(state)


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    '''Read from a stream of values until two consecutive values satisfy
    the given done function.

    Will error out if the stream runs out before the predicate is
    satisfied (including streams with 0 or 1 values) and will loop
    forever if the stream doesn't end *or* converge.

    '''
    a = next(values)

    for b in values:
        if done(a, b):
            break
        else:
            a = b
            yield b


def condition_evaluate_mrp(a1: np.ndarray, a2: np.ndarray) -> bool:
    return max(abs(a1 - a2)) < DEFAULT_TOLERANCE


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> V[S]:
    '''Calculate the value function for the given Markov Reward
    Process.
    '''
    def update(v: np.ndarray) -> np.ndarray:
        return mrp.reward_function_vec + gamma * mrp.transition_matrix.dot(v)

    v_0: np.ndarray = np.zeros(len(mrp.non_terminal_states))

    vf_array = list(converge(
        iterate(update, v_0),
        done=condition_evaluate_mrp
    ))[-1]
    return {mrp.non_terminal_states[i]: v for i, v in enumerate(vf_array)}


def greedy_policy_from_vf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: V[S],
    gamma: float
) -> FinitePolicy[S, A]:
    greedy_policy_dict: Dict[S, FiniteDistribution[A]] = {}

    for s in mdp.non_terminal_states:

        q_values: List[Tuple[A, float]] = []
        action_map = mdp.mapping[s]

        for a in mdp.actions(s):
            q_val: float = 0.
            for (next_s, r), p in action_map[a].table():
                next_state_vf = vf[next_s]\
                    if mdp.mapping[next_s] is not None else 0.
                q_val += p * (r + gamma * next_state_vf)
            q_values.append((a, q_val))

        greedy_policy_dict[s] = Categorical([(
            max(q_values, key=operator.itemgetter(1))[0],
            1.
        )])

    return FinitePolicy(greedy_policy_dict)


def condition_policy_iteration(
    x1: Tuple[V[S], FinitePolicy[S, A]],
    x2: Tuple[V[S], FinitePolicy[S, A]]
) -> bool:
    return max(
        abs(x1[0][s] - x2[0][s]) for s in x1[0].keys()
    ) < DEFAULT_TOLERANCE


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False
) -> Tuple[V[S], FinitePolicy[S, A]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]])\
            -> Tuple[V[S], FinitePolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: V[S] = {mrp.non_terminal_states[i]: v for i, v in
                           enumerate(mrp.get_value_function_vec())}\
            if matrix_method_for_mrp_eval else evaluate_mrp(mrp, gamma)

        improved_pi: FinitePolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )

        return policy_vf, improved_pi

    v_0 = {s: 0.0 for s in mdp.non_terminal_states}
    pi_0 = FinitePolicy({
        s: Categorical([(a, 1. / len(mdp.actions(s))) for a in mdp.actions(s)])
        for s in mdp.non_terminal_states
    })
    vf_pi_0 = (v_0, pi_0)
    return list(converge(
        iterate(update, vf_pi_0),
        done=condition_policy_iteration
    ))[-1]


def bellman_opt_update(
    v: V[S],
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> V[S]:
    '''Do one update of the value function for a given MDP.'''
    def update_s(s: S) -> float:
        q_values: List[float] = []
        action_map = mdp.mapping[s]

        for a in mdp.actions(s):
            q_val: float = 0.
            for (next_s, r), p in action_map[a].table():
                next_state_vf = v[next_s]\
                    if mdp.mapping[next_s] is not None else 0.
                q_val += p * (r + gamma * next_state_vf)
            q_values.append(q_val)

        return max(q_values)

    return {s: update_s(s) for s in v.keys()}


def condition_value_iteration(v1: V[S], v2: V[S]) -> bool:
    return max([abs(v1[s] - v2[s]) for s in v1.keys()]) < DEFAULT_TOLERANCE


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Tuple[V[S], FinitePolicy[S, A]]:
    '''Calculate the value function (V*) of the given MDP by applying the
    value_update function repeatedly until the values start
    converging.
    '''
    def update(v: V[S]) -> V[S]:
        return bellman_opt_update(v, mdp, gamma)

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    opt_vf: V[S] = list(converge(
        iterate(update, v_0),
        done=condition_value_iteration
    ))[-1]

    opt_policy: FinitePolicy[S, A] = greedy_policy_from_vf(
        mdp,
        opt_vf,
        gamma
    )

    return opt_vf, opt_policy


if __name__ == '__main__':

    from pprint import pprint

    transition_reward_map = {
        1: Categorical([((1, 7.0), 0.6), ((2, 5.0), 0.3), ((3, 2.0), 0.1)]),
        2: Categorical([((1, -2.0), 0.1), ((2, 4.0), 0.2), ((3, 0.0), 0.7)]),
        3: Categorical([((1, 3.0), 0.2), ((2, 8.0), 0.6), ((3, 4.0), 0.2)])
    }
    gamma = 0.9

    fmrp = FiniteMarkovRewardProcess(transition_reward_map)
    fmrp.display_stationary_distribution()
    fmrp.display_reward_function()
    fmrp.display_value_function(gamma=gamma)
    pprint(evaluate_mrp(fmrp, gamma=gamma))
