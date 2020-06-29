from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)
import numpy as np
from typing import Callable, Mapping, Iterator, TypeVar, List

A = TypeVar('A')
S = TypeVar('S')
X = TypeVar('X')

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]


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
            yield b
        else:
            a = b

    raise Exception('Iterator too for converge.')


def converged(v1: V[S], v2: V[S]) -> bool:
    return max([abs(v1[s] - v2[s]) for s in v1.keys()]) < 1e-5


def bellman_opt_update(
    v: V[S],
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> V[S]:
    '''Do one update of the value function for a given MDP.'''
    def update_s(s: S) -> float:
        outcomes: List[float] = []
        action_map = mdp.mapping[s]

        for a in mdp.actions(s):
            for (next_s, r), p in action_map[a].table():
                next_state_vf = v[next_s]\
                    if mdp.mapping[next_s] is not None else 0.
                outcomes.append(p * (r + gamma * next_state_vf))

        return max(outcomes)

    return {s: update_s(s) for s in v.keys()}


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> V[S]:
    '''Calculate the value function (V*) of the given MDP by applying the
    value_update function repeatedly until the values start
    converging.

    '''
    def update(v: V[S]) -> V[S]:
        return bellman_opt_update(v, mdp, gamma)

    v_0 = {s: 0.0 for s in mdp.non_terminal_states}
    return list(converge(iterate(update, v_0), done=converged))[-1]


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> V[S]:
    '''Calculate the value function V* for the given Markov Reward
    Process.

    '''
    def update(v: np.ndarray) -> np.ndarray:
        return mrp.reward_function_vec + gamma * mrp.transition_matrix.dot(v)

    v_0 = np.zeros(len(mrp.non_terminal_states))

    vf_array = list(converge(
        iterate(update, v_0),
        done=lambda x, y: max(abs(x-y)) < 1e-5
    ))[-1]
    return {mrp.non_terminal_states: v for i, v in enumerate(vf_array)}


if __name__ == '__main__':

    from rl.distribution import Categorical
    from pprint import pprint

    transition_reward_map = {
        1: Categorical([((1, 7.0), 0.6), ((2, 7.0), 0.3), ((3, 7.0), 0.1)]),
        2: Categorical([((1, 10.0), 0.1), ((2, 10.0), 0.2), ((3, 10.0), 0.7)]),
        3: None
    }
    gamma = 0.9

    fmrp = FiniteMarkovRewardProcess(transition_reward_map)
    fmrp.display_reward_function()
    fmrp.display_value_function(gamma=gamma)
    pprint(evaluate_mrp(fmrp, gamma=gamma))
