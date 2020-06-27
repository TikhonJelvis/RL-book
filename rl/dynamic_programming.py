from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)

from typing import Callable, Dict, Iterator, TypeVar

A = TypeVar('A')
S = TypeVar('S')
X = TypeVar('X')

# A representation of a value function for a finite MDP with states of
# type S
V = Dict[S, float]


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

    raise Exception('Iterator too  for converge.')


def converged(v1: V[S], v2: V[S]) -> bool:
    return max([abs(v1[s] - v2[s]) for s in v1.keys()]) < 0.1


def value_update(v: V[S], mdp: FiniteMarkovDecisionProcess[S, A]) -> V[S]:
    '''Do one update of the value function for a given MDP.'''
    def update_s(s: S) -> float:
        outcomes = []

        for a in mdp.actions(s):
            next_states = mdp.mapping[s][a].table()
            for ((next_s, r), p) in next_states:
                outcomes += [p * (r + v[next_s])]

        return max(outcomes)

    return {s: update_s(s) for s in v.keys()}


def value_iteration(mdp: FiniteMarkovDecisionProcess[S, A]) -> V[S]:
    '''Calculate the value function (V*) of the given MDP by applying the
    value_update function repeatedly until the values start
    converging.

    '''
    def update(v: V[S]) -> V[S]:
        return value_update(v, mdp)

    v_0 = {s: 0.0 for s in mdp.states()}
    return list(converge(iterate(update, v_0), done=converged))[-1]


def evaluate_mrp(mrp: FiniteMarkovRewardProcess[S]) -> V[S]:
    '''Calculate the value function V* for the given Markov Reward
    Process.

    '''
    def update(v: V[S]) -> V[S]:
        def update_s(s: S) -> float:
            next_states = mrp.transition_reward(s).table()
            return sum(p * (r + v[next_s]) for ((next_s, r), p) in next_states)

        return {s: update_s(s) for s in v.keys()}

    v_0 = {s: 0.0 for s in mrp.states()}
    return list(converge(iterate(update, v_0), done=converged))[-1]
