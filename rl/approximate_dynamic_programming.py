from typing import Iterator, Mapping, Tuple, TypeVar

from rl.function_approx import FunctionApprox
from rl.iterate import converged, iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess)

S = TypeVar('S')

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]


def evaluate_finite_mrp(
        mrp: FiniteMarkovRewardProcess[S],
        γ: float,
        approx_0: FunctionApprox[S]) -> Iterator[FunctionApprox[S]]:
    '''Iteratively calculate the value function for the give Markov reward
    process, using the given FunctionApprox to approximate the value
    function at each step.

    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        vs = v.evaluate(mrp.non_terminal_states)
        updated = mrp.reward_function_vec + γ *\
            mrp.get_transition_matrix().dot(vs)
        return v.update(zip(mrp.states(), updated))

    return iterate(update, approx_0)


def evaluate_mrp(
        mrp: MarkovRewardProcess[S],
        γ: float,
        approx_0: FunctionApprox[S],
        n: int) -> Iterator[FunctionApprox[S]]:

    '''Iteratively calculate the value function for the give Markov reward
    process, using the given FunctionApprox to approximate the value
    function at each step for a random sample of the process's states.

    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        states = mrp.sample_states().sample_n(n)

        def return_(s_r: Tuple[S, float]) -> float:
            s, r = s_r
            return r + γ * v.evaluate([s]).item()

        return v.update([(s, mrp.transition_reward(s).expectation(return_))
                         for s in states])

    return iterate(update, approx_0)
