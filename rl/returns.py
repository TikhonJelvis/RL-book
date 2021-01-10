import itertools
import math
from typing import Iterable, Iterator, TypeVar, overload

import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.iterate as iterate


S = TypeVar('S')
A = TypeVar('A')


@overload
def returns(
        trace: Iterable[mp.TransitionStep[S]],
        γ: float,
        tolerance: float
) -> Iterator[mp.ReturnStep[S]]:
    ...


@overload
def returns(
        trace: Iterable[mdp.TransitionStep[S, A]],
        γ: float,
        tolerance: float
) -> Iterator[mdp.ReturnStep[S, A]]:
    ...


def returns(trace, γ, tolerance):
    '''Given an iterator of states and rewards, calculate the return of
    the first N states.

    Arguments:
    rewards -- instantaneous rewards
    γ -- the discount factor (0 < γ ≤ 1)
    tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    '''
    trace = iter(trace)

    max_steps = round(math.log(tolerance) / math.log(γ)) if γ < 1 else None
    if max_steps is not None:
        trace = itertools.islice(trace, max_steps * 2)

    *transitions, last_transition = list(trace)

    return_steps = iterate.accumulate(
        reversed(transitions),
        func=lambda next, curr: curr.add_return(γ, next.return_),
        initial=last_transition.add_return(γ, 0)
    )
    return_steps = reversed(list(return_steps))

    if max_steps is not None:
        return_steps = itertools.islice(return_steps, max_steps)

    return return_steps
