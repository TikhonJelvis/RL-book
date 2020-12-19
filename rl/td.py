'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

import itertools
from typing import Iterable, Iterator, TypeVar

from rl.function_approx import FunctionApprox
import rl.markov_process as mp

S = TypeVar('S')


def evaluate_mrp(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        γ: float,
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using TD(0) using the given sequence of
    transitions.

    Each value this function yields represents the approximated value
    function for the MRP after an additional transition.

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)

    '''
    def step(v, transition):
        return v.update([(transition.state,
                          transition.reward + γ * v(transition.next_state))])

    return itertools.accumulate(transitions, step, initial=approx_0)
