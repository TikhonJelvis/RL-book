'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

import itertools
import math
from typing import Iterator, TypeVar

from rl.distribution import Constant, Distribution
from rl.function_approx import FunctionApprox
from rl.markov_decision_process import MarkovRewardProcess

S = TypeVar('S')


def td_0(
        mrp: MarkovRewardProcess[S],
        states: Distribution[S],
        approx_0: FunctionApprox[S],
        γ: float,
        tolerance: float
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using TD(0), simulating episodes of the given
    number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mrp -- the Markov Reward Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      α -- learning rate, either a constant (0 < α ≤ 1) or a function
           from # of updates to a learning rate, default: 1
      γ -- discount rate (0 < γ ≤ 1)
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance
    '''
    v = approx_0

    max_steps = None
    if γ < 1:
        max_steps = round(math.log(tolerance) / math.log(γ))

    while True:
        start = states.sample()
        episode = mrp.simulate_reward(Constant(start))
        if max_steps is not None:
            episode = itertools.islice(episode, max_steps)

        state = start
        updates = []
        for next_state, reward in episode:
            diff = v.evaluate([next_state]) - v.evaluate([state])
            updates += [(state, reward + γ * diff)]
            state = next_state

        v = v.update(updates)
        yield v
