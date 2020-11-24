'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Iterator, TypeVar

from rl.distribution import Constant, Distribution
from rl.function_approx import FunctionApprox
import rl.iterate as iterate
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

    while True:
        start = states.sample()
        episode = iterate.discount_tolerance(
            iter(mrp.simulate_reward(Constant(start))),
            γ,
            tolerance
        )

        state = start
        for next_state, reward in episode:
            diff = v.evaluate([next_state]) - v.evaluate([state])
            v = v.update([(state, reward + γ * diff)])
            state = next_state
            yield v
