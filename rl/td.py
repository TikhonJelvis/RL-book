'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

import itertools
from typing import Callable, Iterator, TypeVar

from rl.distribution import Constant, Distribution
from rl.function_approx import FunctionApprox
from rl.markov_decision_process import MarkovRewardProcess

S = TypeVar('S')


def td_0(
        mrp: MarkovRewardProcess[S],
        max_steps: int,
        states: Distribution[S],
        approx_0: FunctionApprox[S],
        α: Callable[[int], float] = lambda _: 1,
        γ: float = 1
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using TD(0), simulating episodes of the given
    number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mrp -- the Markov Reward Process to evaluate
      max_steps -- max steps to take in an episode
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      α -- learning rate, either a constant (0 < α ≤ 1) or a function
           from # of updates to a learning rate, default: 1
      γ -- discount rate (0 < γ ≤ 1), default: 1

    '''
    n = 0
    v = approx_0

    while True:
        start = states.sample()
        episode =\
            itertools.islice(mrp.simulate_reward(Constant(start)), max_steps)

        state = start
        for next_state, reward in episode:
            diff = v.evaluate([next_state]) - v.evaluate([state])
            v = v.update([(state, α(n) * (reward + γ * diff))])
            n += 1

        yield v
