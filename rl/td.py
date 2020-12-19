'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

import itertools
from typing import Iterator, TypeVar

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
from rl.markov_decision_process import MarkovRewardProcess

S = TypeVar('S')


def td_0(
        mrp: MarkovRewardProcess[S],
        states: Distribution[S],
        approx_0: FunctionApprox[S],
        γ: float,
        episode_length: int
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
      episode_length -- number of steps in each episode
    '''
    v = approx_0

    while True:
        episode = itertools.islice(
            iter(mrp.simulate_reward(states)),
            episode_length
        )

        for step in episode:
            v = v.update([(
                step.state,
                step.reward + γ * v.evaluate([step.next_state])[0]
            )])
            yield v
