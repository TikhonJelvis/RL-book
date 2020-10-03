'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from collections import defaultdict
import math
import itertools
from typing import Callable, Dict, Iterator, TypeVar

from rl.distribution import Constant, Distribution
from rl.function_approx import FunctionApprox
import rl.iterate as iterate
from rl.markov_decision_process import MarkovRewardProcess

S = TypeVar('S')


def evaluate_mrp(
        mrp: MarkovRewardProcess[S],
        states: Distribution[S],
        approx_0: FunctionApprox[S],
        γ: float,
        max_steps: int = 1,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mrp -- the Markov Reward Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1), default: 1
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    '''
    v = approx_0

    for trace in mrp.reward_traces(states):
        if γ < 1:
            stop_at = max_steps + round(math.log(tolerance) / math.log(γ))
            episode =\
                itertools.islice(mrp.simulate_reward(states), stop_at)
        else:
            episode = mrp.simulate_reward(states)

        v = v.update(list(iterate.returns(episode, γ, n_states=max_steps)))
        yield v
