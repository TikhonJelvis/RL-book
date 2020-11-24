'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Iterator, Optional, Tuple, TypeVar

from rl.distribution import Bernoulli, Choose, Constant, Distribution
from rl.function_approx import FunctionApprox
import rl.iterate as iterate
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import (MarkovRewardProcess,
                                        MarkovDecisionProcess,
                                        Policy)

S = TypeVar('S')
A = TypeVar('A')


def evaluate_mrp(
        mrp: MarkovRewardProcess[S],
        states: Distribution[S],
        approx_0: FunctionApprox[S],
        γ: float,
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

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    v = approx_0

    for trace in mrp.reward_traces(states):
        v = v.update(list(iterate.returns(trace, γ, tolerance)))
        yield v


def evaluate_mdp(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        γ: float,
        ϵ: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mrp -- the Markov Reward Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1), default: 1
      ϵ -- the fraction of the actions where we explore rather
      than following the optimal policy
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    q = approx_0
    p = markov_decision_process.policy_from_q(q, mdp)

    while True:
        trace = mdp.simulate_actions(states, p)
        q = q.update(
            [((trace.state, trace.action), trace.reward)
             for trace in markov_decision_process.returns(trace, γ, tolerance)]
        )
        p = markov_decision_process.policy_from_q(q, mdp, ϵ)
        yield q
