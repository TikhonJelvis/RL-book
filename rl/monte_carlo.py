'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

import math
import itertools
from typing import Iterator, Optional, Tuple, TypeVar

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
import rl.iterate as iterate
from rl.markov_decision_process import (returns,
                                        MarkovRewardProcess,
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
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    q = approx_0
    p = policy_from_q(q, mdp)

    for trace in mdp.action_traces(states, p):
        q = q.update(
            [((trace.state, trace.action), trace.reward)
             for trace in returns(trace, γ, tolerance)]
        )
        p = policy_from_q(q, mdp)
        yield q


def policy_from_q(
        q: FunctionApprox[Tuple[S, A]],
        mdp: MarkovDecisionProcess[S, A]
) -> Policy[S, A]:
    '''Return a policy that chooses the action that maximizes the reward
    for each state in the given Q function.

    '''
    class QPolicy(Policy[S, A]):
        def act(self, s: S) -> Optional[Distribution[A]]:
            return max(q.evaluate([(s, a) for a in mdp.actions(s)]))

    return QPolicy()
