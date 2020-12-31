'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

import itertools
from typing import Callable, Iterable, Iterator, TypeVar, Tuple

from rl.function_approx import FunctionApprox
import rl.markov_process as mp
import rl.markov_decision_process as mdp

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


A = TypeVar('A')


# TODO: More specific name (ie experience replay?)
def evaluate_mdp(
        transitions: Iterable[mdp.TransitionStep[S, A]],
        actions: Callable[[S], Iterable[A]],
        approx_0: FunctionApprox[Tuple[S, A]],
        γ: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    '''Return policies that try to maximize the reward based on the given
    set of experiences.

    Arguments:
      transitions -- a sequence of state, action, reward, state (S, A, R, S')
      actions -- a function returning the possible actions for a given state
      approx_0 -- initial approximation of q function
      γ -- discount rate (0 < γ ≤ 1)

    Returns:
      an itertor of approximations of the q function based on the
      transitions given as input

    '''
    def step(q, transition):
        next_reward = max(
            q((transition.next_state, a))
            for a in actions(transition.next_state)
        )
        return q.update([
            ((transition.state, transition.action),
             transition.reward + γ * next_reward)
        ])

    return itertools.accumulate(transitions, step, initial=approx_0)
