'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Iterable, Iterator, Tuple, TypeVar

from rl.function_approx import FunctionApprox
import rl.markov_process as mp
import rl.markov_decision_process as mdp
from rl.returns import returns

S = TypeVar('S')
A = TypeVar('A')


def evaluate_mrp(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: FunctionApprox[S],
        γ: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1), default: 1
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    episodes: Iterable[Iterable[mp.ReturnStep[S]]] =\
        (returns(trace, γ, tolerance) for trace in traces)

    return approx_0.iterate_updates(
        ((step.state, step.return_) for step in episode)
        for episode in episodes
    )


def evaluate_mdp(
        traces: Iterable[Iterable[mdp.TransitionStep[S, A]]],
        approx_0: FunctionApprox[Tuple[S, A]],
        γ: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MDP
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    episodes: Iterable[Iterable[mdp.ReturnStep[S, A]]] =\
        (returns(trace, γ, tolerance) for trace in traces)

    return approx_0.iterate_updates(
        (((step.state, step.action), step.return_) for step in episode)
        for episode in episodes
    )
