'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Iterable, Iterator, Tuple, TypeVar, Callable

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
import rl.markov_process as mp
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.markov_decision_process import policy_from_q, TransitionStep
from rl.returns import returns

S = TypeVar('S')
A = TypeVar('A')


def mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: FunctionApprox[S],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1), default: 1
      episode_length_tolerance -- stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)

    return approx_0.iterate_updates(
        ((step.state, step.return_) for step in episode)
        for episode in episodes
    )


def glie_mc_control(
    mdp: MarkovDecisionProcess[S, A],
    states: Distribution[S],
    approx_0: FunctionApprox[Tuple[S, A]],
    γ: float,
    ϵ_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mdp -- the Markov Decision Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 ≤ γ ≤ 1)
      ϵ_as_func_of_episodes -- a function from the number of episodes
      to epsilon. epsilon is the fraction of the actions where we explore
      rather than following the optimal policy
      episode_length_tolerance -- stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    q: FunctionApprox[Tuple[S, A]] = approx_0
    p: Policy[S, A] = policy_from_q(q, mdp)
    yield q

    num_episodes: int = 0
    while True:
        trace: Iterable[TransitionStep[S, A]] = \
            mdp.simulate_actions(states, p)
        num_episodes += 1
        q = q.update(
            ((step.state, step.action), step.return_)
            for step in returns(trace, γ, episode_length_tolerance)
        )
        p = policy_from_q(q, mdp, ϵ_as_func_of_episodes(num_episodes))
        yield q
