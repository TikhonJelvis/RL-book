'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Callable, Iterable, Iterator, TypeVar, Tuple, Mapping, Set
from rl.function_approx import FunctionApprox
import rl.markov_process as mp
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.markov_decision_process import TransitionStep
import rl.iterate as iterate
from rl.distribution import Distribution, Categorical
from operator import itemgetter

S = TypeVar('S')


def td_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        γ: float
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
    def step(
            v: FunctionApprox[S],
            transition: mp.TransitionStep[S]
    ) -> FunctionApprox[S]:
        return v.update([(
            transition.state,
            transition.reward + γ * v(transition.next_state)
        )])

    return iterate.accumulate(transitions, step, initial=approx_0)


A = TypeVar('A')
QType = Mapping[S, Mapping[A, float]]


def epsilon_greedy_action(
    q: FunctionApprox[Tuple[S, A]],
    nt_state: S,
    actions: Set[A],
    ϵ: float
) -> A:
    '''
    given a non-terminal state, a Q-Value Function (in the form of a
    FunctionApprox: (state, action) -> Value, and epislon, return
    an action sampled from the probability distribution implied by an
    epsilon-greedy policy that is derived from the Q-Value Function.
    '''
    greedy_action: A = max(
        ((a, q((nt_state, a))) for a in actions),
        key=itemgetter(1)
    )[0]
    return Categorical(
        {a: ϵ / len(actions) +
         (1 - ϵ if a == greedy_action else 0.) for a in actions}
    ).sample()


def glie_sarsa(
    mdp: MarkovDecisionProcess[S, A],
    states: Distribution[S],
    approx_0: FunctionApprox[Tuple[S, A]],
    γ: float,
    ϵ_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    q: FunctionApprox[Tuple[S, A]] = approx_0
    yield q
    num_episodes: int = 0
    while True:
        num_episodes += 1
        ϵ: float = ϵ_as_func_of_episodes(num_episodes)
        state: S = states.sample()
        action: A = epsilon_greedy_action(
            q=q,
            nt_state=state,
            actions=set(mdp.actions(state)),
            ϵ=ϵ
        )
        steps: int = 0
        while not mdp.is_terminal(state) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if mdp.is_terminal(next_state):
                q = q.update([((state, action), reward)])
            else:
                next_action: A = epsilon_greedy_action(
                    q=q,
                    nt_state=next_state,
                    actions=set(mdp.actions(next_state)),
                    ϵ=ϵ
                )
                q = q.update([(
                    (state, action),
                    reward + γ * q((next_state, next_action))
                )])
                action = next_action
            yield q
            steps += 1
            state = next_state


PolicyFromQType = Callable[
    [FunctionApprox[Tuple[S, A]], MarkovDecisionProcess[S, A]],
    Policy[S, A]
]


def q_learning(
    mdp: MarkovDecisionProcess[S, A],
    policy_from_q: PolicyFromQType,
    states: Distribution[S],
    approx_0: FunctionApprox[Tuple[S, A]],
    γ: float,
    max_episode_length: int
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    q: FunctionApprox[Tuple[S, A]] = approx_0
    yield q
    while True:
        state: S = states.sample()
        steps: int = 0
        while not mdp.is_terminal(state) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(q, mdp)
            action: A = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            q = q.update([(
                (state, action),
                reward + γ * (max(q((next_state, a)) for a in
                              mdp.actions(next_state))
                              if not mdp.is_terminal(next_state) else 0)
            )])
            yield q
            steps += 1
            state = next_state


# TODO: More specific name (ie experience replay?)
def td_control(
        transitions: Iterable[TransitionStep[S, A]],
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
    def step(
            q: FunctionApprox[Tuple[S, A]],
            transition: TransitionStep[S, A]
    ) -> FunctionApprox[Tuple[S, A]]:
        next_reward = max(
            q((transition.next_state, a))
            for a in actions(transition.next_state)
        )
        return q.update([
            ((transition.state, transition.action),
             transition.reward + γ * next_reward)
        ])

    return iterate.accumulate(transitions, step, initial=approx_0)
