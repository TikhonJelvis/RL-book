from __future__ import annotations

from itertools import groupby
import dataclasses
from dataclasses import dataclass
from operator import itemgetter
from typing import Dict, List, Generic, Sequence, Tuple, TypeVar, Iterator

from rl.distribution import FiniteDistribution
from rl.dynamic_programming import V, extended_vf
from rl.markov_process import (FiniteMarkovRewardProcess, RewardTransition,
                               StateReward, NonTerminal, Terminal, State)
from rl.markov_decision_process import (
    ActionMapping, FiniteMarkovDecisionProcess, StateActionMapping)
from rl.policy import FiniteDeterministicPolicy

S = TypeVar('S')


@dataclass(frozen=True)
class WithTime(Generic[S]):
    '''A wrapper that augments a state of type S with a time field.

    '''
    state: S
    time: int = 0

    def step_time(self) -> WithTime[S]:
        return dataclasses.replace(self, time=self.time + 1)


RewardOutcome = FiniteDistribution[Tuple[WithTime[S], float]]


# Finite-horizon Markov reward processes
def finite_horizon_MRP(
    process: FiniteMarkovRewardProcess[S],
    limit: int
) -> FiniteMarkovRewardProcess[WithTime[S]]:
    '''Turn a normal FiniteMarkovRewardProcess into one with a finite horizon
    that stops after 'limit' steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    transition_map: Dict[WithTime[S], RewardOutcome] = {}

    # Non-terminal states
    for time in range(limit):

        for s in process.non_terminal_states:
            result: StateReward[S] = process.transition_reward(s)
            s_time = WithTime(state=s.state, time=time)

            transition_map[s_time] = result.map(
                lambda sr: (WithTime(state=sr[0].state, time=time + 1), sr[1])
            )

    return FiniteMarkovRewardProcess(transition_map)


# TODO: Better name...
def unwrap_finite_horizon_MRP(
    process: FiniteMarkovRewardProcess[WithTime[S]]
) -> Sequence[RewardTransition[S]]:
    '''Given a finite-horizon process, break the transition between each
    time step (starting with 0) into its own data structure. This
    representation makes it easier to implement backwards
    induction.

    '''
    def time(x: WithTime[S]) -> int:
        return x.time

    def single_without_time(
        s_r: Tuple[State[WithTime[S]], float]
    ) -> Tuple[State[S], float]:
        if isinstance(s_r[0], NonTerminal):
            ret: Tuple[State[S], float] = (
                NonTerminal(s_r[0].state.state),
                s_r[1]
            )
        else:
            ret = (Terminal(s_r[0].state.state), s_r[1])
        return ret

    def without_time(arg: StateReward[WithTime[S]]) -> StateReward[S]:
        return arg.map(single_without_time)

    return [{NonTerminal(s.state): without_time(
        process.transition_reward(NonTerminal(s))
    ) for s in states} for _, states in groupby(
        sorted(
            (nt.state for nt in process.non_terminal_states),
            key=time
        ),
        key=time
    )]


def evaluate(
    steps: Sequence[RewardTransition[S]],
    gamma: float
) -> Iterator[V[S]]:
    '''Evaluate the given finite Markov reward process using backwards
    induction, given that the process stops after limit time steps.

    '''

    v: List[V[S]] = []

    for step in reversed(steps):
        v.append({s: res.expectation(
            lambda s_r: s_r[1] + gamma * (
                extended_vf(v[-1], s_r[0]) if len(v) > 0 else 0.
            )
        ) for s, res in step.items()})

    return reversed(v)


# Finite-horizon Markov decision processes

A = TypeVar('A')


def finite_horizon_MDP(
    process: FiniteMarkovDecisionProcess[S, A],
    limit: int
) -> FiniteMarkovDecisionProcess[WithTime[S], A]:
    '''Turn a normal FiniteMarkovDecisionProcess into one with a finite
    horizon that stops after 'limit' steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    mapping: Dict[WithTime[S], Dict[A, FiniteDistribution[
        Tuple[WithTime[S], float]]]] = {}

    # Non-terminal states
    for time in range(0, limit):
        for s in process.non_terminal_states:
            s_time = WithTime(state=s.state, time=time)
            mapping[s_time] = {a: result.map(
                lambda sr: (WithTime(state=sr[0].state, time=time + 1), sr[1])
            ) for a, result in process.mapping[s].items()}

    return FiniteMarkovDecisionProcess(mapping)


def unwrap_finite_horizon_MDP(
    process: FiniteMarkovDecisionProcess[WithTime[S], A]
) -> Sequence[StateActionMapping[S, A]]:
    '''Unwrap a finite Markov decision process into a sequence of
    transitions between each time step (starting with 0). This
    representation makes it easier to implement backwards induction.

    '''
    def time(x: WithTime[S]) -> int:
        return x.time

    def single_without_time(
        s_r: Tuple[State[WithTime[S]], float]
    ) -> Tuple[State[S], float]:
        if isinstance(s_r[0], NonTerminal):
            ret: Tuple[State[S], float] = (
                NonTerminal(s_r[0].state.state),
                s_r[1]
            )
        else:
            ret = (Terminal(s_r[0].state.state), s_r[1])
        return ret

    def without_time(arg: ActionMapping[A, WithTime[S]]) -> \
            ActionMapping[A, S]:
        return {a: sr_distr.map(single_without_time)
                for a, sr_distr in arg.items()}

    return [{NonTerminal(s.state): without_time(
        process.mapping[NonTerminal(s)]
    ) for s in states} for _, states in groupby(
        sorted(
            (nt.state for nt in process.non_terminal_states),
            key=time
        ),
        key=time
    )]


def optimal_vf_and_policy(
    steps: Sequence[StateActionMapping[S, A]],
    gamma: float
) -> Iterator[Tuple[V[S], FiniteDeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step

    '''
    v_p: List[Tuple[V[S], FiniteDeterministicPolicy[S, A]]] = []

    for step in reversed(steps):
        this_v: Dict[NonTerminal[S], float] = {}
        this_a: Dict[S, A] = {}
        for s, actions_map in step.items():
            action_values = ((res.expectation(
                lambda s_r: s_r[1] + gamma * (
                    extended_vf(v_p[-1][0], s_r[0]) if len(v_p) > 0 else 0.
                )
            ), a) for a, res in actions_map.items())
            v_star, a_star = max(action_values, key=itemgetter(0))
            this_v[s] = v_star
            this_a[s.state] = a_star
        v_p.append((this_v, FiniteDeterministicPolicy(this_a)))

    return reversed(v_p)
