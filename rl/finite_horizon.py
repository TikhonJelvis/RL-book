from __future__ import annotations

from itertools import groupby
import dataclasses
from dataclasses import dataclass
from operator import itemgetter
from typing import (
    Dict, List, Generic, Optional, Sequence, Tuple, TypeVar, Iterator)

from rl.distribution import Constant, FiniteDistribution
from rl.dynamic_programming import V
from rl.markov_process import (
    FiniteMarkovRewardProcess, RewardTransition, StateReward)
from rl.markov_decision_process import (
    ActionMapping, FiniteMarkovDecisionProcess, FinitePolicy,
    StateActionMapping)

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
    transition_map: Dict[WithTime[S], Optional[RewardOutcome]] = {}

    # Non-terminal states
    for time in range(0, limit):

        for s in process.states():
            result: Optional[StateReward[S]] = process.transition_reward(s)
            s_time = WithTime(state=s, time=time)

            transition_map[s_time] = None if result is None else result.map(
                lambda s_r: (WithTime(state=s_r[0], time=time + 1), s_r[1])
            )

    # Terminal states
    for s in process.states():
        transition_map[WithTime(state=s, time=limit)] = None

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

    def without_time(
        arg: Optional[StateReward[WithTime[S]]]
    ) -> Optional[StateReward[S]]:
        return None if arg is None else arg.map(
            lambda s_r: (s_r[0].state, s_r[1])
        )

    return [{s.state: without_time(process.transition_reward(s))
             for s in states} for _, states in groupby(
                 sorted(process.states(), key=time),
                 key=time
             )][:-1]


def evaluate(
    steps: Sequence[RewardTransition[S]],
    gamma: float
) -> Iterator[V[S]]:
    '''Evaluate the given finite Markov reward process using backwards
    induction, given that the process stops after limit time steps.

    '''

    v: List[Dict[S, float]] = []

    for step in reversed(steps):
        v.append({s: res.expectation(
            lambda s_r: s_r[1] + gamma * (v[-1][s_r[0]] if
                                          len(v) > 0 and s_r[0] in v[-1]
                                          else 0.)
            ) for s, res in step.items() if res is not None})

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
    mapping: Dict[WithTime[S], Optional[Dict[A, StateReward[WithTime[S]]]]] =\
        {}

    # Non-terminal states
    for time in range(0, limit):
        for s in process.states():
            s_time = WithTime(state=s, time=time)
            actions_map = process.action_mapping(s)
            if actions_map is None:
                mapping[s_time] = None
            else:
                mapping[s_time] = {a: result.map(
                    lambda s_r: (WithTime(state=s_r[0], time=time + 1), s_r[1])
                ) for a, result in actions_map.items()}

    # Terminal states
    for s in process.states():
        mapping[WithTime(state=s, time=limit)] = None

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

    def without_time(
        arg: Optional[ActionMapping[A, WithTime[S]]]
    ) -> Optional[ActionMapping[A, S]]:
        return None if arg is None else {
            a: sr_distr.map(lambda s_r: (s_r[0].state, s_r[1]))
            for a, sr_distr in arg.items()
        }

    return [{s.state: without_time(process.action_mapping(s))
             for s in states} for _, states in groupby(
                sorted(process.states(), key=time),
                key=time
             )][:-1]


def optimal_vf_and_policy(
    steps: Sequence[StateActionMapping[S, A]],
    gamma: float
) -> Iterator[Tuple[V[S], FinitePolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step

    '''
    v_p: List[Tuple[Dict[S, float], FinitePolicy[S, A]]] = []

    for step in reversed(steps):
        this_v: Dict[S, float] = {}
        this_a: Dict[S, FiniteDistribution[A]] = {}
        for s, actions_map in step.items():
            if actions_map is not None:
                action_values = ((res.expectation(
                    lambda s_r: s_r[1] + gamma * (v_p[-1][0][s_r[0]] if
                                                  len(v_p) > 0 and
                                                  s_r[0] in v_p[-1][0]
                                                  else 0.)
                ), a) for a, res in actions_map.items())
                v_star, a_star = max(action_values, key=itemgetter(0))
                this_v[s] = v_star
                this_a[s] = Constant(a_star)
        v_p.append((this_v, FinitePolicy(this_a)))

    return reversed(v_p)
