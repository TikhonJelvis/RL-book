from __future__ import annotations

from itertools import groupby
import dataclasses
from dataclasses import dataclass
from typing import (
    Dict, List, Generic, Optional, Sequence, Tuple, TypeVar)

from rl.distribution import Constant, FiniteDistribution, Categorical
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
    all_states = {s for s in process.states()}
    for time in range(0, limit):
        def set_time(s_r: Tuple[S, float]) -> Tuple[WithTime[S], float]:
            return WithTime(state=s_r[0], time=time + 1), s_r[1]

        for s in all_states:
            result = process.transition_reward(s)
            s_time = WithTime(state=s, time=time)

            transition_map[s_time] = None if result is None else Categorical({
                    (WithTime(state=s1, time=time + 1), r): p
                    for (s1, r), p in result
                })

    # Terminal states
    for s in all_states:
        transition_map[WithTime(state=s, time=limit)] = None

    return FiniteMarkovRewardProcess(transition_map)


def sr_distribution_without_time(
    arg: Optional[StateReward[WithTime[S]]]
) -> Optional[StateReward[S]]:
    return None if arg is None else Categorical(
        {(s.state, r): p for (s, r), p in arg}
    )


# TODO: Better name...
def unwrap_finite_horizon_MRP(
    process: FiniteMarkovRewardProcess[WithTime[S]]
) -> Sequence[RewardTransition[S]]:
    '''Given a finite-horizon process, break the transition between each
    time step (starting with 0) into its own data structure. This
    representation makes it easier to implement backwards
    induction.

    '''
    def f(x: WithTime[S]) -> int:
        return x.time

    return [{s.state: sr_distribution_without_time(
        process.transition_reward(s)) for s in states}
            for _, states in groupby(sorted(process.states(), key=f), key=f)]


def evaluate_state_reward(
    v: V[S],
    result: Optional[StateReward[S]]
) -> float:
    if result is None:
        return 0.0
    else:
        return result.expectation(lambda s_r: v[s_r[0]] + s_r[1])


def evaluate(steps: Sequence[RewardTransition[S]]) -> Sequence[V[S]]:
    '''Evaluate the given finite Markov reward process using backwards
    induction, given that the process stops after limit time steps.

    '''

    length = len(steps) - 1
    v: List[Dict[S, float]] = [{} for _ in range(length)]
    for i in range(length - 1, -1, -1):
        for s, res in steps[i].items():
            v[i][s] = res.expectation(
                lambda x: (v[i + 1][x[0]] if i < length - 1 else 0.) + x[1]
            )

    return v


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
    all_states = [s for s in process.states()]
    for time in range(0, limit):
        for s in all_states:
            s_time = WithTime(state=s, time=time)
            actions_map = process.action_mapping(s)
            if actions_map is None:
                mapping[s_time] = None
            else:
                mapping[s_time] = {a: Categorical({
                    (WithTime(state=s1, time=time + 1), r): p
                    for (s1, r), p in result
                }) for a, result in actions_map.items()}

    # Terminal states
    for s in all_states:
        mapping[WithTime(state=s, time=limit)] = None

    return FiniteMarkovDecisionProcess(mapping)


def action_mapping_without_time(
    arg: Optional[ActionMapping[A, WithTime[S]]]
) -> Optional[ActionMapping[A, S]]:
    return None if arg is None else {a: Categorical(
        {(s.state, r): p for (s, r), p in sr_distr}
    ) for a, sr_distr in arg.items()}


def unwrap_finite_horizon_MDP(
    process: FiniteMarkovDecisionProcess[WithTime[S], A]
) -> Sequence[StateActionMapping[S, A]]:
    '''Unwrap a finite Markov decision process into a sequence of
    transitions between each time step (starting with 0). This
    representation makes it easier to implement backwards induction.

    '''
    def f(x: WithTime[S]) -> int:
        return x.time

    return [{s.state: action_mapping_without_time(process.action_mapping(s))
             for s in states}
            for _, states in groupby(sorted(process.states(), key=f), key=f)]


def optimal_vf_and_policy(
    steps: Sequence[StateActionMapping[S, A]]
) -> Tuple[Sequence[V[S]], Sequence[FinitePolicy[S, A]]]:
    '''Use backwards induction to find the optimal policy for the given
    finite Markov decision process.

    '''
    length = len(steps) - 1
    v: List[Dict[S, float]] = [{} for _ in range(length)]
    p: List[FinitePolicy[S, A]] = [FinitePolicy({}) for _ in range(length)]

    def best_action(actions: ActionMapping[A, S]) -> Tuple[A, float]:
        action_values =\
            ((a, evaluate_state_reward(v, actions[a])) for a in actions)
        return max(action_values, key=lambda a_v: a_v[1])

    for i in range(length - 1, -1, -1):
        this_p: Dict[S, FiniteDistribution[A]] = {}
        for s, actions_map in steps[i].items():
            action_values = ((a, res.expectation(
                lambda x: (v[i + 1][x[0]] if i < length - 1 else 0.) + x[1]
            )) for a, res in actions_map.items())
            a, r = max(action_values, key=lambda x: x[1])
            v[i][s] = r
            this_p[s] = Constant(a)
        p[i] = FinitePolicy(this_p)

    return v, p
