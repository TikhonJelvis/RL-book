from __future__ import annotations

from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import (Dict, Generic, List, Optional,
                    Protocol, Sequence, Tuple, TypeVar)

from rl.distribution import Constant, FiniteDistribution
from rl.dynamic_programming import V
from rl.markov_process import (
    FiniteMarkovRewardProcess, RewardTransition, StateReward)
from rl.markov_decision_process import (
    ActionMapping, FiniteMarkovDecisionProcess, FinitePolicy,
    StateActionMapping)


# States with time
class HasTime(Protocol):
    '''In our current design, finite-horizon processes have to have time
    as part of the state.

    A finite-horizon process also an end time T. The idea is that time
    starts with 0 and increments with every step the process takes;
    states where time is equal to T are the terminal states of the
    finite-horizon process.

    '''
    time: int


S = TypeVar('S')
S_time = TypeVar('S_time', bound=HasTime, covariant=True)


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
        limit: int) -> FiniteMarkovRewardProcess[WithTime[S]]:
    '''Turn a normal FiniteMarkovRewardProcess into one with a finite horizon
    that stops after 'limit' steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    transition_map: Dict[WithTime[S],
                         Optional[RewardOutcome]] = {}

    # Non-terminal states
    for time in range(0, limit):
        def set_time(s_r: Tuple[S, float]) -> Tuple[WithTime[S], float]:
            return (WithTime(state=s_r[0], time=time + 1), s_r[1])

        for s in process.states():
            result = process.transition_reward(s)
            s_time = WithTime(state=s, time=time)

            transition_map[s_time] = \
                None if result is None else result.map(set_time)

    # Terminal states
    for s in process.states():
        transition_map[WithTime(state=s, time=limit)] = None

    return FiniteMarkovRewardProcess(transition_map)


# TODO: Better name...
def unwrap_finite_horizon_MRP(
        process: FiniteMarkovRewardProcess[S_time],
        limit: int
) -> Sequence[RewardTransition[S_time]]:
    '''Given a finite-horizon process, break the transition between each
    time step (starting with 0) into its own data structure. This
    representation makes it easier to implement backwards
    induction.

    '''
    states: Dict[int, List[S_time]] = defaultdict(list)
    for state in process.states():
        states[state.time] += [state]

    def transition_from(time: int) -> RewardTransition[S_time]:
        return {state: process.transition_reward(state)
                for state in states[time]}

    return [transition_from(time) for time in range(0, limit)]


def evaluate_state_reward(v: V[S],
                          result: Optional[StateReward[S]]) -> float:
    if result is None:
        return 0.0
    else:
        return result.expectation(lambda s_r: v[s_r[0]] + s_r[1])


def evaluate(steps: Sequence[RewardTransition[S_time]]) -> V[S_time]:
    '''Evaluate the given finite Markov reward process using backwards
    induction, given that the process stops after limit time steps.

    '''
    v: Dict[S_time, float] = defaultdict(float)

    for step in reversed(steps):
        for s in step:
            v[s] = evaluate_state_reward(v, step[s])

    return v


# Finite-horizon Markov decision processes

A = TypeVar('A')


def finite_horizon_MDP(
        process: FiniteMarkovDecisionProcess[S, A],
        limit: int) -> FiniteMarkovDecisionProcess[WithTime[S], A]:
    '''Turn a normal FiniteMarkovDecisionProcess into one with a finite
    horizon that stops after 'limit' steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    mapping: Dict[WithTime[S], Optional[Dict[A, StateReward[WithTime[S]]]]] =\
        {}

    def set_time(s_r: Tuple[S, float]) -> Tuple[WithTime[S], float]:
        return (WithTime(state=s_r[0], time=time + 1), s_r[1])

    # Non-terminal states
    for time in range(0, limit):
        for s in process.states():
            s_time = WithTime(state=s, time=time)
            actions = process.action_mapping(s)
            if actions is None:
                mapping[s_time] = None
            else:
                mapping[s_time] =\
                    {a: actions[a].map(set_time) for a in actions}

    # Terminal states
    for s in process.states():
        mapping[WithTime(state=s, time=limit)] = None

    return FiniteMarkovDecisionProcess(mapping)


def unwrap_finite_horizon_MDP(
        process: FiniteMarkovDecisionProcess[S_time, A],
        limit: int) -> Sequence[StateActionMapping[S_time, A]]:
    '''Unwrap a finite Markov decision process into a sequence of
    transitions between each time step (starting with 0). This
    representation makes it easier to implement backwards induction.

    '''
    # TODO: Extract this into a group_by function, or even find a
    # library function that does the same thing.
    states: Dict[int, List[S_time]] = defaultdict(list)
    for state in process.states():
        states[state.time] += [state]

    def transition_from(time: int) -> StateActionMapping[S_time, A]:
        return {state: process.action_mapping(state) for state in states[time]}

    return [transition_from(time) for time in range(0, limit)]


def optimal_policy(
        steps: Sequence[StateActionMapping[S_time, A]]
) -> FinitePolicy[S_time, A]:
    '''Use backwards induction to find the optimal policy for the given
    finite Markov decision process.

    '''
    p: Dict[S_time, Optional[FiniteDistribution[A]]] = {}
    v: Dict[S_time, float] = defaultdict(float)

    def best_action(actions: ActionMapping[A, S_time]) -> Tuple[A, float]:
        action_values =\
            ((a, evaluate_state_reward(v, actions[a])) for a in actions)
        return max(action_values, key=lambda a_v: a_v[1])

    for step in reversed(steps):
        for s in step:
            actions = step[s]
            if actions is None:
                p[s] = None
                v[s] = 0.0
            else:
                a, r = best_action(actions)
                p[s] = Constant(a)
                v[s] = r

    return FinitePolicy(p)
