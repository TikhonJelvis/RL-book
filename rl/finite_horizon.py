from __future__ import annotations

from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import (Dict, Generic, List, Optional,
                    Protocol, Sequence, Tuple, TypeVar)

from rl.distribution import FiniteDistribution
from rl.dynamic_programming import V
from rl.markov_process import FiniteMarkovRewardProcess, RewardTransition


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


def finite_horizon_markov_process(
        process: FiniteMarkovRewardProcess[S],
        limit: int) -> FiniteMarkovRewardProcess[WithTime[S]]:
    '''Turn a normal FiniteMarkovRewardProcess into one with a finite horizon
    that stops after limit steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    transition_map: Dict[WithTime[S],
                         Optional[RewardOutcome]] = {}

    for time in range(0, limit):
        def set_time(s_r: Tuple[S, float]) -> Tuple[WithTime[S], float]:
            return (WithTime(state=s_r[0], time=time + 1), s_r[1])

        for s in process.transition_reward_map:
            outcome = process.transition_reward(s)
            s_time = WithTime(state=s, time=time)

            transition_map[s_time] = \
                None if outcome is None else outcome.map(set_time)

    return FiniteMarkovRewardProcess(transition_map)


# TODO: Better name...
def unwrap_finite_horizon_markov_process(
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


def evaluate(steps: Sequence[RewardTransition[S_time]]) -> V[S_time]:
    '''Evaluate the given finite Markov reward process using backwards
    induction, given that the process stops after limit time steps.

    '''
    v: Dict[S_time, float] = defaultdict(float)

    for step in reversed(steps):
        for s in step:
            outcome = step[s]
            if outcome is None:
                v[s] = 0.0  # 0 reward in terminal state
            else:
                for (next_s, r), p in outcome:
                    v[s] += p * (v[next_s] + r)

    return v
