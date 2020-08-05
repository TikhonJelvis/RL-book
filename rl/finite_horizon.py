from __future__ import annotations

from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import (Dict, Generic, List, Mapping,
                    Optional, Protocol, Sequence, TypeVar)

from rl.distribution import FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, Transition


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


def finite_horizon_markov_process(
        process: FiniteMarkovProcess[S],
        limit: int) -> FiniteMarkovProcess[WithTime[S]]:
    '''Turn a normal FiniteMarkovProcess into one with a finite horizon
    that stops after limit steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    transition_map: Dict[WithTime[S],
                         Optional[FiniteDistribution[WithTime[S]]]] = {}

    for time in range(0, limit):
        def set_time(s: S) -> WithTime[S]:
            return WithTime(state=s, time=time + 1)

        for s in process.transition_map:
            s_next = process.transition(s)
            s_time = WithTime(state=s, time=time)

            transition_map[s_time] = \
                None if s_next is None else s_next.map(set_time)

    return FiniteMarkovProcess(transition_map)


# TODO: Better name...
def unwrap_finite_horizon_markov_process(
        process: FiniteMarkovProcess[S_time],
        limit: int
) -> Sequence[Transition[S_time]]:
    '''Given a finite-horizon process, break the transition between each
    time step (starting with 0) into its own data structure. This
    representation makes it easier to implement backwards
    induction.

    '''
    states: Dict[int, List[S_time]] = defaultdict(list)
    for state in process.states():
        states[state.time] += [state]

    def transition_from(time: int) -> Transition[S_time]:
        return {state: process.transition(state) for state in states[time]}

    return [transition_from(time) for time in range(0, limit)]
