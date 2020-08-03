from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Generic, Optional, Protocol, TypeVar

from rl.distribution import FiniteDistribution
from rl.markov_process import FiniteMarkovProcess


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


class FiniteHorizon(Protocol[S_time]):
    '''A finite-horizon process has two properties: a state space that
    keeps track of time (HasTime) and a time limit that determines
    when the process ends. The time in the state is an int that starts
    and 0 and increments with every step; states where the time is
    equal to the time limit are terminal states.

    '''
    time_limit: int


# Types for applying a finite horizon to existing processes:

@dataclass
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
