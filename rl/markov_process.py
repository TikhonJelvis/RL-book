from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Generic, Tuple, TypeVar

from rl.distribution import Bernoulli, Distribution

S = TypeVar('S')


class MarkovProcess(ABC, Generic[S]):
    '''A Markov process with states of type S.

    '''

    state: S

    @abstractmethod
    def step(self) -> S:
        pass

    def simulate(self) -> Iterable[S]:
        '''Run a simulation trace of this Markov process, generating the
        states visited during the trace.

        This yields the start state first, then continues yielding
        subsequent states forever.

        '''

        while True:
            yield self.state
            self.state = self.step()


class MarkovRewardProcess(MarkovProcess[S]):
    def step(self) -> S:
        '''Steps the Markov Reward Process, ignoring the generated reward
        (which makes this just a normal Markov Process).

        '''
        self.step_reward()[0]

    @abstractmethod
    def step_reward(self) -> Tuple[S, float]:
        '''Step the process, providing both the next step and the reward for
        that step.
        '''
        pass


class FlipFlop(MarkovProcess[bool]):
    '''A simple example Markov chain with two states, flipping from one to
    the other with probability p and staying at the same state with
    probability 1 - p.

    '''

    state: bool

    def __init__(self, p, start_state=True):
        self.p = p
        self.state = start_state

    def step(self):
        switch_states = Bernoulli(self.p).sample()

        if switch_states:
            return not self.state
        else:
            return self.state
