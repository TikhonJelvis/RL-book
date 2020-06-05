from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Generic, Tuple, TypeVar

from rl.distribution import Bernoulli, Distribution, SampledDistribution

S = TypeVar('S')


class MarkovProcess(ABC, Generic[S]):
    '''A Markov process with states of type S.

    '''

    state: S

    def __init__(self, start_state: S):
        self.state = start_state

    @abstractmethod
    def simulate_transition(self) -> S:
        pass

    def transition(self) -> Distribution[S]:
        '''Given the current state of the process, returns a distribution of the next states.

        '''
        return SampledDistribution(self.simulate_transition)

    def simulate(self) -> Iterable[S]:
        '''Run a simulation trace of this Markov process, generating the
        states visited during the trace.

        This yields the start state first, then continues yielding
        subsequent states forever.

        '''

        while True:
            yield self.state
            self.state = self.transition().sample()


class MarkovRewardProcess(MarkovProcess[S]):
    def simulate_transition(self) -> S:
        '''Transitions the Markov Reward Process, ignoring the generated reward
        (which makes this just a normal Markov Process).

        '''
        return self.transition_reward()[0]

    @abstractmethod
    def transition_reward(self) -> Tuple[S, float]:
        '''Transition the process, providing both the next transition and the reward for
        that transition.
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

    def simulate_transition(self) -> bool:
        switch_states = Bernoulli(self.p).sample()

        if switch_states:
            return not self.state
        else:
            return self.state
