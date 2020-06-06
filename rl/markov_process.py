from abc import ABC, abstractmethod
from typing import Dict, Iterable, Generic, List, Tuple, TypeVar

from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution)

S = TypeVar('S')


class MarkovProcess(ABC, Generic[S]):
    '''A Markov process with states of type S.

    '''

    state: S

    def __init__(self, start_state: S):
        self.state = start_state

    @abstractmethod
    def transition(self) -> Distribution[S]:
        '''Given the current state of the process, returns a distribution of
        the next states.

        '''
        pass

    def simulate(self) -> Iterable[S]:
        '''Run a simulation trace of this Markov process, generating the
        states visited during the trace.

        This yields the start state first, then continues yielding
        subsequent states forever.

        '''

        while True:
            yield self.state
            self.state = self.transition().sample()


class FiniteMarkovProcess(MarkovProcess[S]):
    '''A Markov Process with a finite state space.

    Having a finite state space lets us use tabular methods to work
    with the process (ie dynamic programming).

    '''

    state_space: List[S]

    transition_matrix: Dict[S, Dict[S, float]]

    def __init__(self, state_space: List[S],
                 transition_matrix: Dict[S, Dict[S, float]]):
        self.state_space = state_space

        self.transition_matrix = transition_matrix

    def transition(self) -> FiniteDistribution[S]:
        return Categorical(self.transition_matrix[self.state].items())


class MarkovRewardProcess(MarkovProcess[S]):
    def transition(self) -> Distribution[S]:
        '''Transitions the Markov Reward Process, ignoring the generated
        reward (which makes this just a normal Markov Process).

        '''
        def next_state():
            state, _ = self.transition_reward().sample()
            return state

        return SampledDistribution(next_state)

    @abstractmethod
    def transition_reward(self) -> Distribution[Tuple[S, float]]:
        '''Given the current state, returns a distribution of the next state
        and reward from transitioning between the states.

        '''
        pass

    def simulate_reward(self) -> Iterable[Tuple[S, float]]:
        '''Simulate the MRP, yielding the new state and reward for each
        transition.

        The trace starts with the start state and a reward of 0.

        '''
        yield self.state, 0

        while True:
            next_state, reward = self.transition_reward().sample()
            self.state = next_state
            yield next_state, reward


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S],
                                MarkovRewardProcess[S]):
    transition_reward_matrix: Dict[S, Dict[Tuple[S, float], float]]

    def __init__(self, state_space: List[S],
                 transition_reward_matrix: Dict[S, Dict[Tuple[S, float],
                                                        float]]):
        self.state_space = state_space

        self.transition_reward_matrix = transition_reward_matrix

        self.transition_matrix = {}
        for state, item in self.transition_reward_matrix.items():
            self.transition_matrix[state] = {}

            for (next_state,
                 _), probability in self.transition_reward_matrix[state].items(
                 ):
                self.transition_matrix[state][next_state] = probability
