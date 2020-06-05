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
    def simulate_transition(self) -> S:
        pass

    def transition(self) -> Distribution[S]:
        '''Given the current state of the process, returns a distribution of
        the next states.

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

    def simulate_transition(self) -> S:
        return self.transition().sample()

    def transition(self) -> FiniteDistribution[S]:
        return Categorical(self.transition_matrix[self.state].items())


class MarkovRewardProcess(MarkovProcess[S]):
    def simulate_transition(self) -> S:
        '''Transitions the Markov Reward Process, ignoring the generated
        reward (which makes this just a normal Markov Process).

        '''
        return self.simulate_transition_reward()[0]

    @abstractmethod
    def simulate_transition_reward(self) -> Tuple[S, float]:
        '''Transition the process, providing both the next transition and the
        reward for that transition.

        '''
        pass

    def transition_reward(self) -> Distribution[Tuple[S, float]]:
        return SampledDistribution(self.simulate_transition_reward)

    # TODO: This starts the simulation *after* the first state, while
    # simulate() starts with the start state
    def simulate_reward(self) -> Iterable[Tuple[S, float]]:
        while True:
            next_state, reward = self.transition_reward().sample()
            self.state = next_state
            yield next_state, reward


class FiniteMarkovRewardProcess(MarkovRewardProcess[S],
                                FiniteMarkovProcess[S]):
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
