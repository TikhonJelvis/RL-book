from abc import ABC, abstractmethod
from typing import Dict, Iterable, Generic, Sequence, Tuple
from rl.gen_utils.type_aliases import S, S_TransType, SR_TransType
from collections import defaultdict
import numpy as np

from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution)


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

    state_space: Sequence[S]
    transition_map: S_TransType
    transition_matrix: np.ndarray

    def __init__(
            self,
            start_state: S,
            state_space: Sequence[S],
            transition_map: S_TransType
    ):
        super().__init__(start_state)
        self.state_space = state_space
        self.transition_map = transition_map
        self.transition_matrix = self.get_transition_matrix()

    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.state_space)
        mat = np.zeros((sz, sz))
        for i, s1 in enumerate(self.state_space):
            for j, s2 in enumerate(self.state_space):
                mat[i, j] = self.transition_map[s1].get(s2, 0.)
        return mat

    def transition(self) -> FiniteDistribution[S]:
        return Categorical(self.transition_map[self.state].items())

    def get_stationary_distribution(self) -> FiniteDistribution[S]:
        eig_vals, eig_vecs = np.linalg.eig(self.transition_matrix.T)
        index_of_first_unit_eig_val = np.where(
            np.abs(eig_vals - 1) < 1e-8)[0][0]
        eig_vec_of_unit_eig_val = np.real(
            eig_vecs[:, index_of_first_unit_eig_val])
        return Categorical([
            (self.state_space[i], ev)
            for i, ev in enumerate(eig_vec_of_unit_eig_val /
                                   sum(eig_vec_of_unit_eig_val))
        ])


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


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S], MarkovRewardProcess[S]):

    transition_reward_map: SR_TransType
    reward_vec: np.ndarray

    def __init__(
            self,
            start_state: S,
            state_space: Sequence[S],
            transition_reward_map: SR_TransType
    ):

        transition_map: Dict[S, Dict[Tuple[S, float], float]] = {}

        for state, trans in self.transition_reward_map.items():
            transition_map[state] = defaultdict(float)
            for (next_state, _), probability in trans.items():
                transition_map[state][next_state] += probability

        super().__init__(start_state, state_space, transition_map)

        self.transition_reward_map = transition_reward_map

        self.reward_vec = np.array(
            [sum(probability * reward for (_, reward), probability in
                 transition_reward_map[state]) for state in state_space]
        )
