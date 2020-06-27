from abc import ABC, abstractmethod
import graphviz
from typing import Dict, Iterable, Generic, Sequence, Tuple, Mapping, TypeVar
from collections import defaultdict
import numpy as np
from pprint import pprint

from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution)

S = TypeVar('S')

Transition = Mapping[S, FiniteDistribution[S]]

RewardTransition = Mapping[S, FiniteDistribution[Tuple[S, float]]]


class MarkovProcess(ABC, Generic[S]):
    '''A Markov process with states of type S.

    '''
    @abstractmethod
    def transition(self, state: S) -> Distribution[S]:
        '''Given a state of the process, returns a distribution of
        the next states.

        '''

    def simulate(self, start_state: S) -> Iterable[S]:
        '''Run a simulation trace of this Markov process, generating the
        states visited during the trace.

        This yields the start state first, then continues yielding
        subsequent states forever.

        '''

        state: S = start_state
        while True:
            yield state
            state = self.transition(state).sample()


class FiniteMarkovProcess(MarkovProcess[S]):
    '''A Markov Process with a finite state space.

    Having a finite state space lets us use tabular methods to work
    with the process (ie dynamic programming).

    '''

    state_space: Sequence[S]
    transition_map: Transition[S]
    transition_matrix: np.ndarray

    def __init__(self, transition_map: Transition[S]):
        self.state_space = list(transition_map.keys())
        self.transition_map = transition_map
        self.transition_matrix = self.get_transition_matrix()

    def __repr__(self) -> str:
        display = ""

        for s, d in self.transition_map.items():
            display += f"From State {s}:\n"
            for s1, p in d.table():
                display += f"  To State {s1} with Probability {p:.3f}\n"

        return display

    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.state_space)
        mat = np.zeros((sz, sz))

        for i, s1 in enumerate(self.state_space):
            for j, s2 in enumerate(self.state_space):
                mat[i, j] = self.transition(s1).probability(s2)

        return mat

    def transition(self, state: S) -> FiniteDistribution[S]:
        return self.transition_map[state]

    def states(self) -> Iterable[S]:
        return self.transition_map.keys()

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

    def display_stationary_distribution(self):
        pprint({
            s: round(p, 3)
            for s, p in self.get_stationary_distribution().table()
        })

    def generate_image(self) -> graphviz.Digraph:
        d = graphviz.Digraph()

        for s in self.state_space:
            d.node(str(s))

        for s, v in self.transition_map.items():
            for s1, p in v.table():
                d.edge(str(s), str(s1), label=str(p))

        return d


class MarkovRewardProcess(MarkovProcess[S]):
    def transition(self, state: S) -> Distribution[S]:
        '''Transitions the Markov Reward Process, ignoring the generated
        reward (which makes this just a normal Markov Process).

        '''
        def next_state(state=state):
            next_s, _ = self.transition_reward(state).sample()
            return next_s

        return SampledDistribution(next_state)

    @abstractmethod
    def transition_reward(self, state: S) -> Distribution[Tuple[S, float]]:
        '''Given a state, returns a distribution of the next state
        and reward from transitioning between the states.

        '''

    def simulate_reward(self, start_state: S) -> Iterable[Tuple[S, float]]:
        '''Simulate the MRP, yielding the new state and reward for each
        transition.

        The trace starts with the start state and a reward of 0.

        '''

        state: S = start_state
        reward: float = 0.

        while True:
            yield state, reward
            state, reward = self.transition_reward(state).sample()


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S],
                                MarkovRewardProcess[S]):

    transition_reward_map: RewardTransition[S]
    reward_function_vec: np.ndarray

    def __init__(self, transition_reward_map: RewardTransition[S]):

        transition_map: Dict[S, FiniteDistribution[S]] = {}

        for state, trans in transition_reward_map.items():
            probabilities: Dict[S, float] = defaultdict(float)

            for (next_state, _), probability in trans.table():
                probabilities[next_state] += probability

            transition_map[state] = Categorical(list(probabilities.items()))

        super().__init__(transition_map)

        self.transition_reward_map = transition_reward_map

        self.reward_function_vec = np.array([
            sum(probability * reward for (
                _,
                reward), probability in transition_reward_map[state].table())
            for state in self.state_space
        ])

    def __repr__(self) -> str:
        display = ""
        for s, d in self.transition_reward_map.items():
            display += f"From State {s}:\n"
            for (s1, r), p in d.table():
                display +=\
                    f"  To [State {s1} and Reward {r:.3f}]"\
                    + f" with Probability {p:.3f}\n"
        return display

    def transition_reward(self, state: S) ->\
            FiniteDistribution[Tuple[S, float]]:
        return self.transition_reward_map[state]

    def get_value_function_vec(self, gamma) -> np.ndarray:
        return np.linalg.inv(
            np.eye(len(self.state_space)) -
            gamma * self.transition_matrix).dot(self.reward_function_vec)

    def display_reward_function(self):
        pprint({
            self.state_space[i]: round(r, 3)
            for i, r in enumerate(self.reward_function_vec)
        })

    def display_value_function(self, gamma):
        pprint({
            self.state_space[i]: round(v, 3)
            for i, v in enumerate(self.get_value_function_vec(gamma))
        })
