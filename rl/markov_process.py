from abc import ABC, abstractmethod
import graphviz
from typing import (Dict, Iterable, Generic, Sequence, Tuple, Mapping,
                    Optional, TypeVar)
from collections import defaultdict
import numpy as np
from pprint import pprint

from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution)

S = TypeVar('S')


class MarkovProcess(ABC, Generic[S]):
    '''A Markov process with states of type S.
    '''
    @abstractmethod
    def transition(self, state: S) -> Optional[Distribution[S]]:
        '''Given a state of the process, returns a distribution of
        the next states.  Returning None means we are in a terminal state.
        '''

    def is_terminal(self, state: S) -> bool:
        '''Return whether the given state is a terminal state.

        The default implementation of is_terminal calculates a transition
        from the current state, so it could be worth overloading this
        method if your process has a cheaper way of determing whether
        a state is terminal.
        '''
        return self.transition(state) is None

    def simulate(self, start_state: S) -> Iterable[S]:
        '''Run a simulation trace of this Markov process, generating the
        states visited during the trace.

        This yields the start state first, then continues yielding
        subsequent states forever or until we hit a terminal state.
        '''

        state: S = start_state
        while True:
            yield state
            next_states = self.transition(state)
            if next_states is None:
                break
            else:
                state = next_states.sample()


Transition = Mapping[S, Optional[FiniteDistribution[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    '''A Markov Process with a finite state space.

    Having a finite state space lets us use tabular methods to work
    with the process (ie dynamic programming).

    '''

    non_terminal_states: Sequence[S]
    transition_map: Transition[S]

    def __init__(self, transition_map: Transition[S]):
        self.non_terminal_states = [s for s, v in transition_map.items()
                                    if v is not None]
        self.transition_map = transition_map

    def __repr__(self) -> str:
        display = ""

        for s, d in self.transition_map.items():
            if d is None:
                display += f"{s} is a Terminal State\n"
            else:
                display += f"From State {s}:\n"
                for s1, p in d:
                    display += f"  To State {s1} with Probability {p:.3f}\n"

        return display

    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.non_terminal_states)
        mat = np.zeros((sz, sz))

        for i, s1 in enumerate(self.non_terminal_states):
            for j, s2 in enumerate(self.non_terminal_states):
                mat[i, j] = self.transition(s1).probability(s2)

        return mat

    def transition(self, state: S) -> Optional[FiniteDistribution[S]]:
        return self.transition_map[state]

    def states(self) -> Iterable[S]:
        return self.transition_map.keys()

    def get_stationary_distribution(self) -> FiniteDistribution[S]:
        eig_vals, eig_vecs = np.linalg.eig(self.get_transition_matrix().T)
        index_of_first_unit_eig_val = np.where(
            np.abs(eig_vals - 1) < 1e-8)[0][0]
        eig_vec_of_unit_eig_val = np.real(
            eig_vecs[:, index_of_first_unit_eig_val])
        return Categorical({
            self.non_terminal_states[i]: ev
            for i, ev in enumerate(eig_vec_of_unit_eig_val /
                                   sum(eig_vec_of_unit_eig_val))
        })

    def display_stationary_distribution(self):
        pprint({
            s: round(p, 3)
            for s, p in self.get_stationary_distribution()
        })

    def generate_image(self) -> graphviz.Digraph:
        d = graphviz.Digraph()

        for s in self.transition_map.keys():
            d.node(str(s))

        for s, v in self.transition_map.items():
            if v is not None:
                for s1, p in v:
                    d.edge(str(s), str(s1), label=str(p))

        return d


class MarkovRewardProcess(MarkovProcess[S]):
    def transition(self, state: S) -> Optional[Distribution[S]]:
        '''Transitions the Markov Reward Process, ignoring the generated
        reward (which makes this just a normal Markov Process).

        '''
        distribution = self.transition_reward(state)
        if distribution is None:
            return None
        else:
            def next_state(distribution=distribution):
                next_s, _ = distribution.sample()
                return next_s

            return SampledDistribution(next_state)

    @abstractmethod
    def transition_reward(self, state: S)\
            -> Optional[Distribution[Tuple[S, float]]]:
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
            next_distribution = self.transition_reward(state)
            if next_distribution is None:
                break
            else:
                state, reward = next_distribution.sample()


RewardTransition = Mapping[S, Optional[FiniteDistribution[Tuple[S, float]]]]


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S],
                                MarkovRewardProcess[S]):

    transition_reward_map: RewardTransition[S]
    reward_function_vec: np.ndarray

    def __init__(self, transition_reward_map: RewardTransition[S]):

        transition_map: Dict[S, Optional[FiniteDistribution[S]]] = {}

        for state, trans in transition_reward_map.items():
            if trans is None:
                transition_map[state] = None
            else:
                probabilities: Dict[S, float] = defaultdict(float)
                for (next_state, _), probability in trans:
                    probabilities[next_state] += probability

                transition_map[state] = Categorical(probabilities)

        super().__init__(transition_map)

        self.transition_reward_map = transition_reward_map

        self.reward_function_vec = np.array([
            sum(probability * reward for (_, reward), probability in
                transition_reward_map[state])
            for state in self.non_terminal_states
        ])

    def __repr__(self) -> str:
        display = ""
        for s, d in self.transition_reward_map.items():
            if d is None:
                display += f"{s} is a Terminal State\n"
            else:
                display += f"From State {s}:\n"
                for (s1, r), p in d:
                    display +=\
                        f"  To [State {s1} and Reward {r:.3f}]"\
                        + f" with Probability {p:.3f}\n"
        return display

    def transition_reward(self, state: S) ->\
            Optional[FiniteDistribution[Tuple[S, float]]]:
        return self.transition_reward_map[state]

    def get_value_function_vec(self, gamma: float) -> np.ndarray:
        return np.linalg.inv(
            np.eye(len(self.non_terminal_states)) -
            gamma * self.get_transition_matrix()
        ).dot(self.reward_function_vec)

    def display_reward_function(self):
        pprint({
            self.non_terminal_states[i]: round(r, 3)
            for i, r in enumerate(self.reward_function_vec)
        })

    def display_value_function(self, gamma: float):
        pprint({
            self.non_terminal_states[i]: round(v, 3)
            for i, v in enumerate(self.get_value_function_vec(gamma))
        })
