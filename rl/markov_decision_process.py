from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (DefaultDict, Dict, Iterable, Generic, Mapping,
                    Tuple, Sequence, TypeVar, Set)

from rl.distribution import (Categorical, Distribution, FiniteDistribution)

from rl.markov_process import (
    FiniteMarkovRewardProcess, MarkovRewardProcess, StateReward, State,
    NonTerminal, Terminal)
from rl.policy import FinitePolicy, Policy

A = TypeVar('A')
S = TypeVar('S')


@dataclass(frozen=True)
class TransitionStep(Generic[S, A]):
    '''A single step in the simulation of an MDP, containing:

    state -- the state we start from
    action -- the action we took at that state
    next_state -- the state we ended up in after the action
    reward -- the instantaneous reward we got for this transition
    '''
    state: NonTerminal[S]
    action: A
    next_state: State[S]
    reward: float

    def add_return(self, γ: float, return_: float) -> ReturnStep[S, A]:
        '''Given a γ and the return from 'next_state', this annotates the
        transition with a return for 'state'.

        '''
        return ReturnStep(
            self.state,
            self.action,
            self.next_state,
            self.reward,
            return_=self.reward + γ * return_
        )


@dataclass(frozen=True)
class ReturnStep(TransitionStep[S, A]):
    '''A Transition that also contains the total *return* for its starting
    state.

    '''
    return_: float


class MarkovDecisionProcess(ABC, Generic[S, A]):
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        mdp = self

        class RewardProcess(MarkovRewardProcess[S]):
            def transition_reward(
                self,
                state: NonTerminal[S]
            ) -> Distribution[Tuple[State[S], float]]:
                actions: Distribution[A] = policy.act(state)
                return actions.apply(lambda a: mdp.step(state, a))

        return RewardProcess()

    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        pass

    @abstractmethod
    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> Distribution[Tuple[State[S], float]]:
        pass

    def simulate_actions(
            self,
            start_states: Distribution[NonTerminal[S]],
            policy: Policy[S, A]
    ) -> Iterable[TransitionStep[S, A]]:
        '''Simulate this MDP with the given policy, yielding the
        sequence of (states, action, next state, reward) 4-tuples
        encountered in the simulation trace.

        '''
        state: State[S] = start_states.sample()

        while isinstance(state, NonTerminal):
            action_distribution = policy.act(state)

            action = action_distribution.sample()
            next_distribution = self.step(state, action)

            next_state, reward = next_distribution.sample()
            yield TransitionStep(state, action, next_state, reward)
            state = next_state

    def action_traces(
            self,
            start_states: Distribution[NonTerminal[S]],
            policy: Policy[S, A]
    ) -> Iterable[Iterable[TransitionStep[S, A]]]:
        '''Yield an infinite number of traces as returned by
        simulate_actions.

        '''
        while True:
            yield self.simulate_actions(start_states, policy)


ActionMapping = Mapping[A, StateReward[S]]
StateActionMapping = Mapping[NonTerminal[S], ActionMapping[A, S]]


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    '''A Markov Decision Process with finite state and action spaces.

    '''

    mapping: StateActionMapping[S, A]
    non_terminal_states: Sequence[NonTerminal[S]]

    def __init__(
        self,
        mapping: Mapping[S, Mapping[A, FiniteDistribution[Tuple[S, float]]]]
    ):
        non_terminals: Set[S] = set(mapping.keys())
        self.mapping = {NonTerminal(s): {a: Categorical(
            {(NonTerminal(s1) if s1 in non_terminals else Terminal(s1), r): p
             for (s1, r), p in v.table().items()}
        ) for a, v in d.items()} for s, d in mapping.items()}
        self.non_terminal_states = list(self.mapping.keys())

    def __repr__(self) -> str:
        display = ""
        for s, d in self.mapping.items():
            display += f"From State {s.state}:\n"
            for a, d1 in d.items():
                display += f"  With Action {a}:\n"
                for (s1, r), p in d1:
                    opt = "Terminal " if isinstance(s1, Terminal) else ""
                    display += f"    To [{opt}State {s1.state} and "\
                        + f"Reward {r:.3f}] with Probability {p:.3f}\n"
        return display

    def step(self, state: NonTerminal[S], action: A) -> StateReward[S]:
        action_map: ActionMapping[A, S] = self.mapping[state]
        return action_map[action]

    def apply_finite_policy(self, policy: FinitePolicy[S, A])\
            -> FiniteMarkovRewardProcess[S]:

        transition_mapping: Dict[S, FiniteDistribution[Tuple[S, float]]] = {}

        for state in self.mapping:
            action_map: ActionMapping[A, S] = self.mapping[state]
            outcomes: DefaultDict[Tuple[S, float], float]\
                = defaultdict(float)
            actions = policy.act(state)
            for action, p_action in actions:
                for (s1, r), p in action_map[action].table().items():
                    outcomes[(s1.state, r)] += p_action * p

            transition_mapping[state.state] = Categorical(outcomes)

        return FiniteMarkovRewardProcess(transition_mapping)

    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        '''All the actions allowed for the given state.

        This will be empty for terminal states.

        '''
        return self.mapping[state].keys()
