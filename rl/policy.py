from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Mapping, TypeVar

from rl.distribution import Choose, Constant, Distribution, FiniteDistribution
from rl.markov_process import NonTerminal

A = TypeVar('A')
S = TypeVar('S')


class Policy(ABC, Generic[S, A]):
    '''A policy is a function that specifies what we should do (the
    action) at a given state of our MDP.

    '''
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        '''A distribution of actions to take from the given non-terminal
        state.

        '''


@dataclass(frozen=True)
class UniformPolicy(Policy[S, A]):
    valid_actions: Callable[[S], Iterable[A]]

    def act(self, state: NonTerminal[S]) -> Choose[A]:
        return Choose(self.valid_actions(state.state))


@dataclass(frozen=True)
class RandomPolicy(Policy[S, A]):
    '''A policy that randomly selects one of several specified policies
    each action.

    Given the right inputs, this could simulate things like ε-greedy
    policies::

        RandomPolicy()

    '''
    policy_choices: Distribution[Policy[S, A]]

    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        policy: Policy[S, A] = self.policy_choices.sample()
        return policy.act(state)


@dataclass(frozen=True)
class DeterministicPolicy(Policy[S, A]):
    action_for: Callable[[S], A]

    def act(self, state: NonTerminal[S]) -> Constant[A]:
        return Constant(self.action_for(state.state))


class Always(DeterministicPolicy[S, A]):
    '''A constant policy: always return the same (specified) action for
    every possible state.

    '''
    action: A

    def __init__(self, action: A):
        self.action = action
        super().__init__(lambda _: action)


@dataclass(frozen=True)
class FinitePolicy(Policy[S, A]):
    ''' A policy where the state and action spaces are finite.

    '''
    policy_map: Mapping[S, FiniteDistribution[A]]

    def __repr__(self) -> str:
        display = ""
        for s, d in self.policy_map.items():
            display += f"For State {s}:\n"
            for a, p in d:
                display += f"  Do Action {a} with Probability {p:.3f}\n"
        return display

    def act(self, state: NonTerminal[S]) -> FiniteDistribution[A]:
        return self.policy_map[state.state]


class FiniteDeterministicPolicy(FinitePolicy[S, A]):
    '''A deterministic policy where the state and action spaces are
    finite.

    '''
    action_for: Mapping[S, A]

    def __init__(self, action_for: Mapping[S, A]):
        self.action_for = action_for
        super().__init__(policy_map={s: Constant(a) for s, a in
                                     self.action_for.items()})

    def __repr__(self) -> str:
        display = ""
        for s, a in self.action_for.items():
            display += f"For State {s}: Do Action {a}\n"
        return display
