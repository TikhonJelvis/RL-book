from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (DefaultDict, Dict, Iterable, Generic, Mapping, Tuple,
                    TypeVar)

from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution)
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess)

A = TypeVar('A')

S = TypeVar('S')


class Policy(ABC, Generic[S, A]):
    '''A policy is a function that specifies what we should do (the
    action) at a given state of our MDP.

    '''
    @abstractmethod
    def act(self, state: S) -> Distribution[A]:
        pass


class FinitePolicy(Policy[S, A]):
    ''' A policy where the state and action spaces are finite.

    '''
    policy_map: Mapping[S, FiniteDistribution[A]]

    def __init__(self, policy_map: Mapping[S, FiniteDistribution[A]]):
        self.policy_map = policy_map

    def act(self, state: S) -> FiniteDistribution[A]:
        return self.policy_map[state]


class MarkovDecisionProcess(ABC, Generic[S, A]):
    @abstractmethod
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        pass


StateReward = FiniteDistribution[Tuple[S, float]]
ActionMapping = Mapping[A, StateReward[S]]


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    '''A Markov Decision Process with finite state and action spaces.

    '''

    mapping: Mapping[S, ActionMapping[A, S]]

    def __init__(self, mapping: Mapping[S, ActionMapping[A, S]]):
        self.mapping = mapping

    # Note: We need both apply_policy and apply_finite_policy because,
    # to be compatible with MarkovRewardProcess, apply_policy has to
    # work even if the policy is *not* finite.
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess:
        class Process(MarkovRewardProcess):
            def transition_reward(self,
                                  state: S) -> Distribution[Tuple[S, float]]:
                def next_state():
                    action = policy.act(state).sample()
                    return self.mapping[state][action].sample()

                return SampledDistribution(next_state)

        return Process()

    def apply_finite_policy(
            self, policy: FinitePolicy[S, A]) -> FiniteMarkovRewardProcess[S]:
        transition_mapping: Dict[S, StateReward[S]] = {}

        for state in self.mapping:
            outcomes: DefaultDict[Tuple[S, float], float] = defaultdict(float)

            for action, p_action in policy.act(state).table():
                for outcome, p_state in self.mapping[state][action].table():
                    outcomes[outcome] += p_action * p_state

            transition_mapping[state] = Categorical(outcomes.items())

        return FiniteMarkovRewardProcess(transition_mapping)

    # Note: For now, this is only available on finite MDPs; this might
    # change in the future.
    def actions(self, state: S) -> Iterable[A]:
        '''All the actions allowed for the given state.

        '''
        return self.mapping[state].keys()
