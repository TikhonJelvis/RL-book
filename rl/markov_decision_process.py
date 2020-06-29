from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (DefaultDict, Dict, Iterable, Generic, Mapping,
                    Tuple, Sequence, TypeVar, Optional)

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
    def act(self, state: S) -> Optional[Distribution[A]]:
        pass


class FinitePolicy(Policy[S, A]):
    ''' A policy where the state and action spaces are finite.

    '''
    policy_map: Mapping[S, Optional[FiniteDistribution[A]]]

    def __init__(
        self,
        policy_map: Mapping[S, Optional[FiniteDistribution[A]]]
    ):
        self.policy_map = policy_map

    def __repr__(self) -> str:
        display = ""
        for s, d in self.policy_map.items():
            if d is None:
                display += f"{s} is a Terminal State\n"
            else:
                display += f"For State {s}:\n"
                for a, p in d.table():
                    display += f"  Do Action {a} with Probability {p:.3f}\n"
        return display

    def act(self, state: S) -> FiniteDistribution[A]:
        return self.policy_map[state]


class MarkovDecisionProcess(ABC, Generic[S, A]):
    @abstractmethod
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        pass


StateReward = FiniteDistribution[Tuple[S, float]]
ActionMapping = Mapping[A, StateReward[S]]
StateActionMapping = Mapping[S, Optional[ActionMapping[A, S]]]


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    '''A Markov Decision Process with finite state and action spaces.

    '''

    mapping: StateActionMapping[S, A]
    non_terminal_states: Sequence[S]

    def __init__(self, mapping: StateActionMapping[S, A]):
        self.mapping = mapping
        self.non_terminal_states = [s for s, v in mapping.items()
                                    if v is not None]

    def __repr__(self) -> str:
        display = ""
        for s, d in self.mapping.items():
            if d is None:
                display += f"{s} is a Terminal State\n"
            else:
                display += f"From State {s}:\n"
                for a, d1 in d.items():
                    display += f"  With Action {a}:\n"
                    for (s1, r), p in d1.table():
                        display += f"    To [State {s} and "\
                            + f"Reward {r:.3f}] with Probability {p:.3f}\n"
        return display

    # Note: We need both apply_policy and apply_finite_policy because,
    # to be compatible with MarkovRewardProcess, apply_policy has to
    # work even if the policy is *not* finite.
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:

        class Process(MarkovRewardProcess[S]):

            def transition_reward(self, state: S)\
                    -> Optional[Distribution[Tuple[S, float]]]:

                action_map: Optional[ActionMapping[A, S]] = self.mapping[state]
                if action_map is None:
                    return None
                else:
                    def next_pair(action_map=action_map):
                        action: A = policy.act(state).sample()
                        return action_map[action].sample()

                    return SampledDistribution(next_pair)

        return Process()

    def apply_finite_policy(self, policy: FinitePolicy[S, A])\
            -> FiniteMarkovRewardProcess[S]:

        transition_mapping: Dict[S, Optional[StateReward[S]]] = {}

        for state in self.mapping:
            action_map: Optional[ActionMapping[A, S]] = self.mapping[state]
            if action_map is None:
                transition_mapping[state] = None
            else:
                outcomes: DefaultDict[Tuple[S, float], float]\
                    = defaultdict(float)
                for action, p_action in policy.act(state).table():
                    for outcome, p_state in action_map[action].table():
                        outcomes[outcome] += p_action * p_state

                transition_mapping[state] = Categorical(outcomes.items())

        return FiniteMarkovRewardProcess(transition_mapping)

    # Note: For now, this is only available on finite MDPs; this might
    # change in the future.
    def actions(self, state: S) -> Optional[Iterable[A]]:
        '''All the actions allowed for the given state.

        '''
        if self.mapping[state] is None:
            return None
        else:
            return self.mapping[state].keys()
