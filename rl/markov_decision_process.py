from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (DefaultDict, Dict, Iterable, Generic, Mapping,
                    Tuple, Sequence, TypeVar, Optional)
from rl.distribution import (Bernoulli, Constant, Categorical, Choose,
                             Distribution, FiniteDistribution)
from rl.function_approx import (FunctionApprox)

from rl.markov_process import (
    FiniteMarkovRewardProcess, MarkovRewardProcess, StateReward
)

A = TypeVar('A')
S = TypeVar('S')


class Policy(ABC, Generic[S, A]):
    '''A policy is a function that specifies what we should do (the
    action) at a given state of our MDP.

    '''
    @abstractmethod
    def act(self, state: S) -> Optional[Distribution[A]]:
        pass


class Always(Policy[S, A]):
    action: A

    def __init__(self, action: A):
        self.action = action

    def act(self, _: S) -> Optional[Distribution[A]]:
        return Constant(self.action)


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
                for a, p in d:
                    display += f"  Do Action {a} with Probability {p:.3f}\n"
        return display

    def act(self, state: S) -> Optional[FiniteDistribution[A]]:
        return self.policy_map[state]

    def states(self) -> Iterable[S]:
        return self.policy_map.keys()


@dataclass(frozen=True)
class TransitionStep(Generic[S, A]):
    '''A single step in the simulation of an MDP, containing:

    state -- the state we start from
    action -- the action we took at that state
    next_state -- the state we ended up in after the action
    reward -- the instantaneous reward we got for this transition
    '''
    state: S
    action: A
    next_state: S
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
                    state: S
            ) -> Optional[Distribution[Tuple[S, float]]]:
                actions: Optional[Distribution[A]] = policy.act(state)

                if actions is None:
                    return None

                # TODO: Handle the case where mdp.step(state, a)
                # returns None
                #
                # Idea: use an exception for termination instead of
                # return None?
                return actions.apply(lambda a: mdp.step(state, a))

        return RewardProcess()

    @abstractmethod
    def actions(self, state: S) -> Iterable[A]:
        pass

    def is_terminal(self, state: S) -> bool:
        '''Is the given state a terminal state?

        We cannot take any actions from a terminal state. This means
        that a state is terminal iff `self.actions(s)` is empty.

        '''
        try:
            next(iter(self.actions(state)))
            return False
        except StopIteration:
            return True

    @abstractmethod
    def step(
            self,
            state: S,
            action: A
    ) -> Optional[Distribution[Tuple[S, float]]]:
        pass

    def simulate_actions(
            self,
            start_states: Distribution[S],
            policy: Policy[S, A]
    ) -> Iterable[TransitionStep[S, A]]:
        '''Simulate this MDP with the given policy, yielding the
        sequence of (states, action, next state, reward) 4-tuples
        encountered in the simulation trace.

        '''
        state: S = start_states.sample()
        reward: float = 0

        while True:
            action_distribution = policy.act(state)
            if action_distribution is None:
                return

            action = action_distribution.sample()
            next_distribution = self.step(state, action)
            if next_distribution is None:
                return

            next_state, reward = next_distribution.sample()
            yield TransitionStep(state, action, next_state, reward)
            state = next_state

    def action_traces(
            self,
            start_states: Distribution[S],
            policy: Policy[S, A]
    ) -> Iterable[Iterable[TransitionStep[S, A]]]:
        '''Yield an infinite number of traces as returned by
        simulate_actions.

        '''
        while True:
            yield self.simulate_actions(start_states, policy)


def policy_from_q(
        q: FunctionApprox[Tuple[S, A]],
        mdp: MarkovDecisionProcess[S, A],
        ϵ: float = 0.0
) -> Policy[S, A]:
    '''Return a policy that chooses the action that maximizes the reward
    for each state in the given Q function.

    Arguments:
      q -- approximation of the Q function for the MDP
      mdp -- the process for which we're generating a policy
      ϵ -- the fraction of the actions where we explore rather
      than following the optimal policy

    Returns a policy based on the given Q function.

    '''
    explore = Bernoulli(ϵ)

    class QPolicy(Policy[S, A]):
        def act(self, s: S) -> Optional[Distribution[A]]:
            if mdp.is_terminal(s):
                return None

            if explore.sample():
                return Choose(set(mdp.actions(s)))

            _, action = q.argmax((s, a) for a in mdp.actions(s))
            return Constant(action)

    return QPolicy()


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
                    for (s1, r), p in d1:
                        display += f"    To [State {s1} and "\
                            + f"Reward {r:.3f}] with Probability {p:.3f}\n"
        return display

    def step(self, state: S, action: A) -> Optional[StateReward]:
        action_map: Optional[ActionMapping[A, S]] = self.mapping[state]

        if action_map is None:
            return None
        return action_map[action]

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

                actions = policy.act(state)
                if actions is not None:
                    for action, p_action in actions:
                        for outcome, p_state_reward in action_map[action]:
                            outcomes[outcome] += p_action * p_state_reward

                transition_mapping[state] = Categorical(outcomes)

        return FiniteMarkovRewardProcess(transition_mapping)

    def action_mapping(self, state: S) -> Optional[ActionMapping[A, S]]:
        return self.mapping[state]

    def actions(self, state: S) -> Iterable[A]:
        '''All the actions allowed for the given state.

        This will be empty for terminal states.

        '''
        actions = self.mapping[state]
        return iter([]) if actions is None else actions.keys()

    def states(self) -> Iterable[S]:
        '''Iterate over all the states in this process—terminal *and*
        non-terminal.

        '''
        return self.mapping.keys()
