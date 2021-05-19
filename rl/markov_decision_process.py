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
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        pass


class Always(Policy[S, A]):
    action: A

    def __init__(self, action: A):
        self.action = action

    def act(self, _: S) -> Distribution[A]:
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


class State(ABC, Generic[S]):
    def on_non_terminal(self, f: Callable[[S], X], default: X) -> X:
        if isinstance(self, NonTerminal):
            return f(self.state)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S


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
        reward: float = 0

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


def epsilon_greedy_policy(
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
        def act(self, s: NonTerminal[S]) -> Distribution[A]:
            if explore.sample():
                return Choose(set(mdp.actions(s)))

            _, action = q.argmax((s, a) for a in mdp.actions(s))
            return Constant(action)

    return QPolicy()


ActionMapping = Mapping[A, StateReward[State[S]]]
StateActionMapping = Mapping[S, Optional[ActionMapping[A, S]]]


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    '''A Markov Decision Process with finite state and action spaces.

    '''

    mapping: StateActionMapping[NonTerminal[S], A]

    def __init__(self, mapping: StateActionMapping[NonTerminal[S], A]):
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

    def step(self, state: NonTerminal[S], action: A) -> StateReward:
        action_map: ActionMapping[A, NonTerminal[S]] = self.mapping[state]

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
