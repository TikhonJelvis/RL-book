from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Callable, Generic, Iterator, Mapping, TypeVar, List,
                    Tuple, Dict)
import operator

from rl.iterate import converge, iterate
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess,
                                        FinitePolicy)
from rl.distribution import FiniteDistribution, Categorical

A = TypeVar('A')
S = TypeVar('S')
X = TypeVar('X')

DEFAULT_TOLERANCE = 1e-5


Self = TypeVar('Self', bound='FunctionRepresentation')


class FunctionRepresentation(ABC, Generic[X]):
    @abstractmethod
    def __getitem__(self, key: X) -> float:
        pass

    @abstractmethod
    def update(self, key, value):
        pass

    @abstractmethod
    def update_all(self, update):
        pass

    @abstractmethod
    def within(self, other, tolerance=DEFAULT_TOLERANCE):
        '''Are all the values in the given FunctionRepresentation within the
        given bound of the values in this FunctionRepresentation?

        '''
        pass


@dataclass
class TabularRepresentation(FunctionRepresentation, Generic[X]):
    mapping: Mapping[X, float]

    def __getitem__(self, key):
        return self.mapping[key]

    def update(self, key, value):
        return TabularRepresentation(
            {x: value if x == key else old_value
             for x, old_value in self.mapping.items()})

    def update_all(self, update):
        return TabularRepresentation({x: update(self, x) for x in self.mapping})

    def within(self, other, tolerance=DEFAULT_TOLERANCE):
        return all(abs(self[k] - other[k]) < DEFAULT_TOLERANCE
                   for k in self.mapping.keys())


# A representation of a value function for a finite MDP with states of
# type S
V = TabularRepresentation[S]


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> Iterator[V[S]]:
    '''Calculate the value function for the given Markov Reward
    Process.
    '''
    def update_s(v, s: S) -> float:
        next_states = mrp.transition_reward(s)

        if next_states is None:
            return 0.0  # terminal state
        else:
            return \
                v[s] + gamma * sum(p * v[s1] for s1, p in next_states.table())

    v_0: V[S] = TabularRepresentation({s: 0. for s in mrp.non_terminal_states})

    return iterate(lambda v: v.update_all(update_s), v_0)


def greedy_policy_from_vf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: V[S],
    gamma: float
) -> FinitePolicy[S, A]:
    greedy_policy_dict: Dict[S, FiniteDistribution[A]] = {}

    for s in mdp.non_terminal_states:

        q_values: List[Tuple[A, float]] = []
        action_map = mdp.mapping[s]

        for a in mdp.actions(s):
            q_val: float = 0.
            for (next_s, r), p in action_map[a].table():
                next_state_vf = vf[next_s]\
                    if mdp.mapping[next_s] is not None else 0.
                q_val += p * (r + gamma * next_state_vf)
            q_values.append((a, q_val))

        greedy_policy_dict[s] = Categorical([(
            max(q_values, key=operator.itemgetter(1))[0],
            1.
        )])

    return FinitePolicy(greedy_policy_dict)


def condition_policy_iteration(
    x1: Tuple[V[S], FinitePolicy[S, A]],
    x2: Tuple[V[S], FinitePolicy[S, A]]
) -> bool:
    return max(
        abs(x1[0][s] - x2[0][s]) for s in x1[0].keys()
    ) < DEFAULT_TOLERANCE


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False
) -> Tuple[V[S], FinitePolicy[S, A]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]])\
            -> Tuple[V[S], FinitePolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: V[S] = {mrp.non_terminal_states[i]: v for i, v in
                           enumerate(mrp.get_value_function_vec())}\
            if matrix_method_for_mrp_eval else evaluate_mrp(mrp, gamma)

        improved_pi: FinitePolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )

        return policy_vf, improved_pi

    v_0 = {s: 0.0 for s in mdp.non_terminal_states}
    pi_0 = FinitePolicy({
        s: Categorical([(a, 1. / len(mdp.actions(s))) for a in mdp.actions(s)])
        for s in mdp.non_terminal_states
    })
    vf_pi_0 = (v_0, pi_0)
    return list(converge(
        iterate(update, vf_pi_0),
        done=condition_policy_iteration
    ))[-1]


def bellman_opt_update(
    v: V[S],
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> V[S]:
    '''Do one update of the value function for a given MDP.'''
    def update_s(s: S) -> float:
        q_values: List[float] = []
        action_map = mdp.mapping[s]

        for a in mdp.actions(s):
            q_val: float = 0.
            for (next_s, r), p in action_map[a].table():
                next_state_vf = v[next_s]\
                    if mdp.mapping[next_s] is not None else 0.
                q_val += p * (r + gamma * next_state_vf)
            q_values.append(q_val)

        return max(q_values)

    return {s: update_s(s) for s in v.keys()}


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Tuple[V[S], FinitePolicy[S, A]]:
    '''Calculate the value function (V*) of the given MDP by applying the
    value_update function repeatedly until the values start
    converging.
    '''
    def update(v: V[S]) -> V[S]:
        return bellman_opt_update(v, mdp, gamma)

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    opt_vf: V[S] = list(converge(
        iterate(update, v_0),
        done=condition_vf_dict
    ))[-1]

    opt_policy: FinitePolicy[S, A] = greedy_policy_from_vf(
        mdp,
        opt_vf,
        gamma
    )

    return opt_vf, opt_policy


if __name__ == '__main__':

    from pprint import pprint

    transition_reward_map = {
        1: Categorical([((1, 7.0), 0.6), ((2, 5.0), 0.3), ((3, 2.0), 0.1)]),
        2: Categorical([((1, -2.0), 0.1), ((2, 4.0), 0.2), ((3, 0.0), 0.7)]),
        3: Categorical([((1, 3.0), 0.2), ((2, 8.0), 0.6), ((3, 4.0), 0.2)])
    }
    gamma = 0.9

    fmrp = FiniteMarkovRewardProcess(transition_reward_map)
    fmrp.display_stationary_distribution()
    fmrp.display_reward_function()
    fmrp.display_value_function(gamma=gamma)
    pprint(evaluate_mrp(fmrp, gamma=gamma))
