import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Dict

import numpy as np

from rl.distribution import Categorical, Choose
from rl.iterate import converged, iterate
from rl.markov_process import NonTerminal, State
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy

A = TypeVar('A')
S = TypeVar('S')

DEFAULT_TOLERANCE = 1e-5

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[NonTerminal[S], float]


def extended_vf(v: V[S], s: State[S]) -> float:
    def non_terminal_vf(st: NonTerminal[S], v=v) -> float:
        return v[st]
    return s.on_non_terminal(non_terminal_vf, 0.0)


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> Iterator[np.ndarray]:
    '''Iteratively calculate the value function for the give Markov reward
    process.

    '''
    def update(v: np.ndarray) -> np.ndarray:
        return mrp.reward_function_vec + gamma * \
            mrp.get_transition_matrix().dot(v)

    v_0: np.ndarray = np.zeros(len(mrp.non_terminal_states))

    return iterate(update, v_0)


def almost_equal_np_arrays(
    v1: np.ndarray,
    v2: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Return whether the two value functions as np.ndarray are within the
    given tolerance of each other.

    '''
    return max(abs(v1 - v2)) < tolerance


def evaluate_mrp_result(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> V[S]:
    v_star: np.ndarray = converged(
        evaluate_mrp(mrp, gamma=gamma),
        done=almost_equal_np_arrays
    )
    return {s: v_star[i] for i, s in enumerate(mrp.non_terminal_states)}


def greedy_policy_from_vf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: V[S],
    gamma: float
) -> FiniteDeterministicPolicy[S, A]:
    greedy_policy_dict: Dict[S, A] = {}

    for s in mdp.non_terminal_states:
        q_values: Iterator[Tuple[A, float]] = \
            ((a, mdp.mapping[s][a].expectation(
                lambda s_r: s_r[1] + gamma * extended_vf(vf, s_r[0])
            )) for a in mdp.actions(s))
        greedy_policy_dict[s.state] = \
            max(q_values, key=operator.itemgetter(1))[0]

    return FiniteDeterministicPolicy(greedy_policy_dict)


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False
) -> Iterator[Tuple[V[S], FinitePolicy[S, A]]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]])\
            -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: V[S] = {mrp.non_terminal_states[i]: v for i, v in
                           enumerate(mrp.get_value_function_vec(gamma))}\
            if matrix_method_for_mrp_eval else evaluate_mrp_result(mrp, gamma)
        improved_pi: FiniteDeterministicPolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )

        return policy_vf, improved_pi

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    pi_0: FinitePolicy[S, A] = FinitePolicy(
        {s.state: Choose(mdp.actions(s)) for s in mdp.non_terminal_states}
    )
    return iterate(update, (v_0, pi_0))


def almost_equal_vf_pis(
    x1: Tuple[V[S], FinitePolicy[S, A]],
    x2: Tuple[V[S], FinitePolicy[S, A]]
) -> bool:
    return max(
        abs(x1[0][s] - x2[0][s]) for s in x1[0]
    ) < DEFAULT_TOLERANCE


def policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    return converged(policy_iteration(mdp, gamma), done=almost_equal_vf_pis)


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Iterator[V[S]]:
    '''Calculate the value function (V*) of the given MDP by applying the
    update function repeatedly until the values converge.

    '''
    def update(v: V[S]) -> V[S]:
        return {s: max(mdp.mapping[s][a].expectation(
            lambda s_r: s_r[1] + gamma * extended_vf(v, s_r[0])
        ) for a in mdp.actions(s)) for s in v}

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    return iterate(update, v_0)


def almost_equal_vfs(
    v1: V[S],
    v2: V[S],
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Return whether the two value function tables are within the given
    tolerance of each other.

    '''
    return max(abs(v1[s] - v2[s]) for s in v1) < tolerance


def value_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    opt_vf: V[S] = converged(
        value_iteration(mdp, gamma),
        done=almost_equal_vfs
    )
    opt_policy: FiniteDeterministicPolicy[S, A] = greedy_policy_from_vf(
        mdp,
        opt_vf,
        gamma
    )

    return opt_vf, opt_policy


if __name__ == '__main__':

    from pprint import pprint

    transition_reward_map = {
        1: Categorical({(1, 7.0): 0.6, (2, 5.0): 0.3, (3, 2.0): 0.1}),
        2: Categorical({(1, -2.0): 0.1, (2, 4.0): 0.2, (3, 0.0): 0.7}),
        3: Categorical({(1, 3.0): 0.2, (2, 8.0): 0.6, (3, 4.0): 0.2})
    }
    gamma = 0.9

    fmrp = FiniteMarkovRewardProcess(transition_reward_map)
    fmrp.display_stationary_distribution()
    fmrp.display_reward_function()
    fmrp.display_value_function(gamma=gamma)
    pprint(evaluate_mrp_result(fmrp, gamma=gamma))
