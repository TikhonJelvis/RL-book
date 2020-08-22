from typing import Mapping, Iterator, TypeVar, List, Tuple, Dict
import operator

from rl.iterate import converged, iterate
from rl.markov_decision_process import (ActionMapping,
                                        FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess,
                                        FinitePolicy)
from rl.distribution import FiniteDistribution, Categorical, Constant, Choose

A = TypeVar('A')
S = TypeVar('S')

DEFAULT_TOLERANCE = 1e-5

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]


def almost_equal_vfs(
    v1: V[S],
    v2: V[S],
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Return whether the two value function tables are within the given
    tolerance of each other.

    '''
    return max([abs(v1[s] - v2[s]) for s in v1.keys()]) < tolerance


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> Iterator[V[S]]:
    '''Iteratively calculate the value function for the give Markov reward
    process.

    '''
    def update(v: V[S]) -> V[S]:
        return {s: mrp.reward_function_vec[i] + gamma *
                sum(p * v.get(s1, 0.) for s1, p in mrp.transition_map[s])
                for i, s in enumerate(mrp.non_terminal_states)}

    v_0: V[S] = {s: 0. for s in mrp.non_terminal_states}

    return iterate(update, v_0)


def evaluate_mrp_result(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> V[S]:
    return converged(evaluate_mrp(mrp, gamma=gamma), done=almost_equal_vfs)


def greedy_policy_from_vf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: V[S],
    gamma: float
) -> FinitePolicy[S, A]:
    greedy_policy_dict: Dict[S, FiniteDistribution[A]] = {}

    for s in mdp.non_terminal_states:

        q_values: List[Tuple[A, float]] = []
        action_map: ActionMapping[A, S] = mdp.mapping[s]

        for a in mdp.actions(s):
            q_val: float = 0.
            for (next_s, r), p in action_map[a]:
                q_val += p * (r + gamma * vf.get(next_s, 0.))
            q_values.append((a, q_val))

        greedy_policy_dict[s] =\
            Constant(max(q_values, key=operator.itemgetter(1))[0])

    return FinitePolicy(greedy_policy_dict)


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False
) -> Iterator[Tuple[V[S], FinitePolicy[S, A]]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]])\
            -> Tuple[V[S], FinitePolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: V[S] = {mrp.non_terminal_states[i]: v for i, v in
                           enumerate(mrp.get_value_function_vec(gamma))}\
            if matrix_method_for_mrp_eval else evaluate_mrp_result(mrp, gamma)
        improved_pi: FinitePolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )

        return policy_vf, improved_pi

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    pi_0: FinitePolicy[S, A] = FinitePolicy(
        {s: Choose(set(mdp.actions(s))) for s in mdp.non_terminal_states}
    )
    return iterate(update, (v_0, pi_0))


def almost_equal_vf_pis(
    x1: Tuple[V[S], FinitePolicy[S, A]],
    x2: Tuple[V[S], FinitePolicy[S, A]]
) -> bool:
    return max(
        abs(x1[0][s] - x2[0][s]) for s in x1[0].keys()
    ) < DEFAULT_TOLERANCE


def policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
) -> Tuple[V[S], FinitePolicy[S, A]]:
    return converged(policy_iteration(mdp, gamma), done=almost_equal_vf_pis)


def bellman_opt_update(
    v: V[S],
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> V[S]:
    '''Do one update of the value function for a given MDP.'''
    def update_s(s: S) -> float:
        return max(sum(p * (r + gamma * v.get(next_s, 0.))
                       for (next_s, r), p in mdp.mapping[s][a])
                   for a in mdp.actions(s))

    return {s: update_s(s) for s in v.keys()}


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Iterator[V[S]]:
    '''Calculate the value function (V*) of the given MDP by applying the
    value_update function repeatedly until the values start
    converging.
    '''
    def update(v: V[S]) -> V[S]:
        return bellman_opt_update(v, mdp, gamma)

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    return iterate(update, v_0)


def value_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Tuple[V[S], FinitePolicy[S, A]]:
    opt_vf: V[S] = converged(
        value_iteration(mdp, gamma),
        done=almost_equal_vfs
    )
    opt_policy: FinitePolicy[S, A] = greedy_policy_from_vf(
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
