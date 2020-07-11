from typing import (Dict, Iterator, List, Mapping, List, Optional,
                    Tuple, TypeVar)
import operator

from rl.iterate import converge, converged, iterate
from rl.markov_decision_process import (ActionMapping,
                                        FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess,
                                        FinitePolicy)
from rl.distribution import FiniteDistribution, Categorical

A = TypeVar('A')
S = TypeVar('S')
X = TypeVar('X')

DEFAULT_TOLERANCE = 1e-5

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]


# TODO: better name?
def v_converged(v1: V[S], v2: V[S],
                tolerance: float = DEFAULT_TOLERANCE) -> bool:
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
    def state_value(state: S, v: V[S]) -> float:
        '''Calculate the value of the given state using the value of each
        possible next state from the given value function v.

        '''
        next_states = mrp.transition(state)

        if next_states is None:
            return 0
        else:
            return sum(p * v[s1] for s1, p in next_states.table())

    def update(v: V[S]) -> V[S]:
        return {s: mrp.reward_function_vec[i] + gamma * state_value(s, v)
                for i, s in enumerate(mrp.non_terminal_states)}

    v_0: V[S] = {s: 0. for s in mrp.non_terminal_states}

    return iterate(update, v_0)


def greedy_policy_from_vf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: V[S],
    gamma: float
) -> FinitePolicy[S, A]:
    greedy_policy_dict: Dict[S, FiniteDistribution[A]] = {}

    for s in mdp.non_terminal_states:

        q_values: List[Tuple[A, float]] = []
        action_map: Optional[ActionMapping[A, S]] = mdp.mapping[s]

        if action_map is not None:
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
        policy_vf: V[S] =\
            converged(evaluate_mrp(mrp, gamma), done=v_converged)\
            if not matrix_method_for_mrp_eval else\
            {mrp.non_terminal_states[i]: v for i, v in
             enumerate(mrp.get_value_function_vec(gamma))}

        improved_pi: FinitePolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )

        return policy_vf, improved_pi

    v_0 = {s: 0.0 for s in mdp.non_terminal_states}
    pi_0 = FinitePolicy({
        s: Categorical([(a, 1. / len(list(mdp.actions(s))))
                        for a in mdp.actions(s)])
        for s in mdp.non_terminal_states
    })
    vf_pi_0 = (v_0, pi_0)
    return iterate(update, vf_pi_0)


def bellman_opt_update(
    v: V[S],
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> V[S]:
    '''Do one update of the value function for a given MDP.'''
    def update_s(s: S) -> float:
        q_values: List[float] = []
        action_map = mdp.mapping[s]

        if action_map is not None:
            for a in mdp.actions(s):
                q_val: float = 0.
                for (next_s, r), p in action_map[a].table():
                    next_state_vf = v[next_s]\
                        if mdp.mapping[next_s] is not None else 0.
                    q_val += p * (r + gamma * next_state_vf)
                q_values.append(q_val)

            return max(q_values)

        else:
            return 0.0

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
    opt_vf: V[S] = converged(iterate(update, v_0), done=v_converged)

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
    pprint(converged(evaluate_mrp(fmrp, gamma=gamma), done=v_converged))
