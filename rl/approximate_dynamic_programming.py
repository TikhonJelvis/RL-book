'''Approximate dynamic programming algorithms are variations on
dynamic programming algorithms that can work with function
approximations rather than exact representations of the process's
state space.

'''

from typing import Iterator, Tuple, TypeVar, Sequence, List
from operator import itemgetter
import numpy as np

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition, NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess,
                                        StateActionMapping)
from rl.policy import DeterministicPolicy

S = TypeVar('S')
A = TypeVar('A')

# A representation of a value function for a finite MDP with states of
# type S
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]


def extended_vf(vf: ValueFunctionApprox[S], s: State[S]) -> float:
    return s.on_non_terminal(vf, 0.0)


def evaluate_finite_mrp(
        mrp: FiniteMarkovRewardProcess[S],
        γ: float,
        approx_0: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:

    '''Iteratively calculate the value function for the give finite Markov
    Reward Process, using the given FunctionApprox to approximate the
    value function at each step.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        vs: np.ndarray = v.evaluate(mrp.non_terminal_states)
        updated: np.ndarray = mrp.reward_function_vec + γ * \
            mrp.get_transition_matrix().dot(vs)
        return v.update(zip(mrp.non_terminal_states, updated))

    return iterate(update, approx_0)


def evaluate_mrp(
    mrp: MarkovRewardProcess[S],
    γ: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:

    '''Iteratively calculate the value function for the given Markov Reward
    Process, using the given FunctionApprox to approximate the value function
    at each step for a random sample of the process' non-terminal states.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(s, mrp.transition_reward(s).expectation(return_))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def value_iteration_finite(
    mdp: FiniteMarkovDecisionProcess[S, A],
    γ: float,
    approx_0: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given finite
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(
                s,
                max(mdp.mapping[s][a].expectation(return_)
                    for a in mdp.actions(s))
            ) for s in mdp.non_terminal_states]
        )

    return iterate(update, approx_0)


def value_iteration(
    mdp: MarkovDecisionProcess[S, A],
    γ: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step for a random sample of the process'
    non-terminal states.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def backward_evaluate_finite(
    step_f0_pairs: Sequence[Tuple[RewardTransition[S],
                                  ValueFunctionApprox[S]]],
    γ: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps.

    '''

    v: List[ValueFunctionApprox[S]] = []

    for i, (step, approx0) in enumerate(reversed(step_f0_pairs)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(v[i-1], s1) if i > 0 else 0.)

        v.append(
            approx0.solve([(s, res.expectation(return_))
                           for s, res in step.items()])
        )

    return reversed(v)


MRP_FuncApprox_Distribution = Tuple[MarkovRewardProcess[S],
                                    ValueFunctionApprox[S],
                                    NTStateDistribution[S]]


def backward_evaluate(
    mrp_f0_mu_triples: Sequence[MRP_FuncApprox_Distribution[S]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps, using
    the given FunctionApprox for each time step for a random sample of the
    time step's states.

    '''
    v: List[ValueFunctionApprox[S]] = []

    for i, (mrp, approx0, mu) in enumerate(reversed(mrp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(v[i-1], s1) if i > 0 else 0.)

        v.append(
            approx0.solve(
                [(s, mrp.transition_reward(s).expectation(return_))
                 for s in mu.sample_n(num_state_samples)],
                error_tolerance
            )
        )

    return reversed(v)


def back_opt_vf_and_policy_finite(
    step_f0s: Sequence[Tuple[StateActionMapping[S, A],
                             ValueFunctionApprox[S]]],
    γ: float,
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step

    '''
    vp: List[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]] = []

    for i, (step, approx0) in enumerate(reversed(step_f0s)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)

        this_v = approx0.solve(
            [(s, max(res.expectation(return_)
                     for a, res in actions_map.items()))
             for s, actions_map in step.items()]
        )

        def deter_policy(state: S) -> A:
            return max(
                ((res.expectation(return_), a) for a, res in
                 step[NonTerminal(state)].items()),
                key=itemgetter(0)
            )[1]

        vp.append((this_v, DeterministicPolicy(deter_policy)))

    return reversed(vp)


MDP_FuncApproxV_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    ValueFunctionApprox[S],
    NTStateDistribution[S]
]


def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxV_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step, using the given FunctionApprox for each time step
    for a random sample of the time step's states.

    '''
    vp: List[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)

        this_v = approx0.solve(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in mu.sample_n(num_state_samples)],
            error_tolerance
        )

        def deter_policy(state: S) -> A:
            return max(
                ((mdp.step(NonTerminal(state), a).expectation(return_), a)
                 for a in mdp.actions(NonTerminal(state))),
                key=itemgetter(0)
            )[1]

        vp.append((this_v, DeterministicPolicy(deter_policy)))

    return reversed(vp)


MDP_FuncApproxQ_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    QValueFunctionApprox[S, A],
    NTStateDistribution[S]
]


def back_opt_qvf(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxQ_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    '''Use backwards induction to find the optimal q-value function  policy at
    each time step, using the given FunctionApprox (for Q-Value) for each time
    step for a random sample of the time step's states.

    '''
    horizon: int = len(mdp_f0_mu_triples)
    qvf: List[QValueFunctionApprox[S, A]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            next_return: float = max(
                qvf[i-1]((s1, a)) for a in
                mdp_f0_mu_triples[horizon - i][0].actions(s1)
            ) if i > 0 and isinstance(s1, NonTerminal) else 0.
            return r + γ * next_return

        this_qvf = approx0.solve(
            [((s, a), mdp.step(s, a).expectation(return_))
             for s in mu.sample_n(num_state_samples) for a in mdp.actions(s)],
            error_tolerance
        )

        qvf.append(this_qvf)

    return reversed(qvf)
