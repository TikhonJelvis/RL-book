from typing import Iterator, Mapping, Tuple, TypeVar, Sequence, List
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from operator import itemgetter
import numpy as np
from rl.distribution import Distribution, Constant
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess, Policy,
                                        MarkovDecisionProcess,
                                        StateActionMapping)

S = TypeVar('S')
A = TypeVar('A')

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]


def evaluate_finite_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    γ: float,
    approx_0: FunctionApprox[S]
) -> Iterator[FunctionApprox[S]]:
    '''Iteratively calculate the value function for the give finite Markov
    Reward Process, using the given FunctionApprox to approximate the value
    function at each step.
    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        vs: np.ndarray = v.evaluate(mrp.non_terminal_states)
        updated: np.ndarray = mrp.reward_function_vec + γ * \
            mrp.get_transition_matrix().dot(vs)
        return v.update(zip(mrp.states(), updated))

    return iterate(update, approx_0)


def evaluate_mrp(
    mrp: MarkovRewardProcess[S],
    γ: float,
    approx_0: FunctionApprox[S],
    non_terminal_states_distribution: Distribution[S],
    num_state_samples: int
) -> Iterator[FunctionApprox[S]]:

    '''Iteratively calculate the value function for the given Markov Reward
    Process, using the given FunctionApprox to approximate the value function
    at each step for a random sample of the process' non-terminal states.

    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        nt_states: Sequence[S] = non_terminal_states_distribution.sample_n(
            num_state_samples
        )

        def return_(s_r: Tuple[S, float]) -> float:
            s1, r = s_r
            return r + γ * v.evaluate([s1]).item()

        return v.update(
            [(s, mrp.transition_reward(s).expectation(return_))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def value_iteration_finite(
    mdp: FiniteMarkovDecisionProcess[S, A],
    γ: float,
    approx_0: FunctionApprox[S]
) -> Iterator[FunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given finite
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step

    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:

        def return_(s_r: Tuple[S, float]) -> float:
            s1, r = s_r
            return r + γ * v.evaluate([s1]).item()

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
    approx_0: FunctionApprox[S],
    non_terminal_states_distribution: Distribution[S],
    num_state_samples: int
) -> Iterator[FunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step for a random sample of the process'
    non-terminal states.

    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        nt_states: Sequence[S] = non_terminal_states_distribution.sample_n(
            num_state_samples
        )

        def return_(s_r: Tuple[S, float]) -> float:
            s1, r = s_r
            return r + γ * v.evaluate([s1]).item()

        return v.update(
            [(s, max(mdp.step(s, a).expectation(return_,)
                     for a in mdp.actions(s)))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def backward_evaluate_finite(
    step_f0_pairs: Sequence[Tuple[RewardTransition[S], FunctionApprox[S]]],
    γ: float
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps.

    '''

    v: List[FunctionApprox[S]] = []
    num_steps: int = len(step_f0_pairs)

    for i, (step, approx0) in enumerate(reversed(step_f0_pairs)):

        def return_(s_r: Tuple[S, float], i=i) -> float:
            s1, r = s_r
            return r + γ * (v[i-1].evaluate([s1]).item() if i > 0 and
                            step_f0_pairs[num_steps - i][0][s1] is not None
                            else 0.)

        v.append(
            approx0.solve([(s, res.expectation(return_))
                           for s, res in step.items() if res is not None])
        )

    return reversed(v)


MRP_FuncApprox_Distribution = \
    Tuple[MarkovRewardProcess[S], FunctionApprox[S], Distribution[S]]


def backward_evaluate(
    mrp_f0_mu_triples: Sequence[MRP_FuncApprox_Distribution[S]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps, using
    the given FunctionApprox for each time step for a random sample of the
    time step's states.

    '''
    v: List[FunctionApprox[S]] = []
    
    num_steps: int = len(mrp_f0_mu_triples)

    for i, (mrp, approx0, mu) in enumerate(reversed(mrp_f0_mu_triples)):

        def return_(s_r: Tuple[S, float], i=i) -> float:
            s1, r = s_r
            return r + γ * (v[i-1].evaluate([s1]).item() if i > 0 and not
                            mrp_f0_mu_triples[num_steps - i][0].is_terminal(s1)
                            else 0.)

        v.append(
            approx0.solve(
                [(s, mrp.transition_reward(s).expectation(return_))
                 for s in mu.sample_n(num_state_samples)
                 if not mrp.is_terminal(s)],
                error_tolerance
            )
        )

    return reversed(v)


def back_opt_vf_and_policy_finite(
    step_f0s: Sequence[Tuple[StateActionMapping[S, A], FunctionApprox[S]]],
    γ: float,
) -> Iterator[Tuple[FunctionApprox[S], Policy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step

    '''
    vp: List[Tuple[FunctionApprox[S], Policy[S, A]]] = []

    num_steps: int = len(step_f0s)

    for i, (step, approx0) in enumerate(reversed(step_f0s)):

        def return_(s_r: Tuple[S, float], i=i) -> float:
            s1, r = s_r
            return r + γ * (vp[i-1][0].evaluate([s1]).item() if i > 0 and
                            step_f0s[num_steps - i][0][s1] is not None else 0.)

        this_v = approx0.solve(
            [(s, max(res.expectation(return_)
                     for a, res in actions_map.items()))
             for s, actions_map in step.items() if actions_map is not None]
        )

        class ThisPolicy(Policy[S, A]):
            def act(self, state: S) -> Constant[A]:
                return Constant(max(
                    ((res.expectation(return_), a)
                     for a, res in step[state].items()),
                    key=itemgetter(0)
                )[1])

        vp.append((this_v, ThisPolicy()))

    return reversed(vp)


MDP_FuncApproxV_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    FunctionApprox[S],
    Distribution[S]
]


def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxV_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[Tuple[FunctionApprox[S], Policy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step, using the given FunctionApprox for each time step
    for a random sample of the time step's states.

    '''
    vp: List[Tuple[FunctionApprox[S], Policy[S, A]]] = []

    num_steps: int = len(mdp_f0_mu_triples)

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[S, float], i=i) -> float:
            s1, r = s_r
            return r + γ * (vp[i-1][0].evaluate([s1]).item() if i > 0 and not
                            mdp_f0_mu_triples[num_steps - i][0].is_terminal(s1)
                            else 0.)

        this_v = approx0.solve(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in mu.sample_n(num_state_samples)
             if not mdp.is_terminal(s)],
            error_tolerance
        )

        class ThisPolicy(Policy[S, A]):
            def act(self, state: S) -> Constant[A]:
                return Constant(max(
                    ((mdp.step(state, a).expectation(return_), a)
                     for a in mdp.actions(state)),
                    key=itemgetter(0)
                )[1])

        vp.append((this_v, ThisPolicy()))

    return reversed(vp)


MDP_FuncApproxQ_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    FunctionApprox[Tuple[S, A]],
    Distribution[S]
]


def back_opt_qvf(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxQ_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    '''Use backwards induction to find the optimal q-value function  policy at
    each time step, using the given FunctionApprox (for Q-Value) for each time
    step for a random sample of the time step's states.

    '''
    horizon: int = len(mdp_f0_mu_triples)
    qvf: List[FunctionApprox[Tuple[S, A]]] = []

    num_steps: int = len(mdp_f0_mu_triples)

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[S, float], i=i) -> float:
            s1, r = s_r
            return r + γ * (
                max(qvf[i-1].evaluate([(s1, a)]).item()
                    for a in mdp_f0_mu_triples[horizon - i][0].actions(s1))
                if i > 0 and
                not mdp_f0_mu_triples[num_steps - i][0].is_terminal(s1)
                else 0.
            )

        this_qvf = approx0.solve(
            [((s, a), mdp.step(s, a).expectation(return_))
             for s in mu.sample_n(num_state_samples)
             if not mdp.is_terminal(s) for a in mdp.actions(s)],
            error_tolerance
        )

        qvf.append(this_qvf)

    return reversed(qvf)
