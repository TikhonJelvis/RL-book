from typing import TypeVar, Callable, Iterator, Sequence, Tuple, Mapping
from rl.function_approx import Tabular
from rl.distribution import Choose
from rl.markov_process import NonTerminal
from rl.markov_decision_process import (
    MarkovDecisionProcess, FiniteMarkovDecisionProcess,
    FiniteMarkovRewardProcess)
from rl.policy import FiniteDeterministicPolicy, FinitePolicy
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import NTStateDistribution
import itertools
import rl.iterate as iterate
from rl.returns import returns
import rl.monte_carlo as mc
import rl.td as td
from rl.function_approx import learning_rate_schedule
from rl.dynamic_programming import V, value_iteration_result
from math import sqrt
from pprint import pprint

S = TypeVar('S')
A = TypeVar('A')


def glie_mc_finite_control_equal_wts(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-5,
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    return mc.glie_mc_control(
        mdp=fmdp,
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(values_map=initial_qvf_dict),
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        episode_length_tolerance=episode_length_tolerance
    )


def glie_mc_control_learning_rate(
    mdp: MarkovDecisionProcess[S, A],
    start_state_distribution: NTStateDistribution,
    initial_func_approx: QValueFunctionApprox[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-5
) -> Iterator[QValueFunctionApprox[S, A]]:
    return mc.glie_mc_control(
        mdp=mdp,
        states=start_state_distribution,
        approx_0=initial_func_approx,
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        episode_length_tolerance=episode_length_tolerance
    )


def glie_mc_finite_control_learning_rate(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-5
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return mc.glie_mc_control(
        mdp=fmdp,
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(
            values_map=initial_qvf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        episode_length_tolerance=episode_length_tolerance
    )


def glie_sarsa_learning_rate(
    mdp: MarkovDecisionProcess[S, A],
    start_state_distribution: NTStateDistribution[S],
    initial_func_approx: QValueFunctionApprox[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    return td.glie_sarsa(
        mdp=mdp,
        states=start_state_distribution,
        approx_0=initial_func_approx,
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        max_episode_length=max_episode_length
    )


def glie_sarsa_finite_learning_rate(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return td.glie_sarsa(
        mdp=fmdp,
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(
            values_map=initial_qvf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        max_episode_length=max_episode_length
    )


def q_learning_learning_rate(
    mdp: MarkovDecisionProcess[S, A],
    start_state_distribution: NTStateDistribution[S],
    initial_func_approx: QValueFunctionApprox[S, A],
    gamma: float,
    epsilon: float,
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    return td.q_learning(
        mdp=mdp,
        policy_from_q=lambda f, m: mc.epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=epsilon
        ),
        states=start_state_distribution,
        approx_0=initial_func_approx,
        γ=gamma,
        max_episode_length=max_episode_length
    )


def q_learning_finite_learning_rate(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon: float,
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return td.q_learning(
        mdp=fmdp,
        policy_from_q=lambda f, m: mc.epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=epsilon
        ),
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(
            values_map=initial_qvf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        max_episode_length=max_episode_length
    )


def get_vf_and_policy_from_qvf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    qvf: QValueFunctionApprox[S, A]
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    opt_vf: V[S] = {
        s: max(qvf((s, a)) for a in mdp.actions(s))
        for s in mdp.non_terminal_states
    }
    opt_policy: FiniteDeterministicPolicy[S, A] = \
        FiniteDeterministicPolicy({
            s.state: qvf.argmax((s, a) for a in mdp.actions(s))[1]
            for s in mdp.non_terminal_states
        })
    return opt_vf, opt_policy


def glie_mc_finite_equal_wts_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float,
    num_episodes: int
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        glie_mc_finite_control_equal_wts(
            fmdp=fmdp,
            gamma=gamma,
            epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
            episode_length_tolerance=episode_length_tolerance
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_episodes))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"GLIE MC Optimal Value Function with {num_episodes:d} episodes")
    pprint(opt_vf)
    print(f"GLIE MC Optimal Policy with {num_episodes:d} episodes")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def glie_mc_finite_learning_rate_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float,
    num_episodes: int
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        glie_mc_finite_control_learning_rate(
            fmdp=fmdp,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            gamma=gamma,
            epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
            episode_length_tolerance=episode_length_tolerance
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_episodes))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"GLIE MC Optimal Value Function with {num_episodes:d} episodes")
    pprint(opt_vf)
    print(f"GLIE MC Optimal Policy with {num_episodes:d} episodes")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def glie_sarsa_finite_learning_rate_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int,
    num_updates: int,
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        glie_sarsa_finite_learning_rate(
            fmdp=fmdp,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            gamma=gamma,
            epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
            max_episode_length=max_episode_length
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_updates))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"GLIE SARSA Optimal Value Function with {num_updates:d} updates")
    pprint(opt_vf)
    print(f"GLIE SARSA Optimal Policy with {num_updates:d} updates")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def q_learning_finite_learning_rate_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon: float,
    max_episode_length: int,
    num_updates: int,
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        q_learning_finite_learning_rate(
            fmdp=fmdp,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            gamma=gamma,
            epsilon=epsilon,
            max_episode_length=max_episode_length
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_updates))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"Q-Learning ptimal Value Function with {num_updates:d} updates")
    pprint(opt_vf)
    print(f"Q-Learning Optimal Policy with {num_updates:d} updates")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def compare_mc_sarsa_ql(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    method_mask: Tuple[bool, bool, bool],
    learning_rates: Sequence[Tuple[float, float, float]],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    q_learning_epsilon: float,
    mc_episode_length_tol: float,
    num_episodes: int,
    plot_batch: int,
    plot_start: int
) -> None:
    true_vf: V[S] = value_iteration_result(fmdp, gamma)[0]
    states: Sequence[NonTerminal[S]] = fmdp.non_terminal_states
    colors: Sequence[str] = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))

    if method_mask[0]:
        for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
            mc_funcs_it: Iterator[QValueFunctionApprox[S, A]] = \
                glie_mc_finite_control_learning_rate(
                    fmdp=fmdp,
                    initial_learning_rate=init_lr,
                    half_life=half_life,
                    exponent=exponent,
                    gamma=gamma,
                    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
                    episode_length_tolerance=mc_episode_length_tol
                )
            mc_errors = []
            batch_mc_errs = []
            for i, mc_qvf in enumerate(
                    itertools.islice(mc_funcs_it, num_episodes)
            ):
                mc_vf: V[S] = {
                    s: max(mc_qvf((s, a)) for a in fmdp.actions(s))
                    for s in states
                }
                batch_mc_errs.append(sqrt(sum(
                    (mc_vf[s] - true_vf[s]) ** 2 for s in states
                ) / len(states)))
                if i % plot_batch == plot_batch - 1:
                    mc_errors.append(sum(batch_mc_errs) / plot_batch)
                    batch_mc_errs = []
            mc_plot = mc_errors[plot_start:]
            label = f"MC InitRate={init_lr:.3f},HalfLife" + \
                f"={half_life:.0f},Exp={exponent:.1f}"
            plt.plot(
                range(len(mc_plot)),
                mc_plot,
                color=colors[k],
                linestyle='-',
                label=label
            )

    sample_episodes: int = 1000
    uniform_policy: FinitePolicy[S, A] = \
        FinitePolicy(
            {s.state: Choose(fmdp.actions(s)) for s in states}
    )
    fmrp: FiniteMarkovRewardProcess[S] = \
        fmdp.apply_finite_policy(uniform_policy)
    td_episode_length: int = int(round(sum(
        len(list(returns(
            trace=fmrp.simulate_reward(Choose(states)),
            γ=gamma,
            tolerance=mc_episode_length_tol
        ))) for _ in range(sample_episodes)
    ) / sample_episodes))

    if method_mask[1]:
        for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
            sarsa_funcs_it: Iterator[QValueFunctionApprox[S, A]] = \
                glie_sarsa_finite_learning_rate(
                    fmdp=fmdp,
                    initial_learning_rate=init_lr,
                    half_life=half_life,
                    exponent=exponent,
                    gamma=gamma,
                    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
                    max_episode_length=td_episode_length,
                )
            sarsa_errors = []
            transitions_batch = plot_batch * td_episode_length
            batch_sarsa_errs = []

            for i, sarsa_qvf in enumerate(
                itertools.islice(
                    sarsa_funcs_it,
                    num_episodes * td_episode_length
                )
            ):
                sarsa_vf: V[S] = {
                    s: max(sarsa_qvf((s, a)) for a in fmdp.actions(s))
                    for s in states
                }
                batch_sarsa_errs.append(sqrt(sum(
                    (sarsa_vf[s] - true_vf[s]) ** 2 for s in states
                ) / len(states)))
                if i % transitions_batch == transitions_batch - 1:
                    sarsa_errors.append(sum(batch_sarsa_errs) /
                                        transitions_batch)
                    batch_sarsa_errs = []
            sarsa_plot = sarsa_errors[plot_start:]
            label = f"SARSA InitRate={init_lr:.3f},HalfLife" + \
                f"={half_life:.0f},Exp={exponent:.1f}"
            plt.plot(
                range(len(sarsa_plot)),
                sarsa_plot,
                color=colors[k],
                linestyle='--',
                label=label
            )

    if method_mask[2]:
        for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
            ql_funcs_it: Iterator[QValueFunctionApprox[S, A]] = \
                q_learning_finite_learning_rate(
                    fmdp=fmdp,
                    initial_learning_rate=init_lr,
                    half_life=half_life,
                    exponent=exponent,
                    gamma=gamma,
                    epsilon=q_learning_epsilon,
                    max_episode_length=td_episode_length,
                )
            ql_errors = []
            transitions_batch = plot_batch * td_episode_length
            batch_ql_errs = []

            for i, ql_qvf in enumerate(
                itertools.islice(
                    ql_funcs_it,
                    num_episodes * td_episode_length
                )
            ):
                ql_vf: V[S] = {
                    s: max(ql_qvf((s, a)) for a in fmdp.actions(s))
                    for s in states
                }
                batch_ql_errs.append(sqrt(sum(
                    (ql_vf[s] - true_vf[s]) ** 2 for s in states
                ) / len(states)))
                if i % transitions_batch == transitions_batch - 1:
                    ql_errors.append(sum(batch_ql_errs) / transitions_batch)
                    batch_ql_errs = []
            ql_plot = ql_errors[plot_start:]
            label = f"Q-Learning InitRate={init_lr:.3f},HalfLife" + \
                f"={half_life:.0f},Exp={exponent:.1f}"
            plt.plot(
                range(len(ql_plot)),
                ql_plot,
                color=colors[k],
                linestyle=':',
                label=label
            )

    plt.xlabel("Episode Batches", fontsize=20)
    plt.ylabel("Optimal Value Function RMSE", fontsize=20)
    plt.title(
        "RMSE as function of episode batches",
        fontsize=20
    )
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()
