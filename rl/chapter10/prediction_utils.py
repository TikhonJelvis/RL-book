from typing import Iterable, TypeVar, Callable, Iterator, Sequence, \
    Tuple, Mapping
from rl.function_approx import FunctionApprox, Tabular
from rl.distribution import Distribution, Choose
from rl.markov_process import (MarkovRewardProcess,
                               FiniteMarkovRewardProcess, TransitionStep)
import itertools
import rl.iterate as iterate
from rl.returns import returns
import rl.monte_carlo as mc
from rl.function_approx import learning_rate_schedule
import rl.td as td
import numpy as np
from math import sqrt
from pprint import pprint

S = TypeVar('S')


def mrp_episodes_stream(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: Distribution[S]
) -> Iterable[Iterable[TransitionStep[S]]]:
    return mrp.reward_traces(start_state_distribution)


def fmrp_episodes_stream(
    fmrp: FiniteMarkovRewardProcess[S]
) -> Iterable[Iterable[TransitionStep[S]]]:
    return mrp_episodes_stream(fmrp, Choose(set(fmrp.non_terminal_states)))


def mc_prediction_equal_wts(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: Distribution[S],
    gamma: float,
    tolerance: float,
    initial_func_approx: FunctionApprox[S]
) -> Iterator[FunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    return mc.mc_prediction(
        traces=episodes,
        approx_0=initial_func_approx,
        γ=gamma,
        tolerance=tolerance
    )


def mc_finite_prediction_equal_wts(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    tolerance: float,
    initial_vf_dict: Mapping[S, float]
) -> Iterator[FunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    return mc.mc_prediction(
        traces=episodes,
        approx_0=Tabular(values_map=initial_vf_dict),
        γ=gamma,
        tolerance=tolerance
    )


def mc_prediction_learning_rate(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: Distribution[S],
    gamma: float,
    tolerance: float,
    initial_func_approx: FunctionApprox[S]
) -> Iterator[FunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    return mc.mc_prediction(
        traces=episodes,
        approx_0=initial_func_approx,
        γ=gamma,
        tolerance=tolerance
    )


def mc_finite_prediction_learning_rate(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    tolerance: float,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[S, float]
) -> Iterator[FunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return mc.mc_prediction(
        traces=episodes,
        approx_0=Tabular(
            values_map=initial_vf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        tolerance=tolerance
    )


def unit_experiences_from_episodes(
    episodes: Iterable[Iterable[TransitionStep[S]]],
    episode_length: int
) -> Iterable[TransitionStep[S]]:
    return itertools.chain.from_iterable(
        itertools.islice(episode, episode_length) for episode in episodes
    )


def td_prediction_learning_rate(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: Distribution[S],
    gamma: float,
    episode_length: int,
    initial_func_approx: FunctionApprox[S]
) -> Iterator[FunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    td_experiences: Iterable[TransitionStep[S]] = \
        unit_experiences_from_episodes(
            episodes,
            episode_length
        )
    return td.td_prediction(
        transitions=td_experiences,
        approx_0=initial_func_approx,
        γ=gamma
    )


def td_finite_prediction_learning_rate(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[S, float]
) -> Iterator[FunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    td_experiences: Iterable[TransitionStep[S]] = \
        unit_experiences_from_episodes(
            episodes,
            episode_length
        )
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return td.td_prediction(
        transitions=td_experiences,
        approx_0=Tabular(
            values_map=initial_vf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma
    )


def mc_finite_equal_wts_correctness(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    tolerance: float,
    num_episodes: int,
    initial_vf_dict: Mapping[S, float]
) -> None:
    mc_vfs: Iterator[FunctionApprox[S]] = \
        mc_finite_prediction_equal_wts(
            fmrp=fmrp,
            gamma=gamma,
            tolerance=tolerance,
            initial_vf_dict=initial_vf_dict
        )
    final_mc_vf: FunctionApprox[S] = \
        iterate.last(itertools.islice(mc_vfs, num_episodes))
    print(f"Equal-Weights-MC Value Function with {num_episodes:d} episodes")
    pprint({s: round(final_mc_vf(s), 3) for s in fmrp.non_terminal_states})
    print("True Value Function")
    fmrp.display_value_function(gamma=gamma)


def mc_finite_learning_rate_correctness(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    tolerance: float,
    num_episodes: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[S, float]
) -> None:
    mc_vfs: Iterator[FunctionApprox[S]] = \
        mc_finite_prediction_learning_rate(
            fmrp=fmrp,
            gamma=gamma,
            tolerance=tolerance,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            initial_vf_dict=initial_vf_dict
        )
    final_mc_vf: FunctionApprox[S] = \
        iterate.last(itertools.islice(mc_vfs, num_episodes))
    print("Decaying-Learning-Rate-MC Value Function with " +
          f"{num_episodes:d} episodes")
    pprint({s: round(final_mc_vf(s), 3) for s in fmrp.non_terminal_states})
    print("True Value Function")
    fmrp.display_value_function(gamma=gamma)


def td_finite_learning_rate_correctness(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length: int,
    num_episodes: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[S, float]
) -> None:
    td_vfs: Iterator[FunctionApprox[S]] = \
        td_finite_prediction_learning_rate(
            fmrp=fmrp,
            gamma=gamma,
            episode_length=episode_length,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            initial_vf_dict=initial_vf_dict
        )
    final_td_vf: FunctionApprox[S] = \
        iterate.last(itertools.islice(td_vfs, episode_length * num_episodes))
    print("Decaying-Learning-Rate-TD Value Function with " +
          f"{num_episodes:d} episodes")
    pprint({s: round(final_td_vf(s), 3) for s in fmrp.non_terminal_states})
    print("True Value Function")
    fmrp.display_value_function(gamma=gamma)


def compare_td_and_mc(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    mc_episode_length_tol: float,
    num_episodes: int,
    learning_rates: Sequence[Tuple[float, float, float]],
    initial_vf_dict: Mapping[S, float],
    plot_batch: int,
    plot_start: int
) -> None:
    true_vf: np.ndarray = fmrp.get_value_function_vec(gamma)
    states: Sequence[S] = fmrp.non_terminal_states
    colors: Sequence[str] = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))

    for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
        mc_funcs_it: Iterator[FunctionApprox[S]] = \
            mc_finite_prediction_learning_rate(
                fmrp=fmrp,
                gamma=gamma,
                tolerance=mc_episode_length_tol,
                initial_learning_rate=init_lr,
                half_life=half_life,
                exponent=exponent,
                initial_vf_dict=initial_vf_dict
            )
        mc_errors = []
        batch_mc_errs = []
        for i, mc_f in enumerate(itertools.islice(mc_funcs_it, num_episodes)):
            batch_mc_errs.append(sqrt(sum(
                (mc_f(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
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
    td_episode_length: int = int(round(sum(
        len(list(returns(
            trace=fmrp.simulate_reward(Choose(set(states))),
            γ=gamma,
            tolerance=mc_episode_length_tol
        ))) for _ in range(sample_episodes)
    ) / sample_episodes))

    for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
        td_funcs_it: Iterator[FunctionApprox[S]] = \
            td_finite_prediction_learning_rate(
                fmrp=fmrp,
                gamma=gamma,
                episode_length=td_episode_length,
                initial_learning_rate=init_lr,
                half_life=half_life,
                exponent=exponent,
                initial_vf_dict=initial_vf_dict
            )
        td_errors = []
        transitions_batch = plot_batch * td_episode_length
        batch_td_errs = []

        for i, td_f in enumerate(
                itertools.islice(td_funcs_it, num_episodes * td_episode_length)
        ):
            batch_td_errs.append(sqrt(sum(
                (td_f(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
            ) / len(states)))
            if i % transitions_batch == transitions_batch - 1:
                td_errors.append(sum(batch_td_errs) / transitions_batch)
                batch_td_errs = []
        td_plot = td_errors[plot_start:]
        label = f"TD InitRate={init_lr:.3f},HalfLife" + \
            f"={half_life:.0f},Exp={exponent:.1f}"
        plt.plot(
            range(len(td_plot)),
            td_plot,
            color=colors[k],
            linestyle='--',
            label=label
        )

    plt.xlabel("Episode Batches", fontsize=20)
    plt.ylabel("Value Function RMSE", fontsize=20)
    plt.title(
        "RMSE of MC and TD as function of episode batches",
        fontsize=25
    )
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()
