from typing import Iterable, TypeVar, Callable, Iterator
from rl.function_approx import FunctionApprox, Tabular
from rl.distribution import Distribution, Choose
from rl.markov_process import (MarkovRewardProcess,
                               FiniteMarkovRewardProcess, TransitionStep)
import itertools
import rl.monte_carlo as mc
import rl.td as td

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


def learning_rate_decay_func(
    base_learning_rate: float,
    learning_rate_decay: float
) -> Callable[[int], float]:
    def lr_func(n: int) -> float:
        return base_learning_rate * (1 + (n - 1) / learning_rate_decay) ** -0.5
    return lr_func


def mc_prediction_equal_wts(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: Distribution[S],
    gamma: float,
    tolerance: float
) -> Iterator[FunctionApprox[S]]:
    traces: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    return mc.evaluate_mrp(
        traces=traces,
        approx_0=Tabular(),
        γ=gamma,
        tolerance=tolerance
    )


def mc_finite_prediction_equal_wts(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    tolerance: float
) -> Iterator[FunctionApprox[S]]:
    traces: Iterable[Iterable[TransitionStep[S]]] = fmrp_episodes_stream(fmrp)
    return mc.evaluate_mrp(
        traces=traces,
        approx_0=Tabular(),
        γ=gamma,
        tolerance=tolerance
    )


def mc_prediction_learning_rate(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: Distribution[S],
    gamma: float,
    tolerance: float,
    base_learning_rate: float,
    learning_rate_decay: float
) -> Iterator[FunctionApprox[S]]:
    traces: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    learning_rate_func: Callable[[int], float] = learning_rate_decay_func(
        base_learning_rate=base_learning_rate,
        learning_rate_decay=learning_rate_decay
    )
    return mc.evaluate_mrp(
        traces=traces,
        approx_0=Tabular(count_to_weight_func=learning_rate_func),
        γ=gamma,
        tolerance=tolerance
    )


def mc_finite_prediction_learning_rate(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    tolerance: float,
    base_learning_rate: float,
    learning_rate_decay: float
) -> Iterator[FunctionApprox[S]]:
    traces: Iterable[Iterable[TransitionStep[S]]] = fmrp_episodes_stream(fmrp)
    learning_rate_func: Callable[[int], float] = learning_rate_decay_func(
        base_learning_rate=base_learning_rate,
        learning_rate_decay=learning_rate_decay
    )
    return mc.evaluate_mrp(
        traces=traces,
        approx_0=Tabular(count_to_weight_func=learning_rate_func),
        γ=gamma,
        tolerance=tolerance
    )


def td_prediction_learning_rate(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: Distribution[S],
    gamma: float,
    episode_length: int,
    base_learning_rate: float,
    learning_rate_decay: float
) -> Iterator[FunctionApprox[S]]:
    traces: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    td_experiences: Iterable[TransitionStep[S]] = \
        itertools.chain.from_iterable(
            itertools.islice(trace, episode_length) for trace in traces
        )
    learning_rate_func: Callable[[int], float] = learning_rate_decay_func(
        base_learning_rate=base_learning_rate,
        learning_rate_decay=learning_rate_decay
    )
    return td.evaluate_mrp(
        transitions=td_experiences,
        approx_0=Tabular(count_to_weight_func=learning_rate_func),
        γ=gamma
    )


def td_finite_prediction_learning_rate(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length: int,
    base_learning_rate: float,
    learning_rate_decay: float
) -> Iterator[FunctionApprox[S]]:
    traces: Iterable[Iterable[TransitionStep[S]]] = fmrp_episodes_stream(fmrp)
    td_experiences: Iterable[TransitionStep[S]] = \
        itertools.chain.from_iterable(
            itertools.islice(trace, episode_length) for trace in traces
        )
    learning_rate_func: Callable[[int], float] = learning_rate_decay_func(
        base_learning_rate=base_learning_rate,
        learning_rate_decay=learning_rate_decay
    )
    return td.evaluate_mrp(
        transitions=td_experiences,
        approx_0=Tabular(count_to_weight_func=learning_rate_func),
        γ=gamma
    )
