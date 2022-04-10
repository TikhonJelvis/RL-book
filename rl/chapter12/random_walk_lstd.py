from rl.chapter10.random_walk_mrp import RandomWalkMRP
from rl.chapter12.laguerre import laguerre_state_features
from rl.td import td_prediction, least_squares_td
from rl.function_approx import LinearFunctionApprox, Tabular, \
    learning_rate_schedule
from rl.approximate_dynamic_programming import NTStateDistribution
import numpy as np
from typing import Iterable, Sequence, Callable
from rl.markov_process import TransitionStep, NonTerminal
from rl.distribution import Choose
import itertools
from rl.gen_utils.plot_funcs import plot_list_of_curves
import rl.iterate as iterate


this_barrier: int = 20
this_p: float = 0.55
random_walk: RandomWalkMRP = RandomWalkMRP(
    barrier=this_barrier,
    p=this_p
)

gamma = 1.0
true_vf: np.ndarray = random_walk.get_value_function_vec(gamma=gamma)

num_transitions: int = 10000

nt_states: Sequence[NonTerminal[int]] = random_walk.non_terminal_states
start_distribution: NTStateDistribution[int] = Choose(nt_states)
traces: Iterable[Iterable[TransitionStep[int]]] = \
    random_walk.reward_traces(start_distribution)
transitions: Iterable[TransitionStep[int]] = \
    itertools.chain.from_iterable(traces)

td_transitions: Iterable[TransitionStep[int]] = \
    itertools.islice(transitions, num_transitions)

initial_learning_rate: float = 0.5
half_life: float = 1000
exponent: float = 0.5
approx0: Tabular[NonTerminal[int]] = Tabular(
    count_to_weight_func=learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
)

td_func: Tabular[NonTerminal[int]] = \
    iterate.last(itertools.islice(
        td_prediction(
            transitions=td_transitions,
            approx_0=approx0,
            γ=gamma
        ),
        num_transitions
    ))
td_vf: np.ndarray = td_func.evaluate(nt_states)

num_polynomials: int = 5
features: Sequence[Callable[[NonTerminal[int]], float]] = \
    laguerre_state_features(num_polynomials)
lstd_transitions: Iterable[TransitionStep[int]] = \
    itertools.islice(transitions, num_transitions)
epsilon: float = 1e-4

lstd_func: LinearFunctionApprox[NonTerminal[int]] = \
    least_squares_td(
        transitions=lstd_transitions,
        feature_functions=features,
        γ=gamma,
        ε=epsilon
    )
lstd_vf: np.ndarray = lstd_func.evaluate(nt_states)

x_vals: Sequence[int] = [s.state for s in nt_states]

plot_list_of_curves(
    [x_vals, x_vals, x_vals],
    [true_vf, td_vf, lstd_vf],
    ["b-", "g.-", "r--"],
    ["True Value Function", "Tabular TD Value Function", "LSTD Value Function"],
    x_label="States",
    y_label="Value Function",
    title="Tabular TD and LSTD versus True Value Function"
)
