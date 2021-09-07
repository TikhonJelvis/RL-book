from typing import Sequence, Iterable, Callable
from rl.function_approx import AdamGradient
from rl.function_approx import LinearFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.distribution import Choose
from rl.markov_decision_process import NonTerminal
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.chapter10.prediction_utils import (
    mc_prediction_learning_rate,
    td_prediction_learning_rate
)
import numpy as np
from itertools import islice


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

gamma: float = 0.9

si_mrp: SimpleInventoryMRPFinite = SimpleInventoryMRPFinite(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)
nt_states: Sequence[NonTerminal[InventoryState]] = si_mrp.non_terminal_states
true_vf: np.ndarray = si_mrp.get_value_function_vec(gamma=gamma)

mc_episode_length_tol: float = 1e-6
num_episodes = 10000

td_episode_length: int = 100
initial_learning_rate: float = 0.03
half_life: float = 1000.0
exponent: float = 0.5

ffs: Sequence[Callable[[NonTerminal[InventoryState]], float]] = \
    [(lambda x, s=s: float(x.state == s.state)) for s in nt_states]

mc_ag: AdamGradient = AdamGradient(
    learning_rate=0.05,
    decay1=0.9,
    decay2=0.999
)

td_ag: AdamGradient = AdamGradient(
    learning_rate=0.003,
    decay1=0.9,
    decay2=0.999
)

mc_func_approx: LinearFunctionApprox[NonTerminal[InventoryState]] = \
    LinearFunctionApprox.create(
        feature_functions=ffs,
        adam_gradient=mc_ag
    )

td_func_approx: LinearFunctionApprox[NonTerminal[InventoryState]] = \
    LinearFunctionApprox.create(
        feature_functions=ffs,
        adam_gradient=td_ag
    )

it_mc: Iterable[ValueFunctionApprox[InventoryState]] = \
    mc_prediction_learning_rate(
        mrp=si_mrp,
        start_state_distribution=Choose(nt_states),
        gamma=gamma,
        episode_length_tolerance=mc_episode_length_tol,
        initial_func_approx=mc_func_approx
    )

it_td: Iterable[ValueFunctionApprox[InventoryState]] = \
    td_prediction_learning_rate(
        mrp=si_mrp,
        start_state_distribution=Choose(nt_states),
        gamma=gamma,
        episode_length=td_episode_length,
        initial_func_approx=td_func_approx
    )

mc_episodes: int = 3000
for i, mc_vf in enumerate(islice(it_mc, mc_episodes)):
    mc_rmse: float = np.sqrt(sum(
        (mc_vf(s) - true_vf[i]) ** 2 for i, s in enumerate(nt_states)
    ) / len(nt_states))
    print(f"MC: Iteration = {i:d}, RMSE = {mc_rmse:.3f}")

td_experiences: int = 300000
for i, td_vf in enumerate(islice(it_td, td_experiences)):
    td_rmse: float = np.sqrt(sum(
        (td_vf(s) - true_vf[i]) ** 2 for i, s in enumerate(nt_states)
    ) / len(nt_states))
    print(f"TD: Iteration = {i:d}, RMSE = {td_rmse:.3f}")
