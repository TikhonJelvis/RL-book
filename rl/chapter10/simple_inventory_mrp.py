from typing import Iterator, Sequence
from itertools import islice
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.function_approx import FunctionApprox
from rl.chapter10.prediction_utils import (
    mc_finite_prediction_equal_wts, mc_finite_prediction_learning_rate,
    td_finite_prediction_learning_rate
)
from rl.iterate import last
from rl.gen_utils.plot_funcs import plot_list_of_curves
from math import sqrt, log, ceil
import numpy as np
from pprint import pprint


def inventory_mrp(
    capacity: int,
    poisson_lambda: float,
    holding_cost: float,
    stockout_cost: float
) -> SimpleInventoryMRPFinite:
    return SimpleInventoryMRPFinite(
        capacity=capacity,
        poisson_lambda=poisson_lambda,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost
    )


def mc_equal_wts_correctness(
    inv_mrp: SimpleInventoryMRPFinite,
    gamma: float,
    tolerance: float,
    num_episodes=int
) -> None:
    mc_vfs: Iterator[FunctionApprox[InventoryState]] = \
        mc_finite_prediction_equal_wts(
            fmrp=inv_mrp,
            gamma=gamma,
            tolerance=tolerance
        )
    final_mc_vf: FunctionApprox[InventoryState] = \
        last(islice(mc_vfs, num_episodes))
    print(f"Equal Weights MC Value Function with {num_episodes:d} episodes")
    pprint({s: round(final_mc_vf(s), 3) for s in inv_mrp.non_terminal_states})
    print("True Value Function")
    inv_mrp.display_value_function(gamma=gamma)


def td_learning_rate_correctness(
    inv_mrp: SimpleInventoryMRPFinite,
    gamma: float,
    episode_length: int,
    num_episodes: int,
    base_learning_rate: float,
    learning_rate_decay: float
) -> None:
    td_vfs: Iterator[FunctionApprox[InventoryState]] = \
        td_finite_prediction_learning_rate(
            fmrp=inv_mrp,
            gamma=gamma,
            episode_length=episode_length,
            base_learning_rate=base_learning_rate,
            learning_rate_decay=learning_rate_decay
        )
    final_td_vf: FunctionApprox[InventoryState] = \
        last(islice(td_vfs, num_episodes))
    print("Constant Learning Rate TD Value Function with " +
          f"{num_episodes:d} episodes")
    pprint({s: round(final_td_vf(s), 3) for s in inv_mrp.non_terminal_states})
    print("True Value Function")
    inv_mrp.display_value_function(gamma=gamma)


def compare_td_and_mc(
    inv_mrp: SimpleInventoryMRPFinite,
    gamma: float,
    mc_episode_length_tol: float,
    num_episodes: int,
    base_learning_rate: float,
    learning_rate_decay: float,
    plot_batch: int
) -> None:
    true_vf: np.ndarray = inv_mrp.get_value_function_vec(gamma)
    states: Sequence[InventoryState] = inv_mrp.non_terminal_states
    mc_funcs_it: Iterator[FunctionApprox[InventoryState]] = \
        mc_finite_prediction_learning_rate(
            fmrp=inv_mrp,
            gamma=gamma,
            tolerance=mc_episode_length_tol,
            base_learning_rate=base_learning_rate,
            learning_rate_decay=learning_rate_decay
        )
    mc_errors = []
    batch_mc_errs = []
    for i, mc_f in enumerate(islice(mc_funcs_it, num_episodes)):
        batch_mc_errs.append(sqrt(sum((mc_f(s) - true_vf[j]) ** 2 for j, s in
                                      enumerate(states)) / len(states)))
        if i % plot_batch == plot_batch - 1:
            mc_errors.append(sum(batch_mc_errs) / plot_batch)
            batch_mc_errs = []

    td_episode_length = int(ceil(log(mc_episode_length_tol) / log(gamma)))
    td_funcs_it: Iterator[FunctionApprox[InventoryState]] = \
        td_finite_prediction_learning_rate(
            fmrp=inv_mrp,
            gamma=gamma,
            episode_length=td_episode_length,
            base_learning_rate=base_learning_rate,
            learning_rate_decay=learning_rate_decay
        )
    td_errors = []
    transitions_batch = plot_batch * td_episode_length
    batch_td_errs = []

    for i, td_f in enumerate(
            islice(td_funcs_it, num_episodes * td_episode_length)
    ):
        batch_td_errs.append(sqrt(sum((td_f(s) - true_vf[j]) ** 2 for j, s in
                                      enumerate(states)) / len(states)))
        if i % transitions_batch == transitions_batch - 1:
            td_errors.append(sum(batch_td_errs) / transitions_batch)
            batch_td_errs = []

    plot_start = 0
    mc_plot = mc_errors[plot_start:]
    td_plot = td_errors[plot_start:]

    plot_list_of_curves(
        [range(len(mc_plot)), range(len(td_plot))],
        [mc_plot, td_plot],
        ["b", "r"],
        ["Monte-Carlo", "Temporal-Difference"],
        x_label="Episode Batches",
        y_label="Value Function RMSE",
        title="RMSE of MC and TD as function of episode batches"
    )


if __name__ == '__main__':
    capacity: int = 2
    poisson_lambda: float = 1.0
    holding_cost: float = 1.0
    stockout_cost: float = 10.0

    gamma: float = 0.9
    mc_episode_length_tol: float = 1e-6
    num_episodes = 10000

    base_learning_rate: float = 0.1
    learning_rate_decay: float = 100

    si_mrp: SimpleInventoryMRPFinite = inventory_mrp(
        capacity=capacity,
        poisson_lambda=poisson_lambda,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost
    )

#     td_episode_length: int = 100
#     mc_equal_wts_correctness(
#         inv_mrp=si_mrp,
#         gamma=gamma,
#         tolerance=mc_episode_length_tol,
#         num_episodes=num_episodes
#     )
#     td_learning_rate_correctness(
#         inv_mrp=si_mrp,
#         gamma=gamma,
#         episode_length=td_episode_length,
#         num_episodes=num_episodes,
#         base_learning_rate=base_learning_rate,
#         learning_rate_decay=learning_rate_decay
#     )

    plot_batch: int = 100
    compare_td_and_mc(
        inv_mrp=si_mrp,
        gamma=gamma,
        mc_episode_length_tol=mc_episode_length_tol,
        num_episodes=num_episodes,
        base_learning_rate=base_learning_rate,
        learning_rate_decay=learning_rate_decay,
        plot_batch=plot_batch
    )
