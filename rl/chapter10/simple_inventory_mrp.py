from typing import Iterator
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.function_approx import Tabular, FunctionApprox
from rl.distribution import Choose
from rl.monte_carlo import evaluate_mrp
from rl.td import td_0
from itertools import islice
from rl.gen_utils.plot_funcs import plot_list_of_curves
from math import sqrt, log, ceil


user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0

user_gamma = 0.9

num_traces = 1000
pool = 1

tol = 1e-6

si_mrp = SimpleInventoryMRPFinite(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda,
    holding_cost=user_holding_cost,
    stockout_cost=user_stockout_cost
)

states = si_mrp.non_terminal_states
true_vf = si_mrp.get_value_function_vec(gamma=user_gamma)

mc_funcs_it: Iterator[FunctionApprox[InventoryState]] = evaluate_mrp(
    mrp=si_mrp,
    states=Choose(set(si_mrp.states())),
    approx_0=Tabular(),
    γ=user_gamma,
    tolerance=tol
)

mc_errors = []
pool_mc_errs = []
for i, mc_f in enumerate(islice(mc_funcs_it, 0, num_traces)):
    pool_mc_errs.append(sqrt(sum((mc_f(s) - true_vf[j]) ** 2 for j, s in
                        enumerate(states)) / len(states)))
    if i % pool == pool - 1:
        mc_errors.append(sum(pool_mc_errs) / pool)
        pool_mc_errs = []

td_episode_length = int(ceil(log(tol) / log(user_gamma)))
alpha = 0.03

td_funcs_it: Iterator[FunctionApprox[InventoryState]] = td_0(
    mrp=si_mrp,
    states=Choose(set(si_mrp.states())),
    approx_0=Tabular(count_to_weight_func=lambda _: alpha),
    γ=user_gamma,
    episode_length=td_episode_length
)

td_errors = []
trans_pool = pool * td_episode_length
pool_td_errs = []

for i, td_f in enumerate(
        islice(td_funcs_it, 0, num_traces * td_episode_length)
):
    pool_td_errs.append(sqrt(sum((td_f(s) - true_vf[j]) ** 2 for j, s in
                        enumerate(states)) / len(states)))
    if i % trans_pool == trans_pool - 1:
        td_errors.append(sum(pool_td_errs) / trans_pool)
        pool_td_errs = []

plot_start = 50
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
