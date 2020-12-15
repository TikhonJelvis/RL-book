from typing import Iterator
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.function_approx import Tabular, FunctionApprox
from rl.distribution import Choose
from rl.iterate import last
from rl.monte_carlo import evaluate_mrp
from itertools import islice
from pprint import pprint


user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0

user_gamma = 0.9

si_mrp = SimpleInventoryMRPFinite(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda,
    holding_cost=user_holding_cost,
    stockout_cost=user_stockout_cost
)

si_mrp.display_value_function(gamma=user_gamma)

it: Iterator[FunctionApprox[InventoryState]] = evaluate_mrp(
    mrp=si_mrp,
    states=Choose(set(si_mrp.states())),
    approx_0=Tabular(),
    γ=user_gamma,
    tolerance=1e-6
)

num_traces = 100000

last_func: FunctionApprox[InventoryState] = last(islice(it, num_traces))
pprint({s: round(last_func.evaluate([s])[0], 3)
        for s in si_mrp.non_terminal_states})
