from typing import Mapping, Iterator, Iterable
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
from rl.markov_process import NonTerminal, TransitionStep
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.td_lambda import lambda_return_prediction
import rl.iterate as iterate
import itertools
from pprint import pprint


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mrp: SimpleInventoryMRPFinite = SimpleInventoryMRPFinite(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)
initial_vf_dict: Mapping[NonTerminal[InventoryState], float] = \
    {s: 0. for s in si_mrp.non_terminal_states}

gamma: float = 0.9
lambda_param = 0.3
num_episodes = 10000

episode_length: int = 100
initial_learning_rate: float = 0.03
half_life: float = 1000.0
exponent: float = 0.5

approx_0: Tabular[NonTerminal[InventoryState]] = Tabular(
    values_map=initial_vf_dict,
    count_to_weight_func=learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
)

episodes: Iterable[Iterable[TransitionStep[InventoryState]]] = \
    si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))
traces: Iterable[Iterable[TransitionStep[InventoryState]]] = \
        (itertools.islice(episode, episode_length) for episode in episodes)

vf_iter: Iterator[Tabular[NonTerminal[InventoryState]]] = \
    lambda_return_prediction(
        traces=traces,
        approx_0=approx_0,
        γ=gamma,
        lambd=lambda_param
    )

vf: Tabular[NonTerminal[InventoryState]] = \
    iterate.last(itertools.islice(vf_iter, num_episodes))

pprint(vf.values_map)
si_mrp.display_value_function(gamma=gamma)





