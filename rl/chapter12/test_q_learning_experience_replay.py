from typing import Iterator
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, \
    InventoryState
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.monte_carlo import epsilon_greedy_policy
from rl.td import q_learning_experience_replay
from rl.dynamic_programming import value_iteration_result
import rl.iterate as iterate
import itertools
from pprint import pprint


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mdp: SimpleInventoryMDPCap = SimpleInventoryMDPCap(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)

gamma: float = 0.9
epsilon: float = 0.3

initial_learning_rate: float = 0.1
learning_rate_half_life: float = 1000
learning_rate_exponent: float = 0.5

episode_length: int = 100
mini_batch_size: int = 1000
time_decay_half_life: float = 3000
num_updates: int = 10000

q_iter: Iterator[QValueFunctionApprox[InventoryState, int]] = \
    q_learning_experience_replay(
        mdp=si_mdp,
        policy_from_q=lambda f, m: epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=epsilon
        ),
        states=Choose(si_mdp.non_terminal_states),
        approx_0=Tabular(
            count_to_weight_func=learning_rate_schedule(
                initial_learning_rate=initial_learning_rate,
                half_life=learning_rate_half_life,
                exponent=learning_rate_exponent
            )
        ),
        γ=gamma,
        max_episode_length=episode_length,
        mini_batch_size=mini_batch_size,
        weights_decay_half_life=time_decay_half_life
    )

qvf: QValueFunctionApprox[InventoryState, int] = iterate.last(
    itertools.islice(
        q_iter,
        num_updates
    )
)
vf, pol = get_vf_and_policy_from_qvf(mdp=si_mdp, qvf=qvf)
pprint(vf)
print(pol)

true_vf, true_pol = value_iteration_result(mdp=si_mdp, gamma=gamma)
pprint(true_vf)
print(true_pol)
