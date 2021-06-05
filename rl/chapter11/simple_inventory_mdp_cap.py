from typing import Tuple, Callable, Sequence
from rl.chapter11.control_utils import glie_mc_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    q_learning_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    glie_sarsa_finite_learning_rate_correctness
from rl.chapter11.control_utils import compare_mc_sarsa_ql
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap


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
mc_episode_length_tol: float = 1e-5
num_episodes = 10000

epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: k ** -0.5
q_learning_epsilon: float = 0.2

td_episode_length: int = 100
initial_learning_rate: float = 0.1
half_life: float = 10000.0
exponent: float = 1.0

glie_mc_finite_learning_rate_correctness(
    fmdp=si_mdp,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    episode_length_tolerance=mc_episode_length_tol,
    num_episodes=num_episodes
)

glie_sarsa_finite_learning_rate_correctness(
    fmdp=si_mdp,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    max_episode_length=td_episode_length,
    num_updates=num_episodes * td_episode_length
)

q_learning_finite_learning_rate_correctness(
    fmdp=si_mdp,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    gamma=gamma,
    epsilon=q_learning_epsilon,
    max_episode_length=td_episode_length,
    num_updates=num_episodes * td_episode_length
)

num_episodes = 500
plot_batch: int = 10
plot_start: int = 0
learning_rates: Sequence[Tuple[float, float, float]] = \
    [(0.05, 1000000, 0.5)]

compare_mc_sarsa_ql(
    fmdp=si_mdp,
    method_mask=[True, True, False],
    learning_rates=learning_rates,
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    q_learning_epsilon=q_learning_epsilon,
    mc_episode_length_tol=mc_episode_length_tol,
    num_episodes=num_episodes,
    plot_batch=plot_batch,
    plot_start=plot_start
)
