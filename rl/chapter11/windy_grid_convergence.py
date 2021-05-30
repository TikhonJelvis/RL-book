from rl.chapter11.windy_grid import WindyGrid, Cell, Move
from rl.chapter11.control_utils import compare_mc_sarsa_ql
from rl.markov_decision_process import FiniteMarkovDecisionProcess

wg = WindyGrid(
    rows=5,
    columns=5,
    blocks={(0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (4, 0)},
    terminals={(3, 4)},
    wind=[(0., 0.9), (0.0, 0.8), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)],
    bump_cost=100000.0
)
valid = wg.validate_spec()
if valid:
    fmdp: FiniteMarkovDecisionProcess[Cell, Move] = wg.get_finite_mdp()
    compare_mc_sarsa_ql(
        fmdp=fmdp,
        method_mask=[False, True, True],
        learning_rates=[(0.03, 1e8, 1.0)],
        gamma=1.,
        epsilon_as_func_of_episodes=lambda k: 1. / k,
        q_learning_epsilon=0.2,
        mc_episode_length_tol=1e-5,
        num_episodes=400,
        plot_batch=10,
        plot_start=0
    )
