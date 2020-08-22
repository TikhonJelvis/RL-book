from rl.markov_decision_process import (
    FiniteMarkovDecisionProcess, FinitePolicy)
from rl.finite_horizon import WithTime
from typing import Sequence, Tuple, Iterator
from scipy.stats import poisson
from rl.distribution import Categorical
from rl.finite_horizon import (
    finite_horizon_MDP, unwrap_finite_horizon_MDP, optimal_vf_and_policy)
from rl.dynamic_programming import V


class ClearancePricingMDP:

    initial_inventory: int
    time_steps: int
    price_lambda_pairs: Sequence[Tuple[float, float]]
    mdp: FiniteMarkovDecisionProcess[WithTime[int], int]

    def __init__(
        self,
        initial_inventory: int,
        time_steps: int,
        price_lambda_pairs: Sequence[Tuple[float, float]]
    ):
        self.initial_inventory = initial_inventory
        self.time_steps = time_steps
        self.price_lambda_pairs = price_lambda_pairs
        distrs = [poisson(l) for _, l in price_lambda_pairs]
        prices = [p for p, _ in price_lambda_pairs]
        stationary_mdp: FiniteMarkovDecisionProcess[int, int] =\
            FiniteMarkovDecisionProcess({
                s: {i: Categorical(
                    {(s - k, prices[i] * k):
                     (distrs[i].pmf(k) if k < s else 1 - distrs[i].cdf(s - 1))
                     for k in range(s + 1)})
                    for i in range(len(prices))}
                for s in range(initial_inventory + 1)
            })
        self.mdp = finite_horizon_MDP(stationary_mdp, time_steps)

    def get_optimal_vf_and_policy(self)\
            -> Iterator[Tuple[V[int], FinitePolicy[int, int]]]:
        return optimal_vf_and_policy(unwrap_finite_horizon_MDP(self.mdp))


if __name__ == '__main__':
    ii = 12
    steps = 8
    pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]
    cp: ClearancePricingMDP = ClearancePricingMDP(
        initial_inventory=ii,
        time_steps=steps,
        price_lambda_pairs=pairs
    )
    print(cp.mdp)
    for t, (vf, policy) in enumerate(cp.get_optimal_vf_and_policy()):
        print(f"Time Step {t:d}")
        print("---------------")
        print(vf)
        print(policy)
