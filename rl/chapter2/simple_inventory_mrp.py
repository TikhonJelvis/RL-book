from typing import Mapping, Tuple, Dict, List
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import RewardTransition
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical
import numpy as np

IntPair = Tuple[int, int]


class SimpleInventoryMRP(MarkovRewardProcess[IntPair]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def transition_reward(
        self,
        state: IntPair
    ) -> SampledDistribution[Tuple[IntPair, float]]:

        def sample_next_state_reward(
            state=state
        ) -> Tuple[IntPair, float]:
            demand_sample = np.random.poisson(self.poisson_lambda)
            alpha, beta = state
            ip = alpha + beta
            next_state = (
                max(ip - demand_sample, 0),
                max(self.capacity - ip, 0)
            )
            reward = - self.holding_cost * alpha\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)


class SimpleInventoryMRPFinite(FiniteMarkovRewardProcess[IntPair]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> RewardTransition[IntPair]:
        d: Dict[IntPair, Categorical[Tuple[IntPair, float]]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                ip = alpha + beta
                beta1 = max(self.capacity - ip, 0)
                base_reward = - self.holding_cost * alpha
                sr_probs_list: List[Tuple[Tuple[IntPair, float], float]] = [
                    (((ip - i, beta1), base_reward),
                     self.poisson_distr.pmf(i)) for i in range(ip)
                ]
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = base_reward - self.stockout_cost *\
                    (probability * (self.poisson_lambda - ip) +
                     ip * self.poisson_distr.pmf(ip))
                sr_probs_list.append(
                    (((0, beta1), reward), probability)
                )
                d[(alpha, beta)] = Categorical(sr_probs_list)
        return d


if __name__ == '__main__':
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

    from rl.markov_process import FiniteMarkovProcess
    print("Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(si_mrp.transition_map))

    print("Transition Reward Map")
    print("---------------------")
    print(si_mrp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mrp.display_stationary_distribution()
    print()

    print("Reward Function")
    print("---------------")
    si_mrp.display_reward_function()
    print()

    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
