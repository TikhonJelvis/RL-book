from typing import Mapping, Tuple
from rl.markov_process import FiniteMarkovRewardProcess
from scipy.stats import poisson

IntPair = Tuple[int, int]
TransType = Mapping[IntPair, Mapping[Tuple[IntPair, float], float]]


class SimpleInventoryMRP(FiniteMarkovRewardProcess[IntPair]):

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

    def get_transition_reward_map(self) -> TransType:
        d = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                ip = alpha + beta
                d1 = {}
                beta1 = max(self.capacity - ip, 0)
                for i in range(ip):
                    next_state = (ip - i, beta1)
                    reward = self.holding_cost * alpha
                    probability = self.poisson_distr.pmf(i)
                    d1[(next_state, reward)] = probability
                next_state = (0, beta1)
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = self.holding_cost * alpha + self.stockout_cost *\
                    (probability * (self.poisson_lambda - ip) +
                     ip * self.poisson_distr.pmf(ip))
                d1[(next_state, reward)] = probability
                d[(alpha, beta)] = d1
        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = -1.0
    user_stockout_cost = -10.0

    user_gamma = 0.9

    si = SimpleInventoryMRP(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    from rl.markov_process import FiniteMarkovProcess
    print("Transition Map")
    print(FiniteMarkovProcess(si.transition_map))

    print("Transition Reward Map")
    print(si)

    print("Stationary Distribution")
    si.display_stationary_distribution()

    print("Reward Function")
    si.display_reward_function()

    print("Value Function")
    si.display_value_function(gamma=user_gamma)
