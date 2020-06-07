from typing import Mapping, Tuple
from rl.markov_process import FiniteMarkovRewardProcess
from scipy.stats import poisson

IntPair = Tuple[int, int]
TransType = Mapping[IntPair, Mapping[Tuple[IntPair, float], float]]


class SimpleInventory:

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
        self.transition_reward_map: TransType = self.get_transition_reward_map()

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

    def get_finite_markov_reward_process(self) -> FiniteMarkovRewardProcess:
        return FiniteMarkovRewardProcess(self.transition_reward_map)


if __name__ == '__main__':
    from pprint import pprint
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = -1.0
    user_stockout_cost = -10.0

    user_gamma = 0.9

    si = SimpleInventory(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    fmrp = si.get_finite_markov_reward_process()
    print("Transition Rewards Map")
    pprint(fmrp.transition_reward_map)
    print("Transition Map")
    pprint(fmrp.transition_map)

    stationary_distribution = {
        s: p for s, p in fmrp.get_stationary_distribution().to_pdf()
    }
    print("Stationary Distribution")
    pprint(stationary_distribution)

    rewards_function = {
        fmrp.state_space[i]: r for i, r in enumerate(fmrp.reward_vec)
    }
    print("Rewards Function")
    pprint(rewards_function)

    value_function = {fmrp.state_space[i]: v for i, v
                      in enumerate(fmrp.value_function_vec(gamma=user_gamma))}
    print("Value Function")
    pprint(value_function)



