from typing import Mapping, Tuple
from rl.markov_process import FiniteMarkovProcess
from scipy.stats import poisson

IntPair = Tuple[int, int]
MPTransType = Mapping[IntPair, Mapping[IntPair, float]]


class SimpleInventoryMP(FiniteMarkovProcess[IntPair]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> MPTransType:
        d = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                ip = alpha + beta
                d1 = {}
                beta1 = max(self.capacity - ip, 0)
                for i in range(ip):
                    next_state = (ip - i, beta1)
                    probability = self.poisson_distr.pmf(i)
                    d1[next_state] = probability
                next_state = (0, beta1)
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                d1[next_state] = probability
                d[(alpha, beta)] = d1
        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0

    si_mp = SimpleInventoryMP(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda
    )

    print("Transition Map")
    print("--------------")
    print(si_mp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mp.display_stationary_distribution()
