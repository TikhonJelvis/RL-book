from typing import Tuple, Dict, List
from rl.distribution import Categorical
from rl.markov_process import Transition, FiniteMarkovProcess
from scipy.stats import poisson

IntPair = Tuple[int, int]


class SimpleInventoryMPFinite(FiniteMarkovProcess[IntPair]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Transition[IntPair]:
        d: Dict[IntPair, Categorical[IntPair]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                ip = alpha + beta
                beta1 = max(self.capacity - ip, 0)
                state_probs_list: List[Tuple[IntPair, float]] = [
                    ((ip - i, beta1), self.poisson_distr.pmf(i))
                    for i in range(ip)
                ]
                state_probs_list.append(
                    ((0, beta1), 1 - self.poisson_distr.cdf(ip - 1))
                )
                d[(alpha, beta)] = Categorical(state_probs_list)
        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0

    si_mp = SimpleInventoryMPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda
    )

    print("Transition Map")
    print("--------------")
    print(si_mp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mp.display_stationary_distribution()
