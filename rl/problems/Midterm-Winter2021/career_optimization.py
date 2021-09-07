from typing import Tuple, Mapping, Dict, Sequence, Iterable
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.dynamic_programming import value_iteration_result
from rl.distribution import Categorical
from scipy.stats import poisson

IntPair = Tuple[int, int]
CareerDecisionsMap = Mapping[int, Mapping[
    IntPair,
    Categorical[Tuple[int, float]]
]]


class CareerOptimization(FiniteMarkovDecisionProcess[int, IntPair]):

    def __init__(
        self,
        hours: int,
        wage_cap: int,
        alpha: float,
        beta: float
    ):
        self.hours = hours
        self.wage_cap = wage_cap
        self.alpha = alpha
        self.beta = beta
        super().__init__(self.get_transitions())

    def get_transitions(self) -> CareerDecisionsMap:
        d: Dict[int, Mapping[IntPair, Categorical[Tuple[int, float]]]] = {}
        for w in range(1, self.wage_cap + 1):
            d1: Dict[IntPair, Categorical[Tuple[int, float]]] = {}
            for s in range(self.hours + 1):
                for t in range(self.hours + 1 - s):
                    pd = poisson(self.alpha * t)
                    prob: float = self.beta * s / self.hours
                    r: float = w * (self.hours - s - t)
                    same_prob: float = (1 - prob) * pd.pmf(0)
                    sr_probs: Dict[Tuple[int, float], float] = {}
                    if w == self.wage_cap:
                        sr_probs[(w, r)] = 1.
                    elif w == self.wage_cap - 1:
                        sr_probs[(w, r)] = same_prob
                        sr_probs[(w + 1, r)] = 1 - same_prob
                    else:
                        sr_probs[(w, r)] = same_prob
                        sr_probs[(w + 1, r)] = prob * pd.pmf(0) + pd.pmf(1)
                        for w1 in range(w + 2, self.wage_cap):
                            sr_probs[(w1, r)] = pd.pmf(w1 - w)
                        sr_probs[(self.wage_cap, r)] = \
                            1 - pd.cdf(self.wage_cap - w - 1)
                    d1[(s, t)] = Categorical(sr_probs)
            d[w] = d1
        return d


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from pprint import pprint
    hours: int = 10
    wage_cap: int = 30
    alpha: float = 0.08
    beta: float = 0.82
    gamma: float = 0.95

    co: CareerOptimization = CareerOptimization(
        hours=hours,
        wage_cap=wage_cap,
        alpha=alpha,
        beta=beta
    )

    _, opt_det_policy = value_iteration_result(co, gamma=gamma)
    wages: Iterable[int] = range(1, co.wage_cap + 1)
    opt_actions: Mapping[int, Tuple[int, int]] = \
        {w: opt_det_policy.action_for[w]
         for w in wages}
    searching: Sequence[int] = [s for _, (s, _) in opt_actions.items()]
    learning: Sequence[int] = [l for _, (_, l) in opt_actions.items()]
    working: Sequence[int] = [co.hours - s - l for _, (s, l) in
                              opt_actions.items()]
    pprint(opt_actions)
    plt.xticks(wages)
    p1 = plt.bar(wages, searching, color='red')
    p2 = plt.bar(wages, learning, color='blue')
    p3 = plt.bar(wages, working, color='green')
    plt.legend((p1[0], p2[0], p3[0]), ('Job-Searching', 'Learning', 'Working'))
    plt.grid(axis='y')
    plt.xlabel("Hourly Wage Level")
    plt.ylabel("Hours Spent")
    plt.title("Career Optimization")
    plt.show()
