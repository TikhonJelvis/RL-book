from dataclasses import dataclass
from typing import Tuple, Mapping, Dict, List
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, ActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical, Constant
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


InvOrderMapping = Mapping[InventoryState, ActionMapping[int, InventoryState]]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

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
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                base_reward = - self.holding_cost * alpha
                d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}

                for order in range(max(self.capacity - ip, 0) + 1):
                    sr_probs_list: List[Tuple[Tuple[InventoryState, float],
                                              float]] =\
                        [((InventoryState(ip - i, order), base_reward),
                          self.poisson_distr.pmf(i)) for i in range(ip)]

                    probability = 1 - self.poisson_distr.cdf(ip - 1)
                    reward = base_reward - self.stockout_cost *\
                        (probability * (self.poisson_lambda - ip) +
                         ip * self.poisson_distr.pmf(ip))
                    sr_probs_list.append(
                        ((InventoryState(0, order), reward), probability)
                    )
                    d1[order] = Categorical(sr_probs_list)

                d[state] = d1
        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    fdp: FinitePolicy[InventoryState, int] = FinitePolicy(
        {InventoryState(alpha, beta):
         Constant(max(user_capacity - (alpha + beta), 0)) for alpha in
         range(user_capacity + 1) for beta in range(user_capacity + 1 - alpha)}
    )

    print("Policy Map")
    print("----------")
    print(fdp)

    implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
        si_mdp.apply_finite_policy(fdp)
    print("Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(implied_mrp.transition_map))

    print("Transition Reward Map")
    print("---------------------")
    print(implied_mrp)

    print("Stationary Distribution")
    print("-----------------------")
    implied_mrp.display_stationary_distribution()
    print()

    print("Reward Function")
    print("---------------")
    implied_mrp.display_reward_function()
    print()

    print("Value Function")
    print("--------------")
    implied_mrp.display_value_function(gamma=user_gamma)
    print()
