from dataclasses import dataclass
from typing import Tuple, Iterator
import itertools
import numpy as np
from scipy.stats import poisson
import random

from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import MarkovRewardProcess
from rl.markov_decision_process import Policy
from rl.distribution import Constant, SampledDistribution


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class SimpleInventoryMDPNoCap(MarkovDecisionProcess[InventoryState, int]):
    def __init__(self, poisson_lambda: float, holding_cost: float,
                 stockout_cost: float):
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def step(
        self,
        state: InventoryState,
        order: int
    ) -> SampledDistribution[Tuple[InventoryState, float]]:

        def sample_next_state_reward(
            state=state,
            order=order
        ) -> Tuple[InventoryState, float]:
            demand_sample: int = np.random.poisson(self.poisson_lambda)
            ip: int = state.inventory_position()
            next_state: InventoryState = InventoryState(
                max(ip - demand_sample, 0),
                order
            )
            reward: float = - self.holding_cost * state.on_hand\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)

    def actions(self, state: InventoryState) -> Iterator[int]:
        return itertools.count(start=0, step=1)

    def fraction_of_days_oos(
        self,
        policy: Policy[InventoryState, int],
        time_steps: int,
        num_traces: int
    ) -> float:
        impl_mrp: MarkovRewardProcess[InventoryState] =\
            self.apply_policy(policy)
        count: int = 0
        high_fractile: int = int(poisson(self.poisson_lambda).ppf(0.98))
        start: InventoryState = random.choice(
            [InventoryState(i, 0) for i in range(high_fractile + 1)])

        for _ in range(num_traces):
            steps = itertools.islice(
                impl_mrp.simulate_reward(Constant(start)),
                time_steps
            )
            for step in steps:
                if step.reward < -self.holding_cost * step.next_state.on_hand:
                    count += 1

        return float(count) / (time_steps * num_traces)


class SimpleInventoryDeterministicPolicy(Policy[InventoryState, int]):
    def __init__(self, reorder_point: int):
        self.reorder_point: int = reorder_point

    def act(self, state: InventoryState) -> Constant[int]:
        return Constant(max(self.reorder_point - state.inventory_position(),
                            0))


class SimpleInventoryStochasticPolicy(Policy[InventoryState, int]):
    def __init__(self, reorder_point_poisson_mean: float):
        self.reorder_point_poisson_mean: float = reorder_point_poisson_mean

    def act(self, state: InventoryState) -> SampledDistribution[int]:
        def action_func(state=state) -> int:
            reorder_point_sample: int = \
                np.random.poisson(self.reorder_point_poisson_mean)
            return max(reorder_point_sample - state.inventory_position(), 0)

        return SampledDistribution(action_func)


if __name__ == '__main__':
    user_poisson_lambda = 2.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_reorder_point = 8
    user_reorder_point_poisson_mean = 8.0

    user_time_steps = 1000
    user_num_traces = 1000

    si_mdp_nocap = SimpleInventoryMDPNoCap(poisson_lambda=user_poisson_lambda,
                                           holding_cost=user_holding_cost,
                                           stockout_cost=user_stockout_cost)

    si_dp = SimpleInventoryDeterministicPolicy(
        reorder_point=user_reorder_point
    )

    oos_frac_dp = si_mdp_nocap.fraction_of_days_oos(policy=si_dp,
                                                    time_steps=user_time_steps,
                                                    num_traces=user_num_traces)
    print(
        f"Deterministic Policy yields {oos_frac_dp * 100:.2f}%"
        + " of Out-Of-Stock days"
    )

    si_sp = SimpleInventoryStochasticPolicy(
        reorder_point_poisson_mean=user_reorder_point_poisson_mean)

    oos_frac_sp = si_mdp_nocap.fraction_of_days_oos(policy=si_sp,
                                                    time_steps=user_time_steps,
                                                    num_traces=user_num_traces)
    print(
        f"Stochastic Policy yields {oos_frac_sp * 100:.2f}%"
        + " of Out-Of-Stock days"
    )
