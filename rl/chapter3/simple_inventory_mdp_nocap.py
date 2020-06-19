from typing import Tuple, Sequence
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import MarkovRewardProcess
from rl.markov_decision_process import Policy
from rl.distribution import Constant, SampledDistribution
import numpy as np

IntPair = Tuple[int, int]


class SimpleInventoryMRPNoCap(MarkovRewardProcess[IntPair]):

    def __init__(
        self,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float,
        policy: Policy[IntPair, int]
    ):
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost
        self.policy: Policy[IntPair, int] = policy

    def transition_reward(
        self,
        state: IntPair
    ) -> SampledDistribution[Tuple[IntPair, float]]:
        order = self.policy.act(state).sample()

        def sample_next_state_reward(
            state=state,
            order=order
        ) -> Tuple[IntPair, float]:
            demand_sample = np.random.poisson(self.poisson_lambda)
            ip = state[0] + state[1]
            next_state = (max(ip - demand_sample, 0), order)
            reward = - self.holding_cost * state[0]\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)


class SimpleInventoryMDPNoCap(MarkovDecisionProcess[IntPair, int]):

    def __init__(
        self,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def apply_policy(
        self,
        policy: Policy[IntPair, int]
    ) -> MarkovRewardProcess[IntPair]:
        return SimpleInventoryMRPNoCap(
            poisson_lambda=self.poisson_lambda,
            holding_cost=self.holding_cost,
            stockout_cost=self.stockout_cost,
            policy=policy
        )


class SimpleInventoryDeterministicPolicy(Policy[IntPair, int]):

    def __init__(self, reorder_point: int):
        self.reorder_point: int = reorder_point

    def act(self, state: IntPair) -> Constant[int]:
        return Constant(
            max(self.reorder_point - (state[0] + state[1]), 0)
        )


class SimpleInventoryStochasticPolicy(Policy[IntPair, int]):

    def __init__(self, reorder_point_poisson_mean: float):
        self.reorder_point_poisson_mean: float = reorder_point_poisson_mean

    def act(self, state: IntPair) -> SampledDistribution[int]:

        def action_func(state=state) -> int:
            reorder_point_sample: int = \
                np.random.poisson(self.reorder_point_poisson_mean)
            return max(reorder_point_sample - (state[0] + state[1]), 0)

        return SampledDistribution(action_func)


if __name__ == '__main__':
    import itertools
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_reorder_point = 2
    user_reorder_point_poisson_mean = 2.0

    user_start_state = (5, 2)

    user_time_steps = 100

    si_dp = SimpleInventoryDeterministicPolicy(
        reorder_point=user_reorder_point
    )

    si_mdp_nocap = SimpleInventoryMDPNoCap(
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    si_mrp_nocap_dp = si_mdp_nocap.apply_policy(si_dp)

    trace_dp: Sequence[Tuple[IntPair, float]] = list(itertools.islice(
        si_mrp_nocap_dp.simulate_reward(user_start_state),
        user_time_steps + 1
    ))
    print(trace_dp)

    si_sp = SimpleInventoryStochasticPolicy(
        reorder_point_poisson_mean=user_reorder_point_poisson_mean
    )

    si_mrp_nocap_sp = si_mdp_nocap.apply_policy(si_sp)

    trace_sp: Sequence[Tuple[IntPair, float]] = list(itertools.islice(
        si_mrp_nocap_sp.simulate_reward(user_start_state),
        user_time_steps + 1
    ))
    print(trace_sp)
