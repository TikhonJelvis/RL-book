import unittest
from rl.approximate_dynamic_programming import (
    evaluate_finite_mrp, evaluate_mrp, value_iteration_finite, value_iteration)
from rl.dynamic_programming import value_iteration_result
from typing import Sequence, Mapping
import numpy as np
from rl.distribution import Constant, Choose
from rl.function_approx import Dynamic, FunctionApprox
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FinitePolicy)
from rl.chapter3.simple_inventory_mdp_cap import (InventoryState,
                                                  SimpleInventoryMDPCap)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        user_capacity = 2
        user_poisson_lambda = 1.0
        user_holding_cost = 1.0
        user_stockout_cost = 10.0

        self.gamma = 0.9

        self.si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
            SimpleInventoryMDPCap(
                capacity=user_capacity,
                poisson_lambda=user_poisson_lambda,
                holding_cost=user_holding_cost,
                stockout_cost=user_stockout_cost
            )

        self.fdp: FinitePolicy[InventoryState, int] = FinitePolicy(
            {InventoryState(alpha, beta):
             Constant(user_capacity - (alpha + beta)) for alpha in
             range(user_capacity + 1) for beta in
             range(user_capacity + 1 - alpha)}
        )

        self.implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
            self.si_mdp.apply_finite_policy(self.fdp)

        self.states: Sequence[InventoryState] = \
            self.implied_mrp.non_terminal_states

    def test_evaluate_mrp(self):
        mrp_vf1: np.ndarray = self.implied_mrp.get_value_function_vec(
            self.gamma
        )
        # print({s: mrp_vf1[i] for i, s in enumerate(self.states)})

        fa = Dynamic({s: 0.0 for s in self.states})
        mrp_finite_fa = FunctionApprox.converged(
            evaluate_finite_mrp(
                self.implied_mrp,
                self.gamma,
                fa
            )
        )
        # print(mrp_finite_fa.values_map)
        mrp_vf2: np.ndarray = mrp_finite_fa.evaluate(self.states)

        self.assertLess(max(abs(mrp_vf1 - mrp_vf2)), 0.001)

        mrp_fa = FunctionApprox.converged(
            evaluate_mrp(
                self.implied_mrp,
                self.gamma,
                fa,
                Choose(self.states),
                num_state_samples=30
            ),
            0.1
        )
        # print(mrp_fa.values_map)
        mrp_vf3: np.ndarray = mrp_fa.evaluate(self.states)
        self.assertLess(max(abs(mrp_vf1 - mrp_vf3)), 1.0)

    def test_value_iteration(self):
        mdp_map: Mapping[InventoryState, float] = value_iteration_result(
            self.si_mdp,
            self.gamma
        )[0]
        # print(mdp_map)
        mdp_vf1: np.ndarray = np.array([mdp_map[s] for s in self.states])

        fa = Dynamic({s: 0.0 for s in self.states})
        mdp_finite_fa = FunctionApprox.converged(
            value_iteration_finite(
                self.si_mdp,
                self.gamma,
                fa
            )
        )
        # print(mdp_finite_fa.values_map)
        mdp_vf2: np.ndarray = mdp_finite_fa.evaluate(self.states)

        self.assertLess(max(abs(mdp_vf1 - mdp_vf2)), 0.001)

        mdp_fa = FunctionApprox.converged(
            value_iteration(
                self.si_mdp,
                self.gamma,
                fa,
                Choose(self.states),
                num_state_samples=30
            ),
            0.1
        )
        # print(mdp_fa.values_map)
        mdp_vf3: np.ndarray = mdp_fa.evaluate(self.states)
        self.assertLess(max(abs(mdp_vf1 - mdp_vf3)), 1.0)
