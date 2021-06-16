import unittest

import numpy as np

from rl.distribution import Choose
from rl.finite_horizon import (
    unwrap_finite_horizon_MRP, finite_horizon_MRP, evaluate,
    unwrap_finite_horizon_MDP, finite_horizon_MDP, optimal_vf_and_policy)
from rl.function_approx import Dynamic, Tabular
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy

from rl.chapter4.clearance_pricing_mdp import ClearancePricingMDP

from rl.approximate_dynamic_programming import (
    backward_evaluate_finite, backward_evaluate,
    back_opt_vf_and_policy_finite, back_opt_vf_and_policy)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        ii = 10
        self.steps = 6
        pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]
        self.cp: ClearancePricingMDP = ClearancePricingMDP(
            initial_inventory=ii,
            time_steps=self.steps,
            price_lambda_pairs=pairs
        )

        def policy_func(x: int) -> int:
            return 0 if x < 2 else (1 if x < 5 else (2 if x < 8 else 3))

        stationary_policy: FiniteDeterministicPolicy[int, int] = \
            FiniteDeterministicPolicy(
                {s: policy_func(s) for s in range(ii + 1)}
            )

        self.single_step_mrp: FiniteMarkovRewardProcess[int] = \
            self.cp.single_step_mdp.apply_finite_policy(stationary_policy)

        self.mrp_seq = unwrap_finite_horizon_MRP(
            finite_horizon_MRP(self.single_step_mrp, self.steps)
        )

        self.single_step_mdp: FiniteMarkovDecisionProcess[int, int] = \
            self.cp.single_step_mdp

        self.mdp_seq = unwrap_finite_horizon_MDP(
            finite_horizon_MDP(self.single_step_mdp, self.steps)
        )

    def test_evaluate_mrp(self):
        vf = evaluate(self.mrp_seq, 1.)
        states = self.single_step_mrp.non_terminal_states
        fa_dynamic = Dynamic({s: 0.0 for s in states})
        fa_tabular = Tabular()
        distribution = Choose(states)
        approx_vf_finite = backward_evaluate_finite(
            [(self.mrp_seq[i], fa_dynamic) for i in range(self.steps)],
            1.
        )
        approx_vf = backward_evaluate(
            [(self.single_step_mrp, fa_tabular, distribution)
             for _ in range(self.steps)],
            1.,
            num_state_samples=120,
            error_tolerance=0.01
        )

        for t, (v1, v2, v3) in enumerate(zip(
                vf,
                approx_vf_finite,
                approx_vf
        )):
            states = self.mrp_seq[t].keys()
            v1_arr = np.array([v1[s] for s in states])
            v2_arr = v2.evaluate(states)
            v3_arr = v3.evaluate(states)
            self.assertLess(max(abs(v1_arr - v2_arr)), 0.001)
            self.assertLess(max(abs(v1_arr - v3_arr)), 1.0)

    def test_value_iteration(self):
        vpstar = optimal_vf_and_policy(self.mdp_seq, 1.)
        states = self.single_step_mdp.non_terminal_states
        fa_dynamic = Dynamic({s: 0.0 for s in states})
        fa_tabular = Tabular()
        distribution = Choose(states)
        approx_vpstar_finite = back_opt_vf_and_policy_finite(
            [(self.mdp_seq[i], fa_dynamic) for i in range(self.steps)],
            1.
        )
        approx_vpstar = back_opt_vf_and_policy(
            [(self.single_step_mdp, fa_tabular, distribution)
             for _ in range(self.steps)],
            1.,
            num_state_samples=120,
            error_tolerance=0.01
        )

        for t, ((v1, _), (v2, _), (v3, _)) in enumerate(zip(
                vpstar,
                approx_vpstar_finite,
                approx_vpstar
        )):
            states = self.mdp_seq[t].keys()
            v1_arr = np.array([v1[s] for s in states])
            v2_arr = v2.evaluate(states)
            v3_arr = v3.evaluate(states)
            self.assertLess(max(abs(v1_arr - v2_arr)), 0.001)
            self.assertLess(max(abs(v1_arr - v3_arr)), 1.0)
