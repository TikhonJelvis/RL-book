from numpy.testing import assert_allclose
import unittest

from rl.approximate_dynamic_programming import (evaluate_mrp,
                                                evaluate_finite_mrp)
from rl.distribution import Categorical, Choose
from rl.finite_horizon import (finite_horizon_MRP, evaluate,
                               unwrap_finite_horizon_MRP, WithTime)
from rl.function_approx import Dynamic
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
import rl.iterate as iterate


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    '''A version of FlipFlop implemented with the FiniteMarkovProcess
    machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FlipFlop(0.7)

    def test_evaluate_finite_mrp(self):
        start = Dynamic({s: 0.0 for s in
                         self.finite_flip_flop.non_terminal_states})
        v = iterate.converged(
            evaluate_finite_mrp(
                self.finite_flip_flop,
                γ=0.99,
                approx_0=start
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        self.assertEqual(len(v.values_map), 2)

        for s in v.values_map:
            self.assertLess(abs(v(s) - 170), 0.1)

    def test_evaluate_mrp(self):
        start = Dynamic({s: 0.0 for s in
                         self.finite_flip_flop.non_terminal_states})

        v = iterate.converged(
            evaluate_mrp(
                self.finite_flip_flop,
                γ=0.99,
                approx_0=start,
                non_terminal_states_distribution=Choose(
                    self.finite_flip_flop.non_terminal_states
                ),
                num_state_samples=5
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        self.assertEqual(len(v.values_map), 2)

        for s in v.values_map:
            self.assertLess(abs(v(s) - 170), 1.0)

        v_finite = iterate.converged(
            evaluate_finite_mrp(
                self.finite_flip_flop,
                γ=0.99,
                approx_0=start
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        assert_allclose(v.evaluate([NonTerminal(True), NonTerminal(False)]),
                        v_finite.evaluate([NonTerminal(True),
                                           NonTerminal(False)]),
                        rtol=0.01)

    def test_compare_to_backward_induction(self):
        finite_horizon = finite_horizon_MRP(self.finite_flip_flop, 10)

        start = Dynamic({s: 0.0 for s in finite_horizon.non_terminal_states})
        v = iterate.converged(
            evaluate_finite_mrp(
                finite_horizon,
                γ=1,
                approx_0=start
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        self.assertEqual(len(v.values_map), 20)

        finite_v =\
            list(evaluate(unwrap_finite_horizon_MRP(finite_horizon), gamma=1))

        for time in range(10):
            self.assertAlmostEqual(
                v(NonTerminal(WithTime(state=True, time=time))),
                finite_v[time][NonTerminal(True)]
            )
            self.assertAlmostEqual(
                v(NonTerminal(WithTime(state=False, time=time))),
                finite_v[time][NonTerminal(False)]
            )
