import unittest

from rl.distribution import Categorical
from rl.dynamic_programming import evaluate_mrp_result
from rl.finite_horizon import (finite_horizon_MRP, evaluate,
                               unwrap_finite_horizon_MRP, WithTime)
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal


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

    def test_evaluate_mrp(self):
        v = evaluate_mrp_result(self.finite_flip_flop, gamma=0.99)

        self.assertEqual(len(v), 2)

        for s in v:
            self.assertLess(abs(v[s] - 170), 0.1)

    def test_compare_to_backward_induction(self):
        finite_horizon = finite_horizon_MRP(self.finite_flip_flop, 10)

        v = evaluate_mrp_result(finite_horizon, gamma=1)
        self.assertEqual(len(v), 20)

        finite_v =\
            list(evaluate(unwrap_finite_horizon_MRP(finite_horizon), gamma=1))

        for time in range(10):
            self.assertAlmostEqual(
                v[NonTerminal(WithTime(state=True, time=time))],
                finite_v[time][NonTerminal(True)]
            )
            self.assertAlmostEqual(
                v[NonTerminal(WithTime(state=False, time=time))],
                finite_v[time][NonTerminal(False)]
            )
