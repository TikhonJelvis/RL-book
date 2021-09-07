import unittest

import random

from rl.distribution import Categorical, Choose
from rl.function_approx import Tabular
import rl.iterate as iterate
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
import rl.monte_carlo as mc


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
        random.seed(42)
        self.finite_flip_flop = FlipFlop(0.7)

    def test_evaluate_finite_mrp(self):
        start = Tabular({s: 0.0 for s in
                         self.finite_flip_flop.non_terminal_states})
        traces = self.finite_flip_flop.reward_traces(Choose({
            NonTerminal(True),
            NonTerminal(False)
        }))
        v = iterate.converged(
            mc.mc_prediction(traces, γ=0.99, approx_0=start),
            # Loose bound of 0.01 to speed up test.
            done=lambda a, b: a.within(b, 0.01)
        )

        self.assertEqual(len(v.values_map), 2)

        for s in v.values_map:
            # Intentionally loose bound—otherwise test is too slow.
            # Takes >1s on my machine otherwise.
            self.assertLess(abs(v(s) - 170), 1.0)
