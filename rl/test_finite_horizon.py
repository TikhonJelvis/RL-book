import unittest

from rl.distribution import Categorical
import rl.test_distribution as distribution
from rl.finite_horizon import (
    finite_horizon_markov_process, unwrap_finite_horizon_markov_process,
    WithTime)
from rl.markov_process import FiniteMarkovRewardProcess


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    ''' A version of FlipFlop implemented with the FiniteMarkovProcess machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestFiniteHorizon(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FlipFlop(0.7)

    def test_finite_horizon_markov_process(self):
        finite = finite_horizon_markov_process(self.finite_flip_flop, 10)

        trues = [WithTime(True, time) for time in range(0, 10)]
        falses = [WithTime(False, time) for time in range(0, 10)]
        expected_states = trues + falses

        expected_transition = {}
        for state in expected_states:
            expected_transition[state] =\
                Categorical({
                    (WithTime(state.state, state.time + 1), 1.0): 0.3,
                    (WithTime(not state.state, state.time + 1), 2.0): 0.7
                })

        for state in expected_states:
            distribution.assert_almost_equal(
                self,
                finite.transition_reward_map[state],
                expected_transition[state])

    def test_unwrap_finite_horizon_markov_process(self):
        finite = finite_horizon_markov_process(self.finite_flip_flop, 10)

        def transition_for(time):
            return {
                WithTime(True, time): Categorical({
                    (WithTime(True, time + 1), 1.0): 0.3,
                    (WithTime(False, time + 1), 2.0): 0.7,
                }),
                WithTime(False, time): Categorical({
                    (WithTime(True, time + 1), 2.0): 0.7,
                    (WithTime(False, time + 1), 1.0): 0.3,
                })
            }

        unwrapped = unwrap_finite_horizon_markov_process(finite, 10)
        expected_transitions = [transition_for(n) for n in range(0, 10)]
        for time in range(0, 10):
            got = unwrapped[time]
            expected = expected_transitions[time]
            distribution.assert_almost_equal(
                self, got[WithTime(True, time)],
                expected[WithTime(True, time)])
            distribution.assert_almost_equal(
                self, got[WithTime(False, time)],
                expected[WithTime(False, time)])
