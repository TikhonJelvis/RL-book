import unittest

import dataclasses

from rl.distribution import Categorical, Constant
import rl.test_distribution as distribution
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_decision_process import (
    ActionMapping, FiniteMarkovDecisionProcess)

from rl.finite_horizon import (finite_horizon_MDP, finite_horizon_MRP,
                               WithTime, unwrap_finite_horizon_MDP,
                               unwrap_finite_horizon_MRP, evaluate,
                               optimal_vf_and_policy)


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    ''' A version of FlipFlop implemented with the FiniteMarkovProcess machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestFiniteMRP(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FlipFlop(0.7)

    def test_finite_horizon_MRP(self):
        finite = finite_horizon_MRP(self.finite_flip_flop, 10)

        trues = [WithTime(True, time) for time in range(0, 10)]
        falses = [WithTime(False, time) for time in range(0, 10)]
        non_terminal_states = set(trues + falses)
        terminal_states = {WithTime(True, 10), WithTime(False, 10)}
        expected_states = non_terminal_states.union(terminal_states)

        self.assertEqual(set(finite.states()), expected_states)

        expected_transition = {}
        for state in non_terminal_states:
            expected_transition[state] =\
                Categorical({
                    (WithTime(state.state, state.time + 1), 1.0): 0.3,
                    (WithTime(not state.state, state.time + 1), 2.0): 0.7
                })

        for state in non_terminal_states:
            distribution.assert_almost_equal(
                self,
                finite.transition_reward(state),
                expected_transition[state])

        for state in terminal_states:
            self.assertEqual(finite.transition(state), None)

    def test_unwrap_finite_horizon_MRP(self):
        finite = finite_horizon_MRP(self.finite_flip_flop, 10)

        def transition_for(time):
            return {
                True: Categorical({
                    (True, 1.0): 0.3,
                    (False, 2.0): 0.7,
                }),
                False: Categorical({
                    (True, 2.0): 0.7,
                    (False, 1.0): 0.3,
                })
            }

        unwrapped = unwrap_finite_horizon_MRP(finite)
        self.assertEqual(len(unwrapped), 10)

        expected_transitions = [transition_for(n) for n in range(0, 10)]
        for time in range(0, 10):
            got = unwrapped[time]
            expected = expected_transitions[time]
            distribution.assert_almost_equal(
                self, got[True],
                expected[True])
            distribution.assert_almost_equal(
                self, got[False],
                expected[False])

    def test_evaluate(self):
        process = finite_horizon_MRP(self.finite_flip_flop, 10)
        vs = list(evaluate(unwrap_finite_horizon_MRP(process), gamma=1))

        self.assertEqual(len(vs), 10)

        self.assertAlmostEqual(vs[0][True], 17)
        self.assertAlmostEqual(vs[0][False], 17)

        self.assertAlmostEqual(vs[5][True], 17 / 2)
        self.assertAlmostEqual(vs[5][False], 17 / 2)

        self.assertAlmostEqual(vs[9][True], 17 / 10)
        self.assertAlmostEqual(vs[9][False], 17 / 10)


class TestFiniteMDP(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FiniteMarkovDecisionProcess({
            True: {
                True: Categorical({(True, 1.0): 0.7, (False, 2.0): 0.3}),
                False: Categorical({(True, 1.0): 0.3, (False, 2.0): 0.7}),
            },
            False: {
                True: Categorical({(False, 1.0): 0.7, (True, 2.0): 0.3}),
                False: Categorical({(False, 1.0): 0.3, (True, 2.0): 0.7}),
            }
        })

    def test_finite_horizon_MDP(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, limit=10)

        self.assertEqual(len(finite.states()), 22)

        for s in finite.states():
            if len(set(finite.actions(s))) > 0:
                self.assertEqual(set(finite.actions(s)), {False, True})

        start = WithTime(state=True, time=0)
        result = finite.action_mapping(start)[False]
        expected_result = Categorical({
            (WithTime(False, time=1), 2.0): 0.7,
            (WithTime(True, time=1), 1.0): 0.3
        })
        distribution.assert_almost_equal(self, result, expected_result)

        self.assertEqual(finite.step(WithTime(True, 10), True), None)

    def test_unwrap_finite_horizon_MDP(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, 10)
        unwrapped = unwrap_finite_horizon_MDP(finite)

        self.assertEqual(len(unwrapped), 10)

        def action_mapping_for(
                s: WithTime[bool]
        ) -> ActionMapping[bool, WithTime[bool]]:
            same = s.step_time()
            different = dataclasses.replace(s.step_time(), state=not s.state)

            return {
                True: Categorical({
                    (same, 1.0): 0.7,
                    (different, 2.0): 0.3
                }),
                False: Categorical({
                    (same, 1.0): 0.3,
                    (different, 2.0): 0.7
                })
            }

        for t in range(0, 10):
            for s in True, False:
                s_time = WithTime(state=s, time=t)
                for a in True, False:
                    distribution.assert_almost_equal(
                        self,
                        finite.action_mapping(s_time)[a],
                        action_mapping_for(s_time)[a])

        self.assertEqual(
            finite.action_mapping(WithTime(state=True, time=10)),
            None
        )

    def test_optimal_policy(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, limit=10)
        steps = unwrap_finite_horizon_MDP(finite)
        *v_ps, (v, p) = optimal_vf_and_policy(steps, gamma=1)

        for s in p.states():
            self.assertEqual(p.act(s), Constant(False))

        self.assertAlmostEqual(v_ps[0][0][True], 17)
        self.assertAlmostEqual(v_ps[5][0][False], 17 / 2)
