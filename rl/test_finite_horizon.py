from typing import Mapping, Sequence
import unittest

import dataclasses

from rl.distribution import Categorical
import rl.test_distribution as distribution
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_decision_process import (ActionMapping,
                                        FiniteMarkovDecisionProcess,
                                        NonTerminal, Terminal)

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

        trues = [NonTerminal(WithTime(True, time)) for time in range(10)]
        falses = [NonTerminal(WithTime(False, time)) for time in range(10)]
        non_terminal_states = set(trues + falses)
        self.assertEqual(set(finite.non_terminal_states), non_terminal_states)

        expected_transition = {}
        for state in non_terminal_states:
            t: int = state.state.time
            st: bool = state.state.state
            if t < 9:
                prob = {
                    (NonTerminal(WithTime(st, t + 1)), 1.0): 0.3,
                    (NonTerminal(WithTime(not st, t + 1)), 2.0): 0.7
                }
            else:
                prob = {
                    (Terminal(WithTime(st, t + 1)), 1.0): 0.3,
                    (Terminal(WithTime(not st, t + 1)), 2.0): 0.7
                }

            expected_transition[state] = Categorical(prob)

        for state in non_terminal_states:
            distribution.assert_almost_equal(
                self,
                finite.transition_reward(state),
                expected_transition[state])

    def test_unwrap_finite_horizon_MRP(self):
        finite = finite_horizon_MRP(self.finite_flip_flop, 10)

        def transition_for(_):
            return {
                True: Categorical({
                    (NonTerminal(True), 1.0): 0.3,
                    (NonTerminal(False), 2.0): 0.7
                }),
                False: Categorical({
                    (NonTerminal(True), 2.0): 0.7,
                    (NonTerminal(False), 1.0): 0.3
                })
            }

        unwrapped = unwrap_finite_horizon_MRP(finite)
        self.assertEqual(len(unwrapped), 10)

        expected_transitions = [transition_for(n) for n in range(10)]
        for time in range(9):
            got = unwrapped[time]
            expected = expected_transitions[time]
            distribution.assert_almost_equal(
                self, got[NonTerminal(True)],
                expected[True]
            )
            distribution.assert_almost_equal(
                self, got[NonTerminal(False)],
                expected[False]
            )

        distribution.assert_almost_equal(
            self, unwrapped[9][NonTerminal(True)],
            Categorical({
                (Terminal(True), 1.0): 0.3,
                (Terminal(False), 2.0): 0.7
            })
        )
        distribution.assert_almost_equal(
            self, unwrapped[9][NonTerminal(False)],
            Categorical({
                (Terminal(True), 2.0): 0.7,
                (Terminal(False), 1.0): 0.3
            })
        )

    def test_evaluate(self):
        process = finite_horizon_MRP(self.finite_flip_flop, 10)
        vs = list(evaluate(unwrap_finite_horizon_MRP(process), gamma=1))

        self.assertEqual(len(vs), 10)

        self.assertAlmostEqual(vs[0][NonTerminal(True)], 17)
        self.assertAlmostEqual(vs[0][NonTerminal(False)], 17)

        self.assertAlmostEqual(vs[5][NonTerminal(True)], 17 / 2)
        self.assertAlmostEqual(vs[5][NonTerminal(False)], 17 / 2)

        self.assertAlmostEqual(vs[9][NonTerminal(True)], 17 / 10)
        self.assertAlmostEqual(vs[9][NonTerminal(False)], 17 / 10)


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

        self.assertEqual(len(finite.non_terminal_states), 20)

        for s in finite.non_terminal_states:
            self.assertEqual(set(finite.actions(s)), {False, True})

        start = NonTerminal(WithTime(state=True, time=0))
        result = finite.mapping[start][False]
        expected_result = Categorical({
            (NonTerminal(WithTime(False, time=1)), 2.0): 0.7,
            (NonTerminal(WithTime(True, time=1)), 1.0): 0.3
        })
        distribution.assert_almost_equal(self, result, expected_result)

    def test_unwrap_finite_horizon_MDP(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, 10)
        unwrapped = unwrap_finite_horizon_MDP(finite)

        self.assertEqual(len(unwrapped), 10)

        def action_mapping_for(s: WithTime[bool]) -> \
                ActionMapping[bool, WithTime[bool]]:
            same = NonTerminal(s.step_time())
            different = NonTerminal(dataclasses.replace(
                s.step_time(),
                state=not s.state
            ))

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

        for t in range(9):
            for s in True, False:
                s_time = WithTime(state=s, time=t)
                for a in True, False:
                    distribution.assert_almost_equal(
                        self,
                        finite.mapping[NonTerminal(s_time)][a],
                        action_mapping_for(s_time)[a]
                    )

        for s in True, False:
            s_time = WithTime(state=s, time=9)
            same = Terminal(s_time.step_time())
            different = Terminal(dataclasses.replace(
                s_time.step_time(),
                state=not s_time.state
            ))
            act_map = {
                True: Categorical({
                    (same, 1.0): 0.7,
                    (different, 2.0): 0.3
                }),
                False: Categorical({
                    (same, 1.0): 0.3,
                    (different, 2.0): 0.7
                })

            }
            for a in True, False:
                distribution.assert_almost_equal(
                    self,
                    finite.mapping[NonTerminal(s_time)][a],
                    act_map[a]
                )

    def test_optimal_policy(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, limit=10)
        steps = unwrap_finite_horizon_MDP(finite)
        *v_ps, (_, p) = optimal_vf_and_policy(steps, gamma=1)

        for _, a in p.action_for.items():
            self.assertEqual(a, False)

        self.assertAlmostEqual(v_ps[0][0][NonTerminal(True)], 17)
        self.assertAlmostEqual(v_ps[5][0][NonTerminal(False)], 17 / 2)
