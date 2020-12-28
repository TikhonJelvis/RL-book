import unittest

import itertools
from typing import cast, Iterable, Tuple

from rl.distribution import Categorical, Choose
from rl.function_approx import Tabular
import rl.iterate as iterate
import rl.markov_decision_process as mdp
import rl.monte_carlo as mc


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.finite_mdp = mdp.FiniteMarkovDecisionProcess({
            True: {
                True: Categorical({(True, 1.0): 0.7, (False, 2.0): 0.3}),
                False: Categorical({(True, 1.0): 0.3, (False, 2.0): 0.7}),
            },
            False: {
                True: Categorical({(False, 1.0): 0.7, (True, 2.0): 0.3}),
                False: Categorical({(False, 1.0): 0.3, (True, 2.0): 0.7}),
            }
        })

        optimal = mdp.FinitePolicy({
            True: Choose({False}),
            False: Choose({False})
        })
        self.finite_flip_flop = self.finite_mdp.apply_finite_policy(optimal)

    def test_evaluate_finite_mrp(self):
        start = Tabular({s: 0.0 for s in self.finite_flip_flop.states()})
        traces = self.finite_flip_flop.reward_traces(Choose({True, False}))
        v = iterate.converged(
            mc.evaluate_mrp(traces, γ=0.99, approx_0=start),
            # Loose bound of 0.025 to speed up test.
            done=lambda a, b: a.within(b, 0.025)
        )

        self.assertEqual(len(v.values_map), 2)

        for s in v.values_map:
            # Intentionally loose bound—otherwise test is too slow.
            # Takes >1s on my machine otherwise.
            self.assertLess(abs(v(s) - 170), 1.0)

    def test_evaluate_finite_mdp(self) -> None:
        q_0: Tabular[Tuple[bool, bool]] = Tabular(
            {(s, a): 0.0
             for s in self.finite_mdp.states()
             for a in self.finite_mdp.actions(s)},
            count_to_weight_func=lambda _: 0.1
        )

        uniform_policy: mdp.FinitePolicy[bool, bool] =\
            mdp.FinitePolicy({
                s: Choose(self.finite_mdp.actions(s))
                for s in self.finite_mdp.states()
            })

        transitions: Iterable[Iterable[mdp.TransitionStep[bool, bool]]] =\
            self.finite_mdp.action_traces(
                Choose(self.finite_mdp.states()),
                uniform_policy
            )

        qs = mc.evaluate_mdp(
            transitions,
            q_0,
            γ=0.99
        )

        q = iterate.last(itertools.islice(qs, 20))

        if q is not None:
            q = cast(Tabular[Tuple[bool, bool]], q)
            self.assertEqual(len(q.values_map), 4)

            for s in [True, False]:
                self.assertLess(abs(q((s, False)) - 170.0), 2)
                self.assertGreater(q((s, False)), q((s, True)))
        else:
            assert False
