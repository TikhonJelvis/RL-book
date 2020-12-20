import unittest

import itertools
from typing import cast, Iterable, Iterator, Optional, Tuple

from rl.distribution import Categorical, Choose
from rl.function_approx import Tabular
import rl.iterate as iterate
from rl.markov_process import FiniteMarkovRewardProcess
import rl.markov_process as mp
from rl.markov_decision_process import FiniteMarkovDecisionProcess
import rl.markov_decision_process as mdp
import rl.td as td


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

        self.finite_mdp = FiniteMarkovDecisionProcess({
            True: {
                True: Categorical({(True, 1.0): 0.7, (False, 2.0): 0.3}),
                False: Categorical({(True, 1.0): 0.3, (False, 2.0): 0.7}),
            },
            False: {
                True: Categorical({(False, 1.0): 0.7, (True, 2.0): 0.3}),
                False: Categorical({(False, 1.0): 0.3, (True, 2.0): 0.7}),
            }
        })

    def test_evaluate_finite_mrp(self) -> None:
        start = Tabular(
            {s: 0.0 for s in self.finite_flip_flop.states()},
            count_to_weight_func=lambda _: 0.1
        )

        episode_length = 20
        episodes: Iterable[Iterable[mp.TransitionStep[bool]]] =\
            self.finite_flip_flop.reward_traces(Choose({True, False}))
        transitions: Iterable[mp.TransitionStep[bool]] =\
            itertools.chain.from_iterable(
                itertools.islice(episode, episode_length)
                for episode in episodes
            )

        vs = td.evaluate_mrp(transitions, γ=0.99, approx_0=start)

        v: Optional[Tabular[bool]] = iterate.last(
            itertools.islice(cast(Iterator[Tabular[bool]], vs), 10000)
        )

        if v is not None:
            self.assertEqual(len(v.values_map), 2)

            for s in v.values_map:
                # Intentionally loose bound—otherwise test is too slow.
                # Takes >1s on my machine otherwise.
                self.assertLess(abs(v(s) - 170), 3.0)
        else:
            assert False

    def test_evaluate_finite_mdp(self) -> None:
        q_0: Tabular[Tuple[bool, bool]] = Tabular(
            {(s, a): 0.0
             for s in self.finite_mdp.states()
             for a in self.finite_mdp.actions(s)},
            count_to_weight_func=lambda _: 0.1
        )

        uniform_policy: mdp.Policy[bool, bool] =\
            mdp.FinitePolicy({
                s: Choose(self.finite_mdp.actions(s))
                for s in self.finite_mdp.states()
            })

        transitions: Iterable[mdp.TransitionStep[bool, bool]] =\
            self.finite_mdp.simulate_actions(
                Choose(self.finite_mdp.states()),
                uniform_policy
            )

        qs = td.evaluate_mdp(
            transitions,
            self.finite_mdp.actions,
            q_0,
            γ=0.99
        )

        q: Optional[Tabular[Tuple[bool, bool]]] =\
            iterate.last(
                cast(Iterator[Tabular[Tuple[bool, bool]]],
                     itertools.islice(qs, 20000))
            )

        if q is not None:
            self.assertEqual(len(q.values_map), 4)

            for s in [True, False]:
                self.assertLess(abs(q((s, False)) - 170.0), 2)
                self.assertGreater(q((s, False)), q((s, True)))
        else:
            assert False
