import itertools
import numpy as np
from typing import Tuple
import unittest

from rl.distribution import (Bernoulli, Categorical, Distribution,
                             SampledDistribution, Constant)
from rl.markov_process import (FiniteMarkovProcess, MarkovProcess,
                               MarkovRewardProcess, NonTerminal, State)


# Example classes:
class FlipFlop(MarkovProcess[bool]):
    '''A simple example Markov chain with two states, flipping from one to
    the other with probability p and staying at the same state with
    probability 1 - p.

    '''

    p: float

    def __init__(self, p: float):
        self.p = p

    def transition(self, state: NonTerminal[bool]) -> \
            Distribution[State[bool]]:
        def next_state(state=state):
            switch_states = Bernoulli(self.p).sample()
            next_st: bool = not state.state if switch_states else state.state
            return NonTerminal(next_st)

        return SampledDistribution(next_state)


class FiniteFlipFlop(FiniteMarkovProcess[bool]):
    ''' A version of FlipFlop implemented with the FiniteMarkovProcess machinery.

    '''
    def __init__(self, p: float):
        transition_map = {
            b: Categorical({not b: p, b: 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_map)


class RewardFlipFlop(MarkovRewardProcess[bool]):
    p: float

    def __init__(self, p: float):
        self.p = p

    def transition_reward(self, state: NonTerminal[bool]) -> \
            Distribution[Tuple[State[bool], float]]:
        def next_state(state=state):
            switch_states = Bernoulli(self.p).sample()

            st: bool = state.state
            if switch_states:
                next_s: bool = not st
                reward = 1 if st else 0.5
                return NonTerminal(next_s), reward
            else:
                return NonTerminal(st), 0.5

        return SampledDistribution(next_state)


class TestMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(
            self.flip_flop.simulate(Constant(NonTerminal(True))),
            10
        ))

        self.assertTrue(all(isinstance(outcome.state, bool)
                            for outcome in trace))

        longer_trace = itertools.islice(
            self.flip_flop.simulate(Constant(NonTerminal(True))),
            10000
        )
        count_trues = len(list(outcome for outcome in longer_trace
                               if outcome.state))

        # If the code is correct, this should fail with a vanishingly
        # small probability
        self.assertTrue(1000 < count_trues < 9000)


class TestFiniteMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FiniteFlipFlop(0.5)

        self.biased = FiniteFlipFlop(0.3)

    def test_flip_flop(self):
        trace = list(itertools.islice(
            self.flip_flop.simulate(Constant(NonTerminal(True))),
            10
        ))

        self.assertTrue(all(isinstance(outcome.state, bool)
                            for outcome in trace))

        longer_trace = itertools.islice(
            self.flip_flop.simulate(Constant(NonTerminal(True))),
            10000
        )
        count_trues = len(list(outcome for outcome in longer_trace
                               if outcome.state))

        # If the code is correct, this should fail with a vanishingly
        # small probability
        self.assertTrue(1000 < count_trues < 9000)

    def test_transition_matrix(self):
        matrix = self.flip_flop.get_transition_matrix()
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_equal(matrix, expected)

        matrix = self.biased.get_transition_matrix()
        expected = np.array([[0.7, 0.3], [0.3, 0.7]])
        np.testing.assert_array_equal(matrix, expected)

    def test_stationary_distribution(self):
        distribution = self.flip_flop.get_stationary_distribution().table()
        expected = [(True, 0.5), (False, 0.5)]
        np.testing.assert_almost_equal(list(distribution.items()), expected)

        distribution = self.biased.get_stationary_distribution().table()
        expected = [(True, 0.5), (False, 0.5)]
        np.testing.assert_almost_equal(list(distribution.items()), expected)

    def test_display(self):
        # Just test that the display functions don't error out.
        try:
            self.flip_flop.display_stationary_distribution()
            self.flip_flop.generate_image()
            self.flip_flop.__repr__()
        except Exception:
            self.fail("Display functions raised an error.")


class TestRewardMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = RewardFlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(
            self.flip_flop.simulate_reward(Constant(NonTerminal(True))),
            10
        ))

        self.assertTrue(
            all(isinstance(step.next_state.state, bool) for step in trace)
        )

        cumulative_reward = sum(step.reward for step in trace)
        self.assertTrue(0 <= cumulative_reward <= 10)
