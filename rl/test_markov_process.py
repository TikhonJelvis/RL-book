import itertools
from typing import Tuple
import unittest

from rl.distribution import Bernoulli, Distribution, SampledDistribution
from rl.markov_process import (FiniteMarkovProcess, MarkovProcess,
                               MarkovRewardProcess)


# Example classes:
class FlipFlop(MarkovProcess[bool]):
    '''A simple example Markov chain with two states, flipping from one to
    the other with probability p and staying at the same state with
    probability 1 - p.

    '''

    state: bool

    p: float

    def __init__(self, p, start_state=True):
        self.p = p
        self.state = start_state

    def transition(self) -> Distribution[bool]:
        def next_state():
            switch_states = Bernoulli(self.p).sample()

            if switch_states:
                return not self.state
            else:
                return self.state

        return SampledDistribution(next_state)


class FiniteFlipFlop(FiniteMarkovProcess[bool]):
    ''' A version of FlipFlop implemented with the FiniteMarkovProcess machinery.

    '''
    def __init__(self, p, start_state=True):
        self.state = start_state

        self.state_space = [False, True]

        self.transition_matrix = {
            True: {
                False: p,
                True: 1 - p
            },
            False: {
                False: 1 - p,
                True: p
            }
        }


class RewardFlipFlop(MarkovRewardProcess[bool]):
    state: bool

    p: float

    def __init__(self, p, start_state=True):
        self.p = p

        self.state = start_state

    def transition_reward(self) -> Distribution[Tuple[bool, float]]:
        def next_state():
            switch_states = Bernoulli(self.p).sample()

            if switch_states:
                next_state = not self.state
                reward = 1 if self.state else 0.5
                return (next_state, reward)
            else:
                return (self.state, 0.5)

        return SampledDistribution(next_state)


class TestMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(self.flip_flop.simulate(), 10))

        self.assertTrue(all(isinstance(outcome, bool) for outcome in trace))

        longer_trace = itertools.islice(self.flip_flop.simulate(), 10000)
        count_trues = len(list(outcome for outcome in longer_trace if outcome))

        # If the code is correct, this should fail with a vanishingly
        # small probability
        self.assertTrue(1000 < count_trues < 9000)


class TestFiniteMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FiniteFlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(self.flip_flop.simulate(), 10))

        self.assertTrue(all(isinstance(outcome, bool) for outcome in trace))

        longer_trace = itertools.islice(self.flip_flop.simulate(), 10000)
        count_trues = len(list(outcome for outcome in longer_trace if outcome))

        # If the code is correct, this should fail with a vanishingly
        # small probability
        self.assertTrue(1000 < count_trues < 9000)


class TestRewardMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = RewardFlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(self.flip_flop.simulate_reward(), 10))

        self.assertTrue(all(isinstance(outcome, bool) for outcome, _ in trace))

        cumulative_reward = sum(reward for _, reward in trace)
        self.assertTrue(0 <= cumulative_reward <= 10)
