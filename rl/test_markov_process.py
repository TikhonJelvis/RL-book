import itertools
from typing import Tuple
import unittest

from rl.distribution import Bernoulli
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

    def simulate_transition(self) -> bool:
        switch_states = Bernoulli(self.p).sample()

        if switch_states:
            return not self.state
        else:
            return self.state


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

    def simulate_transition_reward(self) -> Tuple[bool, float]:
        switch_states = Bernoulli(self.p).sample()

        if switch_states:
            next_state = not self.state
            reward = 1 if self.state else 0.5
            return (next_state, reward)
        else:
            return (self.state, 0.5)


class TestMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(self.flip_flop.simulate(), 10))

        self.assertTrue(all(isinstance(outcome, bool) for outcome in trace))


class TestFiniteMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FiniteFlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(self.flip_flop.simulate(), 10))

        self.assertTrue(all(isinstance(outcome, bool) for outcome in trace))


class TestRewardMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = RewardFlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(itertools.islice(self.flip_flop.simulate_reward(), 10))

        self.assertTrue(all(isinstance(outcome, bool) for outcome, _ in trace))

        cumulative_reward = sum(reward for _, reward in trace)
        self.assertTrue(0 <= cumulative_reward <= 10)


if __name__ == "__main__":
    flip_flop_trace = \
        list(itertools.islice(FlipFlop(0.5).simulate(), 10))

    finit_trace = \
        list(itertools.islice(FiniteFlipFlop(0.5).simulate(), 10))

    reward_trace = \
        list(itertools.islice(RewardFlipFlop(0.5).simulate_reward(), 10))

    print("FlipFlop:\n {flip_flop_trace}")
    print("FiniteFlipFlop:\n {finite_trace}")
    print("RewardFlipFlop:\n {reward_trace}")
