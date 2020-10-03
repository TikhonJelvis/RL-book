import itertools
import unittest

from rl.iterate import (iterate, last, converge, converged, returns)


class TestIterate(unittest.TestCase):
    def test_iterate(self):
        ns = iterate(lambda x: x + 1, start=0)
        self.assertEqual(list(itertools.islice(ns, 5)), list(range(0, 5)))


class TestLast(unittest.TestCase):
    def test_last(self):
        self.assertEqual(last(range(0, 5)), 4)
        self.assertEqual(last(range(0, 10)), 9)

        self.assertEqual(last([]), None)


class TestConverge(unittest.TestCase):
    def test_converge(self):
        def close(a, b):
            return abs(a - b) < 0.1

        ns = (1.0 / n for n in iterate(lambda x: x + 1, start=1))
        self.assertAlmostEqual(converged(ns, close), 0.33, places=2)

        ns = (1.0 / n for n in iterate(lambda x: x + 1, start=1))
        all_ns = [1.0, 0.5, 0.33]
        for got, expected in zip(converge(ns, close), all_ns):
            self.assertAlmostEqual(got, expected, places=2)

    def test_converge_end(self):
        '''Check that converge ends the iterator at the right place when the
        underlying iterator ends before converging.

        '''
        def close(a, b):
            return abs(a - b) < 0.1

        ns = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.assertAlmostEqual(converged(iter(ns), close), 2.0)

        for got, expected in zip(converge(iter(ns), close), ns):
            self.assertAlmostEqual(got, expected)


class TestReturns(unittest.TestCase):
    def test_simple(self):
        simple = [(l, 1.0) for l in "abcdefg"]

        self.assertEqual(set(returns(simple, γ=1.0, n_states=1)), {('a', 7.0)})
        self.assertEqual(set(returns(simple, γ=1.0, n_states=3)),
                         {('a', 7.0), ('b', 6.0), ('c', 5.0)})
        self.assertEqual(set(returns(simple, γ=1.0, n_states=7)),
                         {(s, 7.0 - n) for s, n
                          in zip("abcdefg", range(0, 7))})

        # Ensure passing in an iterator (vs a list/set/etc) works
        self.assertEqual(set(returns(iter(simple), γ=1.0, n_states=3)),
                         {('a', 7.0), ('b', 6.0), ('c', 5.0)})
