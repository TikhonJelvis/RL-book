import itertools
import unittest

from rl.iterate import (iterate, last, converge, converged)


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
        self.assertAlmostEqual(converged(ns, close), 0.25, places=2)

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
