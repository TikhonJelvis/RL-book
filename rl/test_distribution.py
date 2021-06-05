from collections import Counter
import unittest

from rl.distribution import (Bernoulli, Categorical, Choose, Constant,
                             Gaussian, SampledDistribution, Uniform)


def assert_almost_equal(test_case, dist_a, dist_b):
    '''Check that two distributions are "almost" equal (ie ignore small
    differences in floating point numbers when comparing them).

    '''
    a_table = dist_a.table()
    b_table = dist_b.table()

    assert a_table.keys() == b_table.keys()

    for outcome in a_table:
        test_case.assertAlmostEqual(a_table[outcome], b_table[outcome])


class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.finite = Choose(range(0, 6))
        self.sampled = SampledDistribution(
            lambda: self.finite.sample(),
            100000
        )

    def test_expectation(self):
        expected_finite = self.finite.expectation(lambda x: x)
        expected_sampled = self.sampled.expectation(lambda x: x)
        self.assertLess(abs(expected_finite - expected_sampled), 0.02)

    def test_sample_n(self):
        samples = self.sampled.sample_n(10)
        self.assertEqual(len(samples), 10)
        self.assertTrue(all(0 <= s < 6 for s in samples))


class TestUniform(unittest.TestCase):
    def setUp(self):
        self.uniform = Uniform(100000)

    def test_expectation(self):
        expectation = self.uniform.expectation(lambda x: x)
        self.assertLess(abs(expectation - 0.5), 0.01)


class TestGaussian(unittest.TestCase):
    def setUp(self):
        self.unit = Gaussian(1.0, 1.0, 100000)
        self.large = Gaussian(10.0, 30.0, 100000)

    def test_expectation(self):
        unit_expectation = self.unit.expectation(lambda x: x)
        self.assertLess(abs(unit_expectation - 1.0), 0.1)

        large_expectation = self.large.expectation(lambda x: x)
        self.assertLess(abs(large_expectation - 10), 0.3)


class TestFiniteDistribution(unittest.TestCase):
    def setUp(self):
        self.die = Choose({1, 2, 3, 4, 5, 6})

        self.ragged = Categorical({0: 0.9, 1: 0.05, 2: 0.025, 3: 0.025})

    def test_map(self):
        plusOne = self.die.map(lambda x: x + 1)
        assert_almost_equal(self, plusOne, Choose({2, 3, 4, 5, 6, 7}))

        evenOdd = self.die.map(lambda x: x % 2 == 0)
        assert_almost_equal(self, evenOdd, Choose({True, False}))

        greaterThan4 = self.die.map(lambda x: x > 4)
        assert_almost_equal(self, greaterThan4,
                            Categorical({True: 1/3, False: 2/3}))

    def test_expectation(self):
        self.assertAlmostEqual(self.die.expectation(float), 3.5)

        even = self.die.map(lambda n: n % 2 == 0)
        self.assertAlmostEqual(even.expectation(float), 0.5)

        self.assertAlmostEqual(self.ragged.expectation(float), 0.175)


class TestConstant(unittest.TestCase):
    def test_constant(self):
        assert_almost_equal(self, Constant(42), Categorical({42: 1.}))
        self.assertAlmostEqual(Constant(42).probability(42), 1.)
        self.assertAlmostEqual(Constant(42).probability(37), 0.)


class TestBernoulli(unittest.TestCase):
    def setUp(self):
        self.fair = Bernoulli(0.5)
        self.unfair = Bernoulli(0.3)

    def test_constant(self):
        assert_almost_equal(
            self, self.fair, Categorical({True: 0.5, False: 0.5}))
        self.assertAlmostEqual(self.fair.probability(True), 0.5)
        self.assertAlmostEqual(self.fair.probability(False), 0.5)

        assert_almost_equal(self, self.unfair,
                            Categorical({True: 0.3, False: 0.7}))
        self.assertAlmostEqual(self.unfair.probability(True), 0.3)
        self.assertAlmostEqual(self.unfair.probability(False), 0.7)


class TestChoose(unittest.TestCase):
    def setUp(self):
        self.one = Choose({1})
        self.six = Choose({1, 2, 3, 4, 5, 6})
        self.repeated = Choose([1,1,1,2])

    def test_choose(self):
        assert_almost_equal(self, self.one, Constant(1))
        self.assertAlmostEqual(self.one.probability(1), 1.)
        self.assertAlmostEqual(self.one.probability(0), 0.)

        categorical_six = Categorical({x: 1/6 for x in range(1, 7)})
        assert_almost_equal(self, self.six, categorical_six)
        self.assertAlmostEqual(self.six.probability(1), 1/6)
        self.assertAlmostEqual(self.six.probability(0), 0.)

    def test_repeated(self):
        counts = Counter(self.repeated.sample_n(1000))
        self.assertLess(abs(counts[1] - 750), 50)
        self.assertLess(abs(counts[2] - 250), 50)

        table = self.repeated.table()
        self.assertAlmostEqual(table[1], 0.75)
        self.assertAlmostEqual(table[2], 0.25)

        counts = Counter(self.repeated.sample_n(1000))
        self.assertLess(abs(counts[1] - 750), 50)
        self.assertLess(abs(counts[2] - 250), 50)


class TestCategorical(unittest.TestCase):
    def setUp(self):
        self.normalized = Categorical({True: 0.3, False: 0.7})
        self.unnormalized = Categorical({True: 3., False: 7.})

    def test_categorical(self):
        assert_almost_equal(self, self.normalized, Bernoulli(0.3))
        self.assertAlmostEqual(self.normalized.probability(True), 0.3)
        self.assertAlmostEqual(self.normalized.probability(False), 0.7)
        self.assertAlmostEqual(self.normalized.probability(None), 0.)

    def test_normalization(self):
        assert_almost_equal(self, self.unnormalized, self.normalized)
        self.assertAlmostEqual(self.unnormalized.probability(True), 0.3)
        self.assertAlmostEqual(self.unnormalized.probability(False), 0.7)
        self.assertAlmostEqual(self.unnormalized.probability(None), 0.)
