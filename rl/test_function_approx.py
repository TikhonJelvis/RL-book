import unittest
import numpy as np

from rl.function_approx import (Dynamic)


class TestDynamic(unittest.TestCase):
    def setUp(self):
        self.dynamic_0 = Dynamic(values_map={0: 0.0, 1: 0.0, 2: 0.0})
        self.dynamic_almost_0 = Dynamic(values_map={0: 0.01, 1: 0.01, 2: 0.01})

        self.dynamic_1 = Dynamic(values_map={0: 1.0, 1: 2.0, 2: 3.0})
        self.dynamic_almost_1 = Dynamic(values_map={0: 1.01, 1: 2.01, 2: 3.01})

    def test_update(self):
        updated = self.dynamic_0.update([(0, 1.0), (1, 2.0), (2, 3.0)])
        self.assertEqual(self.dynamic_1, updated)

        partially_updated = self.dynamic_0.update([(1, 3.0)])
        expected = {0: 0.0, 1: 3.0, 2: 0.0}
        self.assertEqual(partially_updated, Dynamic(values_map=expected))

    def test_evaluate(self):
        np.testing.assert_array_almost_equal(
            self.dynamic_0.evaluate([0, 1, 2]),
            np.array([0.0, 0.0, 0.0])
        )

        np.testing.assert_array_almost_equal(
            self.dynamic_1.evaluate([0, 1, 2]),
            np.array([1.0, 2.0, 3.0])
        )

    def test_call(self):
        for i in range(0, 3):
            self.assertEqual(self.dynamic_0(i), 0.0)
            self.assertEqual(self.dynamic_1(i), float(i + 1))

    def test_within(self):
        self.assertTrue(self.dynamic_0.within(self.dynamic_0, tolerance=0.0))
        self.assertTrue(self.dynamic_0.within(self.dynamic_almost_0,
                                              tolerance=0.011))

        self.assertTrue(self.dynamic_1.within(self.dynamic_1, tolerance=0.0))
        self.assertTrue(self.dynamic_1.within(self.dynamic_almost_1,
                                              tolerance=0.011))
