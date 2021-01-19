import unittest

import itertools

import chapter4


class TestChapter4(unittest.TestCase):
    def test_iterate(self):
        result = itertools.islice(
            chapter4.iterate(lambda x: x + 1, 0), 5
        )
        self.assertEqual(list(result), list(range(5)))
