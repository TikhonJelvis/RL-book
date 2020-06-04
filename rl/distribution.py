from abc import ABC, abstractmethod
import random
from typing import Callable, Generic, List, Tuple, TypeVar

A = TypeVar('A')

B = TypeVar('B')


class Distribution(ABC, Generic[A]):
    '''A probability distribution that we can sample.

    '''
    @abstractmethod
    def sample(self) -> A:
        '''Return a random sample from this distribution.

        '''
        pass


class FiniteDistribution(Distribution[A], ABC):
    '''A probability distribution with a finite number of outcomes, which
    means we can render it as a PDF or CDF table.

    '''
    @abstractmethod
    def to_pdf(self) -> List[Tuple[A, float]]:
        '''Returns a tabular representaiton of the probability density
        function (PDF) for this distribution.

        '''
        pass


class Bernoulli(FiniteDistribution[bool]):
    '''A distribution with two outcomes. Returns True with probability p
    and False with probability 1 - p.

    '''
    def __init__(self, p: float):
        self.p = p

    def sample(self) -> bool:
        return random.uniform(0, 1) < self.p

    def to_pdf(self) -> List[Tuple[bool, float]]:
        return [(True, self.p), (False, 1 - self.p)]


class Choose(FiniteDistribution[A]):
    '''Select an element of the given list uniformly at random.

    '''
    def __init__(self, options: List[A]):
        self.options = options

    def sample(self) -> A:
        return self.options[random.randrange(len(self.options))]

    def to_pdf(self) -> List[Tuple[A, float]]:
        length = len(self.options)
        return [(x, 1.0 / length) for x in self.options]
