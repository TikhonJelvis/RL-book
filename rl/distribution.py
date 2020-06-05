from abc import ABC, abstractmethod
import numpy as np
import random
from typing import Callable, Generic, Iterable, List, Tuple, TypeVar

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


class SampledDistribution(Distribution[A]):
    '''A distribution defined by a function to sample it.

    '''
    def __init__(self, sampler: Callable[[], A]):
        self.sampler = sampler

    def sample(self):
        return self.sampler()


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

    options: List[A]

    def __init__(self, options: List[A]):
        self.options = options

    def sample(self) -> A:
        return self.options[random.randrange(len(self.options))]

    def to_pdf(self) -> List[Tuple[A, float]]:
        length = len(self.options)
        return [(x, 1.0 / length) for x in self.options]


class Categorical(FiniteDistribution[A]):
    '''Select from a finite set of outcomes with the specified
    probabilities.

    '''

    outcomes: List[A]
    probabilities: List[float]

    def __init__(self, distribution: Iterable[Tuple[A, float]]):
        self.outcomes = []
        self.probabilities = []
        
        for outcome, probability in distribution:
            self.outcomes += [outcome]
            self.probabilities += [probability]

    def sample(self) -> A:
        return np.random.default_rng().choice(self.outcomes,
                                              size=1,
                                              p=self.probabilities)[0]

    def to_pdf(self) -> List[Tuple[A, float]]:
        return list(zip(self.outcomes, self.probabilities))
