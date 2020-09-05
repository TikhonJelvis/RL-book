from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import random
from typing import (Callable, Dict, Generic, Iterator,
                    Mapping, Set, Sequence, Tuple, TypeVar)

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

    def sample_n(self, n: int) -> Sequence[A]:
        '''Return n samples from this distribution.'''
        return [self.sample() for _ in range(n)]

    def expectation(self, f: Callable[[A], float]) -> float:
        '''Return an approximation of the expected value of f(X) for some f.

        The default implementation of this method samples the
        underlying distribution some number of times to calcuate the
        expectation.

        '''
        # TODO: Revisit # of samples
        samples = 100000
        return sum(f(self.sample()) for _ in range(0, samples)) / samples


class SampledDistribution(Distribution[A]):
    '''A distribution defined by a function to sample it.

    '''

    def __init__(self, sampler: Callable[[], A]):
        self.sampler = sampler

    def sample(self) -> A:
        return self.sampler()


class Uniform(Distribution[float]):
    '''Sample a uniform float between 0 and 1.

    '''

    def sample(self):
        return random.uniform(0, 1)


class Poisson(Distribution[int]):
    '''A poisson distribution with the given parameter.

    '''

    λ: float

    def __init__(self, λ: float):
        self.λ = λ

    def sample(self):
        return np.random.poisson(lam=self.λ)


class Gaussian(Distribution[float]):
    '''A Gaussian distribution with the given μ and σ.'''

    μ: float
    σ: float

    def __init__(self, μ: float, σ: float):
        self.μ = μ
        self.σ = σ

    def sample(self):
        return np.random.normal(loc=self.μ, scale=self.σ)


class FiniteDistribution(Distribution[A], ABC):
    '''A probability distribution with a finite number of outcomes, which
    means we can render it as a PDF or CDF table.

    '''
    @abstractmethod
    def table(self) -> Mapping[A, float]:
        '''Returns a tabular representation of the probability density
        function (PDF) for this distribution.

        '''
        pass

    def probability(self, outcome: A) -> float:
        '''Returns the probability of the given outcome according to this
        distribution.

        '''
        return self.table()[outcome]

    def map(self, f: Callable[[A], B]) -> FiniteDistribution[B]:
        '''Return a new distribution that is the result of applying a function
        to each element of this distribution.

        '''
        result: Dict[B, float] = defaultdict(float)

        for x, p in self:
            result[f(x)] += p

        return Categorical(result)

    # TODO: Can we get rid of f or make it optional? Right now, I
    # don't think that's possible with mypy.
    def expectation(self, f: Callable[[A], float]) -> float:
        '''Calculate the expected value of the distribution, using the given
        function to turn the outcomes into numbers.

        '''
        return sum(p * f(x) for x, p in self)

    def __iter__(self) -> Iterator[Tuple[A, float]]:
        return iter(self.table().items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FiniteDistribution):
            return self.table() == other.table()
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.table())


# TODO: Rename?
class Constant(FiniteDistribution[A]):
    '''A distribution that has a single outcome with probability 1.

    '''
    value: A

    def __init__(self, value: A):
        self.value = value

    def sample(self) -> A:
        return self.value

    def table(self) -> Mapping[A, float]:
        return {self.value: 1}

    def probability(self, outcome: A) -> float:
        return 1. if outcome == self.value else 0.


class Bernoulli(FiniteDistribution[bool]):
    '''A distribution with two outcomes. Returns True with probability p
    and False with probability 1 - p.

    '''

    def __init__(self, p: float):
        self.p = p

    def sample(self) -> bool:
        return random.uniform(0, 1) <= self.p

    def table(self) -> Mapping[bool, float]:
        return {True: self.p, False: 1 - self.p}

    def probability(self, outcome: bool) -> float:
        return self.p if outcome else 1 - self.p


class Choose(FiniteDistribution[A]):
    '''Select an element of the given list uniformly at random.

    '''

    options: Set[A]

    def __init__(self, options: Set[A]):
        self.options = options

    def sample(self) -> A:
        return random.choice(list(self.options))

    def table(self) -> Mapping[A, float]:
        length = len(self.options)
        return {x: 1.0 / length for x in self.options}

    def probability(self, outcome: A) -> float:
        p = 1.0 / len(self.options)
        return p if outcome in self.options else 0.0


class Categorical(FiniteDistribution[A]):
    '''Select from a finite set of outcomes with the specified
    probabilities.

    '''

    probabilities: Mapping[A, float]

    def __init__(self, distribution: Mapping[A, float]):
        total = sum(distribution.values())
        # Normalize probabilities to sum to 1
        self.probabilities = {outcome: probability / total
                              for outcome, probability in distribution.items()}

    def sample(self) -> A:
        outcomes = list(self.probabilities.keys())
        weights = list(self.probabilities.values())
        return random.choices(outcomes, weights=weights)[0]

    def table(self) -> Mapping[A, float]:
        return self.probabilities

    def probability(self, outcome: A) -> float:
        return self.probabilities.get(outcome, 0.)
