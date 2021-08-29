from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
import random
from typing import (Callable, Dict, Generic, Iterator, Iterable,
                    Mapping, Optional, Sequence, Tuple, TypeVar)

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

    @abstractmethod
    def expectation(
        self,
        f: Callable[[A], float]
    ) -> float:
        '''Return the expecation of f(X) where X is the
        random variable for the distribution and f is an
        arbitrary function from X to float

        '''
        pass

    def map(
        self,
        f: Callable[[A], B]
    ) -> Distribution[B]:
        '''Apply a function to the outcomes of this distribution.'''
        return SampledDistribution(lambda: f(self.sample()))

    def apply(
        self,
        f: Callable[[A], Distribution[B]]
    ) -> Distribution[B]:
        '''Apply a function that returns a distribution to the outcomes of
        this distribution. This lets us express *dependent random
        variables*.

        '''
        def sample():
            a = self.sample()
            b_dist = f(a)
            return b_dist.sample()

        return SampledDistribution(sample)


class SampledDistribution(Distribution[A]):
    '''A distribution defined by a function to sample it.

    '''
    sampler: Callable[[], A]
    expectation_samples: int

    def __init__(
        self,
        sampler: Callable[[], A],
        expectation_samples: int = 10000
    ):
        self.sampler = sampler
        self.expectation_samples = expectation_samples

    def sample(self) -> A:
        return self.sampler()

    def expectation(
        self,
        f: Callable[[A], float]
    ) -> float:
        '''Return a sampled approximation of the expectation of f(X) for some f.

        '''
        return sum(f(self.sample()) for _ in
                   range(self.expectation_samples)) / self.expectation_samples


class Uniform(SampledDistribution[float]):
    '''Sample a uniform float between 0 and 1.

    '''
    def __init__(self, expectation_samples: int = 10000):
        super().__init__(
            sampler=lambda: random.uniform(0, 1),
            expectation_samples=expectation_samples
        )


class Poisson(SampledDistribution[int]):
    '''A poisson distribution with the given parameter.

    '''

    λ: float

    def __init__(self, λ: float, expectation_samples: int = 10000):
        self.λ = λ
        super().__init__(
            sampler=lambda: np.random.poisson(lam=self.λ),
            expectation_samples=expectation_samples
        )


class Gaussian(SampledDistribution[float]):
    '''A Gaussian distribution with the given μ and σ.'''

    μ: float
    σ: float

    def __init__(self, μ: float, σ: float, expectation_samples: int = 10000):
        self.μ = μ
        self.σ = σ
        super().__init__(
            sampler=lambda: np.random.normal(loc=self.μ, scale=self.σ),
            expectation_samples=expectation_samples
        )


class Gamma(SampledDistribution[float]):
    '''A Gamma distribution with the given α and β.'''

    α: float
    β: float

    def __init__(self, α: float, β: float, expectation_samples: int = 10000):
        self.α = α
        self.β = β
        super().__init__(
            sampler=lambda: np.random.gamma(shape=self.α, scale=1/self.β),
            expectation_samples=expectation_samples
        )


class Beta(SampledDistribution[float]):
    '''A Beta distribution with the given α and β.'''

    α: float
    β: float

    def __init__(self, α: float, β: float, expectation_samples: int = 10000):
        self.α = α
        self.β = β
        super().__init__(
            sampler=lambda: np.random.beta(a=self.α, b=self.β),
            expectation_samples=expectation_samples
        )


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

    def sample(self) -> A:
        outcomes = list(self.table().keys())
        weights = list(self.table().values())
        return random.choices(outcomes, weights=weights)[0]

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


@dataclass(frozen=True)
class Constant(FiniteDistribution[A]):
    '''A distribution that has a single outcome with probability 1.

    '''
    value: A

    def sample(self) -> A:
        return self.value

    def table(self) -> Mapping[A, float]:
        return {self.value: 1}

    def probability(self, outcome: A) -> float:
        return 1. if outcome == self.value else 0.


@dataclass(frozen=True)
class Bernoulli(FiniteDistribution[bool]):
    '''A distribution with two outcomes. Returns True with probability p
    and False with probability 1 - p.

    '''
    p: float

    def sample(self) -> bool:
        return random.uniform(0, 1) <= self.p

    def table(self) -> Mapping[bool, float]:
        return {True: self.p, False: 1 - self.p}

    def probability(self, outcome: bool) -> float:
        return self.p if outcome else 1 - self.p


@dataclass
class Range(FiniteDistribution[int]):
    '''Select a random integer in the range [low, high), with low
    inclusive and high exclusive. (This works exactly the same as the
    normal range function, but differently from random.randit.)

    '''
    low: int
    high: int

    def __init__(self, a: int, b: Optional[int] = None):
        if b is None:
            b = a
            a = 0

        assert b > a

        self.low = a
        self.high = b

    def sample(self) -> int:
        return random.randint(self.low, self.high - 1)

    def table(self) -> Mapping[int, float]:
        length = self.high - self.low
        return {x: 1 / length for x in range(self.low, self.high)}


class Choose(FiniteDistribution[A]):
    '''Select an element of the given list uniformly at random.

    '''

    options: Sequence[A]
    _table: Optional[Mapping[A, float]] = None

    def __init__(self, options: Iterable[A]):
        self.options = list(options)

    def sample(self) -> A:
        return random.choice(self.options)

    def table(self) -> Mapping[A, float]:
        if self._table is None:
            counter = Counter(self.options)
            length = len(self.options)
            self._table = {x: counter[x] / length for x in counter}

        return self._table

    def probability(self, outcome: A) -> float:
        return self.table().get(outcome, 0.0)


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

    def table(self) -> Mapping[A, float]:
        return self.probabilities

    def probability(self, outcome: A) -> float:
        return self.probabilities.get(outcome, 0.)
