from abc import ABC, abstractmethod
import random
from typing import Callable, Generic, Iterable, List, Set, Tuple, TypeVar

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
    def table(self) -> List[Tuple[A, float]]:
        '''Returns a tabular representaiton of the probability density
        function (PDF) for this distribution.

        '''
        pass

    @abstractmethod
    def probability(self, outcome: A) -> float:
        '''Returns the probability of the given outcome according to this
        distribution.

        '''
        pass


class Constant(FiniteDistribution[A]):
    '''A distribution that has a single outcome with probability 1.

    '''
    value: A

    def __init__(self, value: A):
        self.value = value

    def sample(self) -> A:
        return self.value

    def table(self) -> List[Tuple[A, float]]:
        return [(self.value, 1)]

    def probability(self, outcome: A) -> float:
        return 1 if outcome == self.value else 0.0


class Bernoulli(FiniteDistribution[bool]):
    '''A distribution with two outcomes. Returns True with probability p
    and False with probability 1 - p.

    '''
    def __init__(self, p: float):
        self.p = p

    def sample(self) -> bool:
        return random.uniform(0, 1) < self.p

    def table(self) -> List[Tuple[bool, float]]:
        return [(True, self.p), (False, 1 - self.p)]

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

    def table(self) -> List[Tuple[A, float]]:
        length = len(self.options)
        return [(x, 1.0 / length) for x in self.options]

    def probability(self, outcome: A) -> float:
        p = 1.0 / len(self.options)
        return p if outcome in self.options else 0.0


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

        # Normalize probabilities to sum to 1
        total = sum(self.probabilities)
        self.probabilities = [p / total for p in self.probabilities]

    def sample(self) -> A:
        return random.choices(self.outcomes, weights=self.probabilities)[0]

    def table(self) -> List[Tuple[A, float]]:
        return list(zip(self.outcomes, self.probabilities))

    def probability(self, outcome: A) -> float:
        try:
            i = self.outcomes.index(outcome)
            return self.probabilities[i]
        except ValueError:
            return 0.0
