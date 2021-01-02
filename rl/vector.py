'''Types and functions for working with vector spaces.'''
from abc import abstractmethod
from typing import TypeVar

Self = TypeVar('Self', bound='VectorSpace')


class VectorSpace:
    '''An interface for types that form a vector space, with floats as
    scalars.

    A type V forms a vector space with some type of scalars F (F =
    float for this interface) if it has the following operations:

      - +: V × V → V
      - *: F × V → V
      - -: V → V

    and a distinguished value 0 : V, satisfying:

      - associativity: u + (v + w) = (u + v) + w
      - commutativity: u + v = v + u
      - identity: v + 0 = v
      - inverse: v + (-v) = 0
      - compatibility: a * (b * v) = (a * b) * v
      - identity: 1 * v = v
      - distributivity: a(u + v) = (a * u) + (a * v)
      - distributivity: (a + b) * v = (a * v) + (b * v)

    where u, v, w are of type V and a, b are scalars.

    '''

    @abstractmethod
    def add(self: Self, other: Self) -> Self:
        '''Addition of two vectors. '''

    @abstractmethod
    def inverse(self: Self) -> Self:
        '''Inverse of this vector.'''

    @abstractmethod
    def scalar_multiply(self: Self, scalar: float) -> Self:
        '''Left or right multiplication by a scalar (float or int).'''

    def __add__(self: Self, other: Self) -> Self:
        return self.add(other)

    def __neg__(self: Self) -> Self:
        return self.inverse()

    def __mul__(self: Self, other: float) -> Self:
        return self.scalar_multiply(other)

    def __rmul__(self: Self, other: float) -> Self:
        return self.scalar_multiply(other)
