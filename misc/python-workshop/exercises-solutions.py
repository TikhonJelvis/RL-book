from typing import Iterator, TypeVar

A = TypeVar("A")

def drop(iterator: Iterator[A], n: int) -> Iterator[A]:
    for _ in range(n):
        next(iterator)

    return iterator

# You can make this work for any iterable, not just iterators
# specifically:
from typing import Iterable

def drop_iterable(iterable: Iterable[A]) -> Iterator[A]:
    iterator = iter(iterable)

    for _ in range(n):
        next(i)

    return iterator


from typing import Tuple

def pairs(iterable: Iterable[A]) -> Iterable[Tuple[A, A]]:
    iterator = iter(iterable)
    try:
        for a in iterator:
            yield (a, next(iterator))
    except StopIteration:
        return

def pairs_window(iterable: Iterable[A]) -> Iterator[Tuple[A, A]]:
    iterator = iter(iterable)
    try:
        prev = next(iterator)
        for a in iterator:
            yield (prev, a)
            prev = a
    except StopIteration:
        return


# Tuple[A, ...] is a special type hint to represent tuples of any size
# with elements of type A—don't worry if you didn't know how to write
# type hints for this one, we never covered it!
def sliding_window(iterable: Iterable[A], n: int) -> Iterator[Tuple[A, ...]]:
    iterator = iter(iterable)

    try:
        window = ()
        for _ in range(n):
            window += (next(iterator),)
        yield window

        for a in iterator:
            window = (window + (a,))[1:]
            yield window
    except StopIteration:
        return


def converged(values: Iterator[float], threshold: float) -> float:
    for x_1, x_2 in pairs_window(values):
        if abs(x_1 - x_2) < threshold:
            return x_2


def chain(a: Iterable[A], b: Iterable[A]) -> Iterable[A]:
    for x in a:
        yield x

    for x in b:
        yield x

def sin() -> Series:
    for n in itertools.count(start=0):
        yield 0
        yield ((-1) ** n) / math.factorial(2 * n + 1)

def cos() -> Series:
    for n in itertools.count(start=0):
        yield ((-1) ** n) / math.factorial(2 * n)
        yield 0

# idea: we can see a series a recursively as its first term a_0 and
# the rest a_rest
#
# a = a_0 + x * a_rest
# b = b_0 + x * b_rest
# a * b = a_0 * b_0 + a_rest * b + a_0 * b_rest
#
# to make this work we need to be able to use an iterator twice, which
# we can do with itertools.tee
def multiply(a: Series, b: Series) -> Series:
    a_0 = next(a)
    a_rest = a

    b, b_ = itertools.tee(b)
    b_0 = next(b_)
    b_rest = b_

    yield a_0 * b_0
    for term in add(multiply(a_rest, b), scale(a_0, b_rest)):
        yield term


# same core idea as multiply, but doing long division algorithm:
#
# a = a_0 + x * a_rest
# b = b_0 + x * b_rest
#
# a / b = q, a = qb
# q = a_0 / b_0 + x * (1 / b_0) * (a_rest - q * b_rest)
def divide(a: Series, b: Series) -> Series:
    a_0 = next(a)
    a_rest = a

    b_0 = next(b)
    b_rest = b

    yield a_0 / b_0
    for term in scale(1/b_0, add(a_rest, scale(-1, mulitply(divide(a, b), b_rest)))):
        yield term


# the recursion gets pretty confusing in this one, so try drawing out
# a step-by-step diagram of how each term in the result gets
# calculated in terms of the two inputs
