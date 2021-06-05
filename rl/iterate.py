'''Finding fixed points of functions using iterators.'''
import itertools
from typing import Callable, Iterable, Iterator, Optional, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')


# It would be more efficient if you iterated in place instead of
# returning a copy of the value each time, but the functional version
# of the code is a lot cleaner and easier to work with.
def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    '''Find the fixed point of a function f by applying it to its own
    result, yielding each intermediate value.

    That is, for a function f, iterate(f, x) will give us a generator
    producing:

    x, f(x), f(f(x)), f(f(f(x)))...

    '''
    state = start

    while True:
        yield state
        state = step(state)


def last(values: Iterator[X]) -> Optional[X]:
    '''Return the last value of the given iterator.

    Returns None if the iterator is empty.

    If the iterator does not end, this function will loop forever.
    '''
    try:
        *_, last_element = values
        return last_element
    except ValueError:
        return None


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    '''Read from an iterator until two consecutive values satisfy the
    given done function or the input iterator ends.

    Raises an error if the input iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.

    '''
    a = next(values, None)
    if a is None:
        return

    yield a

    for b in values:
        yield b
        if done(a, b):
            return

        a = b


def converged(values: Iterator[X],
              done: Callable[[X, X], bool]) -> X:
    '''Return the final value of the given iterator when its values
    converge according to the done function.

    Raises an error if the iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    result = last(converge(values, done))

    if result is None:
        raise ValueError("converged called on an empty iterator")

    return result


def accumulate(
        iterable: Iterable[X],
        func: Callable[[Y, X], Y],
        *,
        initial: Optional[Y]
) -> Iterator[Y]:
    '''Make an iterator that returns accumulated sums, or accumulated
    results of other binary functions (specified via the optional func
    argument).

    If func is supplied, it should be a function of two
    arguments. Elements of the input iterable may be any type that can
    be accepted as arguments to func. (For example, with the default
    operation of addition, elements may be any addable type including
    Decimal or Fraction.)

    Usually, the number of elements output matches the input
    iterable. However, if the keyword argument initial is provided,
    the accumulation leads off with the initial value so that the
    output has one more element than the input iterable.

    '''
    if initial is not None:
        iterable = itertools.chain([initial], iterable)  # type: ignore

    return itertools.accumulate(iterable, func)  # type: ignore


if __name__ == '__main__':
    import numpy as np
    x = 0.0
    values = converge(
        iterate(lambda y: np.cos(y), x),
        lambda a, b: np.abs(a - b) < 1e-3
    )
    for i, v in enumerate(values):
        print(f"{i}: {v:.4f}")
