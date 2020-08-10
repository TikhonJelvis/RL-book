from typing import Callable, Iterator, TypeVar

X = TypeVar('X')


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


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    '''Read from a stream of values until two consecutive values satisfy
    the given done function.
    Will error out if the stream runs out before the predicate is
    satisfied (including streams with 0 or 1 values) and will loop
    forever if the stream doesn't end *or* converge.
    '''
    a = next(values)
    yield a

    for b in values:
        yield b
        if done(a, b):
            break
        else:
            a = b


if __name__ == '__main__':
    import numpy as np
    x = 0.0
    values = converge(
        iterate(lambda y: np.cos(y), x),
        lambda a, b: np.abs(a - b) < 1e-3
    )
    for i, v in enumerate(values):
        print(f"{i}: {v:.3f}")
