# start snippet iterate
from typing import Callable, Iterator, TypeVar
X = TypeVar('X')


def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    state = start

    while True:
        yield state
        state = step(state)
# end snippet iterate


# start snippet converge
def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    a = next(values)
    yield a

    for b in values:
        if done(a, b):
            break

        a = b
        yield b
# end snippet converge
