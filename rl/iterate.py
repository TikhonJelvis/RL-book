'''Finding fixed points of functions using iterators.'''
import itertools
from typing import Callable, Iterator, Tuple, TypeVar

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


def last(values: Iterator[X]) -> X:
    '''Return the last value of the given iterator.

    Raises an error if the iterator is empty.

    If the iterator does not end, this function will loop forever.
    '''
    *_, last_element = values
    return last_element


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
        if done(a, b):
            return

        a = b
        yield b


def converged(values: Iterator[X], done: Callable[[X, X], bool]) -> X:
    '''Return the final value of the given iterator when its values
    converge according to the done function.

    Raises an error if the iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    return last(converge(values, done))


def cumulative_reward(rewards: Iterator[Tuple[X, float]],
                      γ: float = 1) -> Iterator[Tuple[X, float]]:
    '''Given an iterator of rewards, calculate the cumulative reward
    (optionally with exponential discounting).

    Arguments:
      rewards -- instantaneous rewards
      γ -- the discount factor (0 < γ ≤ 1), default: 1
    '''
    def accum(s_r_acc, s_r):
        _, r_acc = s_r_acc
        s, r = s_r
        return s, r_acc + γ * r

    return itertools.accumulate(rewards, accum)
