'''Finding fixed points of functions using iterators.'''
import functools
import itertools
import math
from typing import (Callable, Iterable, Iterator, Optional, Tuple, TypeVar)

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
        if done(a, b):
            return

        a = b
        yield b


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


# TODO: Unify with mdp.returns (using a protocol)?
def returns(
        rewards: Iterable[Tuple[X, float]],
        γ: float = 1,
        tolerance: float = 1e-6
) -> Iterator[Tuple[X, float]]:
    '''Given an iterator of states and rewards, calculate the return of
    the first N states.

    Arguments:
    rewards -- instantaneous rewards
    γ -- the discount factor (0 < γ ≤ 1), default: 1
    n_states -- how many states to calculate the return for, default: 1

    '''
    # Ensure that this logic works correctly whether rewards is an
    # iterator or an iterable (ie a list).
    rewards = iter(rewards)

    max_steps = None
    if γ < 1:
        max_steps = round(math.log(tolerance) / math.log(γ))
        rewards = itertools.islice(rewards, 2 * max_steps)

    *initial, (last_s, last_r) = list(itertools.islice(rewards, max_steps))

    def accum(r_acc, r):
        return r_acc + γ * r
    final_return = functools.reduce(accum, (r for _, r in rewards), 0.0)

    def update(acc, point):
        _, return_ = acc
        s, reward = point

        return (s, reward + γ * return_)
    return itertools.accumulate(reversed(initial), update,
                                initial=(last_s, last_r + γ * final_return))
