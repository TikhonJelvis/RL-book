# Why do we want to learn "modern" Python?
  - learn some **tools** and **habits** to write "cleaner" code
    - make your life easier in the long term—how often is "one-off"
      code really a one-off?
    - make your life easier in the short term
      - easier to write and debug initially
      - easier to try different things
        - how much extra work would it be to run a few variations on
          your experiment?
        - how hard is it to go back and fix a major mistake you found
          at the last moment?
        - how likely are you to spot that mistake at all?
  - concretely:
    - break complex code up into small, simple, self-contained pieces
    - write code in a way that actively helps you catch mistakes
    - organize code around abstractions that map to how you're
      thinking about the problem
      - or even give you *better* ways to think about the problem

## Schedule:

### Day 1: Abstracting over Computation
    - we'll look at how to abstract over computation: how can we
      organize code for algorithms?
      - iterative algorithms
        - generators
        - iterators
        - iterator operations
      - first-class functions
        - using first-class functions (+ lambdas) on iterators

### Day 2: Abstracting over Data
    - we'll look at how to abstract over data: how do we structure the
      data our code works with
      - classes/types/etc

# Abstracting over Computation

## Iterative Algorithms
   - Many algorithms follow a simple pattern: do some set step
     repeatedly until we hit some stopping condition
   - Examples:
     - Gradient descent
     - Root finding
     - Dynamic programming/reinforcement learning
     - Fixed-point iteration
     - ...etc

### Sqrt
   - Iterative approximation of $\sqrt{n}$:
     $$
     x_{n + 1} = \frac{x_n + \frac{a}{x_n}}{2}
     $$
     - TODO: illustration
     - calculate $x_{n + 1}$ until $|x_{n + 1} - x| < \epsilon$
   - Simple code:

``` python
def sqrt(a: float) -> float:
    x = a / 2 # initial guess
    x_n = a
    while abs(x_n - x) > 0.01:
        x = x_n
        x_n = (x + (a / x)) / 2
    return x_n
```

**Side note**: type annotations let us specify what types of arguments a function takes and what type of values it returns. In base Python type annotations just act as documentation, but external tools can also use them to find inconsistencies in your code. For example, a typechecker would flag `sqrt("10")` as an error without needing to run the code.

We'll use type annotations in our example code both because it makes the inputs/outputs clear *and* because it is a useful habit when writing Python.

We could make ϵ a parameter:

``` python
def sqrt(a: float, threshold: float = 0.01) -> float:
    x = a / 2 # initial guess
    x_n = a
    while abs(x_n - x) > threshold:
        x = x_n
        x_n = (x + (a / x)) / 2
    return x_n
```

But what if this code is taking too many iterations to converge? This particular algorithm converges well, but real-world algorithms can have more complex behavior like getting stuck or thrashing around. Can we just get the value after a maximum number of iterations?

``` python
def sqrt(a: float, threshold: float = 0.01, iterations = None) -> float:
    steps = 0
    x = a / 2 # initial guess
    x_n = a
    while abs(x_n - x) > threshold:
        steps += 1
        if iterations and steps > iterations:
            break
        x = x_n
        x_n = (x + (a / x)) / 2
    return x_n
```

This is not great!

  - Our code to decide when to stop iterating takes more lines
    than the algorithm itself!
    - Easy source of bugs totally separate from the logic we
      actually care about
  - We'd have to copy-paste this for each new iterative algorithm
    we want to try
    - makes it harder to add new algorithms
    - makes it *much* harder to add new termination conditions!
      - this can get legitimately annoying for more complex
        real-world problems where the algorithms are less
        well-behaved
  - When the algorithm doesn't converge, how do we debug it?
    - How would you graph the steps of =sqrt(n)=?

### Separate Concepts

The fundamental problem is that our code has conflated two concepts that we want to consider separately: how we iterate at each step and how we stop iterating. Can we separate the code for *producing* values from the code for *consuming* them?

Python gives us two related tools to fix this: **iterators** and **generators**.

#### Iterators

Iterators are a way to *iteratively* produce values in Python. You might not realize it, but chances are your Python code uses iterators all the time. Python's `for` loop uses an iterator under the hood to get each value it's looping over—this is how `for` loops work for lists, dictionaries, sets, ranges and even custom types. Try it out:

``` python
for x in [3, 2, 1]: print(x)
for x in {3, 2, 1}: print(x)
for x in range(3): print(x)
```

Note how the iterator for the set (`{3, 2, 1}`) prints `1 2 3` rather than `3 2 1`—sets do not preserve the order in which elements are added, so they iterate over elements in some kind of internally defined order instead.

We can iterate over more than containers and numbers:

``` python
for char in "abc": print(char)
for line in open("example.txt"): print(line)
```

Note: none of these things are iterators *themselves*. Instead, values that we can iterate over are **iterables**, which means that we can get an iterator for the value.

``` python
x_list = [1,2,3]
print(x_list)

x_iterator = iter(list)
print(x_iterator)
```

Any value that supports `iter` is an **iterable**. Iterators are iterables themselves, where `iter` returns a reference to the same iterator (*not* a fresh copy).

``` python
print(iter(x_iterator))
```

#### Approximate `sqrt` as an Iterator

We can write our own custom iterators as well. Unlike data structures that store data directly, iterators can create values one-by-one—we don't have to have everything in memory all at once and we can produce an infinite (well, *unbounded*) number of values.

The first part of our solution for `sqrt`:

  1. Make the code for calculating subsequent $x_{n + 1}$ values into an iterator
 2. Make the code for deciding when to stop into functions that take an iterator as an input.

 To do 1, we'll have to understand what iterators *are* and how we can write our own.

#### Generators

Under the hood, a Python iterator is an object that implements a `__next__` method. `__next__` returns the next value or throws a `StopIteration` exception if there are no more values to return. In principle, we could write an iterator for our `sqrt` function this way:

```
class Sqrt:
    def __init__(self, a: float):
        self.a = a
        self.x = a / 2 # initial guess
        self.x_n = a

    def __next__(self) -> float:
        self.x = self.x_n
        self.x_n = (self.x + (self.a / self.x)) / 2
        return self.x_n
```

However, this is pretty awkward: now `Sqrt` is a class rather than a function for some reason and we have a bunch of distracting boilerplate in the definition. This solves the problems with our initial `sqrt` code but at the cost of making the actual algorithm code a lot more awkward.

Luckily, Python provides a much better way to write this style of code: **generators**. Generators are functions that behave like iterators, so we can write our iterative algorithm in a normal state, but instead of returning one value at the end, we can output a bunch of values as an iterator.

``` python
def sqrt(a: float) -> Iterator[float]:
    x = a / 2 # initial guess
    while True: # loop forever!
        x = (x + (a / x)) / 2
        yield x
```

**Side note**: Iterators can produce different types of values. To
express this in our type annotations, the `Iterator` type can take a
type *argument* that specifies what kind of values the iterator
produces (in this case, `float` values, hence `Iterator[float]`).

When you call `sqrt(n)`, you don't get a number out and it doesn't do any computation. Instead, you can get an iterator where each element corresponds to one iteration of the algorithm. Another way to think about it is that `yield` gives us a point where we can *pause* and *resume* the function: when you ask for a value from the iterator, it will run the code in the function until it hits a `yield` and will return that value. Then, when you get the *next* value from the iterator, the code will start again at that `yield` and run until it hits a `yield` again. (Aside: this is an example of a more general feature called a "coroutine" which other languages support as well.)

``` python
>>> approx = sqrt(37)
>>> next(approx)
10.25
>>> next(approx)
6.929878048780488
>>> next(approx)
6.134538672432479
>>> next(approx)
6.082981028300877
>>> next(approx)
6.082762534222396
```

Ultimately, this code is nice because we can write the iterative part of our algorithm in a natural style, but still separate the logic for *iterating* from the logic for what we *do* with each iteration—we can iterate until we hit some stopping point, graph intermediate values, print every 100th iteration... etc.

#### Iterator Functions

We can write functions that operate on iterators. For example, we might want a function that gets us just the first *n* values from an iterator:

``` python
from typing import TypeVar

A = TypeVar("A")

def take(iterator: Iterator[A], n: int) -> Iterator[A]:
    for _ in range(n):
        yield next(iterator)
```

**Side note**: Iterators have some type for their values, but functions like `take` work for *any* type value. The values produced by the iterator returned from `take` will have the same type as the input iterator. We need to create a "type variable" `A` to reflect that behavior.

We can try this with our `sqrt` example above:

``` python
>>> take(sqrt(37), 5)
<generator object take at 0x7f976590a510>
```

This gives us a generator object; Python does not print the *values* of the generator by default. To see all the values, we can turn the generator into a list:


``` python
>>> list(take(sqrt(37), 5))
[10.25, 6.929878048780488, 6.134538672432479, 6.082981028300877, 6.082762534222396]
```

##### Exercise

Implement `drop` that *drops* the first *n* elements of an iterator, returning an iterator that starts from the n + 1th element:

``` python
def drop(iterator: Iterator[A], n: int) -> Iterator[A]:
    ...
```

Write a `pairs` function that returns subsequent pairs of elements from an iterator, such that:

``` python
>>> list(pairs(iter(range(6))))
[(0, 1), (2, 3), (4, 5)]
>>> list(pairs(iter(range(5))))
[(0, 1), (2, 3)]
```

Hint: you'll have to handle the case where the input iterator runs out values. You can do this by catching the `StopIteration` exception or providing a default value to `next`:

``` python
>>> next(iter(range(0)))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
>>> next(iter(range(0)), None) == None
True
```

The Python `itertools` package has a number of useful functions for working with iterators. Let's try implementing a few of these functions ourselves to get some practice with iterators.

Implement `chain` which takes two iterators and outputs the values from the first iterator followed by the values from the second iterator. This is like append for iterators:

``` python
>>> list(chain(iter([1,2,3]), iter([4,5,6])))
[1, 2, 3, 4, 5, 6]

def chain(a: Iterator[A], b: Iterator[A]) -> Iterator[A]:
    ...
```

Implement `product` which gives us all possible pairs of values from an iterator (the Cartesian product), such that:

``` python
>>> list(product(iter([1,2])))
[(1, 1), (1, 2), (2, 1), (2, 2)]
```

``` python
from typing import Tuple

def product(a: Iterator[A]) -> Iterator[Tuple[A, A]]:
    ...
```

#### First-Class Functions

Another way we can abstract over computation in Python is by using functions *as values*. We can pass functions into other functions as arguments or store them in variables and data structures.

A number of standard library functions follow this pattern. For example, `map` and `filter` let us use a function to operate on the elements of an iterable like a list. Note that the result of a `map` or `filter` is an iterable itself rather than the type of container that was passed in:

``` python
def add1(x: float) -> float:
    return x + 1

>>> map(add1, [1,2,3])
<map object at 0x7f1820fe2350>
>>> map(add1, [1,2,3])
[2, 3, 4]
list(map(add1, {1,3,2}))
[2, 3, 4]
```

``` python
def is_even(x: float) -> bool:
    return x % 2 == 0

>>> list(filter(is_even, [1,2,3,4]))
[2, 4]
```

To make using functions like `map` and `filter` more convenient, Python has a shorthand syntax for defining a function without giving it a name. Functions like this are called "lambdas" (thanks to [the lambda calculus][lambda]), so Python uses the `lambda` keyword to introduce them:

``` python
>>> list(map(lambda x: x + 1, [1,2,3]))
[2, 3, 4]
>>> list(filter(lambda x: x % 2 == 0, [1,2,3,4]))
[2, 4]
```

#### Convergence

So far, we've seen one get a finite result from our infinite iterator: taking *n* values with `take`. However, we don't always know how many steps of the algorithm to run. Let's write a `converge` function that takes an iterator and stops producing values as soon as $|x_n - x_{n + 1}| \le \epsilon|$.

``` python
def converge(iterator, epsilon):
    ...
```

We can use this function to evaluate our approximate square root to some epsilon:

```
>>> converge(sqrt(37), 0.01)
6.08276253029822
```

Try implementing a version of `converge`. Hint: you can use the `pairs` function we implemented earlier to make this easier.

With functions like `take` and `converge`, the *user* of our `sqrt` function can decide how to approximate the final answer—when we're implementing `sqrt`, we don't have to know ahead of time how much precision our users will need.

Another advantage is that we can combine multiple functions like this. For example, we can use *both* `take` *and* `converge` so that we get an answer after 1000 steps even if it has not converged yet:

``` python
>>> converge(take(1000, sqrt(37)))
...
```

In more complicated algorithms, we might want to converge based on some criterion other than a fixed epsilon value. Try writing a generalized version of converge that takes a comparison function instead of a threshold.

## Iterators as Values

So far, I've talked about iterators as representing *computation* rather than *data*, but that's really just a matter of perspective. Iterators themselves are values that we can operate on just like any other value in Python, and we can use them to represent constructs we don't normally think of as computations.

For example, we can use an iterator to represent a series (ie an infinite polynomial). (Idea inspired by Douglas McIlroy's *Power Series, Power Serious* paper.)

Consider $$e^x = \sum_{n = 0}^{\infty} x^n / n! = 1 + x + \frac{1}{2}x^2 + \frac{1}{6}x^3 + \cdots$$

We can express this as an iterator where each value is a subsequent coefficient of the series:

``` python
import math

Series = Iterator[float]

def exp() -> Series:
    for n in itertools.count(start=0):
        yield 1 / math.factorial(n)
```

We can then write useful functions that take these iterators as inputs and produce them as outputs. For example, we can add two infinite series:

``` python
def add(a: Series, b: Series) -> Series:
    for a_i, b_i in zip(a, b):
        yield a_i + b_i
```

We can also implement other operations like multiplying by a constant or getting the first derivative of a series. Give it at try:

``` python
def scale(n: float, a: Series) -> Series:
    ...

def deriv(a: Series) -> Series:
    ...
```

Finally, for this to be useful, we want some way to approximate `f(x)` by taking `n` terms from a series:

``` python
def evaluate(a: Series, x: float, steps: int) -> float:
    ...
```

Bonus challenge: implement the series for `sin` and `cos`:

$$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$$

$$\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$$

Can you implement one in terms of the other?

Bonus (hard) challenge: implement multiplication (the [Cauchy Product][cauchy]) of two power series.

``` python
def multiply(a: Series, b: Series) -> Series:
    ...
```

Hint: consider how we could do this with operations we've previously defined for `Series`.

[cauchy]: https://en.wikipedia.org/wiki/Cauchy_product
