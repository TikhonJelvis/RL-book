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

Iterators don't have to iterate over a concrete data structure like a list, set or range: we can write our own custom logic to produce values instead. Unlike lists, iterators can create values one-by-one—we don't have to have everything in memory all at once and we can produce an infinite (well, *unbounded*) number of values.

The first part of our solution for `sqrt`:

  1. Make the code for calculating subsequent $x_{n + 1}$ values into an iterator
 2. Make the code for deciding when to stop into functions that take an iterator as an input.

#### Convergence

#### Generators

Under the hood, a Python iterator is an object that implements a `__next__` method. `__next__` returns the next value or throws a `StopIteration` exception if there are no more values to return. In principle, we could write an iterator for our `sqrt` function this way:

```
class Sqrt:
    def __init__(self, a: float):
        self.a = a
        self.x = a / 2 # initial guess
        self.x_n = a

    def __next__(self):
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

When you call `sqrt(n)`, you don't get a number out and it doesn't do any computation. Instead, you can get an iterator where each element corresponds to one iteration of the algorithm. Another way to think about it is that `yield` gives us a point where we can *pause* and *resume* the function: when you ask for a value from the iterator, it will run the code in the function until it hits a `yield` and will return that value. Then, when you get the *next* value from the iterator, the code will start again at that `yield` and run until it hits a `yield` again. (Aside: this is an example of a more general feature called a "coroutine" which other languages support as well.)

Ultimately, this code is nice because we can write the iterative part of our algorithm in a natural style, but still separate the logic for *iterating* from the logic for what we *do* with each iteration—we can iterate until we hit some stopping point, graph intermediate values, print every 100th iteration... etc.

## Iterators as Values

So far, I've talked about iterators as representing *computation* rather than *data*, but that's really just a matter of perspective. Iterators themselves are values that we can operate on just like any other value in Python, and we can use them to represent constructs we don't normally think of as computations.

For example, we can use an iterator to represent a series (ie an infinite polynomial). Consider $$e^x = \sum_{n = 0}^{\infty} x^n / n! = 1 + x + \frac{1}{2}x^2 + \frac{1}{6}x^3 + \cdots$$

We can express this as an iterator where each value is a subsequent coefficient of the polynomial:

``` python
def exp(x) -> Iterator[float]:
    for n in itertools.count(start=0):
        yield (x ** n) / fact(n) # not efficient!
```
