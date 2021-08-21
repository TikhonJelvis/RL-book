## Programming and Design {#sec:python .unnumbered}

> The programmer, like the poet, works only slightly removed from pure thought-stuff. He builds his castles in the air, from air, creating by exertion of the imagination. Few media of creation are so flexible, so easy to polish and rework, so readily capable of realizing grand conceptual structures.

- Fred Brooks, *The Mythical Man-Month*

Programming requires creativity with few constraints: imagine something and you can probably build it—in *many* different ways. Liberating and gratifying, but also challenging. Just like starting a novel from a blank page or a painting from a blank canvas, a new program is so open that it's a bit intimidating. Where do you start? What will the system look like? How will you get it *right*? How do you split your problem up? How do you prevent your code from evolving into a complete mess?

There's no easy answer. Programming is inherently iterative—we rarely get the right design at first, but we can always edit code and refactor over time. But iteration itself is not enough; just like a painter needs technique and composition, a programmer needs patterns and design.

Existing teaching resources tend to deemphasize programming techniques and design. Theory- and algorithm-heavy books show algorithms as self-contained procedures written in pseudocode, without the broader context—and corresponding design questions—of a real codebase. Newer AI and ML materials sometimes take a different tack and provide real code examples using industry-strength frameworks, but rarely touch on software design questions.

In this book, we take a third approach. Starting *from scratch*, we build a Python framework that reflects the key ideas and algorithms in the text. The abstractions we define map to the key concepts we introduce; how we structure the code maps to the relationships between those concepts.

Unlike the pseudocode approach, we do not implement algorithms in a vacuum; rather, each algorithm builds on abstractions introduced earlier in the book. By starting from scratch—rather than using an existing ML framework—we keep the code reasonably simple, without needing to worry about specific examples going out of date. We can focus on the concepts important to the text while teaching programming and design *in situ*, demonstrating an intentional approach to code design.

### Code Design

How can take a complex domain like reinforcement learning and turn it into code that is easy to understand, debug and extend? How can we split this problem into manageable pieces? How do those pieces interact?

There is no single easy answer to these questions. As in any creative endeavor, no two programming challenges are identical and people can reasonably find different solutions. A solid design is not going to be completely clear up-front; as we worked on the code for this book, we constantly revisited previous design decisions as we either came up with better solutions or found new requirements to address.

We might have no easy answers, but we do have general patterns and principles that—in our experience—consistently produce quality code. Taken together, these ideas form a philosophy of code design oriented around defining and combining **abstractions** that reflect how we think about our domain. Since code itself can point to specific design ideas and capabilities, there's a feedback loop: expanding the abstractions we've designed can help us find new algorithms and functionality, improving our understanding of the domain.

Just what *is* an abstraction? An appropriately abstract question! An abstraction is a "compound idea": a single concept that combines multiple separate ideas into one. Humans have an inherently limited working memory; we can only keep a small number of distinct things in mind at any given time. If we think of working memory as having some small number of "slots", an abstraction lets us take several distinct ideas that would normally take up several slots of working memory and consider them as a single unit, taking up just *one* slot. Abstractions let us generalize beyond individual concrete objects—we can consider the abstract idea of horses in general rather than needing to consider every single horse as a separate, unique animal. The only way we can understand and interact with anything meaningfully complex is by developing mental abstractions to manage the complexity.

How does this translate to programming? Just as we need to organize complex ideas to think about them, we need to organize complex code to write and understand it. The computer itself does not need structure or organization to run code—it is happy mindlessly following exact instructions with no extra structure. Modern CPUs run *billions* of instructions a second, dealing with billions and billions of bits of information. Humans can't keep up!

The same way our limited working memory pushes us to use mental abstractions, it pushes us to organize code around abstractions as well. How do you understand code? Do you run the code in your head? This is how most people start, but it's difficult at smaller scales and quickly becomes infeasible as the amount of logic and information grows too large to mentally track. Code abstractions let us overcome this limit by logically grouping information and logic so that we can think about it holistically rather than tracking every pieces separately.

The details may differ, but designing code around abstractions that correspond to a solid mental model of the domain works well in any area and with any programming language. Designing and using abstractions like that isn't always easy and can take more up-front thinking than other approaches—but, done well, it pays massive dividends over the life of a project. The goal of writing clean code with well-designed abstraction is to make it easier for ourselves to deal with complexity; this matters as much for "one-off" experimental code as it does for large software engineering efforts done in teams!

### Probability

Let's jump into a real example of code design to see the process in practice. One of the most important building blocks for reinforcement learning—really statistics and machine learning in general—is probability. How do we want to work with probability and probability distributions in our code?

One approach would be to keep probability implicit. Whenever we have a random variable, we could call a function and get a random result. If we were writing a Monopoly game with two six-side dice, we would define it like this:

``` python
from random import randint

def six_sided()
    return randint(1, 6)

def roll_dice():
    return six_sided() + six_sided()
```

This works, but it's pretty limited. We can't do anything except get one outcome at a time. More importantly, this code doesn't reflect how we *think* about probabilities: there's *randomness* but we never even mentioned distributions. We have outcomes and we have a function we can call repeatedly, but there's no way to tell that function apart from a function that has nothing to do with probability but just happens to return an integer.

How can we write code to get the expected value of a distribution? If we have a parametric distribution—a distribution like Poisson or Gaussian characterized by parameters—can we get the parameters out if we need them?

Since distributions are implicit in the code, the *intentions* of the code aren't clear and it is hard to write code that generalizes over distributions. Distributions are absolutely crucial for machine learning, so this is not a great starting point.

#### A Distribution Interface

To address these problems, let's define an abstraction for probability distributions.

How do we represent a distribution in code? What can we *do* with distributions? That depends on exactly what kind of distribution we're working with. If we know something about the structure of a distribution—perhaps it's a Poisson distribution where λ=5, perhaps it's an empirical distribution where we've measured probabilities for each outcome—we might be able to do quite a bit. We might be able to produce an exact PDF or CDF, calculate expectations exactly and do various operations efficiently. But that isn't the case for all the distributions we work with! What if the distribution comes from a complicated simulation? At the extreme, we might not be able to do anything except draw samples from the distribution.

The least common denominator is sampling. We can sample distributions were we don't know enough to do anything else, and we can sample distributions where we know the exact form and parameters. In Python, we can express this idea with a class:

``` python
from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass
```

This class gives us a minimal **interface**: a definition of what we require for something to qualify as a distribution. Any kind of distribution we implement in the future will be able to, at minimum, generate samples; when we write functions that sample distributions, they can require their inputs to inherit from `Distribution`.

Note how this class itself does not actually implement `sample`. When we have a specific kind of distribution, we will be able to sample it, but the `Distribution` class defines the *abstract idea* of what constitutes a distribution rather than defining a *specific* kind of distribution. To reflect this in Python, we've made `Distribution` an **ABC** (Abstract Base Class), with `sample` being an *abstract* method—a method without an implementation. Like their names imply, abstract classes and abstract methods are features that Python provides to help us define abstractions. We can define the `Distribution` class to structure the rest of our probability code before we define any specific distributions.

#### A Concrete Distribution

Let's say that we wanted to model six-sided dice, just like we did with our original example. We could do this by defining a `Die` class that represents an n-sided die and inherits from `Distribution`:

``` python
class Die(Distribution):
    def __init__(self, sides):
        self.sides = sides
        
    def sample(self):
        return random.randint(1, self.sides)


six_sided = Die(6)

def roll_dice():
    return six_sided.sample() + six_sided.sample()
```

So far, we've written a bunch of extra code to... exactly recreate the function we started with. What was the point?

The key difference from the original code is that we now have a value that represents the *distribution* of rolling a die, not just the outcome of a single roll. When we come across a `Die` object in the code we know it represents an n-sided die. We now have a place to add other functionality for n-sided dice, like adding a `__repr__` method so that the object has a readable string representation:

``` python
class Die(Distribution):
    ...
    def __repr__(self):
        return f"Die(sides={self.sides})"
```

When `six_sided` was a function, `print(six_sided)` would give us:

``` python
>>> print(six_sided)
<function six_sided at 0x7f00ea3e3040>
```

With a `Die` class and a `__repr__` method, we get[^f-strings]:

``` python
>>> print(six_sided)
Die(sides=6)
```

This seems small but makes debugging *much* easier, especially as the codebase gets larger and more complex.

[^f-strings]: Our definition of `__repr__` used a Python feature called an "f-string". By putting an `f` in front of a string literal, we can inject the value of a Python expression like `self.sides` into the string by putting the expression between `{` and `}`.

#### Types

With dice, we know the outcome will be an `int`. If we have a coin flip, it will be "heads" or "tails". A normal distribution produces a `double`. In general, the outcome can be anything. Our abstract `Distribution` class did not say anything about this—we have an abstract `sample` method but nothing telling us what that method returns. This works, but it gets confusing: some code we write will work for any kind of distribution, some code needs distributions that return numbers, other code will need something else... No matter what, `sample` better return *something*, otherwise the distribution doesn't really make sense!

We could specify this information in the docstring for `Distribution` and its subclasses. That would be enough to tell programmers what the type should be, but not enough to integrate with tools or check for incompatibilities. Luckily, Python gives us a language feature to help with this: **type annotations**. When we define a function or a method, we can specify the types of inputs it needs and the type of value it returns. For example, `sides` in the `Die` class needs to be an `int` and sample will always return an `int`. Annotations let us make this explicit:

``` python
class Die(Distribution):
    def __init__(self, sides: int):
        self.sides = sides
        
    def sample(self) -> int:
        return random.randint(1, sides)
```

Right away, this acts as documentation—anybody who wants to construct a `Die` object will know they have to specify an `int` for `sides`. However, Python doesn't do anything to prevent you from passing in an incompatible value for `sides`; you won't realize the mistake until you call `sample` and get an error—with a confusing error message to boot:

``` python
>>> bad_die = Die("foo")
>>> bad_die.sample()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../rl/probability.py", line 15, in sample
    return random.randint(1, self.sides)
  File ".../lib/python3.8/random.py", line 248, in randint
    return self.randrange(a, b+1)
TypeError: can only concatenate str (not "int") to str
```

The code fails somewhere inside the implementation of `random.randint`. Even on this small scale, it would take a bit of detective work to figure out what the *actual* problem is; imagine what this would be like in a much larger codebase!

Luckily, we can use type annotations with an external **type checker** like mypy or pyright to catch these problems *before we even run our code*. If we annotate that the `sides` argument needs to be an `int` and later write `Die("foo")`, we would get an error right in our editor:

> probability.py:21: error: Argument 1 to "Die" has incompatible type "str"; expected "int"

Unlike the runtime exception which pointed us to some line inside the Python standard library, this error points to exactly the part of the code where the problems occurs.

By using type annotations throughout our code, we can both make the *intentions* of the code clearer to the reader while also giving enough structure for automated tools like mypy to catch mismatches like this for us.

#### Type Variables

Type annotations are pretty clear when a function always returns a specific type: we always know that sampling a `Die` will give us an `Int`. But how do we handle general-purpose abstractions like `Distribution`? While *specific* distributions will have a *specific* type of outcome, the type will vary depending on the distribution. To deal with this, we need **type variables**: variables that stand in for *some* type that might be different each time the code is used. Type variables are also known as "generics" because they let us write classes that generically work for any type.

To add annotations to the abstract `Distribution` class, we will need to define a type variable for the outcomes of the distribution, then tell Python that `Distribution` is "generic" in that type:

``` python
from typing import Generic, TypeVar

A = TypeVar("A") # Defining a type variable named "A"

class Distribution(ABC, Generic[A]): # Distribution is "generic in A"
    @abstractmethod
    def sample(self) -> A: # Sampling returns a value of type A
        pass
```

Traditionally, type variables have one-letter capitalized names—although it's perfectly fine to use full words if that would make the code clearer. In this code, we've defined a type variable `A` and used `Generic[A]` to specify that `Distribution` uses this variable. The type `Distribution[Int]` would cover distributions that have integer outcomes, `Distribution[Double]` would cover distributions with doubles as outcomes... etc. The `sample` method returns a value of type `A`, which means that it will always return a value that has the type of outcome specified for a `Distribution`. After doing this, we would updated our definition of `Die` to specify that it is a subclass of `Distribution[Int]` in particular:

``` python
class Die(Distribution[int]):
    ...
```
