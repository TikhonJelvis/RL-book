## Programming and Design {#sec:python .unnumbered}

> The programmer, like the poet, works only slightly removed from pure thought-stuff. He builds his castles in the air, from air, creating by exertion of the imagination. Few media of creation are so flexible, so easy to polish and rework, so readily capable of realizing grand conceptual structures.

- Fred Brooks, *The Mythical Man-Month*

Programming is creative work with few constraints: imagine something and you can probably build it—in *many* different ways. Liberating and gratifying, but also challenging. Just like starting a novel from a blank page or a painting from a blank canvas, a new program is so open that it's a bit intimidating. Where do you start? What will the system look like? How will you get it *right*? How do you split your problem up? How do you prevent your code from evolving into a complete mess?

There's no easy answer. Programming is inherently iterative—we rarely get the right design at first, but we can always edit code and refactor over time. But iteration itself is not enough; just like a painter needs technique and composition, a programmer needs patterns and design.

Existing teaching resources tend to deemphasize programming techniques and design. Theory- and algorithm-heavy books show algorithms as self-con\-tain\-ed procedures written in pseudocode, without the broader context—and corresponding design questions—of a real codebase. Newer AI and ML materials sometimes take a different tack and provide real code examples using industry-strength frameworks, but rarely touch on software design questions.

In this book, we take a third approach. Starting *from scratch*, we build a Python framework that reflects the key ideas and algorithms in the text. The abstractions we define map to the key concepts we introduce; how we structure the code maps to the relationships between those concepts.

Unlike the pseudocode approach, we do not implement algorithms in a vacuum; rather, each algorithm builds on abstractions introduced earlier in the book. By starting from scratch—rather than using an existing ML framework—we keep the code reasonably simple, without needing to worry about specific examples going out of date. We can focus on the concepts important to the text while teaching programming and design *in situ*, demonstrating an intentional approach to code design.

### Code Design

How can take a complex domain like reinforcement learning and turn it into code that is easy to understand, debug and extend? How can we split this problem into manageable pieces? How do those pieces interact?

There is no single easy answer to these questions. No two programming challenges are identical and the same challenge has many reasonable solutions. A solid design will not be completely clear up-front; it helps to have a clear direction in mind, but expect to revisit specific decisions over time. That's exactly what we did with the code for this book: we had a vision for a Python reinforcement learning framework that matched the topics we present, but as we wrote more and more of the book, we revised the framework code as we came up with better ideas or found new requirements our previous design did not cover.

We might have no easy answers, but we do have patterns and principles that—in our experience—consistently produce quality code. Taken together, these ideas form a philosophy of code design oriented around defining and combining **abstractions** that reflect how we think about our domain. Since code itself can point to specific design ideas and capabilities, there's a feedback loop: expanding the abstractions we've designed can help us find new algorithms and functionality, improving our understanding of the domain.

Just what *is* an abstraction? An appropriately abstract question! An abstraction is a "compound idea": a single concept that combines multiple separate ideas into one. The human mind can only handle so many distinct ideas at a time—we have an inherently limited working memory. A rather simplified model is that we only have a handful of "slots" in working memory and we simply can't track more independent thoughts at the same time. The way we overcome this limitation is by coming up with *new* ideas—new *abstractions*—that combine multiple distinct concepts into one.

We want to organize code around abstractions for the same reason that we use abstractions to understand more complex ideas. How do you understand code? Do you run the program in your head? That's a natural starting point and it works for simple programs but it quickly becomes difficult and then impossible. A computer doesn't have working-memory limitations and can run *billions* of instructions a second—we can't possibly keep up. The computer doesn't need structure or abstraction in the code it runs, but we need it to have any hope of writing or understanding anything beyond the simplest of programs. Abstractions in our code group information and logic so that we can think about rich concepts rather than tracking every single bit of information and every single instruction separately.

The details may differ, but designing code around abstractions that correspond to a solid mental model of the domain works well in any area and with any programming language. It might take some extra up-front thought but, done well, this style of design pays dividends. Our goal is to write code that makes life easier *for ourselves*; this helps for everything from "one-off" experimental code through software engineering efforts with large teams.

### Probability

But what does designing clean abstractions actually entail? There are always two parts to answering this question:

  1. Understanding the domain concept that you are modeling.
  2. Figuring out how to express that concept with features and patterns provided by your programming language.

Let's jump into an extended example to see exactly what this means. One of the key building blocks for reinforcement learning—all of statistics and machine learning, really—is probability. How are we going to handle uncertainty and randomness in our code?

One approach would be to keep probability implicit. Whenever we have a random variable, we could call a function and get a random result. If we were writing a Monopoly game with two six-side dice, we would define it like this:

``` python
from random import randint

def six_sided()
    return randint(1, 6)

def roll_dice():
    return six_sided() + six_sided()
```

This works, but it's pretty limited. We can't do anything except get one outcome at a time. More importantly, this only captures a slice of how we *think* about probability: there's *randomness* but we never even mentioned distributions. We have outcomes and we have a function we can call repeatedly, but there's no way to tell that function apart from a function that has nothing to do with probability but just happens to return an integer.

How can we write code to get the expected value of a distribution? If we have a parametric distribution—a distribution like Poisson or Gaussian characterized by parameters—can we get the parameters out if we need them?

Since distributions are implicit in the code, the *intentions* of the code aren't clear and it is hard to write code that generalizes over distributions. Distributions are absolutely crucial for machine learning, so this is not a great starting point.

#### A Distribution Interface

To address these problems, let's define an abstraction for probability distributions.

How do we represent a distribution in code? What can we *do* with distributions? That depends on exactly what kind of distribution we're working with. If we know something about the structure of a distribution—perhaps it's a Poisson distribution where λ=5, perhaps it's an empirical distribution with set probabilities for each outcome—we could do quite a bit: produce an exact PDF or CDF, calculate expectations and do various operations efficiently. But that isn't the case for all the distributions we work with! What if the distribution comes from a complicated simulation? At the extreme, we might not be able to do anything except draw samples from the distribution.

Sampling is the least common denominator. We can sample distributions were we don't know enough to do anything else, and we can sample distributions where we know the exact form and parameters. Any abstraction we start with for probability needs to cover sampling, and any abstraction that requires more than just sampling will not let us handle all the distributions we care about.

In Python, we can express this idea with a class:

``` python
from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass
```

This class defines an **interface**: a definition of what we require for something to qualify as a distribution. Any kind of distribution we implement in the future will be able to, at minimum, generate samples; when we write functions that sample distributions, they can require their inputs to inherit from `Distribution`.

The class itself does not actually implement `sample`. `Distribution` captures the *abstract concept* of distributions that we can sample, but we would need to specify a specific distribution to actually sample anything. To reflect this in Python, we've made `Distribution` an **abstract base class** (ABC), with `sample` as an *abstract* method—a method without an implementation. Abstract classes and abstract methods are features that Python provides to help us define interfaces for abstractions. We can define the `Distribution` class to structure the rest of our probability code before we define any specific distributions.

#### A Concrete Distribution

Now that we have an interface, what do we do with it? An interface can be approached from two sides:

  * Something that **requires** the interface. This will be code that uses operations specified in the interface and work with *any* value that satisfies those requirements.
  * Something that **provides** the interface. This will be some value that has supports the operations specified in the interface.

If we have some code that requires an interface and some other code that satisfies the interface, we know that we can put the two together and get something that works—even if the two sides were written without any knowledge or reference to each other. The interface manages how the two sides interact.

To use our `Distribution` class, we can start by providing a **concrete class**[^concrete] that implements the interface. Let's say that we wanted to model dice—perhaps for a game of D&D or Monopoly. We could do this by defining a `Die` class that represents an n-sided die and inherits from `Distribution`:

[^concrete]: In this context, a concrete class is any class that is not an abstract class. More generally, "concrete" is the opposite of "abstract"—when an abstraction can represent multiple more specific concepts, we call any of the specific concepts "concrete".

``` python
import random

class Die(Distribution):
    def __init__(self, sides):
        self.sides = sides

    def sample(self):
        return random.randint(1, self.sides)


six_sided = Die(6)

def roll_dice():
    return six_sided.sample() + six_sided.sample()
```

This version of `roll_dice` has exactly the same behavior as `roll_dice` in the previous section, but it took a bunch of extra code to get there. What was the point?

The key difference is that we now have a value that represents the *distribution* of rolling a die, not just the outcome of a roll. The code is easier to understand—when we come across a `Die` object, the meaning and intention behind it is clear—and it gives us a place to add additional die-specific functionality. For example, it would be useful for debugging if we could print not just the *outcome* of rolling a die but the die itself—otherwise, how would we know if we rolled a die with the right number of sides for the given situation?

If we were using a function to represent our die, printing it would not be useful:

``` python
>>> print(six_sided)
<function six_sided at 0x7f00ea3e3040>
```

That said, the `Die` class we've defined so far isn't much better:

``` python
>>> print(Die(6))
<__main__.Die object at 0x7ff6bcadc190>
```

With a class—and unlike a function—we can fix this. Python lets us change some of the built-in behavior of objects by overriding special methods. To change how the class is printed, we can override `__repr__`: [^f-strings]

``` python
class Die(Distribution):
    ...
    def __repr__(self):
        return f"Die(sides={self.sides})"
```

Much better:

``` python
>>> print(Die(6))
Die(sides=6)
```

This seems small but makes debugging *much* easier, especially as the codebase gets larger and more complex.

[^f-strings]: Our definition of `__repr__` used a Python feature called an "f-string". Introduced in Python 3.6, f-strings make it easier to inject Python values into strings. By putting an `f` in front of a string literal, we can include a Python value in a string: `f"{1 + 1}"` gives us the string `"2"`.

##### Dataclasses

The `Die` class we wrote is intentionally simple. Our die is defined by a single property: the number of sides it has. The `__init__` method takes the number of sides as an input and puts it into a field; once a `Die` object is created, there is no reason to change this value—if we need a die with a different number of sides, we can just create a new object. Abstractions do not have to be complex to be useful.

Unfortunately, some of the default behavior of Python classes isn't well-suited to simple classes. We've already seen that we need to override `__repr__` to get useful behavior, but that's not the only default that's inconvenient. Python's default way to compare objects for equality—the `__eq__` method—uses the `is` operator, which means it compares objects *by identity*. This makes sense for classes in general which can change over time, but it is a poor fit for simple abstraction like `Die`. Two `Die` objects with the same number of sides have the same behavior and represent the same probability distribution, but with the default version of `__eq__`, two `Die` objects declared separately will never be equal:

``` python
>>> six_sided = Die(6)
>>> six_sided == six_sided
True
>>> six_sided == Die(6)
False
>>> Die(6) == Die(6)
False
```

This behavior is inconvenient and confusing, the sort of edge-case that leads to hard-to-spot bugs. Just like we overrode `__repr__`, we can fix this by overriding `__eq__`:

``` python
def __eq__(self, other):
    return self.sides == other.sides
```

This fixes the weird behavior we saw earlier:

``` python
>>> Die(6) == Die(6)
True
```

However, this simple implementation will lead to errors if we use `==` to compare a `Die` with a non-`Die` value:

``` python
>>> Die(6) == None
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../rl/chapter1/probability.py", line 18, in __eq__
    return self.sides == other.sides
AttributeError: 'NoneType' object has no attribute 'sides'
```

We generally won't be comparing values of different types with `==`—for `None`, `Die(6) is None` would be more idiomatic—but the usual expectation in Python is that `==` on different types will return `False` rather than raising an exception. We can fix by explicitly checking the type of `other`:

``` python
def __eq__(self, other):
    if isinstance(other, Die):
        return self.sides == other.sides

    return False
```

``` python
>>> Die(6) == None
False
```

Most of the classes we will define in the rest of the book follow this same pattern—they're defined by a small number of parameters, all that `__init__` does is set a few fields and they need custom `__repr__` and `__eq__` methods. Manually defining `__init__`, `__repr__` and `__eq__` for every single class isn't *too* bad—the definitions are entirely systematic—but it carries some real costs:

  * Extra code without important content makes it harder to *read* and *navigate* through a codebase.
  * It's easy for mistakes to sneak in. For example, if you add a field to a class but forget to add it to its `__eq__` method, you won't get an error—`==` will just ignore that field. Unless you have tests that explicitly check how `==` handles your new field, this oversight can sneak through and lead to weird behavior in code that uses your class.
  * Frankly, writing these methods by hand is just *tedious*.

Luckily, Python 3.7 introduced a feature that fixes all of these
problems: **dataclasses**. The `dataclasses` module provides a
decorator[^decorators] that lets us write a class that behaves like
`Die` without needing to manually implement `__init__`, `__repr__` or
`__eq__`. We still have access to "normal" class features like
inheritance (`(Distribution)`) and custom methods (`sample`):

``` python
from dataclasses import dataclass

@dataclass
class Die(Distribution):
    sides: int

    def sample(self):
        return random.randint(1, self.sides)
```

[^decorators]: Python decorators are modifiers that can be applied to class, function and method definitions. A decorator is written *above* the definition that it applies to, starting with a `@` symbol. Examples include `abstractmethod`—which we saw earlier—and `dataclass`.

This version of `Die` has the exact behavior we want in a way that's easier to write and—more importantly—*far* easier to read. For comparison, here's the code we would have needed *without* dataclasses:

``` python
class Die(Distribution):
    def __init__(self, sides):
        self.sides = sides

    def __repr__(self):
        return f"Die(sides={self.sides})"

    def __eq__(self, other):
        if isinstance(other, Die):
            return self.sides == other.sides

        return False

    def sample(self):
        return random.randint(1, self.sides)
```

As you can imagine, the difference would be even starker for classes with more fields!

Dataclasses provide such a useful foundation for classes in Python that the *majority* of the classes we define in this book are dataclasses—we use dataclasses unless we have a *specific* reason not to.

##### Immutability

Once we've created a `Die` object, it does not make sense to change its number of sides—if we need a distribution for a different die, we can create a new object instead. If we change the `sides` of a `Die` object in one part of our code, it will also change in every other part of the codebase that uses that object, in ways that are hard to track. Even if the change made sense in one place, chance are it is not expected in other parts of the code. Changing state can create invisible connections between seemingly separate parts of the codebase which becomes hard to mentally track. A sure recipe for bugs!

Normally, we avoid this kind of problem in Python purely by convention: nothing *stops* us from changing `sides` on a `Die` object, but we know not to do that. This is doable, but hardly ideal; just like it is better to rely on seatbelts rather than pure driver skill, it is better to have the language prevent us from doing the wrong thing than relying on pure convention. Normal Python classes don't have a convenient way to stop fields from changing, but luckily dataclasses do:

``` python
@dataclass(frozen=True)
class Die(Distribution):
    ...
```

With `frozen=True`, attempting to change `sides` will raise an exception:

``` python
>>> d = Die(6)
>>> d.sides = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 4, in __setattr__
dataclasses.FrozenInstanceError: cannot assign to field 'sides'
```

An object that we cannot change is called **immutable**. Instead of changing the object *in place*, we can return a fresh copy with the field changed; `dataclasses` provides a `replace` function that makes this easy:

``` python
>>> import dataclasses
>>> d6 = Die(6)
>>> d20 = dataclasses.replace(d6, sides=20)
>>> d20
Die(sides=20)
```

This example is a bit convoluted—with such a simple object, we would just write `d20 = Die(20)`—but `dataclasses.replace` becomes a lot more useful with more complex objects that have multiple fields.

Returning a fresh copy of data rather than modifying in place is a common pattern in Python libraries. For example, the majority of Pandas operations—like `drop` or `fillna`—return a *copy* of the dataframe rather than modifying the dataframe in place. These methods have an `inplace` argument as an option, but this leads to enough confusing behavior that the Pandas team is currently deliberating on [deprecating `inplace`][inplace] altogether.

[inplace]: https://github.com/pandas-dev/pandas/issues/16529

Apart from helping prevent odd behavior and bugs, `frozen=True` has an important bonus: we can use immutable objects as dictionary keys and set elements. Without `frozen=True`, we would get a `TypeError` because non-frozen dataclasses do not implement `__hash__`:

``` python
>>> d = Die(6)
>>> {d : "abc"}
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'Die'
>>> {d}
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'Die'
```

With `frozen=True`, dictionaries and sets work as expected:

``` python
>>> d = Die(6)
>>> {d : "abc"}
{Die(sides=6): 'abc'}
>>> {d}
{Die(sides=6)}
```

Immutable dataclass objects act like plain data—not too different from strings and ints. In this book, we follow the same practice with `frozen=True` as we do with dataclasses in general: we set `frozen=True` unless there is a specific reason not to.

#### Checking Types

A die has to have an int number of sides—`0.5` sides or `"foo"` sides simply doesn't make sense. Python will not stop us from *trying* `Die("foo")`, but we would get a `TypeError` if we tried sampling it:

``` python
>>> foo = Die("foo")
>>> foo.sample()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../rl/chapter1/probability.py", line 37, in sample
    return random.randint(1, self.sides)
  File ".../lib/python3.8/random.py", line 248, in randint
    return self.randrange(a, b+1)
TypeError: can only concatenate str (not "int") to str
```

The types of an object's fields are a useful indicator of how the object should be used. Python's dataclasses let us use **type annotations** to specify the type of each field:

``` python
@dataclass(frozen=True)
class Die(Distribution):
    sides: int
```

In normal Python, these type annotations exist primarily for documentation—a user can see the types of each field at a glance, but the language does not raise an error when an object is created with the wrong types in a field. External tools—IDEs and typecheckers—can catch type mismatches in annotated Python code without running the code. With a type-aware editor, `Die("foo")` would be underlined with an error message:

> Argument of type `"Literal['foo']"` cannot be assigned to parameter `"sides"` of type `"int"` in function `"__init__"`
>
> `"Literal['foo']"` is incompatible with `"int"` `[reportGeneralTypeIssues]`

This particular message comes from **pyright** running over the [language server protocol][lsp] (LSP), but Python has a number of different typecheckers available[^typecheckers].

[lsp]: https://microsoft.github.io/language-server-protocol/

[^typecheckers]: Python has a number of external typecheckers, including:

      * **mypy**
      * **pyright**
      * **pytype**
      * **pyre**

    The PyCharm IDE also has a propriety typchecker built-in.

    These tools can be run from the command line or integrated into editors. Different checkers *mostly* overlap in functionality and coverage, but have slight differences in the sort of errors they detect and the style of error messages they generate.

Instead of needing to call `sample` to see an error—which we then have to carefully read to track back to the source of the mistake—the mistake is highlighted for us without even needing to run the code.

##### Static Typing?

Being able to find type mismatches *without running code* is called **static typing**. Some languages—like Java and Haskell—require *all* code to be statically typed; Python does not. In fact, Python started out as a **dynamically typed** languages with no type annotations and not typechecking. With older versions of Python, type errors could only ever happen at runtime.

Python is still *primarily* a dynamically typed language—type annotations are optional in most places and there is no built-in checking for annotations. In the `Die("foo")` example, we only got an error when we ran code that passed `sides` into a function that *required* an `int` (`random.randint`). We can get static checking with external tools, but even then it remains *optional*—even statically checked Python code runs dynamic type checks, and we can freely mix statically checked and "normal" Python. Optional static typing on top of a dynamically typed languages is called **gradual typing** because we can incrementally add static types to an existing dynamically typed codebase.

Dataclass fields are not the only place where knowing types is useful; it would also be handy for function parameters, return values and variables. Python supports *optional* annotations on all of these; dataclasses are the only language construct where annotations are *required*. To help mix annotated and unannotated code, typecheckers will report mismatches in code that is explicitly annotated, but will usually not try to guess types for unannotated code.

How would we add type annotations to our example code? So far, we've defined two classes:

  * `Distribution`, an abstract class defining interfaces for probability distributions in general
  * `Die`, a concrete class for the distribution of an n-sided die

We've already annotated the `sides` in `Die` has to be an `int`. We also know that the *outcome* of a die roll is an `int`. We can annotate this by adding `-> int` after `def sample(...)`:

``` python
@dataclass(frozen=True)
class Die(Distribution):
    sides: int

    def sample(self) -> int:
        return random.randint(1, self.sides)
```

Other kinds of concrete distributions would have other sorts of outcomes. A coin flip would either be `"heads"` or `"tails"`; a normal distribution would produce a `float`.

#### Type Variables

Annotating `sample` for specific cases like `Die`—where outcomes are always `int`—is pretty straightforward. But what can we do about the abstract `Distribution` class itself? In general, a distribution can have any kind of outcomes. The abstract `Distribution` class we wrote earlier does not tell us anything about what `sample` returns:

``` python
class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass
```

This works—annotations are optional, after all—but it can get confusing: some code we write will work for any kind of distribution, some code needs distributions that return numbers, other code will need something else... In every instance `sample` better return *something*, but even that isn't explicitly annotated.

While *specific* distributions will have a *specific* type of outcome, the type will vary depending on the distribution. To deal with this, we need **type variables**: variables that stand in for *some* type that might be different each time the code is used. Type variables are also known as "generics" because they let us write classes that generically work for any type.

To add annotations to the abstract `Distribution` class, we will need to define a type variable for the outcomes of the distribution, then tell Python that `Distribution` is "generic" in that type:

``` python
from typing import Generic, TypeVar

# Defining a type variable named "A"
A = TypeVar("A")


# Distribution is "generic in A"
class Distribution(ABC, Generic[A]):

    # Sampling must return a value of type A
    @abstractmethod
    def sample(self) -> A:
        pass
```

In this code, we've defined a type variable `A`[^type-variable-names] and used `Generic[A]` to specify that `Distribution` uses this variable. We can now write type annotations for distributions *with specific types of outcomes*: for example, `Die` would be an instance of `Distribution[int]` since the outcome of a die roll is always an `int`. We can make this explicit in the class definition:

``` python
class Die(Distribution[int]):
    ...
```

This lets us write specialized functions that only work with certain kinds of distributions. Let's say we wanted to write a function that approximated the expected value of a distribution by sampling repeatedly and calculating the mean. This function works for distributions that have numeric outcomes—`float` or `int`—but not other kinds of distributions. (How would we calculate an average for a coin flip that could be `"heads"` or `"tails"`?) We can annotate this explicitly by using `Distribution[float]`: [^float]

[^float]: The `float` type in Python *also* covers `int`, so we can pass a `Distribution[int]` anywhere that a `Distribution[float]` is required.

``` python
import statistics

def expected_value(d: Distribution[float], n: int) -> float:
    return statistics.mean(d.sample() for _ in range(n))
```

With this function:

  * `expected_value(Die(6))` would be fine
  * `expected_value(Coin())` (where `Coin` is a `Distribution[str]`) would be a type error

Using `expected_value` on a distribution with non-numeric outcomes would raise a `TypeError` at runtime. Having this highlighted in the editor can save us time---we see the mistake right away, rather than waiting for tests to run---and will catch the problem even if our test suite doesn't.

[^type-variable-names]: Traditionally, type variables have one-letter capitalized names—although it's perfectly fine to use full words if that would make the code clearer.
