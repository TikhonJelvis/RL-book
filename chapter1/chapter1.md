## Programming and Design {#sec:python .unnumbered}

> The programmer, like the poet, works only slightly removed from pure thought-stuff. He builds his castles in the air, from air, creating by exertion of the imagination. Few media of creation are so flexible, so easy to polish and rework, so readily capable of realizing grand conceptual structures.

- Fred Brooks, *The Mythical Man-Month*

Programming is, at heart, creative with few constraints: imagine something and you can probably build it, and build it in *many* different ways. This lack of constraints is liberating and gratifying, but it's also challenging. Just like starting a novel from a blank page or a painting from a blank canvas, a blank programming project is so open that it's a bit intimidating. Where do you start? What will the whole system look like? How do you break your problem into manageable pieces? How do you prevent your code from evolving into an impossible-to-understand mess? What if you start strong but program yourself into a corner?

There's no easy answer. Programming is inherently iterative—we can always try something, then edit our code and refactor if our original idea didn't work. But iteration itself is not enough; just like a painter needs technique and composition, a programmer needs patterns and design.

Existing teaching resources tend to deemphasize programming techniques and design. Theory- and algorithm-heavy books show algorithms as self-contained procedures written in pseudocode, without the broader context—and corresponding design questions—of a real codebase. Newer AI and ML materials sometimes take a different tack and provide real code examples using industry-strength frameworks, but the books rarely touch on the design of the frameworks themselves or larger codebases in general.

In this book, we take a third approach. Starting *from scratch*, we build a Python framework that reflects the key ideas and algorithms in the text. The abstractions we define map to the key concepts we introduce; how we structure the code maps to the relationships between those concepts.

Unlike the pseudocode approach, we do not implement algorithms in a vacuum; rather, each algorithm builds on abstractions introduced earlier in the book. By starting from scratch—rather than using an existing ML framework—we keep the code reasonably simple, without needing to worry about specific examples going out of date. We can focus on the concepts important to the text while teaching programming and design *in situ*, demonstrating an intentional approach to code design.

### A Philosophy of Abstraction

How can take a complex domain like reinforcement learning and turn it into code that is easy to understand, debug and extend? How can we split this problem into manageable pieces? How do those pieces interact?

There is no single easy answer to these questions. As in any creative endeavor, no two programming challenges are identical and people can reasonably find different solutions. A solid design is not going to be completely clear up-front; as we worked on the code for this book, we constantly revisited previous design decisions as we either came up with better solutions or found new requirements to address.

We might have no easy answers, but we do have general patterns and principles that—in our experience—consistently produce quality code. Taken together, these ideas form a philosophy of code design oriented around defining and combining **abstractions** that reflect the terms and concepts we use to think about our domain. This is not only a powerful technique for designing code, but it is also a way to better-understand the domain itself; designing the right code abstractions can also lead to novel algorithms and functionality.

Just what *is* an abstraction? An appropriately abstract question! An abstraction is a "compound idea": a single concept that combines multiple separate ideas into one. Humans have an inherently limited working memory; we can only keep a small number of distinct things in mind at any given time. If we think of working memory as having some small number of "slots", an abstraction lets us take several distinct ideas that would normally take up several slots of working memory and consider them as a single unit, taking up just *one* slot. Abstractions let us generalize beyond individual concrete objects—we can consider the abstract idea of horses in general rather than needing to consider every single horse as a separate, unique animal. The only way we can understand and interact with anything meaningfully complex is by developing mental abstractions to manage the complexity.

How does this translate to programming? Just as we need to organize complex ideas to think about them, we need to organize complex code to write and understand it. The computer itself does not need structure or organization to run code—it is happy mindlessly following the exact instructions we give it. Modern CPUs happily run *billions* of instructions a second, dealing with billions and billions of bits of information. Humans can't keep up!

The same way our limited working memory pushes us to use mental abstractions, it pushes us to organize code around abstractions as well. How do you understand code? Do you run the code in your head? This is how most people start, but it's difficult at smaller scales and quickly becomes infeasible as the amount of logic and information grows too large to mentally track. Code abstractions let us overcome this limit by logically grouping information and logic so that we can think about it holistically rather than tracking every pieces separately.

Some level of abstraction in programming is inevitable, but not all abstractions are equal. What makes some abstractions a pleasure to work with, and others a pain?

<!-- TODO: What properties do we want our abstractions to have? -->

Simple, composable abstractions corresponding to domain-specific concepts work well in any area, with any programming languages—although the details may differ. Designing and using abstractions like that isn't always easy and can take more up-front thinking than other approaches—but, done well, it pays massive dividends over the life of a project.

### Probability
