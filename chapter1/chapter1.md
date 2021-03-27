## Code Design and Abstraction in Python {#sec:python .unnumbered}

This book introduces the theory and algorithms behind Reinforcement Learning. What kind of programming and code design do you need to handle these topics?

Traditional presentations of theory- and algorithm-heavy topics deemphasize code, expressing algorithms as self-contained procedures written in pseudocode. Newer materials on AI and machine learning take the opposite tack, implementing algorithms in real-world machine learning frameworks.

This book intentionally takes a different approach: we build up a high-level Reinforcement Learning framework *from scratch* in Python. The code works—this is not pseudocode—but it is also not meant to be "production quality". Rather, the goal is to develop clean abstractions that reflect the concepts covered in the text, without adding extra complexity to improve performance or integrate with existing systems. With this approach, example implementations of algorithms not only make the *algorithm itself* more concrete but also reinforce the key ideas shared *between different algorithms*.

Defining simple, composable abstractions directly corresponding to domain-specific concepts is a powerful approach that works well in any area, not just Reinforcement Learning. Learning to program in this style isn't always easy and it can take more up-front thinking and effort than other approach—but, done well, it pays massive dividends over the life of a project. This book weaves abstraction-oriented design into its presentation of Reinforcement Learning;
