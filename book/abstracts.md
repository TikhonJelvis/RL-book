# Programming and Design

Chapter author: Tikhon Jelvis (https://orcid.org/0000-0002-2532-3215)

## Abstract

Machine learning—like scientific computing in general—requires a substantial amount of programming. Programming effectively is not only a matter of writing code that works, but also organizing code along conceptual lines so that it is easier to understand, modify and debug. This chapter introduces software engineering concepts useful for writing higher-quality code, focused on designing domain-specific abstractions and interfaces. Each concept is illustrated with Python code, which also introduces core Python features used throughout the rest of the book: classes, inheritance, abstract classes, static types, generic programming, dataclasses, immutability, first-class functions, lambdas, iterators and generators.

# Chapter 1: Markov Processes

Chapter author: Ashwin Rao

## Abstract

Reinforcement learning involves making sequential decisions under sequential uncertainty. Markov processes (Markov chains) and Markov reward processes provide a formal framework for modeling sequential uncertainty without decision-making. This chapter starts by introducing processes defined in terms of states and transition functions, then expands to Markov processes by formalizing what it means for a process to be "time-homogenous" or "time-invariant" (the Markov property). Next, the chapter extends Markov processes to Markov reward processes by introducing rewards as part of the transition function and finally covering the value function of a Markov reward process in terms of the Bellman equation. The Python definitions in this chapter set up the programming framework that acts as a foundation for the examples and algorithms in the rest of the book.

# Chapter 2: Markov Decision Processes

Chapter author: Ashwin Rao

## Abstract

Markov processes provide a framework for sequential uncertainty but do not have any mechanism for making decisions or interacting with the system. Markov decision processes address this limitation by extending Markov processes with actions that can be taken at each step in the process. This chapter starts by introducing Markov decision processes as an extension of the Markov processes as defined in chapter 1. Next, it introduces policies and the value function for a policy to evaluate policies in terms of their expected rewards, and what it means to find the optimal policy for a Markov decision process with the Bellman policy and optimality equations, as well as a proof that an optimal policy exists for discrete-time, countable-space, time-homogenous Markov decision processes.

# Chapter 3: Dynamic Programming Algorithms

Chapter author: Ashwin Rao

## Abstract

Dynamic programming is a set of algorithmic techniques for evaluating
and optimizing policies for finite Markov decision processes. This chapter starts by introducing the general concept of fixed-points and then introducing several provably-optimal dynamic programming algorithms. The first two algorithms are policy iteration (along with the Bellman policy operator) and value iteration (along with the Bellman optimality operator). Both of these algorithms can be unified into generalized policy iteration, which provides the structure used by dynamic programming and reinforcement learning algorithms in general. Finally, the chapter covers an important special case of finite-horizon processes which can be efficiently handling with the backwards induction algorithm.

# Chapter 4:  Function Approximation and Approximate Dynamic Programming

Chapter author: Ashwin Rao

## Abstract

Tabular dynamic programming algorithms which have an exact representation of a Markov decision processes's state work well for finite processes with smaller states, but do not scale to larger or infinite state spaces. This chapter introduces approximate dynamic programming as a technique for handling larger and more complex problems. The chapter starts by introducing a mathematical and programming framework for function approximation which provides a way to efficiently approximate large or infinite state spaces. It then describes two classes of function approximations: linear function approximations and deep neural nets. The exact dynamic programming algorithms covered in the previous chapter (policy iteration, value iteration, backward induction) can be generalized to approximate versions by using a function approximation in place of the exact tabular state representation.

# Chapter 5: Utility Theory

Chapter author: Ashwin Rao

## Abstract

This chapter introduces utility theory, which provides a mathematical foundation for modeling rewards in financial applications. Utility functions formalize the tradeoff between financial risk and return in a way that translates naturally to the Markov decision process framework. The chapter defines utility functions and how the shape of a utility function affects decision-making. Different utility functions correspond to different sorts of risk-aversion: risk neutrality, constant absolute risk aversion (CARA) and constant relative risk aversion (CRRA). Utility functions in finance need to account for both expected returns and risk, which is captured in the concept of risk-adjusted reward.

# Chapter 6: Dynamic Asset-Allocation and Consumption

Chapter author: Ashwin Rao

## Abstract

Dynamic asset allocation is the problem of periodically deciding how to split investments across assets with different risk-return profiles—an important problem that applies for both companies and individuals. This chapter formalizes the dynamic asset allocation problem and covers Merton's portfolio problem and solution, which provides an optimal algorithm for asset allocation under certain simplifying assumptions. A general form of the dynamic asset allocation problem can be modeled as a Markov decision process and solved with finite-horizon dynamic programming algorithms. However, in typical real-world situations, the state space of this formulation is too large to be feasible and the distributions of asset returns are unknown, which points to using reinforcement learning algorithms suited for large state spaces rather than dynamic programming.

# Chapter 7: Derivatives Pricing and Hedging

Chapter author: Ashwin Rao

## Abstract

Markov decision processes can model derivatives pricing and hedging (managing risks with derivatives). This chapter covers finding the optimal time and state to exercise American options and identifying optimal hedging strategies in real-world conditions. These problems fit well into the Markov decision process framework and can be tackled with dynamic programming or reinforcement learning algorithms. The chapter starts with an introduction to European and American options then covers the first and second fundamental theorems of asset pricing. The chapter finishes by covering two problems formulated as MDP control problems: optimal exercise of American options (an example of an optimal stopping time problem) as well as pricing and hedging derivatives in an incomplete market.

# Chapter 8: Order-book Trading Algorithms

Chapter author: Ashwin Rao

## Abstract

This chapter introduces problems from algorithmic trading that can be modeled as Markov decision processes. It starts by covering the concepts of an order book with limit orders and market orders, and the price impact of orders. This provides the foundation for two control problems: optimal order execution and optimal market making, which are both formulated as Markov decision processes that can be optimized with either dynamic programming or reinforcement learning algorithms.
