# Chapter 1: Overview

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This overview chapter starts by going over the pedagogical aspects of learning applied mathematics. Then it covers the learnings to be acquired from this book, background required to read this book, and a high-level overview of stochastic control, Markov decision processes, value function, Bellman equation, dynamic programming and reinforcement learning. The chapter ends with an outline of chapters in this book. 

# Chapter 2: Programming and Design

Chapter author: Tikhon Jelvis (https://orcid.org/0000-0002-2532-3215)

## Abstract

Machine learning—like scientific computing in general—requires a substantial amount of programming. Programming effectively is not only a matter of writing code that works but also organizing code along conceptual lines so that it is easier to understand, modify and debug. This chapter introduces software engineering concepts useful for writing higher-quality code, focused on designing domain-specific abstractions and interfaces. Each concept is illustrated with Python code, which also introduces core Python features used throughout the rest of the book: classes, inheritance, abstract classes, static types, generic programming, dataclasses, immutability, first-class functions, lambdas, iterators and generators.

# Chapter 3: Markov Processes

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

Reinforcement learning involves making sequential decisions under sequential uncertainty. Markov processes (MP) and Markov reward processes (MRP) provide a formal framework for modeling sequential uncertainty without decision-making. This chapter starts by introducing the Markov property and formalizes Markov processes in terms of states and transition functions, then considers special cases of time-homogeneous Markov processes and finite Markov processes. Next, the chapter extends Markov processes to Markov reward processes by introducing rewards as part of the transition function and finally covering the MRP value function expressed as expected accumulated rewards and the MRP Bellman equation expressed as a recursive formulation of the value function. The Python definitions in this chapter set up the programming framework that acts as a foundation for the examples and algorithms in the rest of the book.

# Chapter 4: Markov Decision Processes

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

Markov reward processes provide a framework for sequential uncertainty with rewards but do not have any mechanism for making decisions or interacting with the system. Markov decision processes (MDP) address this limitation by extending Markov reward processes with actions that can be taken at each step in the process. This chapter starts by formalizing Markov decision processes as an extension of the Markov process definitions in chapter 1. Next, it introduces policies and the value function for a policy expressed as expected accumulated rewards when the MDP is executed with a fixed policy. Finally, the chapter covers the concepts of optimal value function and optimal policy, along with the all-important Bellman optimality equations and proof that an optimal policy exists for discrete-time, countable-space, time-homogeneous Markov decision processes.

# Chapter 5: Dynamic Programming Algorithms

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

Dynamic programming is a set of algorithmic techniques for evaluating
and optimizing policies for finite Markov decision processes. This chapter starts by introducing the general concept of fixed-points and then presents several dynamic programming algorithms whose correctness is proven using the Banach fixed-point theorem. The policy evaluation algorithm is presented in terms of the Bellman policy operator. The algorithms for calculating the optimal policy are policy iteration and value iteration (presented in terms of the Bellman optimality operator). Both of these algorithms can be unified into generalized policy iteration, which provides the structure used by dynamic programming and reinforcement learning algorithms in general. Finally, the chapter covers an important special case of finite-horizon processes, which can be handled efficiently with the backward induction algorithm.

# Chapter 6:  Function Approximation and Approximate Dynamic Programming

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

Tabular dynamic programming algorithms which have an exact representation of a Markov decision processes' state work well for finite processes with smaller state spaces but do not scale to larger or infinite state spaces. This chapter introduces approximate dynamic programming as a technique for handling larger and more complex problems. The chapter starts by introducing a mathematical and programming framework for function approximation which provides a way to efficiently approximate large or infinite state spaces. It then describes two classes of function approximations: linear function approximations and deep neural nets. The exact dynamic programming algorithms covered in the previous chapter (policy iteration, value iteration, backward induction) can be generalized to approximate versions by using a function approximation in place of the exact tabular state representation.

# Chapter 7: Utility Theory

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter introduces utility theory, which provides a mathematical foundation for modeling risk-aversion. Utility functions formalize the tradeoff between financial risk and return in a way that translates naturally to the Markov decision process framework. The chapter introduces utility functions in terms of the associated concepts of risk, risk-aversion, risk-premium and certainty-equivalent value. A couple of standard utility function formulations are covered: constant absolute risk aversion (CARA) and constant relative risk aversion (CRRA). Utility functions in finance need to account for both expected returns and risk, which is captured in the concept of risk-adjusted reward, an offshoot of utility function formulations.

# Chapter 8: Dynamic Asset-Allocation and Consumption

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

Dynamic asset allocation is the problem of periodically deciding how to split investments across assets with different risk-return profiles—an important problem that applies for both companies and individuals. Dynamic consumption is the problem of periodically deciding how much of one's wealth to consume (versus invest). This chapter formalizes the joint problem of dynamic asset allocation and consumption, covers Merton's portfolio problem and solution, which provides an optimal closed-form solution under certain simplifying assumptions. A general form of the dynamic asset allocation and consumption problem can be modeled as a Markov decision process and solved with finite-horizon dynamic programming algorithms. However, in typical real-world situations, the state space of this formulation is too large to be feasible, and the distributions of asset returns are unknown, which points to using reinforcement learning algorithms suited for large state spaces rather than dynamic programming.

# Chapter 9: Derivatives Pricing and Hedging

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

The theory of pricing and hedging of financial derivatives is an important topic in Mathematical Finance, and this chapter starts with an overview of financial derivatives. A significant portion of this chapter covers derivatives pricing and hedging theory in a simple single-period setting by developing the concepts of arbitrage, replication, market completeness, risk-neutral measure, and proving the two fundamental theorems of asset pricing (in this simple setting). A brief overview of the theory in the general setting is also provided. This is followed by coverage of the problem of optimal exercise of american options, tackled as a special case of the optimal stopping problem modeled as an MDP. A backward induction algorithm is developed to solve this MDP. Finally, this chapter covers an MDP formulation of the problem of pricing and hedging of derivatives when incorporating real-world frictions (technically known as an incomplete market).

# Chapter 10: Order-Book Trading Algorithms

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter introduces problems from algorithmic trading that can be modeled as Markov decision processes. It starts by covering the concepts of an order book with limit orders and market orders, and the price impact of market orders. An order book simulator is implemented to demonstrate how the order book responds to various sequences of limit orders and market orders. This provides the foundation for two control problems: optimal order execution and optimal market-making, which are both formulated as Markov decision processes. Closed-form solutions for simple classical formulations of optimal order execution are developed, and an approximate backward induction algorithm is developed for this problem in more realistic settings. The chapter ends with Avellaneda and Stoikov's continuous-time formulation and solution of the optimal market-making problem.

# Chapter 11: Monte-Carlo and Temporal-Difference for Prediction

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter provides the entry into the world of Reinforcement Learning (RL) algorithms. The chapter starts by restricting the RL problem to a very simple one—where the state space is small and manageable as a table enumeration (known as tabular RL) and where one has to calculate the Value Function for a Fixed Policy (known as the Prediction problem). The restriction to Tabular Prediction makes it very easy to understand the core concepts of Monte-Carlo (MC) and Temporal-Difference (TD). This is then extended to the problem of Prediction with function approximation, enabling the handling of large state spaces. Comparisons and contrasts are made between the MC and TD techniques. The chapter ends with coverage of the TD(lambda) algorithm, which enables one to find a continuum of algorithms between MC and TD by tuning the lambda parameter and playing the bias-variance tradeoff. 

# Chapter 12: Monte-Carlo and Temporal-Difference for Control

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter makes the natural extension from RL Prediction to RL Control (finding the optimal policy), while initially remaining in the tabular setting. The previous chapter's investments made in understanding the core concepts of MC and TD bear fruit in this chapter as important Control Algorithms such as SARSA and Q-learning emerge with enormous clarity along with implementations of these algorithms from scratch in Python. This chapter also introduces a very important concept for the future success of RL in the real world: off-policy learning (Q-Learning is the simplest off-policy learning algorithm, and it has had good success in various applications). Importance Sampling (an off-policy technique) is briefly covered. The chapter ends with a summary of the convergence of RL Prediction and Control algorithms: MC versus TD, on-policy versus off-policy, tabular versus function approximation. 

# Chapter 13: Batch RL, Experience-Replay, DQN, LSPI, Gradient TD

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter moves on from basic and more traditional RL algorithms to recent innovations in RL. The chapter starts with the important ideas of Batch RL and Experience-Replay, which makes more efficient use of data by storing data as it comes and re-using it in batches throughout the learning process of the algorithm. The Deep Q-Networks algorithm is covered that leverages Experience-Replay and Deep Neural Networks for Q-Learning. This is followed by coverage of an important Batch RL technique using linear function approximation—Least-Squares Temporal Difference (for Prediction) and Least-Squares Policy Iteration (for Control). The later part of this chapter provides deeper insights into the core mathematics underpinning RL algorithms, back to the basics of Bellman equation and understanding value function geometry. This sheds light on how to break out of the so-called deadly triad. The chapter ends with a coverage of the state-of-the-art Gradient TD Algorithm, which resists the deadly triad by constructing an appropriate loss function.

# Chapter 14: Policy Gradient Algorithms

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter introduces a very different class of RL algorithms that are based on improving the policy using the gradient of the policy function approximation. When action spaces are large or continuous, policy gradient tends to be the only option, and so, this chapter is useful to overcome many real-world challenges (including those in many financial applications) where the action space is indeed large. The policy gradient theorem is proved, and a few policy gradient algorithms are implemented from scratch. The chapter covers state-of-the-art Actor-Critic methods and a couple of specialized algorithms (Natural PG and Deterministic PG) that have worked well in practice. The chapter ends with coverage of evolutionary strategies, an important algorithm that looks quite similar to policy gradient algorithms but is technically not an RL Algorithm.

# Chapter 15: Multi-Armed Bandits: Exploration versus Exploitation

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter is a deep dive into the topic of balancing exploration and exploitation, a topic of great importance in RL algorithms. Exploration versus exploitation is best understood in the simpler setting of the multi-armed bandit (MAB) problem. The chapter starts with simple MAB algorithms such as epsilon-greedy and optimistic initialization. Then, various state-of-the-art MAB algorithms are covered: Upper Confidence Bound, Thompson Sampling, Gradient Bandit and Information State Space MDP. These algorithms are implemented in Python, and various useful simulations and visualizations are generated from the code. The chapter ends with coverage of Contextual Bandits, an extension of MAB where each arm's reward distribution is dependent on a context.

# Chapter 16: Blending Learning and Planning

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This chapter brings the various pieces of planning and learning concepts learned in this book together. The chapter starts by illustrating a methodology that works well in practice—creatively blending planning and learning (a technique known as Model-based RL). This is followed by coverage of the Monte Carlo Tree Search (MCTS) algorithm that was highly popularized when it solved the Game of GO, a problem that was thought to be insurmountable by current AI technology. The chapter ends with coverage of the Adaptive Multi-stage Sampling algorithm, which can be considered to be the "spiritual origin" of MCTS.

# Chapter 17: Summary and Real-World Considerations

Chapter author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This concluding chapter summarizes the key learnings from this book and provides some commentary on how to take the learnings from this book into practice. The chapter specifically focuses on the challenges one faces in the real-world—modeling difficulties, problem-size difficulties, operational challenges, data challenges (access, cleaning, organization), and also change-management challenges as one shifts an enterprise from legacy systems to an AI system. The chapter provides some guidance on how to go about building an end-to-end system based on RL.

# Appendix A: Moment Generating Function and Its Applications

Appendix author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This appendix introduces moment generating functions (MGF) and derives some key results pertaining to MGFs that are useful in evaluating some expectations under the normal distribution. The appendix also derives some results on minimizing the MGF for a normal distribution and for a symmetric binary distribution.

# Appendix B: Portfolio Theory

Appendix author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This appendix is on portfolio theory covering the mathematical foundations of balancing return versus risk in portfolios and the much-celebrated Capital Asset Pricing Model (CAPM).

# Appendix C: Introduction to and Overview of Stochastic Calculus Basics

Appendix author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This appendix covers the basics of the theory of stochastic calculus as some of this theory (Ito Integral, Ito's Lemma etc.) is required in the derivations in a couple of the chapters in Module II.

# Appendix D: The Hamilton-Jacobi-Bellman (HJB) Equation

Appendix author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This appendix is on the Hamilton-Jacobi-Bellman (HJB) equation, which is a key part of the derivation of the closed-form solutions for 2 of the 5 financial applications covered in Module II.

# Appendix E: Black-Scholes Equation and Its Solution for Call/Put Options

Appendix author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This appendix covers the derivation of the famous Black-Scholes equation, and its solution for Call/Put Options.

# Appendix F: Function Approximations as Affine Spaces

Appendix author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This appendix is a technical perspective of function approximations as affine spaces, which helps develop a deeper mathematical understanding of function approximations.

# Appendix G: Conjugate Priors for Gaussian and Bernoulli Distributions

Appendix author: Ashwin Rao (https://orcid.org/0000-0003-0620-3100)

## Abstract

This appendix covers the formulas for bayesian updates to conjugate prior distributions for the parameters of Gaussian and Bernoulli data distributions.
