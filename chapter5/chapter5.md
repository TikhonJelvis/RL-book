# Approximate Dynamic Programming

In the previous chapter, we covered Dynamic Programming algorithms where the MDP is specified in the form of a finite data structure and the Value Function is represented as a finite "table" of states and values. These Dynamic Programming algorithms made a sweep through all states in each iteration to update the value function. But when the state space is large (as is the case in real-world applications), these Dynamic Programming algorithm won't work because:

1. Large state spaces could disallow a "tabular" representation of the MDP or Value Function, due to storage limits
2. Large state spaces would be time-prohibitive in terms of sweeping through all states (or simple impossible, in the case of infinite state spaces)

When the state space is very large, we need to resort to function approximation of the Value Function and the Dynamic Programming algorithms would be suitably modified to their Approximate Dynamic Programming (abbreviated as ADP) form. It's not hard to modify each of the (tabular) Dynamic Programming algorithms such that instead of sweeping through all the states at each step, we simply sample an appropriate subset of states, update the Value Function for those states (with the same Bellman Operator calculations as for the case of tabular), and then create a function approximation for the Value Function using just the updated values for the sample of states. The fundamental structure of the algorithms and the fundamental principles (Fixed-Point and Bellman Operators) would still be the same.

So, in this chapter, we do a quick review of function approximation, write some code for a couple for a couple of standard function approximation methods, and then utilize these function approximation methods to developed Approximate Dynamic Programming (in particular, Approximate Value Iteration and Approximate Backward Induction). If you are reading this book, it's highly likely that you are already familiar with the simple and standard function approximation methods such as linear function approximation and function approximation using neural networks supervised learning. So we shall go through the background on linear function approximation and neural networks supervised learning in a quick and terse manner, with the goal of developing some code for these methods that we can use not just for the ADP algorithms for this chapter, but also for RL algorithms later in the book. Note also that apart from function approximation of Value Functions $\mathcal{N} \rightarrow \mathbb{R}$, these function approximation methods can also be used for approximation of Stochastic Policies $\mathcal{N} \times \mathcal{A} \rightarrow [0, 1]$ in Policy-based RL algorithms.

## Function Approximation
In this section, we describe function approximation in a fairly generic setting (not specific to Value Functions or Policies).

## Linear Function Approximation

## Neural Network Function Approximation

## Approximate Value Iteration

## Approximate Backward Induction

## Key Takeaways from this Chapter
