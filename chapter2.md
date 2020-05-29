
# Chapter 1: Markov Processes

This book is about "Sequential Decisioning under Sequential Uncertainty".
In this chapter, we will ignore the "sequential decisioning" aspect and focus
just on the "sequential uncertainty" aspect.

## The Concept of *State* in a System

For a gentle introduction to the concept of *State*, let us consider a system
that generates a sequence of random outcomes at discrete time steps that we'll
index by a time variable $t = 0, 1, 2, \ldots$. To understand and reason about
the random evolution of such a system, we are typically interested in the
internal representation of the system at each point in time $t$. We efer
to this internal representation of the system at time $t$ as the (random) *state*
of the system at time $t$ and denote it as $S_t$. Specifically, we are interested
in the probability of the next state $S_{t+1}$, given the present state $S_t$ and
the past states $S_0, S_1, \ldots, S_{t-1}$, i.e., $Pr[S_{t+1}|S_t, S_{t-1}, 
\ldots S_0]$. The internal representation (*state*) could be any data type -
it could be something as simple as a single stock price at the end of a day, or
it could be something quite elaborate like the number of shares of all publicly
traded stocks held by all banks in the U.S. at the end of a week. 

## Markov States

We will be learning about Markov Processes in this chapter and these processes
have what are called *Markov States*. So we will now learn about the *Markov
Property* of *States*. Before we provide the formal definition of the Markov Property, let
us develop intuition for this concept with some examples of random evolution
of stock prices over time. 

To aid with the intuition, let us pretend that stock prices take on only integer
values and that it's acceptable to have zero or negative stock prices. Let us
denote the stock price at time $t$ as $X_t$. Let us assume that from time step $t$
to the next time step $t+1$, the stock price can either go up by 1 or go down by 1,
i.e., the only two outcomes for $X_{t+1}$ are $X_t + 1$ or $X_t - 1$. To
understand the random evolution of the stock prices in time, we just need to
quantify the probability of an up move $Pr[X_{t+1} = X_t + 1]$ since the probability
of a down move $Pr[X_{t+1} = X_t - 1] = 1 - Pr[X_{t+1} = X_t + 1]$. 

