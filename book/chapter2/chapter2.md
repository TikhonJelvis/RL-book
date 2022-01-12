# Processes and Planning Algorithms

## Markov Processes {#sec:mrp-chapter}

This book is about "Sequential Decisioning under Sequential Uncertainty". In this chapter, we will ignore the "sequential decisioning" aspect and focus just on the "sequential uncertainty" aspect.

### The Concept of *State* in a Process

For a gentle introduction to the concept of *State*, we start with an informal notion of the terms *Process* and *State* (this will be formalized later in this chapter). Informally, think of a Process as producing a sequence of random outcomes at discrete time steps that we'll index by a time variable $t = 0, 1, 2, \ldots$. The random outcomes produced by a process might be key financial/trading/business metrics one cares about, such as prices of financial derivatives or the value of a portfolio held by an investor. To understand and reason about the evolution of these random outcomes of a process, it is beneficial to focus on the internal representation of the process at each point in time $t$, that is fundamentally responsible for driving the outcomes produced by the process. We refer to this internal representation of the process at time $t$ as the (random) *state* of the process at time $t$ and denote it as $S_t$. Specifically, we are interested in the probability of the next state $S_{t+1}$, given the present state $S_t$ and the past states $S_0, S_1, \ldots, S_{t-1}$, i.e., $\mathbb{P}[S_{t+1}|S_t, S_{t-1}, \ldots, S_0]$. So to clarify, we distinguish between the internal representation (*state*) and the output (outcomes) of the Process. The *state* could be any data type - it could be something as simple as the daily closing price of a single stock, or it could be something quite elaborate like the number of shares of each publicly traded stock held by each bank in the U.S., as noted at the end of each week. 

### Understanding Markov Property from Stock Price Examples

We will be learning about Markov Processes in this chapter and these processes have *States* that possess a property known as the *Markov Property*. So we will now learn about the *Markov Property of States*. Let us develop some intuition for this property with some examples of random evolution of stock prices over time. 

To aid with the intuition, let us pretend that stock prices take on only integer values and that it's acceptable to have zero or negative stock prices. Let us denote the stock price at time $t$ as $X_t$. Let us assume that from time step $t$ to the next time step $t+1$, the stock price can either go up by 1 or go down by 1, i.e., the only two outcomes for $X_{t+1}$ are $X_t + 1$ or $X_t - 1$. To understand the random evolution of the stock prices in time, we just need to quantify the probability of an up-move $\mathbb{P}[X_{t+1} = X_t + 1]$ since the probability of a down-move $\mathbb{P}[X_{t+1} = X_t - 1] = 1 - \mathbb{P}[X_{t+1} = X_t + 1]$. We will consider 3 different processes for the evolution of stock prices. These 3 processes will prescribe $\mathbb{P}[X_{t+1} = X_t + 1]$ in 3 different ways.

**Process 1**:
$$\mathbb{P}[X_{t+1} = X_t + 1] = \frac 1 {1 + e^{-\alpha_1(L - X_t)}}$$
where $L$ is an arbitrary reference level and $\alpha_1 \in \mathbb{R}_{\geq 0}$ is a "pull strength" parameter. Note that this probability is defined as a [logistic function](https://en.wikipedia.org/wiki/Logistic_function) of $L - X_t$ with the steepness of the logistic function controlled by the parameter $\alpha_1$ (see Figure \ref{fig:logistic})

<div style="text-align:center" markdown="1">
![Logistic Curves \label{fig:logistic}](./chapter2/logistic_curves.png "Logistic Curves")
</div>
 The way to interpret this logistic function of $L-X_t$ is that if $X_t$ is greater than the reference level $L$ (making $\mathbb{P}[X_{t+1} = X_t + 1] < 0.5$), then there is more of a down-pull than an up-pull. Likewise, if $X_t$ is less than $L$, then there is more of an up-pull. The extent of the pull is controlled by the magnitude of the parameter $\alpha_1$. We refer to this behavior as *mean-reverting behavior*, meaning the stock price tends to revert to the "mean" (i.e., to the reference level $L$).

We can model the state $S_t = X_t$ and note that the probabilities of the next state $S_{t+1}$ depend only on the current state $S_t$ and not on the previous states $S_0, S_1, \ldots, S_{t-1}$. Informally, we phrase this property as: "The future is independent of the past given the present". Formally, we can state this property of the states as:
$$\mathbb{P}[S_{t+1}|S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[S_{t+1}|S_t]\text{ for all } t \geq 0$$
This is a highly desirable property since it helps make the mathematics of such processes much easier and the computations much more tractable. We call this the *Markov Property* of States, or simply that these are *Markov States*. 

Let us now code this up. First, we create a dataclass to represent the dynamics of this process. As you can see in the code below, the dataclass `Process1` contains two attributes `level_param: int` and `alpha1: float = 0.25` to represent $L$ and $\alpha_1$ respectively. It contains the method `up_prob` to calculate $\mathbb{P}[X_{t+1} = X_t + 1]$ and the method `next_state`, which samples from a Bernoulli distribution (whose probability is obtained from the method `up_prob`) and creates the next state $S_{t+1}$ from the current state $S_t$. Also, note the nested dataclass `State` meant to represent the state of Process 1 (it's only attribute `price: int` reflects the fact that the state consists of only the current price, which is an integer).


```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: State) -> float:
        return 1. / (1 + np.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)
```


Next, we write a simple simulator using Python's generator functionality (using `yield`) as follows:

```python
def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)
```

Now we can use this simulator function to generate sampling traces. In the following code, we generate `num_traces` number of sampling traces over `time_steps` number of time steps starting from a price $X_0$ of `start_price`. The use of Python's generator feature lets us do this "lazily" (on-demand) using the ``itertools.islice`` function.

```python
import itertools

def process1_price_traces(
    start_price: int,
    level_param: int,
    alpha1: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = Process1(level_param=level_param, alpha1=alpha1)
    start_state = Process1.State(price=start_price)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])
```

The entire code is in the file [rl/chapter2/stock_price_simulations.py][stock_price_simulations.py]. We encourage you to play with this code with different ``start_price, level_param, alpha1, time_steps, num_traces``. You can plot graphs of sampling traces of the stock price, or plot graphs of the terminal distributions of the stock price at various time points (both of these plotting functions are made available for you in this code).

Now let us consider a different process.

**Process 2**:
$$\mathbb{P}[X_{t+1} = X_t + 1] =
\begin{cases}
0.5 (1 - \alpha_2(X_t - X_{t-1})) & \text{if } t > 0\\
0.5 & \text{if } t = 0
\end{cases}
$$
where $\alpha_2$ is a "pull strength" parameter in the closed interval $[0, 1]$. The intuition here is that the direction of the next move $X_{t+1} - X_t$ is biased in the reverse direction of the previous move $X_t - X_{t-1}$ and the extent of the bias is controlled by the parameter $\alpha_2$. 

We note that if we model the state $S_t$ as $X_t$, we won't satisfy the Markov Property because the probabilities of $X_{t+1}$ depend on not just $X_t$ but also on $X_{t-1}$. However, we can perform a little trick here and create an augmented state $S_t$ consisting of the pair
$(X_t, X_t - X_{t-1})$. In case $t=0$, the state $S_0$ can assume the value $(X_0, Null)$ where $Null$ is just a symbol denoting the fact that there have been no stock price movements thus far. With the state $S_t$ as this pair $(X_t, X_t - X_{t-1})$ , we can see that the Markov Property is indeed satisfied:
$$\mathbb{P}[(X_{t+1}, X_{t+1} - X_t)|(X_t, X_t - X_{t-1}), (X_{t-1}, X_{t-1} - X_{t-2}), \ldots, (X_0, Null)]$$
$$= \mathbb{P}[(X_{t+1}, X_{t+1} - X_t)|(X_t, X_t - X_{t-1})] = 0.5(1 - \alpha_2(X_{t+1} - X_t)(X_t - X_{t-1}))$$

It would be natural to wonder why the state doesn't comprise of simply $X_t - X_{t-1}$ - in other words, why is $X_t$ also required to be part of the state. It is true that knowledge of simply $X_t - X_{t-1}$ fully determines the probabilities of $X_{t+1} - X_t$. So if we set the state to be simply $X_t - X_{t-1}$ at any time step $t$, then we do get a Markov Process with just two states +1 and -1 (with probability transitions between them). However, this simple Markov Process doesn't tell us what the value of stock price $X_t$ is by looking at the state $X_t - X_{t-1}$ at time $t$. In this application, we not only care about Markovian state transition probabilities, but we also care about knowledge of stock price at any time $t$ from knowledge of the state at time $t$. Hence, we model the state as the pair $(X_t, X_{t-1})$.
 
Note that if we had modeled the state $S_t$ as the entire stock price history $(X_0, X_1, \ldots, X_t)$, then the Markov Property would be satisfied trivially. The Markov Property would also be satisfied if we had modeled the state $S_t$ as the pair $(X_t, X_{t-1})$ for $t > 0$ and $S_0$ as $(X_0, Null)$. However, our choice of $S_t := (X_t, X_t - X_{t-1})$ is the "simplest/minimal" internal representation. In fact, in this entire book, our endeavor in modeling states for various processes is to ensure the Markov Property with the "simplest/minimal" representation for the state.
 The corresponding dataclass for Process 2 is shown below:
 
 ```python
handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}

@dataclass
class Process2:
    @dataclass
    class State:
        price: int
        is_prev_move_up: Optional[bool]

    alpha2: float = 0.75  # strength of reverse-pull (value in [0,1])

    def up_prob(self, state: State) -> float:
        return 0.5 * (1 + self.alpha2 * handy_map[state.is_prev_move_up])

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process2.State(
            price=state.price + up_move * 2 - 1,
            is_prev_move_up=bool(up_move)
        )
```

The code for generation of sampling traces of the stock price is almost identical to the code we wrote for Process 1.

```python
def process2_price_traces(
    start_price: int,
    alpha2: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = Process2(alpha2=alpha2)
    start_state = Process2.State(price=start_price, is_prev_move_up=None)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])
```
 
 Now let us look at a more complicated process.

**Process 3**: This is an extension of Process 2 where the probability of next movement depends not only on the last movement, but on all past movements. Specifically, it depends on the number of past up-moves (call it $U_t = \sum_{i=1}^t \max(X_i - X_{i-1}, 0)$) relative to the number of past down-moves (call it $D_t = \sum_{i=1}^t \max(X_{i-1} - X_i, 0)$) in the following manner:
$$\mathbb{P}[X_{t+1} = X_t + 1] =
\begin{cases}
\frac 1 {1 + (\frac {U_t + D_t} {D_t} - 1)^{\alpha_3}} & \text{if } t > 0\\
0.5 & \text{if } t = 0
\end{cases}
$$
where $\alpha_3 \in \mathbb{R}_{\geq 0}$ is a "pull strength" parameter. Let us view the above probability expression as:
$$f(\frac {D_t} {U_t + D_t}; \alpha_3)$$
where $f: [0, 1] \rightarrow [0, 1]$ is a sigmoid-shaped function
 $$f(x; \alpha) = \frac 1 {1 + (\frac 1 x - 1)^{\alpha}}$$
 whose steepness at $x=0.5$ is controlled by the parameter $\alpha$ (note: values of $\alpha < 1$ will produce an inverse sigmoid as seen in Figure \ref{fig:unit_sigmoid} which shows unit-sigmoid functions $f$ for different values of $\alpha$). 

<div style="text-align:center" markdown="1">
![Unit-Sigmoid Curves \label{fig:unit_sigmoid}](./chapter2/unit_sigmoid_curves.png "Unit-Sigmoid Curves")
</div>
 The probability of next up-movement is fundamentally dependent on the quantity $\frac {D_t} {U_t + D_t}$ (the function $f$ simply serves to control the extent of the "reverse pull"). $\frac {D_t} {U_t + D_t}$ is the fraction of past time steps when there was a down-move. So, if number of down-moves in history are greater than number of up-moves in history, then there will be more of an up-pull than a down-pull for the next price movement $X_{t+1} - X_t$ (likewise, the other way round when $U_t > D_t$). The extent of this "reverse pull" is controlled by the "pull strength" parameter $\alpha_3$ (governed by the sigmoid-shaped function $f$).

Again, note that if we model the state $S_t$ as $X_t$, we won't satisfy the Markov Property because the probabilities of next state $S_{t+1} = X_{t+1}$ depends on the entire history of stock price moves and not just on the current state $S_t = X_t$. However, we can again do something clever and create a compact enough state $S_t$ consisting of simply the pair $(U_t, D_t)$. With this representation for the state $S_t$, the Markov Property is indeed satisfied:
$$\mathbb{P}[(U_{t+1}, D_{t+1})|(U_t, D_t), (U_{t-1}, D_{t-1}), \ldots, (U_0, D_0)]
= \mathbb{P}[(U_{t+1}, D_{t+1})|(U_t, D_t)]$$
$$=
\begin{cases}
f(\frac {D_t} {U_t + D_t}; \alpha_3) & \text{if }U_{t+1} = U_t + 1, D_{t+1} = D_t\\ 
f(\frac {U_t} {U_t + D_t}; \alpha_3) & \text{if }U_{t+1} = U_t, D_{t+1} = D_t + 1 
\end{cases}
$$
It is important to note that unlike Processes 1 and 2, the stock price $X_t$ is actually not part of the state $S_t$ in Process 3. This is because $U_t$ and $D_t$ together contain sufficient information to capture the stock price $X_t$ (since $X_t = X_0 + U_t - D_t$, and noting that $X_0$ is provided as a constant).

The corresponding dataclass for Process 3 is shown below:

```python
@dataclass
class Process3:
    @dataclass
    class State:
        num_up_moves: int
        num_down_moves: int

    alpha3: float = 1.0  # strength of reverse-pull (non-negative value)

    def up_prob(self, state: State) -> float:
        total = state.num_up_moves + state.num_down_moves
        if total == 0:
            return 0.5
        elif state.num_down_moves == 0:
            return state.num_down_moves ** self.alpha3
        else:
            return 1. / (1 + (total / state.num_down_moves - 1) ** self.alpha3)

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process3.State(
            num_up_moves=state.num_up_moves + up_move,
            num_down_moves=state.num_down_moves + 1 - up_move
        )
```
The code for generation of sampling traces of the stock price is shown below:

```python
def process3_price_traces(
    start_price: int,
    alpha3: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = Process3(alpha3=alpha3)
    start_state = Process3.State(num_up_moves=0, num_down_moves=0)
    return np.vstack([
        np.fromiter((start_price + s.num_up_moves - s.num_down_moves
                    for s in itertools.islice(simulation(process, start_state),
                                              time_steps + 1)), float)
        for _ in range(num_traces)])
```

As suggested for Process 1, you can plot graphs of sampling traces of the stock price, or plot graphs of the probability distributions of the stock price at various terminal time points $T$ for Processes 2 and 3, by playing with the [code in rl/chapter2/stock_price_simulations.py][stock_price_simulations.py].

[stock_price_simulations.py]: https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/stock_price_simulations.py

 Figure \ref{fig:single_trace_mp} shows a single sampling trace of stock prices for each of the 3 processes. Figure \ref{fig:terminal_distribution_mp} shows the probability distribution of the stock price at terminal time $T=100$ over 1000 traces.

<div style="text-align:center" markdown="1">
![Single Sampling Trace \label{fig:single_trace_mp}](./chapter2/single_traces.png "Single Sampling Trace")
</div>

<div style="text-align:center" markdown="1">
![Terminal Distribution \label{fig:terminal_distribution_mp}](./chapter2/terminal_distribution.png "Terminal Distribution")
</div>

Having developed the intuition for the Markov Property of States, we are now ready to formalize the notion of Markov Processes (some of the literature refers to Markov Processes as Markov Chains, but we will stick with the term Markov Processes).

### Formal Definitions for Markov Processes

Our formal definitions in this book will be restricted to Discrete-Time Markov Processes, where time moves forward in discrete time steps $t=0, 1, 2, \ldots$. Also for ease of exposition, our formal definitions in this book will be restricted to sets of states that are [countable](https://en.wikipedia.org/wiki/Countable_set). A countable set can be either a finite set or an infinite set of the same cardinality as the set of natural numbers, i.e., a set that is enumerable. This book will cover examples of continuous-time Markov Processes, where time is a continuous variable [^continuous-time]. This book will also cover examples of sets of states that are uncountable [^uncountable]. However, for ease of exposition, our definitions and development of the theory in this book will be restricted to discrete-time and countable sets of states. The definitions and theory can be analogously extended to continuous-time or to uncountable sets of states (we request you to self-adjust the definitions and theory accordingly when you encounter continuous-time or uncountable sets of states in this book).

[^continuous-time]: Markov Processes in continuous-time often go by the name *Stochastic Processes*, and their calculus goes by the name *Stochastic Calculus* (see Appendix [-@sec:stochasticcalculus-appendix]).

[^uncountable]: Uncountable sets are those with cardinality larger than the set of natural numbers, eg: the set of real numbers, which are not enumerable.

\begin{definition}
A {\em Markov Process} consists of:
\begin{itemize}
\item A countable set of states $\mathcal{S}$ (known as the State Space) and a set $\mathcal{T} \subseteq \mathcal{S}$ (known as the set of Terminal States)
 \item A time-indexed sequence of random states $S_t \in \mathcal{S}$ for time steps $t=0, 1, 2, \ldots$ with each state transition satisfying the Markov Property: $\mathbb{P}[S_{t+1}|S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[S_{t+1}|S_t]$ for all $t \geq 0$.
 \item Termination: If an outcome for $S_T$ (for some time step $T$) is a state in the set $\mathcal{T}$, then this sequence outcome terminates at time step $T$.
 \end{itemize}
 \end{definition}

 We refer to $\mathbb{P}[S_{t+1}|S_t]$ as the transition probabilities for time $t$.

\begin{definition}
A {\em Time-Homogeneous Markov Process} is a Markov Process with the additional property that
$\mathbb{P}[S_{t+1}|S_t]$ is independent of $t$.
 \end{definition}

This means, the dynamics of a Time-Homogeneous Markov Process can be fully specified with the function $$\mathcal{P}: (\mathcal{S} - \mathcal{T}) \times \mathcal{S} \rightarrow [0,1]$$ defined as:
$$\mathcal{P}(s, s') = \mathbb{P}[S_{t+1}=s'|S_t=s] \text{ for time steps } t = 0, 1, 2, \ldots, \text{ for all } s \in \mathcal{S} - \mathcal{T}, s' \in \mathcal{S}$$
such that
$$\sum_{s'\in \mathcal{S}} \mathcal{P}(s,s') = 1 \text{ for all } s \in \mathcal{S} - \mathcal{T}$$
We refer to the function $\mathcal{P}$ as the transition probability function of the Time-Homogeneous Markov Process, with the first argument to $\mathcal{P}$ to be thought of as the "source" state and the second argument as the "destination" state.

Note that the arguments to $\mathcal{P}$ in the above specification are devoid of the time index $t$ (hence, the term *Time-Homogeneous* which means "time-invariant"). Moreover, note that a Markov Process that is not time-homogeneous can be converted to a Time-Homogeneous Markov Process by augmenting all states with the time index $t$. This means if the original state space of a Markov Process that is not time-homogeneous is $\mathcal{S}$, then the state space of the corresponding Time-Homogeneous Markov Process is $\mathbb{Z}_{\geq 0} \times \mathcal{S}$ (where $\mathbb{Z}_{\geq 0}$ denotes the domain of the time index). This is because each time step has it's own unique set of (augmented) states, which means the entire set of states in $\mathbb{Z}_{\geq 0} \times \mathcal{S}$ can be covered by time-invariant transition probabilities, thus qualifying as a Time-Homogeneous Markov Process. Therefore, henceforth, any time we say *Markov Process*, assume we are refering to a *Discrete-Time, Time-Homogeneous Markov Process with a Countable State Space* (unless explicitly specified otherwise), which in turn will be characterized by the transition probability function $\mathcal{P}$. Note that the stock price examples (all 3 of the Processes we covered) are examples of a (Time-Homogeneous) Markov Process, even without requiring augmenting the state with the time index.

The classical definitions and theory of Markov Processes model "termination" with the idea of *Absorbing States*. A state $s$ is called an absorbing state if $\mathcal{P}(s,s) = 1$. This means, once we reach an absorbing state, we are "trapped" there, hence capturing the notion of "termination". So the classical definitions and theory of Markov Processes typically don't include an explicit specification of states as terminal and non-terminal. However, when we get to Markov Reward Processes and Markov Decision Processes (frameworks that are extensions of Markov Processes), we will need to explicitly specify states as terminal and non-terminal states, rather than model the notion of termination with absorbing states. So, for consistency in definitions and in the development of the theory, we are going with a framework where states in a Markov Process are explicitly specified as terminal or non-terminal states. We won't consider an absorbing state as a terminal state as the Markov Process keeps moving forward in time forever when it gets to an absorbing state. We will refer to $\mathcal{S} - \mathcal{T}$ as the set of Non-Terminal States $\mathcal{N}$ (and we will refer to a state in $\mathcal{N}$ as a non-terminal state). The sequence $S_0, S_1, S_2, \ldots$ terminates at time step $t=T$ if $S_T \in \mathcal{T}$.

#### Starting States

Now it's natural to ask the question: How do we "start" the Markov Process (in the stock price examples, this was the notion of the start state)? More generally, we'd like to specify a probability distribution of start states so we can perform simulations and (let's say) compute the probability distribution of states at specific future time steps. While this is a relevant question, we'd like to separate the following two specifications:

* Specification of the transition probability function $\mathcal{P}$
* Specification of the probability distribution of start states (denote this as $\mu: \mathcal{N} \rightarrow [0,1]$)

We say that a Markov Process is fully specified by $\mathcal{P}$ in the sense that this gives us the transition probabilities that govern the complete dynamics of the Markov Process. A way to understand this is to relate specification of $\mathcal{P}$ to the specification of rules in a game (such as chess or monopoly). These games are specified with a finite (in fact, fairly compact) set of rules that is easy for a newbie to the game to understand. However, when we want to *actually play* the game, we need to specify the starting position (one could start these games at arbitrary, but legal, starting positions and not just at some canonical starting position). The specification of the start state of the game is analogous to the specification of $\mu$. Given $\mu$ together with $\mathcal{P}$ enables us to generate sampling traces of the Markov Process (analogously, *play* games like chess or monopoly). These sampling traces typically result in a wide range of outcomes due to sampling and long-running of the Markov Process (versus compact specification of transition probabilities). These sampling traces enable us to answer questions such as probability distribution of states at specific future time steps or expected time of first occurrence of a specific state etc., given a certain starting probability distribution $\mu$.

Thinking about the separation between specifying the rules of the game versus actually playing the game helps us understand the need to separate the notion of dynamics specification $\mathcal{P}$ (fundamental to the time-homogeneous character of the Markov Process) and the notion of starting distribution $\mu$ (required to perform sampling traces). Hence, the separation of concerns between $\mathcal{P}$ and $\mu$ is key to the conceptualization of Markov Processes. Likewise, we separate the concerns in our code design as well, as evidenced by how we separated the ``next_state`` method in the Process dataclasses and the ``simulation`` function.

#### Terminal States

Games are examples of Markov Processes that terminate at specific states (based on rules for winning or losing the game). In general, in a Markov Process, termination might occur after a variable number of time steps (like in the games examples), or like we will see in many financial application examples, termination might occur after a fixed number of time steps, or like in the stock price examples we saw earlier, there is in fact no termination.

If all random sequences of states (sampling traces) reach a terminal state, then we say that these random sequences of the Markov Process are *Episodic* (otherwise we call these sequences as *Continuing*). The notion of episodic sequences is important in Reinforcement Learning since some Reinforcement Learning algorithms require episodic sequences.

When we cover some of the financial applications later in this book, we will find that the Markov Process terminates after a fixed number of time steps, say $T$. In these applications, the time index $t$ is part of the state representation, each state with time index $t=T$ is labeled a terminal state, and all states with time index $t<T$ will transition to states with time index $t+1$.

Now we are ready to write some code for Markov Processes, where we illustrate how to specify that certain states are terminal states.

#### Markov Process Implementation

The first thing we do is to create separate classes for non-terminal states $\mathcal{N}$ and terminal states $\mathcal{T}$, with an abstract base class for all states $\mathcal{S}$ (terminal or non-terminal). In the code below, the abstract base class (`ABC`) `State` represents $\mathcal{S}$. The class `State` is parameterized by a generic type (`TypeVar('S')`) representing a generic state space `Generic[S]`.

The concrete class `Terminal` represents $\mathcal{T}$ and the concrete class `NonTerminal` represents $\mathcal{N}$. The method `on_non_terminal` will prove to be very beneficial in the implementation of various algorithms we shall be writing for Markov Processes and also for Markov Reward Processes and Markov Decision Processes (which are extensions of Markov Processes). The method `on_non_terminal` enables us to calculate a value for all states in $\mathcal{S}$ even though the calculation is defined only for all non-terminal states $\mathcal{N}$. The argument `f` to `on_non_terminal` defines this value through a function from $\mathcal{N}$ to an arbitrary value-type `X`. The argument `default` provides the default value for terminal states $T$ so that `on_non_terminal` can be used on any object in `State` (i.e. for any state in $\mathcal{S}$, terminal or non-terminal). As an example, let's say you want to calculate the expected number of states one would traverse after a certain state and before hitting a terminal state. Clearly, this calculation is well-defined for non-terminal states and the function `f` would implement this by either some kind of analytical method or by sampling state-transition sequences and averaging the counts of non-terminal states traversed across those sequences. By defining (`default`ing) this value to be 0 for terminal states, we can then invoke such a calculation for all states $\mathcal{S}$, terminal or non-terminal, and embed this calculation in an algorithm without worrying about special handing in the code for the edge case of being a terminal state.


```python
from abc import ABC
from dataclasses import dataclass
from typing import Generic, Callable

S = TypeVar('S')
X = TypeVar('X')

class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(
        self,
        f: Callable[[NonTerminal[S]], X],
        default: X
    ) -> X:
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default

@dataclass(frozen=True)
class Terminal(State[S]):
    state: S

@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S
```

Now we are ready to write a class to represent Markov Processes. We create an abstract class `MarkovProcess` parameterized by a generic type (`TypeVar('S')`) representing a generic state space `Generic[S]`. The abstract class has an `@abstractmethod` called `transition` that is meant to specify the transition probability distribution of next states, given a current non-terminal state. We know that `transition` is well-defined only for non-terminal states and hence, it's argument is clearly type-annotated as `NonTerminal[S]`. The return type of `transition` is `Distribution[State[S]]`, which as we know from the Chapter on *Programming and Design*, represents the probability distribution of next states. We also have a method `simulate` that enables us to generate an `Iterable` (generator) of sampled states, given as input a `start_state_distribution: Distribution[NonTerminal[S]]` (from which we sample the starting state). The sampling of next states relies on the implementation of the `sample` method for  the `Distribution[State[S]]` object produced by the `transition` method. Here's the full body of the abstract class `MarkovProcess`:

```python
from abc import abstractmethod
from rl.distribution import Distribution
from typing import Iterable

class MarkovProcess(ABC, Generic[S]):
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        pass

    def simulate(
        self,
        start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[State[S]]:
        state: State[S] = start_state_distribution.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state
```

The above code is in the file [rl/markov_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py).

### Stock Price Examples modeled as Markov Processes

So if you have a mathematical specification of the transition probabilities of a Markov Process, all you need to do is to create a concrete class that implements the interface of the abstract class `MarkovProcess` (specifically by implementing the  `@abstractmethod transition`) in a manner that captures your mathematical specification of the transition probabilities. Let us write this for the case of Process 3 (the 3rd example of stock price transitions we covered earlier). We name the concrete class as `StockPriceMP3`. Note that the generic state space `S` is now replaced with a specific state space represented by the type `@dataclass StateMP3`. The code should be self-explanatory since we implemented this process as a standalone in the previous section. Note the use of the `Categorical` distribution in the `transition` method to capture the 2-outcomes probability distribution of next states (for movements up or down).

```python
from rl.distribution import Categorical
from rl.gen_utils.common_funcs import get_unit_sigmoid_func

@dataclass
class StateMP3:
    num_up_moves: int
    num_down_moves: int

@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):

    alpha3: float = 1.0  # strength of reverse-pull (non-negative value)

    def up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(
            state.num_down_moves / total
        ) if total else 0.5

    def transition(
        self,
        state: NonTerminal[StateMP3]
    ) -> Categorical[State[StateMP3]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP3(
                state.state.num_up_moves + 1, state.state.num_down_moves
            )): up_p,
            NonTerminal(StateMP3(
                state.state.num_up_moves, state.state.num_down_moves + 1
            )): 1 - up_p
        })
```

To generate sampling traces, we write the following function:

```python
from rl.distribution import Constant
import numpy as np

def process3_price_traces(
    start_price: int,
    alpha3: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP3(alpha3=alpha3)
    start_state_distribution = Constant(
        NonTerminal(StateMP3(num_up_moves=0, num_down_moves=0))
    )
    return np.vstack([np.fromiter(
        (start_price + s.state.num_up_moves - s.state.num_down_moves for s in
         itertools.islice(
             mp.simulate(start_state_distribution),
             time_steps + 1
         )),
        float
    ) for _ in range(num_traces)])
```

We leave it to you as an exercise to similarly implement Stock Price Processes 1 and 2 that we had covered earlier. The complete code along with the driver to set input parameters, run all 3 processes and create plots is in the file [rl/chapter2/stock_price_mp.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/stock_price_mp.py). We encourage you to change the input parameters in `__main__` and get an intuitive feel for how the simulation results vary with the changes in parameters.

### Finite Markov Processes

Now let us consider Markov Processes with a finite state space. So we can represent the state space as $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$. Assume the set of non-terminal states $\mathcal{N}$ has $m\leq n$ states. Let us refer to Markov Processes with finite state spaces as Finite Markov Processes. Since Finite Markov Processes are a subclass of Markov Processes, it would make sense to create a concrete class `FiniteMarkovProcess` that implements the interface of the abstract class `MarkovProcess` (specifically implement the `@abstractmethod transition`). But first let's think about the data structure required to specify an instance of a `FiniteMarkovProcess` (i.e., the data structure we'd pass to the `__init__` method of `FiniteMarkovProcess`). One choice is a $m \times n$ 2D numpy array representation, i.e., matrix elements representing transition probabilities
$$\mathcal{P} : \mathcal{N} \times \mathcal{S} \rightarrow [0, 1]$$
However, we often find that this matrix can be sparse since one often transitions from a given state to just a few set of states. So we'd like a sparse representation and we can accomplish this by conceptualizing $\mathcal{P}$ in an [equivalent curried form](https://en.wikipedia.org/wiki/Currying)[^currying] as follows:
$$\mathcal{N} \rightarrow (\mathcal{S} \rightarrow [0, 1])$$

[^currying]: Currying is the technique of converting a function that takes multiple arguments into a sequence of functions that each takes a single argument, as illustrated above for the $\mathcal{P}$ function.

With this curried view, we can represent the outer $\rightarrow$ as a map (in Python, as a dictionary of type `Mapping`) whose keys are the non-terminal states $\mathcal{N}$, and each non-terminal-state key maps to a `FiniteDistribution[S]` type that represents the inner $\rightarrow$, i.e. a finite probability distribution of the next states transitioned to from a given non-terminal state.

Note that the `FiniteDistribution[S]` will only contain the set of states transitioned to with non-zero probability. To make things concrete, here's a toy Markov Process data structure example of a city with highly unpredictable weather outcomes from one day to the next (note: `Categorical` type inherits from `FiniteDistribution` type in the code at [rl/distribution.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/distribution.py)):

```python
from rl.distribution import Categorical
{
  "Rain": Categorical({"Rain": 0.3, "Nice": 0.7}),
  "Snow": Categorical({"Rain": 0.4, "Snow": 0.6}),
  "Nice": Categorical({"Rain": 0.2, "Snow": 0.3})
}
```

It is common to view this Markov Process representation as a directed graph, as depicted in Figure \ref{fig:weather_mp}. The nodes are the states and the directed edges are the probabilistic state transitions, with the transition probabilities labeled on them.

<div style="text-align:center" markdown="1">
![Weather Markov Process \label{fig:weather_mp}](./chapter2/weather_mp.png "Weather Markov Process")
</div>

Our goal now is to define a `FiniteMarkovProcess` class that is a concrete class implementation of the abstract class `MarkovProcess`. This requires us to wrap the states in the keys/values of the `FiniteMarkovProcess` dictionary with the appropriate `Terminal` or `NonTerminal` wrapping. Let's create an alias called `Transition` for this wrapped dictionary data structure since we will use this wrapped data structure often:

```python
from typing import Mapping
from rl.distribution import FiniteDistribution

Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]
```


To create a `Transition` data type from the above example of the weather Markov Process, we'd need to wrap each of the "Rain", "Snow" and "Nice" strings with `NonTerminal`. 

Now we are ready to write the code for the `FiniteMarkovProcess` class. The `__init__` method (constructor) takes as argument a `transition_map` whose type is similar to `Transition[S]` except that we use the `S` type directly in the `Mapping` representation instead of `NonTerminal[S]` or `State[S]` (this is convenient for users to specify their Markov Process in a succinct `Mapping` representation without the burden of wrapping each `S` with a `NonTerminal[S]` or `Terminal[S]`). The dictionary we created above for the weather Markov Process can be used as the `transition_map` argument. However, this means the `__init__` method needs to wrap the specified `S` states as `NonTerminal[S]` or `Terminal[S]` when creating the attribute `self.transition_map`. We also have an attribute `self.non_terminal_states: Sequence[NonTerminal[S]]` that is an ordered sequence of the non-terminal states. We implement the `transition` method by simply returning the `FiniteDistribution[State[S]]` the given `state: NonTerminal[S]` maps to in the attribute `self.transition_map: Transition[S]`. Note that along with the `transition` method, we have implemented the `__repr__` method for a well-formatted display of `self.transition_map`.

```python
from typing import Sequence
from rl.distribution import FiniteDistribution, Categorical

class FiniteMarkovProcess(MarkovProcess[S]):
    non_terminal_states: Sequence[NonTerminal[S]]
    transition_map: Transition[S]

    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        non_terminals: Set[S] = set(transition_map.keys())
        self.transition_map = {
            NonTerminal(s): Categorical(
                {(NonTerminal(s1) if s1 in non_terminals else Terminal(s1)): p
                 for s1, p in v.table().items()}
            ) for s, v in transition_map.items()
        }
        self.non_terminal_states = list(self.transition_map.keys())

    def __repr__(self) -> str:
        display = ""

        for s, d in self.transition_map.items():
            display += f"From State {s.state}:\n"
            for s1, p in d:
                opt = "Terminal " if isinstance(s1, Terminal) else ""
                display += f"  To {opt}State {s1.state} with Probability {p:.3f}\n"

        return display

    def transition(self, state: NonTerminal[S])\
            -> FiniteDistribution[State[S]]:
        return self.transition_map[state]
```

The above code is in the file [rl/markov_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py).

### Simple Inventory Example

To help conceptualize Finite Markov Processes, let us consider a simple example of changes in inventory at a store. Assume you are the store manager and that you are tasked with controlling the ordering of inventory from a supplier. Let us focus on the inventory of a particular type of bicycle. Assume that each day there is random (non-negative integer) demand for the bicycle with the probabilities of demand following a Poisson distribution (with Poisson parameter $\lambda \in \mathbb{R}_{\geq 0}$), i.e. demand $i$ for each $i = 0, 1, 2, \ldots$ occurs with probability
$$f(i) = \frac {e^{-\lambda} \lambda^i} {i!}$$
Denote $F: \mathbb{Z}_{\geq 0} \rightarrow [0, 1]$ as the poisson cumulative probability distribution function, i.e.,
 $$F(i) = \sum_{j=0}^i f(j)$$

Assume you have storage capacity for at most $C \in \mathbb{Z}_{\geq 0}$ bicycles in your store. Each evening at 6pm when your store closes, you have the choice to order a certain number of bicycles from your supplier (including the option to not order any bicycles, on a given day). The ordered bicycles will arrive 36 hours later (at 6am the day after the day after you order - we refer to this as *delivery lead time* of 36 hours). Denote the *State* at 6pm store-closing each day as $(\alpha, \beta)$, where $\alpha$ is the inventory in the store (refered to as On-Hand Inventory at 6pm) and $\beta$ is the inventory on a truck from the supplier (that you had ordered the previous day) that will arrive in your store the next morning at 6am ($\beta$ is refered to as On-Order Inventory at 6pm). Due to your storage capacity constraint of at most $C$ bicycles, your ordering policy is to order $C-(\alpha + \beta)$ if $\alpha + \beta < C$ and to not order if $\alpha + \beta \geq C$. The precise sequence of events in a 24-hour cycle is:

* Observe the $(\alpha, \beta)$ *State* at 6pm store-closing (call this state $S_t$)
* Immediately order according to the ordering policy described above
* Receive bicycles at 6am if you had ordered 36 hours ago
* Open the store at 8am
* Experience random demand from customers according to demand probabilities stated above (number of bicycles sold for the day will be the minimum of demand on the day and inventory at store opening on the day)
* Close the store at 6pm and observe the state (this state is $S_{t+1}$)

If we let this process run for a while, in steady-state we ensure that $\alpha + \beta \leq C$. So to model this process as a Finite Markov Process, we shall only consider the steady-state (finite) set of states
$$\mathcal{S} = \{(\alpha, \beta) | \alpha \in \mathbb{Z}_{\geq 0}, \beta \in \mathbb{Z}_{\geq 0}, 0 \leq \alpha + \beta \leq C\}$$
So restricting ourselves to this finite set of states, our order quantity equals $C - (\alpha + \beta)$ when the state is $(\alpha, \beta)$.

If current state $S_t$ is $(\alpha, \beta)$, there are only $\alpha + \beta + 1$ possible next states $S_{t+1}$ as follows:
$$(\alpha + \beta - i, C - (\alpha + \beta)) \text{ for } i =0, 1, \ldots, \alpha + \beta$$
with transition probabilities governed by the Poisson probabilities of demand as follows:
$$\mathcal{P}((\alpha, \beta), (\alpha + \beta - i, C - (\alpha + \beta))) = f(i)\text{ for } 0 \leq i \leq \alpha + \beta - 1$$
$$\mathcal{P}((\alpha, \beta), (0, C - (\alpha + \beta))) = \sum_{j=\alpha+\beta}^{\infty} f(j) = 1 - F(\alpha + \beta - 1)$$
Note that the next state's ($S_{t+1}$) On-Hand can be zero resulting from any of infinite possible demand outcomes greater than or equal to $\alpha + \beta$. 

So we are now ready to write code for this simple inventory example as a Markov Process. All we have to do is to create a derived class inherited from `FiniteMarkovProcess` and write a method to construct the `transition_map: Transition`. Note that the generic state type `S` is replaced here with the `@dataclass InventoryState` consisting of the pair of On-Hand and On-Order inventory quantities comprising the state of this Finite Markov Process.

```python
from rl.distribution import Categorical
from scipy.stats import poisson

@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order

class SimpleInventoryMPFinite(FiniteMarkovProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[InventoryState, FiniteDistribution[InventoryState]]:
        d: Dict[InventoryState, Categorical[InventoryState]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                state_probs_map: Mapping[InventoryState, float] = {
                    InventoryState(ip - i, beta1):
                    (self.poisson_distr.pmf(i) if i < ip else
                     1 - self.poisson_distr.cdf(ip - 1))
                    for i in range(ip + 1)
                }
                d[InventoryState(alpha, beta)] = Categorical(state_probs_map)
        return d
```

Let us utilize the `__repr__` method written previously to view the transition probabilities for the simple case of $C=2$ and $\lambda = 1.0$ (this code is in the file [rl/chapter2/simple_inventory_mp.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/simple_inventory_mp.py))

```python
user_capacity = 2
user_poisson_lambda = 1.0

si_mp = SimpleInventoryMPFinite(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda
)

print(si_mp)
```

The output we get is nicely displayed as:

```
From State InventoryState(on_hand=0, on_order=0):
  To State InventoryState(on_hand=0, on_order=2) with Probability 1.000
From State InventoryState(on_hand=0, on_order=1):
  To State InventoryState(on_hand=1, on_order=1) with Probability 0.368
  To State InventoryState(on_hand=0, on_order=1) with Probability 0.632
From State InventoryState(on_hand=0, on_order=2):
  To State InventoryState(on_hand=2, on_order=0) with Probability 0.368
  To State InventoryState(on_hand=1, on_order=0) with Probability 0.368
  To State InventoryState(on_hand=0, on_order=0) with Probability 0.264
From State InventoryState(on_hand=1, on_order=0):
  To State InventoryState(on_hand=1, on_order=1) with Probability 0.368
  To State InventoryState(on_hand=0, on_order=1) with Probability 0.632
From State InventoryState(on_hand=1, on_order=1):
  To State InventoryState(on_hand=2, on_order=0) with Probability 0.368
  To State InventoryState(on_hand=1, on_order=0) with Probability 0.368
  To State InventoryState(on_hand=0, on_order=0) with Probability 0.264
From State InventoryState(on_hand=2, on_order=0):
  To State InventoryState(on_hand=2, on_order=0) with Probability 0.368
  To State InventoryState(on_hand=1, on_order=0) with Probability 0.368
  To State InventoryState(on_hand=0, on_order=0) with Probability 0.264
```

For a graphical view of this Markov Process, see Figure \ref{fig:inventory_mp}. The nodes are the states, labeled with their corresponding $\alpha$ and $\beta$ values. The directed edges are the probabilistic state transitions from 6pm on a day to 6pm on the next day, with the transition probabilities labeled on them.

<div style="text-align:center" markdown="1">
![Simple Inventory Markov Process \label{fig:inventory_mp}](./chapter2/simple_inv_mp.png "Simple Inventory Markov Process")
</div>

We can perform a number of interesting experiments and calculations with this simple Markov Process and we encourage you to play with this code by changing values of the capacity $C$ and poisson mean $\lambda$, performing simulations and probabilistic calculations of natural curiosity for a store owner.

There is a rich and interesting theory for Markov Processes. However, we won't go into this theory as our coverage of Markov Processes so far is a sufficient building block to take us to the incremental topics of Markov Reward Processes and Markov Decision Processes. However, before we move on, we'd like to show just a glimpse of the rich theory with the calculation of *Stationary Probabilities* and apply it to the case of the above simple inventory Markov Process.

### Stationary Distribution of a Markov Process
\begin{definition} 
 The {\em Stationary Distribution} of a (Discrete-Time, Time-Homogeneous) Markov Process with state space $\mathcal{S} = \mathcal{N}$ and transition probability function $\mathcal{P}: \mathcal{N} \times \mathcal{N} \rightarrow [0, 1]$ is a probability distribution function $\pi: \mathcal{N} \rightarrow [0, 1]$ such that:
  $$\pi(s) = \sum_{s'\in \mathcal{N}} \pi(s) \cdot \mathcal{P}(s, s') \text{ for all } s \in \mathcal{N}$$
\end{definition}

The intuitive view of the stationary distribution $\pi$ is that (under specific conditions we are not listing here) if we let the Markov Process run forever, then in the long run the states occur at specific time steps with relative frequencies (probabilities) given by a distribution $\pi$ that is independent of the time step. The probability of occurrence of a specific state $s$ at a time step (asymptotically far out in the future) should be equal to the sum-product of probabilities of occurrence of all the states at the previous time step and the transition probabilities from those states to $s$. But since the states' occurrence probabilities are invariant in time, the $\pi$ distribution for the previous time step is the same as the $\pi$ distribution for the time step we considered. This argument holds for all states $s$, and that is exactly the statement of the definition of *Stationary Distribution* formalized above.

If we specialize this definition of *Stationary Distribution* to Finite-States, Discrete-Time, Time-Homogeneous Markov Processes with state space $\mathcal{S} = \{s_1, s_2, \ldots, s_n\} = \mathcal{N}$, then we can express the Stationary Distribution $\pi$ as follows:
$$\pi(s_j) = \sum_{i=1}^n \pi(s_i) \cdot \mathcal{P}(s_i, s_j) \text{ for all } j = 1, 2, \ldots n$$

Below we use bold-face notation to represent functions as vectors and matrices (since we assume finite states). So, $\bm{\pi}$ is a column vector of length $n$ and $\bm{\mathcal{P}}$ is the $n \times n$ transition probability matrix (rows are source states, columns are destination states with each row summing to 1).
Then, the statement of the above definition can be succinctly expressed as:
$$\bm{\pi}^T = \bm{\pi}^T \cdot \bm{\mathcal{P}}$$
which can be re-written as:
$$\bm{\mathcal{P}}^T \cdot \bm{\pi} = \bm{\pi}$$
But this is simply saying that $\bm{\pi}$ is an eigenvector of $\bm{\mathcal{P}}^T$ with eigenvalue of 1. So then, it should be easy to obtain the stationary distribution $\bm{\pi}$ from an eigenvectors and eigenvalues calculation of $\bm{\mathcal{P}}^T$. 

Let us write code to compute the stationary distribution. We shall add two methods in the `FiniteMarkovProcess` class, one for setting up the transition probability matrix $\bm{\mathcal{P}}$ (`get_transition_matrix` method) and another to calculate the stationary distribution $\bm{\pi}$ (`get_stationary_distribution`) from this transition probability matrix. Note that $\bm{\mathcal{P}}$ is restricted to $\mathcal{N} \times \mathcal{N} \rightarrow [0, 1]$ (rather than $\mathcal{N} \times \mathcal{S} \rightarrow [0, 1]$) because these probability transitions suffice for all the calculations we will be performing for Finite Markov Processes. Here's the code for the two methods (the full code for `FiniteMarkovProcess` is in the file [`rl/markov_process.py`](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py)):

```python
import numpy as np
from rl.distribution import FiniteDistribution, Categorical

    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.non_terminal_states)
        mat = np.zeros((sz, sz))

        for i, s1 in enumerate(self.non_terminal_states):
            for j, s2 in enumerate(self.non_terminal_states):
                mat[i, j] = self.transition(s1).probability(s2)

        return mat

    def get_stationary_distribution(self) -> FiniteDistribution[S]:
        eig_vals, eig_vecs = np.linalg.eig(self.get_transition_matrix().T)
        index_of_first_unit_eig_val = np.where(
            np.abs(eig_vals - 1) < 1e-8)[0][0]
        eig_vec_of_unit_eig_val = np.real(
            eig_vecs[:, index_of_first_unit_eig_val])
        return Categorical({
            self.non_terminal_states[i].state: ev
            for i, ev in enumerate(eig_vec_of_unit_eig_val /
                                   sum(eig_vec_of_unit_eig_val))
        })
```

We skip the theory that tells us about the conditions under which a stationary distribution is well-defined, or the conditions under which there is a unique stationary distribution. Instead, we just go ahead with this calculation here assuming this Markov Process satisfies those conditions (it does!). So, we simply seek the index of the `eig_vals` vector with eigenvalue equal to 1 (accounting for floating-point error). Next, we pull out the column of the `eig_vecs` matrix at the `eig_vals` index calculated above, and convert it into a real-valued vector (eigenvectors/eigenvalues calculations are, in general, complex numbers calculations - see the reference for the `np.linalg.eig` function). So this gives us the real-valued eigenvector with eigenvalue equal to 1.  Finally, we have to normalize the eigenvector so it's values add up to 1 (since we want probabilities), and return the probabilities as a `Categorical` distribution).

Running this code for the simple case of capacity $C=2$ and poisson mean $\lambda = 1.0$ (instance of `SimpleInventoryMPFinite`) produces the following output for the stationary distribution $\pi$:

```
{InventoryState(on_hand=0, on_order=0): 0.117,
 InventoryState(on_hand=0, on_order=1): 0.279,
 InventoryState(on_hand=0, on_order=2): 0.117,
 InventoryState(on_hand=1, on_order=0): 0.162,
 InventoryState(on_hand=1, on_order=1): 0.162,
 InventoryState(on_hand=2, on_order=0): 0.162}
```

This tells us that On-Hand of 0 and On-Order of 1 is the state occurring most frequently (28% of the time) when the system is played out indefinitely.   

Let us summarize the 3 different representations we've covered:

* Functional Representation: as given by the `transition` method, i.e., given a non-terminal state, the `transition` method returns a probability distribution of next states. This representation is valuable when performing simulations by sampling the next state from the returned probability distribution of next states. This is applicable to the general case of Markov Processes (including infinite state spaces).
* Sparse Data Structure Representation: as given by `transition map: Transition`, which is convenient for compact storage and useful for visualization (eg: `__repr__` method display or as a directed graph figure). This is applicable only to Finite Markov Processes.
* Dense Data Structure Representation: as given by the `get_transition_matrix` 2D numpy array, which is useful for performing linear algebra that is often required to calculate mathematical properties of the process (eg: to calculate the stationary distribution). This is applicable only to Finite Markov Processes.

Now we are ready to move to our next topic of *Markov Reward Processes*. We'd like to finish this section by stating that the Markov Property owes its name to a mathematician from a century ago - [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov). Although the Markov Property seems like a simple enough concept, the concept has had profound implications on our ability to compute or reason with systems involving time-sequenced uncertainty in practice. There are several good books to learn more about Markov Processes - we recommend [the book by Paul Gagniuc](https://www.amazon.com/Markov-Chains-Theory-Implementation-Experimentation/dp/1119387558) [@Gagniuc2017MarkovCF].

### Formalism of Markov Reward Processes

As we've said earlier, the reason we covered Markov Processes is because we want to make our way to Markov Decision Processes (the framework for Reinforcement Learning algorithms) by adding incremental features to Markov Processes. Now we cover an intermediate framework between Markov Processes and Markov Decision Processes, known as Markov Reward Processes. We essentially just include the notion of a numerical *reward* to a Markov Process each time we transition from one state to the next. These rewards are random, and all we need to do is to specify the probability distributions of these rewards as we make state transitions. 

The main purpose of Markov Reward Processes is to calculate how much reward we would accumulate (in expectation, from each of the non-terminal states) if we let the Process run indefinitely, bearing in mind that future rewards need to be discounted appropriately (otherwise the sum of rewards could blow up to $\infty$). In order to solve the problem of calculating expected accumulative rewards from each non-terminal state, we will first set up some formalism for Markov Reward Processes, develop some (elegant) theory on calculating rewards accumulation, write plenty of code (based on the theory), and apply the theory and code to the simple inventory example (which we will embellish with rewards equal to negative of the costs incurred at the store).

\begin{definition}
A {\em Markov Reward Process} is a Markov Process, along with a time-indexed sequence of {\em Reward} random variables $R_t \in \mathcal{D}$ (a countable subset of $\mathbb{R}$) for time steps $t=1, 2, \ldots$, satisfying the Markov Property (including Rewards): $\mathbb{P}[(R_{t+1}, S_{t+1}) | S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[(R_{t+1}, S_{t+1}) | S_t]$ for all $t \geq 0$.
\end{definition}

It pays to emphasize again (like we emphasized for Markov Processes), that the definitions and theory of Markov Reward Processes we cover (by default) are for discrete-time, for countable state spaces and countable set of pairs of next state and reward transitions (with the knowledge that the definitions and theory are analogously extensible to continuous-time and uncountable spaces/transitions). In the more general case, where states or rewards are uncountable, the same concepts apply except that the mathematical formalism needs to be more detailed and more careful. Specifically, we'd end up with integrals instead of summations, and probability density functions (for continuous probability distributions) instead of probability mass functions (for discrete probability distributions). For ease of notation and more importantly, for ease of understanding of the core concepts (without being distracted by heavy mathematical formalism), we've chosen to stay with discrete-time, countable $\mathcal{S}$ and countable $\mathcal{D}$ (by default). However, there will be examples of Markov Reward Processes in this book involving continuous-time and uncountable $\mathcal{S}$ and $\mathcal{D}$ (please adjust the definitions and formulas accordingly).

Since we commonly assume Time-Homogeneity of Markov Processes, we shall also (by default) assume Time-Homogeneity for Markov Reward Processes, i.e., $\mathbb{P}[(R_{t+1}, S_{t+1}) | S_t]$ is independent of $t$. 

With the default assumption of time-homogeneity, the transition probabilities of a Markov Reward Process can be expressed as a transition probability function:
$$\mathcal{P}_R: \mathcal{N} \times \mathcal{D} \times \mathcal{S} \rightarrow [0,1]$$
defined as:
$$\mathcal{P}_R(s,r,s') = \mathbb{P}[(R_{t+1}=r, S_{t+1}=s') | S_t=s] \text{ for time steps } t= 0, 1, 2, \ldots,$$
$$\text{for all } s \in \mathcal{N}, r \in \mathcal{D}, s' \in \mathcal{S}, \text{ such that } \sum_{s'\in \mathcal{S}} \sum_{r \in \mathcal{D}} \mathcal{P}_R(s,r,s') = 1 \text{ for all } s \in \mathcal{N}$$

The subsection on *Start States* we had covered for Markov Processes naturally applies to Markov Reward Processes as well. So we won't repeat the section here, rather we simply highlight that when it comes to simulations, we need a separate specification of the probability distribution of start states. Also, by inheriting from our framework of Markov Processes, we model the notion of a "process termination" by explicitly specifying states as terminal states or non-terminal states. The sequence $S_0, R_1, S_1, R_2, S_2, \ldots$ terminates at time step $t=T$ if $S_T \in \mathcal{T}$, with $R_T$ being the final reward in the sequence.

If all random sequences of states in a Markov Reward Process terminate, we refer to it as *episodic* sequences (otherwise, we refer to it as *continuing* sequences). 

Let's write some code that captures this formalism. We create a derived `@abstractclass MarkovRewardProcess` that inherits from the `@abstractclass MarkovProcess`. Analogous to `MarkovProcess`'s `@abstractmethod transition` (that represents $\mathcal{P}$), `MarkovRewardProcess` has an `@abstractmethod transition_reward` that represents $\mathcal{P}_R$. Note that the return type of `transition_reward` is `Distribution[Tuple[State[S], float]]`, representing the probability distribution of (next state, reward) pairs transitioned to.

Also, analogous to `MarkovProcess`'s `simulate` method, `MarkovRewardProcess` has the method `simulate_reward` which generates a stream of `TransitionStep[S]` objects. Each `TransitionStep[S]` object consists of a 3-tuple: (state, next state, reward) representing the sampled transitions within the generated sampling trace. Here's the actual code:

```python
@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: NonTerminal[S]
    next_state: State[S]
    reward: float
    
class MarkovRewardProcess(MarkovProcess[S]):

    @abstractmethod
    def transition_reward(self, state: NonTerminal[S])\
            -> Distribution[Tuple[State[S], float]]:
        pass

    def simulate_reward(
        self,
        start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[TransitionStep[S]]:
        state: State[S] = start_state_distribution.sample()
        reward: float = 0.

        while isinstance(state, NonTerminal):
            next_distribution = self.transition_reward(state)

            next_state, reward = next_distribution.sample()
            yield TransitionStep(state, next_state, reward)

            state = next_state
```

So the idea is that if someone wants to model a Markov Reward Process, they'd simply have to create a concrete class that implements the interface of the `@abstractclass MarkovRewardProcess` (specifically implement the `@abstractmethod transition_reward`). But note that the `@abstractmethod transition` of `MarkovProcess` also needs to be implemented to make the whole thing concrete. However, we don't have to implement it in the concrete class implementing the interface of `MarkovRewardProcess` - in fact, we can implement it in the `MarkovRewardProcess` class itself by tapping the method `transition_reward`. Here's the code for the `transition` method in `MarkovRewardProcess`:

```python
from rl.distribution import Distribution, SampledDistribution

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        distribution = self.transition_reward(state)

        def next_state(distribution=distribution):
            next_s, _ = distribution.sample()
            return next_s

        return SampledDistribution(next_state)
```

Note that since the `transition_reward` method is abstract in `MarkovRewardProcess`, the only thing the `transition` method can do is to tap into the `sample` method of the abstract `Distribution` object produced by `transition_reward` and return a `SampledDistribution`. The full code for the `MarkovRewardProcess` class shown above is in the file [rl/markov_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py).


Now let us develop some more theory. Given a specification of $\mathcal{P}_R$, we can extract:
\begin{itemize}
\item The transition probability function $\mathcal{P}: \mathcal{N} \times \mathcal{S} \rightarrow [0,1]$ of the implicit Markov Process defined as:
$$\mathcal{P}(s, s') = \sum_{r\in \mathcal{D}} \mathcal{P}_R(s,r,s')$$
\item The reward transition function:
$$\mathcal{R}_T: \mathcal{N} \times \mathcal{S} \rightarrow \mathbb{R}$$
defined as:
$$\mathcal{R}_T(s,s') = \mathbb{E}[R_{t+1}|S_{t+1}=s',S_t=s] = \sum_{r\in \mathcal{D}} \frac {\mathcal{P}_R(s,r,s')} {\mathcal{P}(s,s')} \cdot r = \sum_{r\in \mathcal{D}} \frac {\mathcal{P}_R(s,r,s')} {\sum_{r\in \mathcal{D}} \mathcal{P}_R(s,r,s')} \cdot r$$
\end{itemize}

The Rewards specification of most Markov Reward Processes we encounter in practice can be directly expressed as the reward transition function $\mathcal{R}_T$ (versus the more general specification of $\mathcal{P}_R$). Lastly, we want to highlight that we can transform either of $\mathcal{P}_R$ or $\mathcal{R}_T$ into a "more compact" reward function that is sufficient to perform key calculations involving Markov Reward Processes. This reward function 
$$\mathcal{R}: \mathcal{N} \rightarrow \mathbb{R}$$
is defined as:
$$\mathcal{R}(s) = \mathbb{E}[R_{t+1}|S_t=s] = \sum_{s' \in \mathcal{S}} \mathcal{P}(s,s') \cdot \mathcal{R}_T(s,s') = \sum_{s'\in \mathcal{S}} \sum_{r\in\mathcal{D}} \mathcal{P}_R(s,r,s') \cdot r$$

We've created a bit of notational clutter here. So it would be a good idea for you to take a few minutes to pause, reflect and internalize the differences between $\mathcal{P}_R$, $\mathcal{P}$ (of the implicit Markov Process), $\mathcal{R}_T$ and $\mathcal{R}$. This notation will analogously re-appear when we learn about Markov Decision Processes in Chapter [-@sec:mdp-chapter]. Moreover, this notation will be used considerably in the rest of the book, so it pays to get comfortable with their semantics.

### Simple Inventory Example as a Markov Reward Process

Now we return to the simple inventory example and embellish it with a reward structure to turn it into a Markov Reward Process (business costs will be modeled as negative rewards). Let us assume that your store business incurs two types of costs:

* Holding cost of $h$ for each bicycle that remains in your store overnight. Think of this as "interest on inventory" - each day your bicycle remains unsold, you lose the opportunity to gain interest on the cash you paid to buy the bicycle. Holding cost also includes the cost of upkeep of inventory.
*  Stockout cost of $p$ for each unit of "missed demand", i.e., for each customer wanting to buy a bicycle that you could not satisfy with available inventory, eg: if 3 customers show up during the day wanting to buy a bicycle each, and you have only 1 bicycle at 8am (store opening time), then you lost two units of demand, incurring a cost of $2p$. Think of the cost of $p$ per unit as the lost revenue plus disappointment for the customer. Typically $p \gg h$.

Let us go through the precise sequence of events, now with incorporation of rewards, in each 24-hour cycle:

* Observe the $(\alpha, \beta)$ *State* at 6pm store-closing (call this state $S_t$)
* Immediately order according to the ordering policy given by: Order quantity $= \max(C - (\alpha + \beta), 0)$
* Record any overnight holding cost incurred as described above
* Receive bicycles at 6am if you had ordered 36 hours ago
* Open the store at 8am
* Experience random demand from customers according to the specified poisson probabilities (poisson mean $=\lambda$)
* Record any stockout cost due to missed demand as described above
* Close the store at 6pm, register the reward $R_{t+1}$ as the negative sum of overnight holding cost and the day's stockout cost, and observe the state (this state is $S_{t+1}$)

Since the customer demand on any day can be an infinite set of possibilities (poisson distribution over the entire range of non-negative integers), we have an infinite set of pairs of next state and reward we could transition to from a given current state. Let's see what the probabilities of each of these transitions looks like. For a given current state $S_t := (\alpha, \beta)$, if customer demand for the day is $i$, then the next state $S_{t+1}$ is:
$$(\max(\alpha + \beta - i, 0), \max(C - (\alpha + \beta), 0))$$
and the reward $R_{t+1}$ is:
$$-h \cdot \alpha - p \cdot \max(i - (\alpha + \beta), 0)$$
Note that the overnight holding cost applies to each unit of on-hand inventory at store closing ($=\alpha$) and the stockout cost applies only to any units of "missed demand" ($=\max(i - (\alpha + \beta), 0)$). Since two different values of demand $i \in \mathbb{Z}_{\geq 0}$ do not collide on any unique pair $(s',r)$ of next state and reward, we can express the transition probability function $\mathcal{P}_R$ for this Simple Inventory Example as a Markov Reward Process as:

$$\mathcal{P}_R((\alpha, \beta), -h \cdot \alpha - p \cdot \max(i - (\alpha + \beta), 0), (\max(\alpha + \beta - i, 0), \max(C - (\alpha + \beta), 0)))$$
$$= \frac {e^{-\lambda} \lambda^i} {i!} \text{ for all } i = 0, 1, 2, \ldots $$

Now let's write some code to implement this simple inventory example as a Markov Reward Process as described above. All we have to do is to create a concrete class implementing the interface of the abstract class `MarkovRewardProcess` (specifically implement the `@abstractmethod transition_reward`). The code below in `transition_reward` method in `class SimpleInventoryMRP` samples the customer demand from a Poisson distribution, uses the above formulas for the pair of next state and reward as a function of the customer demand sample, and returns an instance of `SampledDistribution`. Note that the generic state type `S` is replaced here with the `@dataclass InventoryState` to represent a state of this Markov Reward Process, comprising of the On-Hand and On-Order inventory quantities.

```python
from rl.distribution import SampledDistribution
import numpy as np

@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order

class SimpleInventoryMRP(MarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def transition_reward(
        self,
        state: NonTerminal[InventoryState]
    ) -> SampledDistribution[Tuple[State[InventoryState], float]]:

        def sample_next_state_reward(state=state) ->\
                Tuple[State[InventoryState], float]:
            demand_sample: int = np.random.poisson(self.poisson_lambda)
            ip: int = state.state.inventory_position()
            next_state: InventoryState = InventoryState(
                max(ip - demand_sample, 0),
                max(self.capacity - ip, 0)
            )
            reward: float = - self.holding_cost * state.on_hand\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return NonTerminal(next_state), reward

        return SampledDistribution(sample_next_state_reward)
```

The above code can be found in the file [rl/chapter2/simple_inventory_mrp.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/simple_inventory_mrp.py). We leave it as an exercise for you to use the `simulate_reward` method inherited by `SimpleInventoryMRP` to perform simulations and analyze the statistics produced from the sampling traces.

### Finite Markov Reward Processes

Certain calculations for Markov Reward Processes can be performed easily if:

* The state space is finite ($\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$), and
* The set of unique pairs of next state and reward transitions from each of the states in $\mathcal{N}$ is finite

If we satisfy the above two characteristics, we refer to the Markov Reward Process as a Finite Markov Reward Process. So let us write some code for a Finite Markov Reward Process. We create a concrete class `FiniteMarkovRewardProcess` that primarily inherits from `FiniteMarkovProcess` (a concrete class) and secondarily implements the interface of the abstract class `MarkovRewardProcess`. Our first task is to think about the data structure required to specify an instance of `FiniteMarkovRewardProcess` (i.e., the data structure we'd pass to the `__init__` method of `FiniteMarkovRewardProcess`). Analogous to how we curried $\mathcal{P}$ for a Markov Process as $\mathcal{N} \rightarrow (\mathcal{S} \rightarrow [0,1])$ (where $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ and $\mathcal{N}$ has $m \leq n$ states), here we curry $\mathcal{P}_R$ as:
$$\mathcal{N} \rightarrow (\mathcal{S} \times \mathcal{D} \rightarrow [0, 1])$$
Since $\mathcal{S}$ is finite and since the set of unique pairs of next state and reward transitions is also finite, this leads to the analog of the `Transition` data type for the case of Finite Markov Reward Processes (named `RewardTransition`) as follows:

```python
StateReward = FiniteDistribution[Tuple[State[S], float]]
RewardTransition = Mapping[NonTerminal[S], StateReward[S]]
```

The `FiniteMarkovRewardProcess` class has 3 responsibilities:

* It needs to accept as input to `__init__` a `Mapping` type similar to `RewardTransition` using simply `S` instead of `NonTerminal[S]` or `State[S]` in order to make it convenient for the user to specify a `FiniteRewardProcess` as a succinct dictionary, without being encumbered with wrapping `S` as `NonTerminal[S]` or `Terminal[S]` types. This means the `__init__` method (constructor) needs to appropriately wrap `S` as `NonTerminal[S]` or `Terminal[S]` types to create the attribute `self.transition_reward_map: RewardTransition[S]`. Also, the `__init__` method needs to create a `transition_map: Transition[S]` (extracted from the input to `__init__`) in order to instantiate its concrete parent class `FiniteMarkovProcess`.
* It needs to implement the `transition_reward` method analogous to the implementation of the `transition` method in `FiniteMarkovProcess`
* It needs to compute the reward fuction $\mathcal{R}: \mathcal{N} \rightarrow \mathbb{R}$ from the transition probability function $\mathcal{P}_R$ (i.e. from `self.transition_reward_map: RewardTransition`) based on the expectation calculation we specified above (as mentioned earlier, $\mathcal{R}$ is key to the relevant calculations we shall soon be performing on Finite Markov Reward Processes). To perform further calculations with the reward function $\mathcal{R}$, we need to produce it as a 1-dimensional numpy array (i.e., a vector) attribute of the class (we name it as `reward_function_vec`).

Here's the code that fulfils the above three responsibilities:

```python
import numpy as np
from rl.distribution import FiniteDistribution, Categorical
from collections import defaultdict
from typing import Mapping, Tuple, Dict, Set

class FiniteMarkovRewardProcess(FiniteMarkovProcess[S],
                                MarkovRewardProcess[S]):

    transition_reward_map: RewardTransition[S]
    reward_function_vec: np.ndarray

    def __init__(
        self,
        transition_reward_map: Mapping[S, FiniteDistribution[Tuple[S, float]]]
    ):
        transition_map: Dict[S, FiniteDistribution[S]] = {}

        for state, trans in transition_reward_map.items():
            probabilities: Dict[S, float] = defaultdict(float)
            for (next_state, _), probability in trans:
                probabilities[next_state] += probability

            transition_map[state] = Categorical(probabilities)

        super().__init__(transition_map)

        nt: Set[S] = set(transition_reward_map.keys())
        self.transition_reward_map = {
            NonTerminal(s): Categorical(
                {(NonTerminal(s1) if s1 in nt else Terminal(s1), r): p
                 for (s1, r), p in v.table().items()}
            ) for s, v in transition_reward_map.items()
        }

        self.reward_function_vec = np.array([
            sum(probability * reward for (_, reward), probability in
                self.transition_reward_map[state])
            for state in self.non_terminal_states
        ])

    def transition_reward(self, state: NonTerminal[S]) -> StateReward[S]:
        return self.transition_reward_map[state]
```

The above code for `FiniteMarkovRewardProcess` (and more) is in the file [rl/markov_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py).   

### Simple Inventory Example as a Finite Markov Reward Process

Now we'd like to model the simple inventory example as a Finite Markov Reward Process so we can take advantage of the algorithms that apply to Finite Markov Reward Processes. As we've noted previously, our ordering policy ensures that in steady-state, the sum of On-Hand (denote as $\alpha$) and On-Order (denote as $\beta$) won't exceed the capacity $C$. So we constrain the set of states such that this condition is satisfied: $0 \leq \alpha + \beta \leq C$ (i.e., finite number of states). Although the set of states is finite, there are an infinite number of pairs of next state and reward outcomes possible from any given current state. This is because there are an infinite set of possibilities of customer demand on any given day (resulting in infinite possibilities of stockout cost, i.e., negative reward, on any day). To qualify as a Finite Markov Reward Process, we'll need to model in a manner such that we have a finite set of pairs of next state and reward outcomes from a given current state. So what we'll do is that instead of considering $(S_{t+1}, R_{t+1})$ as the pair of next state and reward, we model the pair of next state and reward to instead be $(S_{t+1}, \mathbb{E}[R_{t+1}|(S_t, S_{t+1})])$ (we know $\mathcal{P}_R$ due to the Poisson probabilities of customer demand, so we can actually calculate this conditional expectation of reward). So given a state $s$, the pairs of next state and reward would be: $(s', \mathcal{R}_T(s, s'))$ for all the $s'$ we transition to from $s$. Since the set of possible next states $s'$ are finite, these newly-modeled rewards associated with the transitions ($\mathcal{R}_T(s,s')$) are also finite and hence, the set of pairs of next state and reward from any current state are also finite. Note that this creative alteration of the reward definition is purely to reduce this Markov Reward Process into a Finite Markov Reward Process. Let's now work out the calculation of the reward transition function $\mathcal{R}_T$.

When the next state's ($S_{t+1}$) On-Hand is greater than zero, it means all of the day's demand was satisfied with inventory that was available at store-opening ($=\alpha + \beta$), and hence, each of these next states $S_{t+1}$ correspond to no stockout cost and only an overnight holding cost of $h \alpha$. Therefore,
$$\mathcal{R}_T((\alpha, \beta), (\alpha + \beta - i, C - (\alpha + \beta))) = - h \alpha \text{ for } 0 \leq i \leq \alpha + \beta - 1$$
When next state's ($S_{t+1}$) On-Hand is equal to zero, there are two possibilities: 

1. The demand for the day was exactly $\alpha + \beta$, meaning all demand was satisifed with available store inventory (so no stockout cost and only overnight holding cost), or
2. The demand for the day was strictly greater than $\alpha + \beta$, meaning there's some stockout cost in addition to overnight holding cost. The exact stockout cost is an expectation calculation involving the number of units of missed demand under the corresponding poisson probabilities of demand exceeding $\alpha + \beta$.

This calculation is shown below:
$$\mathcal{R}_T((\alpha, \beta), (0, C - (\alpha + \beta))) = - h \alpha - p (\sum_{j=\alpha+\beta+1}^{\infty} f(j) \cdot (j - (\alpha + \beta)))$$
 $$= - h \alpha - p (\lambda (1 - F(\alpha + \beta - 1)) -  (\alpha + \beta)(1 - F(\alpha + \beta)))$$ 

So now we have a specification of $\mathcal{R}_T$, but when it comes to our coding interface, we are expected to specify $\mathcal{P}_R$ as that is the interface through which we create a `FiniteMarkovRewardProcess`. Fear not - a specification of $\mathcal{P}_R$ is easy once we have a specification of $\mathcal{R}_T$. We simply create 4-tuples $(s,r,s',p)$ for all $s \in \mathcal{N}, s' \in \mathcal{S}$ such that $r=\mathcal{R}_T(s, s')$ and $p=\mathcal{P}(s,s')$ (we know $\mathcal{P}$ along with $\mathcal{R}_T$), and the set of all these 4-tuples (for all $s \in \mathcal{N}, s' \in \mathcal{S}$) constitute the specification of $\mathcal{P}_R$, i.e., $\mathcal{P}_R(s,r,s') = p$. This turns our reward-definition-altered mathematical model of a Finite Markov Reward Process into a programming model of the `FiniteMarkovRewardProcess` class. This reward-definition-altered model enables us to gain from the fact that we can leverage the algorithms we'll be  writing for Finite Markov Reward Processes (including some simple and elegant linear-algebra-solver-based solutions). The downside of this reward-definition-altered model is that it prevents us from generating samples of the specific rewards encountered when transitioning from one state to another (because we no longer capture the probabilities of individual reward outcomes). Note that we can indeed generate sampling traces, but each transition step in the sampling trace will only show us the "mean reward" (specifically, the expected reward conditioned on current state and next state).

In fact, most Markov Processes you'd encounter in practice can be modeled as a combination of $\mathcal{R}_T$ and $\mathcal{P}$, and you'd simply follow the above $\mathcal{R}_T$ to $\mathcal{P}_R$ representation transformation drill to present this information in the form of $\mathcal{P}_R$ to instantiate a `FiniteMarkovRewardProcess`. We designed the interface to accept $\mathcal{P}_R$ as input since that is the most general interface for specifying Markov Reward Processes.

So now let's write some code for the simple inventory example as a Finite Markov Reward Process as described above. All we have to do is to create a derived class inherited from `FiniteMarkovRewardProcess` and write a method to construct the `transition_reward_map` input to the constructor (`__init__`) of `FiniteMarkovRewardProcess` (i.e., $\mathcal{P}_R$). Note that the generic state type `S` is replaced here with the `@dataclass InventoryState` to represent the inventory state, comprising of the On-Hand and On-Order inventory quantities.

```python
from scipy.stats import poisson

@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order

class SimpleInventoryMRPFinite(FiniteMarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> \
            Mapping[
                InventoryState,
                FiniteDistribution[Tuple[InventoryState, float]]
            ]:
        d: Dict[InventoryState, Categorical[Tuple[InventoryState, float]]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                base_reward = - self.holding_cost * state.on_hand
                sr_probs_map: Dict[Tuple[InventoryState, float], float] =\
                    {(InventoryState(ip - i, beta1), base_reward):
                     self.poisson_distr.pmf(i) for i in range(ip)}
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = base_reward - self.stockout_cost *\
                    (probability * (self.poisson_lambda - ip) +
                     ip * self.poisson_distr.pmf(ip))
                sr_probs_map[(InventoryState(0, beta1), reward)] = probability
                d[state] = Categorical(sr_probs_map)
        return d
```

The above code is in the file [rl/chapter2/simple_inventory_mrp.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/simple_inventory_mrp.py)). We encourage you to play with the inputs to `SimpleInventoryMRPFinite` in `__main__` and view the transition probabilities and rewards of the constructed Finite Markov Reward Process.


### Value Function of a Markov Reward Process

Now we are ready to formally define the main problem involving Markov Reward Processes. As we've said earlier, we'd like to compute the "expected accumulated rewards" from any non-terminal state. However, if we simply add up the rewards in a sampling trace following time step $t$ as $\sum_{i=t+1}^{\infty} R_i = R_{t+1} + R_{t+2} + \ldots$, the sum would often diverge to infinity. So we allow for rewards accumulation to be done with a discount factor $\gamma \in [0, 1]$: We define the (random) *Return* $G_t$ as the "discounted accumulation of future rewards" following time step $t$. Formally,
$$G_t = \sum_{i=t+1}^{\infty} \gamma^{i-t-1} \cdot R_i = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots$$

We use the above definition of *Return* even for a terminating sequence (say terminating at $t=T$, i.e., $S_T \in \mathcal{T}$), by treating $R_i = 0$ for all $i > T$.

Note that $\gamma$ can range from a value of 0 on one extreme (called "myopic") to a value of 1 on another extreme (called "far-sighted"). "Myopic" means the Return is the same as Reward (no accumulation of future Rewards in the Return). With "far-sighted" ($\gamma = 1$), the Return calculation can diverge for continuing (non-terminating) Markov Reward Processes but "far-sighted" is indeed applicable for episodic Markov Reward Processes (where all random sequences of the process terminate). Apart from the Return divergence consideration, $\gamma < 1$ helps algorithms become more tractable (as we shall see later when we get to Reinforcement Learning). We should also point out that the reason to have $\gamma < 1$ is not just for mathematical convenience or computational tractability - there are valid modeling reasons to discount Rewards when accumulating to a Return. When Reward is modeled as a financial quantity (revenues, costs, profits etc.), as will be the case in most financial applications, it makes sense to incorporate [time-value-of-money](https://en.wikipedia.org/wiki/Time_value_of_money) which is a fundamental concept in Economics/Finance that says there is greater benefit in receiving a dollar now versus later (which is the economic reason why interest is paid or earned). So it is common to set $\gamma$ to be the discounting based on the prevailing interest rate ($\gamma = \frac 1 {1+r}$ where $r$ is the interest rate over a single time step). Another technical reason for setting $\gamma < 1$ is that our models often don't fully capture future uncertainty and so, discounting with $\gamma$ acts to undermine future rewards that might not be accurate (due to future uncertainty modeling limitations). Lastly, from an AI perspective, if we want to build machines that acts like humans, psychologists have indeed demonstrated that human/animal behavior prefers immediate reward over future reward.

Note that we are (as usual) assuming the fact that the Markov Reward Process is time-homogeneous (time-invariant probabilities of state transitions and rewards).

As you might imagine now, we'd want to identify non-terminal states with large expected returns and those with small expected returns. This, in fact, is the main problem involving a Markov Reward Process - to compute the "Expected Return" associated with each non-terminal state in the Markov Reward Process. Formally, we are interested in computing the *Value Function*
$$V: \mathcal{N} \rightarrow \mathbb{R}$$
defined as:
$$V(s) = \mathbb{E}[G_t|S_t=s] \text{ for all } s \in \mathcal{N}, \text{ for all } t = 0, 1, 2, \ldots$$

For the rest of the book, we will assume that whenever we are talking about a Value Function, the discount factor $\gamma$ is appropriate to ensure that the Expected Return from each state is finite.

Now we show a creative piece of mathematics due to [Richard Bellman](https://en.wikipedia.org/wiki/Richard_E._Bellman). [Bellman noted](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) [@Bellman1957] that the Value Function has a recursive structure. Specifically,

\begin{equation}
\begin{split}
V(s) & = \mathbb{E}[R_{t+1}|S_t=s] + \gamma \cdot \mathbb{E}[R_{t+2}|S_t=s] + \gamma^2 \cdot \mathbb{E}[R_{t+3}|S_t=s] + \ldots \\
& = \mathcal{R}(s) + \gamma \cdot \sum_{s'\in \mathcal{N}} \mathbb{P}[S_{t+1}=s'|S_t=s] \cdot \mathbb{E}[R_{t+2}|S_{t+1}=s'] \\
& \hspace{4mm} + \gamma^2 \cdot \sum_{s' \in \mathcal{N}} \mathbb{P}[S_{t+1}=s'|S_t=s] \sum_{s'' \in \mathcal{N}} \mathbb{P}[S_{t+2}=s''|S_{t+1}=s'] \cdot \mathbb{E}[R_{t+3}|S_{t+2}=s''] \\
& \hspace{4mm} + \ldots \\
& = \mathcal{R}(s) + \gamma \cdot \sum_{s'\in \mathcal{N}} \mathcal{P}(s, s') \cdot \mathcal{R}(s') + \gamma^2 \cdot \sum_{s' \in \mathcal{N}} \mathcal{P}(s, s') \sum_{s'' \in \mathcal{N}} \mathcal{P}(s', s'') \cdot \mathcal{R}(s'') + \ldots \\
& = \mathcal{R}(s) + \gamma \cdot \sum_{s' \in \mathcal{N}} \mathcal{P}(s,s')\cdot ( \mathcal{R}(s') + \gamma \cdot \sum_{s'' \in \mathcal{N}} \mathcal{P}(s', s'') \cdot \mathcal{R}(s'') + \ldots ) \\
& = \mathcal{R}(s) + \gamma \cdot \sum_{s' \in \mathcal{N}} \mathcal{P}(s, s') \cdot V(s') \text{ for all } s \in \mathcal{N}
\end{split}
\label{eq:mrp_bellman_eqn}
\end{equation} 

Note that although the transitions to random states $s',s'', \ldots$ are in the state space of $\mathcal{S}$ rather than $\mathcal{N}$, the right-hand-side above sums over states $s', s'', \ldots$ only in $\mathcal{N}$ because transitions to terminal states (in $\mathcal{T} = \mathcal{S} - \mathcal{N}$) don't contribute any reward beyond the rewards produced *before reaching* the terminal state.

We refer to this recursive equation \eqref{eq:mrp_bellman_eqn} for the Value Function as the Bellman Equation for Markov Reward Processes. Figure \ref{fig:mrp_bellman_tree} is a convenient visualization aid of this important equation. In the rest of the book, we will depict quite a few of these type of state-transition visualizations to aid with creating mental models of key concepts.

<div style="text-align:center" markdown="1">
![Visualization of MRP Bellman Equation \label{fig:mrp_bellman_tree}](./chapter2/mrp_bellman_tree.png "Visualization of MRP Bellman Equation")
</div>

For the case of Finite Markov Reward Processes, assume $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ and assume $\mathcal{N}$ has $m \leq n$ states. Below we use bold-face notation to represent functions as column vectors and matrices since we have finite states/transitions. So, $\bv$ is a column vector of length $m$, $\bm{\mathcal{P}}$ is an $m \times m$ matrix, and $\bm{\mathcal{R}}$ is a column vector of length $m$ (rows/columns corresponding to states in $\mathcal{N}$), so we can express the above equation in vector and matrix notation as follows:

$$\bv = \bm{\mathcal{R}} + \gamma \bm{\mathcal{P}} \cdot \bv$$
Therefore,
\begin{equation}
\Rightarrow \bv = (\bm{I_m} - \gamma \bm{\mathcal{P}})^{-1} \cdot \bm{\mathcal{R}}
\label{eq:mrp_bellman_linalg_solve}
\end{equation}
where $\bm{I_m}$ is the $m \times m$ identity matrix.

Let us write some code to implement the calculation of Equation \eqref{eq:mrp_bellman_linalg_solve}. In the `FiniteMarkovRewardProcess` class, we implement the method `get_value_function_vec` that performs the above calculation for the Value Function $V$ in terms of the reward function $\mathcal{R}$ and the transition probability function $\mathcal{P}$ of the implicit Markov Process. The Value Function $V$ is produced as a 1D numpy array (i.e. a vector). Here's the code:

```python
    def get_value_function_vec(self, gamma: float) -> np.ndarray:
        return np.linalg.solve(
            np.eye(len(self.non_terminal_states)) -
            gamma * self.get_transition_matrix(),
            self.reward_function_vec
        )
```

Invoking this `get_value_function_vec` method on `SimpleInventoryMRPFinite` for the simple case of capacity $C=2$, poisson mean $\lambda = 1.0$, holding cost $h=1.0$, stockout cost $p=10.0$, and discount factor $\gamma=0.9$ yields the following result:

```
{NonTerminal(state=InventoryState(on_hand=0, on_order=0)): -35.511,
 NonTerminal(state=InventoryState(on_hand=0, on_order=1)): -27.932,
 NonTerminal(state=InventoryState(on_hand=0, on_order=2)): -28.345,
 NonTerminal(state=InventoryState(on_hand=1, on_order=0)): -28.932,
 NonTerminal(state=InventoryState(on_hand=1, on_order=1)): -29.345,
 NonTerminal(state=InventoryState(on_hand=2, on_order=0)): -30.345}
 ```

The corresponding values of the attribute `reward_function_vec` (i.e., $\mathcal{R}$) are:

```
{NonTerminal(state=InventoryState(on_hand=1, on_order=0)): -3.325,
 NonTerminal(state=InventoryState(on_hand=0, on_order=0)): -10.0,
 NonTerminal(state=InventoryState(on_hand=0, on_order=1)): -2.325,
 NonTerminal(state=InventoryState(on_hand=0, on_order=2)): -0.274,
 NonTerminal(state=InventoryState(on_hand=1, on_order=1)): -1.274,
 NonTerminal(state=InventoryState(on_hand=2, on_order=0)): -2.274}  
```

This tells us that On-Hand of 0 and On-Order of 2 has the highest expected reward. However, the Value Function is highest for On-Hand of 0 and On-Order of 1.

This computation for the Value Function works if the state space is not too large (the size of the square linear system of equations is equal to number of non-terminal states). When the state space is large, this direct method of solving a linear system of equations won't scale and we have to resort to numerical methods to solve the recursive Bellman Equation. This is the topic of Dynamic Programming and Reinforcement Learning algorithms that we shall learn in this book. 

### Summary of Key Learnings from this Chapter

Before we end this chapter, we'd like to highlight the two highly important concepts we learnt in this chapter:

* Markov Property: A concept that enables us to reason effectively and compute efficiently in practical systems involving sequential uncertainty
* Bellman Equation: A mathematical insight that enables us to express the Value Function recursively - this equation (and its Optimality version covered in Chapter [-@sec:mdp-chapter]) is in fact the core idea within all Dynamic Programming and Reinforcement Learning algorithms.
