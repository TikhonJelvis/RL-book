
# Chapter 1: Markov Processes

This book is about "Sequential Decisioning under Sequential Uncertainty". In this chapter, we will ignore the "sequential decisioning" aspect and focus just on the "sequential uncertainty" aspect.

## The Concept of *State* in a Process

For a gentle introduction to the concept of *State*, let us consider a process that generates a sequence of random outcomes at discrete time steps that we'll index by a time variable $t = 0, 1, 2, \ldots$. To understand and reason about the random evolution of such a process, we are typically interested in the internal representation of the process at each point in time $t$. We refer to this internal representation of the process at time $t$ as the (random) *state* of the process at time $t$ and denote it as $S_t$. Specifically, we are interested in the probability of the next state $S_{t+1}$, given the present state $S_t$ and the past states $S_0, S_1, \ldots, S_{t-1}$, i.e., $\mathbb{P}[S_{t+1}|S_t, S_{t-1}, \ldots, S_0]$. The internal representation (*state*) could be any data type - it could be something as simple as a single stock price at the end of a day, or it could be something quite elaborate like the number of shares of all publicly traded stocks held by all banks in the U.S. at the end of a week. 

## Understanding Markov Property from Stock Price Examples

We will be learning about Markov Processes in this chapter and these processes have what are called *Markov States*. So we will now learn about the *Markov Property* of *States*. Let us develop intuition for this property with some examples of random evolution of stock prices over time. 

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
 
 Let us now code this up. First, we will create a dataclass to represent the dynamics of this process. As you can see in the code below, the dataclass `Process1` contains two data members `level_param: int` and `alpha1: float = 0.25` to represent $L$ and $\alpha_3$ respectively. It contains the method `up_prob` to calculate $\mathbb{P}[X_{t+1} = X_t + 1]$ and the method `next_state`, which samples from a Bernoulli distribution (whose probability is obtained from the method `up_prob`) and creates the next state $S_{t+1}$ from the current state $S_t$. Also, note the nested dataclass `State` meant to represent the state of Process 1 (it's only data member `price: int` reflects the fact that the state consists of only the current price, which is an integer).
 
 
```python
import numpy as np

@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: State) -> float:
        return 1. / (1 + np.exp(-alpha * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)
```


Next, we write a simple simulator using Python's generator functionality (using `yield') as follows:

```python
def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)
```

Now we can use this simulator function to generate simulation traces. In the following code, we generate `num_traces` number of simulation traces over `time_steps` number of time steps starting from a price $X_0$ of `start_price`. The use of Python's generator feature lets us do this "lazily" (on-demand) using the ``itertools.islice`` function.

```python
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

The entire code is [here][stock_price_mp.py]. We encourage you to play with this code with different ``start_price, level_param, alpha1, time_steps, num_traces``. You can plot graphs of simulation traces of the stock price, or plot graphs of the terminal distributions of the stock price at various time points (both of these plotting functions are made available for you in this code).

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

Note that if we had modeled the state $S_t$ as the entire stock price history $(X_0, X_1, \ldots, X_t)$, then the Markov Property would be satisfied trivially. The Markov Property would also be satisfied if we had modeled the state $S_t$ as the pair $(X_t, X_{t-1})$ for $t > 0$ and $S_0$ as $(X_0, Null)$. However, our choice of $S_t := (X_t, X_t - X_{t-1})$ is the "simplest/minimal" internal representation. In fact, in this entire book, our endeavor in modeling states for various processes will be to ensure the Markov Property with the "simplest/minimal" representation for the state.
 
 The corresponding dataclass for Process 2 is shown below:
 
 ```python
@dataclass
class Process2:
    @dataclass
    class State:
        price: int
        is_prev_move_up: Optional[bool]

    alpha2: float = 0.75  # strength of reverse-pull (value in  [0,1])

    def up_prob(self, state: State) -> float:
        return 0.5 * (1 + self.alpha2 * handy_map[state.is_prev_move_up])

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process2.State(
            price=state.price + up_move * 2 - 1,
            is_prev_move_up=bool(up_move)
        )
```

The code for generation of simulation traces of the stock price is almost identical to the code we wrote for Process 1.

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

**Process 3**: This is an extension of Process 2 where the probability of next movement depends not only on the last movement, but on all past movements. Specifically, it depends on the ratio of all past up-moves (call it $U_t = \sum_{i=1}^t \max(X_i - X_{i-1}, 0)$) and all past down-moves (call it $D_t = \sum_{i=1}^t \max(X_{i-1} - X_i, 0)$) in the following manner:
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

The corresponding dataclass for Process 2 is shown below:

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
The code for generation of simulation traces of the stock price is shown below:

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

As suggested for Process 1, you can plot graphs of simulation traces of the stock price, or plot graphs of the terminal distributions of the stock price at various time points for Processes 2 and 3, by playing with the [code][stock_price_mp.py].

[stock_price_mp.py]: https://github.com/TikhonJelvis/RL-book/blob/master/src/chapter2/stock_price_mp.py

 Figure \ref{fig:single_trace_mp} shows a single simulation trace of stock prices for each of the 3 processes. Figure \ref{fig:terminal_distribution_mp} shows the distribution of the stock price at time $T=100$ over 1000 traces.

<div style="text-align:center" markdown="1">
![Single Simulation Trace \label{fig:single_trace_mp}](./chapter2/single_traces.png "Single Simulation Trace")
</div>

<div style="text-align:center" markdown="1">
![Terminal Distribution \label{fig:terminal_distribution_mp}](./chapter2/terminal_distribution.png "Terminal Distribution")
</div>

Having developed the intuition for the Markov Property of States, we are now ready to formalize the notion of Markov Processes (some of the literature refers to Markov Processes as Markov Chains, but we will stick with the term Markov Processes).

## Formal Definitions

Here we will consider Discrete-Time Markov Processes, where time moves forward in discrete time steps $t=0, 1, 2, \ldots$. This book will also consider a few cases of continuous-time Markov Processes, where time is a continuous variable (this leads to stochastic calculus, which is the foundation of some of the ground-breaking work in Mathematical Finance). However, for now, we will formally define Discrete-Time Markov Processes as they are fairly common and also much easier to develop intuition for.

### Discrete-Time Markov Processes

\begin{definition}
A {\em Discrete-Time Markov Process} consists of:
\begin{itemize}
\item A countable set of states $\mathcal{S}$
 \item A time-indexed sequence of random variables $S_t$ for each time $t=0, 1, 2, \ldots$, with each $S_t$ taking values in the set $\mathcal{S}$
 \item $\mathbb{P}[S_{t+1}|S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[S_{t+1}|S_t]$ for all $t \geq 0$
 \end{itemize}
 \end{definition}
 
 We refer to $\mathbb{P}[S_{t+1}|S_t]$ as the transition probabilities for time $t$.

### Discrete-Time Stationary Markov Processes

\begin{definition}
A {\em Discrete-Time Stationary Markov Process} is a Discrete-Time Markov Process with the additional property that
$\mathbb{P}[S_{t+1}|S_t]$ is independent of $t$.
 \end{definition}
 
 This means, the dynamics of a Discrete-Time Stationary Markov Process can be fully specified with the function $$\mathcal{P}: \mathcal{S} \times \mathcal{S} \rightarrow [0,1]$$ such that $\mathcal{P}(s, s') = \mathbb{P}[S_{t+1}=s'|S_t=s]$ for all $s, s' \in \mathcal{S}$. Hence, $\sum_{s'\in \mathcal{S}} \mathcal{P}(s,s') = 1$ for all $s \in \mathcal{S}$. We refer to the function $\mathcal{P}$ as the transition probabilities function of the Stationary Markov Process, with the first argument to $\mathcal{P}$ to be thought of as the "source" state and the second argument as the "destination" state.
 
Note that this specification is devoid of the time index $t$ (hence, the term *Stationary* which means "time-invariant"). Moreover, note that a non-Stationary Markov Process can be converted to a Stationary Markov Process by augmenting all states with the time index $t$. This means if the original state space of a non-Stationary Markov Process was $\mathcal{S}$, then the state space of the corresponding Stationary Markov Process is $\mathbb{Z}_{\geq 0} \times \mathcal{S}$ (where $\mathbb{Z}_{\geq 0}$ denotes the domain of the time index). This is because each time step has it's own unique set of (augmented) states, which means the entire set of states in $\mathbb{Z}_{\geq 0} \times \mathcal{S}$ can be covered by time-invariant transition probabilities, thus qualifying as a Stationary Markov Process. Therefore, henceforth, any time we say *Markov Process*, assume we are refering to a Discrete-Time Stationary Markov Process (unless explicitly specified otherwise), which in turn will be characterized by the transition probabilities function $\mathcal{P}$. Note that the stock price examples (all 3 of the Processes we covered) are examples of a (Discrete-Time Stationary) Markov Process, even without requiring augmenting the state with the time index.

### Starting States

Now it's natural to ask the question how do we "start" the Markov Process (in the stock price examples, this was the notion of the start state). More generally, we'd like to specify a probability distribution of start states so we can perform simulations and (let's say) compute the probability distribution of states at specific future time steps. While this is a relevant question, we'd like to separate the following two specifications:

* Specification of the transition probability function $\mathcal{P}$
* Specification of the probability distribution of start states (denote this as $\mu: \mathcal{S} \rightarrow [0,1]$)

We say that a Markov Process is fully specified by $\mathcal{P}$ in the sense that this gives us the transition probabilities that govern the complete dynamics of the Markov Process. A way to understand this is to relate specification of $\mathcal{P}$ to the specification of rules in a game (such as chess or monopoly). These games are specified with a finite (in fact, fairly compact) set of rules that is easy for a newbie to the game to understand. However, when we want to *actually play* the game, we need to specify the starting position (one could start these games at arbitrary, but legal, starting positions and not just at some canonical starting position). The specification of the start state of the game is analogous to the specification of $\mu$. Given $\mu$ together with $\mathcal{P}$ enables us to generate simulate traces of the Markov Process (analogously, *play* games like chess or monopoly). These simulation traces typically result in a wide range of outcomes due to sampling and long-running of the Markov Process (versus compact specification of transition probabilities). These simulation traces enable us to answer questions such as probability distribution of states at specific future time steps or expected time of first occurrence of a specific state etc., given a certain starting probability distribution $\mu$.
 
 Thinking about the separation between specifying the rules of the game versus actually playing the game helps us understand the need to separate the notion of dynamics specification $\mathcal{P}$ (fundamental to the stationary character of the Markov Chain) and the notion of starting distribution $\mu$ (required to perform simulation traces). Hence, the separation of concerns between $\mathcal{P}$ and $\mu$ is key to the conceptualization of Markov Chains. Likewise, we separate concerns in our code design as well, as evidenced by how we separated the ``next state`` method in the Process dataclasses and the ``simulation`` function.

### Absorbing States

Thinking about games might make you wonder how we'd represent the fact that games have *ending rules* (rules for winning or losing the game). This brings up the notion of "terminal states". "Terminal states" might occur at any of a variety of time steps (like in the games examples), or like we will see in many financial application examples, termination might occur after a fixed number of time steps. So do we need to specify that certain states are "terminal states"? Yes, we do, but we won't explicitly mark them as "terminal states". Instead, we will build this "termination" feature in $\mathcal{P}$ as follows (note that the technical term for "terminal states" is *Absorbing States* due to the following construction of $\mathcal{P}$).

\begin{definition}[Absorbing States]
A state $s\in \mathcal{S}$ is an *Absorbing State* if $\mathcal{P}(s,s) = 1$
\end{definition}

So instead of thinking of the Markov Process as "terminating", we can simply imagine that the Markov Process keeps cycling with 100% probability at this "terminal state". This notion of being trapped in the state (not being able to escape to another state) is the reason we call it an Absorbing State. 

When we consider some of the financial applications later in this book, we will find that the Markov Process "terminates" after a fixed number of time steps, say $T$. In these applications, the time index $t$ is part of the state and each state with the time index $t=T$ will be constructed to be an absorbing state. All other states with time index $t<T$ will transition to states with time index $t+1$. In fact, you could take each of the 3 Processes seen earlier for stock price movement and add a feature that the forward movement in time terminates at some fixed time step $T$. Then, we'd have to include $t$ in the state representation simply to specify that states with time index $T$ will transition to themselves with 100% probability (note that in these examples the time index $t$ doesn't influence the transition probabilities for states with $t<T$, so these processes are stationary until $t=T-1$.)

### Finite State Space
Now let us consider the case of the state space being finite, i.e., $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$. Finite state space enables us to represent $\mathcal{P}$ in a finite data structure in our code, as a dictionary or as a matrix or as a directed graph. This is rather convenient for visualization and also for performing certain types of calculations involving Markov Processes. The directed graph view is quite common in helping visualize Markov Processes. Also, the $n \times n$ matrix representation (representing transition probabilities as the elements in the matrix) is very useful in answering common questions about the dynamics of a Markov Process (we shall soon see examples of this). 

To help conceptualize this, let us consider a simple inventory example involving just 5 states. Assume you are the store manager for a furniture store and tasked with controlling the ordering of inventory from a supplier. Let us focus on the inventory of a particular type of dining table. Assume that each day there is random demand (denoted as $D$) for the dining table with the probabilities of demand as follows:
$$\mathbb{P}[D=0] = 0.2, \mathbb{P}[D=0] = 0.6, \mathbb{P}[D=2] = 0.2$$
Assume that you can order only a single dining table on any day because the delivery truck cannot carry more than one dining table. Each evening at 6pm when your store closes, you decide to either order a single dining table or not order. If you order, the dining table delivery will arrive 36 hours later (at 6am the day after the day after you order - we refer to this as *delivery lead time* of 36 hours). Assume your ordering policy is as follows: you don't order if you have at least one dining table in your store and you order a single dining table if you have no dining tables left in your store. This inventory system can be modeled as a Markov Chain where tha *State* is given by a pair of integers $(OH, OO)$ where $OH$ refers to the on-hand units of the dining table on a given evening at 6pm ($OH=0$ or $1$ or $2$) and $OO$ refers to the on-order units of the dining table ($OO=0$ or $1$). If $OO=1$, this means a single unit of the dining table was ordered 24 hours ago and that now it's on a truck due to arrive 12 hours later (at 6am the next morning). The sequence of events is:

* Observe the $(OH, OO)$ *State* at 6pm
* Order according to the policy described above
* Receive a dining table at 6am if you had ordered a dining table 36 hours ago
* Open your store at 8am
* Experience random demand from customers according to demand probabilities listed above (sales units for the day is the minimum of demand on the day and inventory at store opening on the day)
* Close your store at 6pm

We leave it as an exercise for you to work out the state transition probabilities, given the above-specified demand probabilities and according to the above-specified ordering policy. You should obtain a Markov Process as depicted in Figure \ref{fig:dining_table_mp}. The nodes are the states, labeled with their corresponding $OH$ and $OO$ values. The directed edges are the probabilistic state transitions from 6pm on a day to 6pm on the next day, with the transition probabilities labeled on the directed edges.

<div style="text-align:center" markdown="1">
![Simple Inventory Markov Process \label{fig:dining_table_mp}](./chapter2/simple_inv_mp.png "Simple Inventory Markov Process")
</div>

### Stationary Distribution
* Conditions under which a Stationary Distribution exists

