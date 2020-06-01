
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
Here we will consider discrete-time Markov Processes, where time moves forward in discrete time steps $t=0, 1, 2, \ldots$. This book will also consider a few cases of continuous-time Markov Processes, where time is  continuous variable (this leads to stochastic calculus, which is the foundation of some of the ground-breaking work in Mathematical Finance). However, for now, we define discrete-time Markov Processes as they are fairly common and also much easier to develop intuition for.

### Discrete-Time Markov Processes
\begin{definition}[Discrete-Time Markov Process]
A Discrete-Time Markov Process consists of:
\begin{itemize}
\item A countable set of states $\mathcal{S}$
 \item A time-indexed sequence of random variables $S_t$ for each time $t=0, 1, 2, \ldots$, with each $S_t$ taking values in the set $\mathcal{S}$
 \item $\mathbb{P}[S_{t+1}|S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[S_{t+1}|S_t]$ for all $t \geq 0$
 \end{itemize}
 \end{definition}

### Discrete-Time Stationary Markov Processes
- Stationary MP is a MP with the additional property that $\mathbb{P}[S_{t+1}=s'|S_t=s]$ is independent of $t$
- Then we can write $\mathcal{P}(s,s') : \mathcal{S} \times \mathcal{S} \rightarrow [0,1]$ with the property that $\sum_{s'\in \mathcal{S}} \mathcal{P}(s,s') = 1$ for all $s \in \mathcal{S}$ 
- Including time in State auotmatically makes it a stationary markov process
- So by default when we say Markov Process, we mean a Discrete-Time Stationary Markov Process and work with the $\mathcal{P}(s,s')$ function

### Initial Distribution

### Absorbing States
- $s$ is an absorbing state if $\mathcal{P}(s,s) = 1$

### Finite states
- Data structure representation
- Tabular algorithms

### Stationary Distribution
- Conditions under which a Stationary Distribution exists

## An Inventory Example of a Markov Process
