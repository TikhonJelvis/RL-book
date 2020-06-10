
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

 Let us now code this up. First, we will create a dataclass to represent the dynamics of this process. As you can see in the code below, the dataclass `Process1` contains two data members `level_param: int` and `alpha1: float = 0.25` to represent $L$ and $\alpha_1$ respectively. It contains the method `up_prob` to calculate $\mathbb{P}[X_{t+1} = X_t + 1]$ and the method `next_state`, which samples from a Bernoulli distribution (whose probability is obtained from the method `up_prob`) and creates the next state $S_{t+1}$ from the current state $S_t$. Also, note the nested dataclass `State` meant to represent the state of Process 1 (it's only data member `price: int` reflects the fact that the state consists of only the current price, which is an integer).


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
        return 1. / (1 + np.exp(-self.alpha1 * (self.level_param - state.price)))

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

## Formal Definitions for Markov Processes

Here we will consider Discrete-Time Markov Processes, where time moves forward in discrete time steps $t=0, 1, 2, \ldots$. This book will also consider a few cases of continuous-time Markov Processes, where time is a continuous variable (this leads to stochastic calculus, which is the foundation of some of the ground-breaking work in Mathematical Finance). However, for now, we will formally define Discrete-Time Markov Processes as they are fairly common and also much easier to develop intuition for.

### Discrete-Time Markov Processes

\begin{definition}
A {\em Discrete-Time Markov Process} consists of:
\begin{itemize}
\item A countable set of states $\mathcal{S}$
 \item A time-indexed sequence of random variables $S_t$ for each time $t=0, 1, 2, \ldots$, with each $S_t$ taking values in the set $\mathcal{S}$
 \item Markov Property: $\mathbb{P}[S_{t+1}|S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[S_{t+1}|S_t]$ for all $t \geq 0$
 \end{itemize}
 \end{definition}

 We refer to $\mathbb{P}[S_{t+1}|S_t]$ as the transition probabilities for time $t$.

### Discrete-Time Stationary Markov Processes

\begin{definition}
A {\em Discrete-Time Stationary Markov Process} is a Discrete-Time Markov Process with the additional property that
$\mathbb{P}[S_{t+1}|S_t]$ is independent of $t$.
 \end{definition}

 This means, the dynamics of a Discrete-Time Stationary Markov Process can be fully specified with the function $$\mathcal{P}: \mathcal{S} \times \mathcal{S} \rightarrow [0,1]$$ such that $\mathcal{P}(s, s') = \mathbb{P}[S_{t+1}=s'|S_t=s]$ for all $s, s' \in \mathcal{S}$. Hence, $\sum_{s'\in \mathcal{S}} \mathcal{P}(s,s') = 1$ for all $s \in \mathcal{S}$. We refer to the function $\mathcal{P}$ as the transition probability function of the Stationary Markov Process, with the first argument to $\mathcal{P}$ to be thought of as the "source" state and the second argument as the "destination" state.

Note that this specification is devoid of the time index $t$ (hence, the term *Stationary* which means "time-invariant"). Moreover, note that a non-Stationary Markov Process can be converted to a Stationary Markov Process by augmenting all states with the time index $t$. This means if the original state space of a non-Stationary Markov Process was $\mathcal{S}$, then the state space of the corresponding Stationary Markov Process is $\mathbb{Z}_{\geq 0} \times \mathcal{S}$ (where $\mathbb{Z}_{\geq 0}$ denotes the domain of the time index). This is because each time step has it's own unique set of (augmented) states, which means the entire set of states in $\mathbb{Z}_{\geq 0} \times \mathcal{S}$ can be covered by time-invariant transition probabilities, thus qualifying as a Stationary Markov Process. Therefore, henceforth, any time we say *Markov Process*, assume we are refering to a Discrete-Time Stationary Markov Process (unless explicitly specified otherwise), which in turn will be characterized by the transition probability function $\mathcal{P}$. Note that the stock price examples (all 3 of the Processes we covered) are examples of a (Discrete-Time Stationary) Markov Process, even without requiring augmenting the state with the time index.

### Starting States

Now it's natural to ask the question how do we "start" the Markov Process (in the stock price examples, this was the notion of the start state). More generally, we'd like to specify a probability distribution of start states so we can perform simulations and (let's say) compute the probability distribution of states at specific future time steps. While this is a relevant question, we'd like to separate the following two specifications:

* Specification of the transition probability function $\mathcal{P}$
* Specification of the probability distribution of start states (denote this as $\mu: \mathcal{S} \rightarrow [0,1]$)

We say that a Markov Process is fully specified by $\mathcal{P}$ in the sense that this gives us the transition probabilities that govern the complete dynamics of the Markov Process. A way to understand this is to relate specification of $\mathcal{P}$ to the specification of rules in a game (such as chess or monopoly). These games are specified with a finite (in fact, fairly compact) set of rules that is easy for a newbie to the game to understand. However, when we want to *actually play* the game, we need to specify the starting position (one could start these games at arbitrary, but legal, starting positions and not just at some canonical starting position). The specification of the start state of the game is analogous to the specification of $\mu$. Given $\mu$ together with $\mathcal{P}$ enables us to generate simulate traces of the Markov Process (analogously, *play* games like chess or monopoly). These simulation traces typically result in a wide range of outcomes due to sampling and long-running of the Markov Process (versus compact specification of transition probabilities). These simulation traces enable us to answer questions such as probability distribution of states at specific future time steps or expected time of first occurrence of a specific state etc., given a certain starting probability distribution $\mu$.

 Thinking about the separation between specifying the rules of the game versus actually playing the game helps us understand the need to separate the notion of dynamics specification $\mathcal{P}$ (fundamental to the stationary character of the Markov Process) and the notion of starting distribution $\mu$ (required to perform simulation traces). Hence, the separation of concerns between $\mathcal{P}$ and $\mu$ is key to the conceptualization of Markov Processes. Likewise, we separate concerns in our code design as well, as evidenced by how we separated the ``next_state`` method in the Process dataclasses and the ``simulation`` function.

### Absorbing States

Thinking about games might make you wonder how we'd represent the fact that games have *ending rules* (rules for winning or losing the game). This brings up the notion of "terminal states". "Terminal states" might occur at any of a variety of time steps (like in the games examples), or like we will see in many financial application examples, termination might occur after a fixed number of time steps. So do we need to specify that certain states are "terminal states"? Yes, we do, but we won't explicitly mark them as "terminal states". Instead, we will build this "termination" feature in $\mathcal{P}$ as follows (note that the technical term for "terminal states" is *Absorbing States* due to the following construction of $\mathcal{P}$).

\begin{definition}[Absorbing States]
A state $s\in \mathcal{S}$ is an {\em Absorbing State} if $\mathcal{P}(s,s) = 1$
\end{definition}

So instead of thinking of the Markov Process as "terminating", we can simply imagine that the Markov Process keeps cycling with 100% probability at this "terminal state". This notion of being trapped in the state (not being able to escape to another state) is the reason we call it an Absorbing State. 

When we consider some of the financial applications later in this book, we will find that the Markov Process "terminates" after a fixed number of time steps, say $T$. In these applications, the time index $t$ is part of the state and each state with the time index $t=T$ will be constructed to be an absorbing state. All other states with time index $t<T$ will transition to states with time index $t+1$. In fact, you could take each of the 3 Processes seen earlier for stock price movement and add a feature that the forward movement in time terminates at some fixed time step $T$. Then, we'd have to include $t$ in the state representation simply to specify that states with time index $T$ will transition to themselves with 100% probability (note that in these examples the time index $t$ doesn't influence the transition probabilities for states with $t<T$, so these processes are stationary until $t=T-1$.)

With this formalism in place, we are now ready to write some code to represent general Markov Processes. We do this with an abstract class `MarkovProcess` parameterized by a generic type (`TypeVar('S')`) representing a generic state space `Generic[S]`. The abstract class has an `abstractmethod` called `transition` that is meant to specify the transition probability distribution of next states, given a current state. The class also has a method `simulate` that enables us to generate a sequence of sampled states starting from a specified `start_state`. The sampling of next states relies on the implementation of the `sample()` method in the `Distribution[S]` object produced by the `transition` method (note that the [`Distribution` class hierarachy](https://github.com/TikhonJelvis/RL-book/blob/master/rl/distribution.py) was covered in the previous chapter). This is the full body of the abstract class `MarkovProcess`:

```python
S = TypeVar('S')
class MarkovProcess(ABC, Generic[S]):

    @abstractmethod
    def transition(self, state: S) -> Distribution[S]:

    def simulate(self, start_state: S) -> Iterable[S]:
        state: S = start_state
        while True:
            yield state
            state = self.transition(state).sample()
```

So if you have a mathematical specification of the transition probabilities of a Markov Process, all you need to do is to create a derived class (inherited from abstract class `MarkovProcess`) and implement the `transition` method in the derived class in a manner that captures your mathematical specification of the transition probabilities. Let us make this concrete for the case of Process 3 (the 3rd example of stock price transitions we covered in the previous section). We will name this derived class (`dataclass`) as `StockPriceMP3`. Note that the generic state space `S` is now replaced with a concrete state space represented by the `dataclass` `StateMP3`. The code should be self-explanatory since we implemented this process as a standalone in the previous section. Note the use of the `Categorical` distribution in the `transition` method to capture the 2-outcomes distribution of next states (for movements up or down).

```python
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

    def transition(self, state: StateMP3) -> Categorical[StateMP3]:
        up_p = self.up_prob(state)

        return Categorical([
            (StateMP3(state.num_up_moves + 1, state.num_down_moves), up_p),
            (StateMP3(state.num_up_moves, state.num_down_moves + 1), 1 - up_p)
        ])
```

To generate simulation traces, we write the following function:

```python
def process3_price_traces(
    start_price: int,
    alpha3: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP3(alpha3=alpha3)
    start_state = StateMP3(num_up_moves=0, num_down_moves=0)
    return np.vstack([
        np.fromiter((start_price + s.num_up_moves - s.num_down_moves
                    for s in itertools.islice(mp.simulate(start_state),
                                              time_steps + 1)), float)
        for _ in range(num_traces)])
```

We leave it to you as an exercise to similarly implement Stock Price Processes 1 and 2 that we had covered in the previous section. The complete code along with the driver to set input parameters, run all 3 processes and create plots is [here](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/stock_price_mp.py). We encourage you to change the input parameters in `__main__` and get an intuitive feel for how the simulation results vary with the changes in parameters.

## Finite Markov Processes

Now let us consider Markov Processes with a finite state space. So we can represent the state space as $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$. Let us refer to Markov Processes with finite state spaces as Finite Markov Processes. Since Finite Markov Processes are a subclass of Markov Processes, it would make sense to create a derived class `FiniteMarkovProcess` that inherits from the abstract class `MarkovProcess`. We'd need to implement the abstract method `transition`. But first let's think about the data structure required to specify an instance of a `FiniteMarkovProcess` (i.e., the data structure we'd pass to the `__init__` method of `FiniteMarkovProcess`). One choice is a $n \times n$ 2D numpy array representation, i.e., matrix elements representing transition probabilities
$$\mathcal{P} : \{s_1, s_2, \ldots, s_n\} \times \{s_1, s_2, \ldots, s_n\} \rightarrow [0, 1]$$
However, we often find that this matrix can be sparse since one often transitions from a given state to just a few set of states. So we'd like a sparse representation and we can accomplish this by conceptualizing $\mathcal{P}$ in an [equivalent curried form](https://en.wikipedia.org/wiki/Currying) as follows:
$$\{s_1, s_2, \ldots, s_n\} \rightarrow (\{s_1, s_2, \ldots, s_n\} \rightarrow [0, 1])$$
With this curried view, we can represent both the outer $\rightarrow$ and the inner $\rightarrow$ as a map (in Python, as a dictionary of type `Mapping`). Let us create an alias for this (called `S_TransType`) since we will use this data structure often:
```python
S_TransType = Mapping[S, Mapping[S, float]]
```
The outer map will have $n$ keys consisting of each of $\{s_1, s_2, \ldots, s_n\}$. The inner map's keys will be only the states transitioned to (from the outer map's state key) with non-zero probability. To make things concrete, here's a toy `S_TransType` example of a city with highly unpredictable weather outcomes from one day to the next:

```python
{
  "Rain": {"Rain": 0.3, "Nice": 0.7},
  "Snow": {"Rain": 0.4, "Snow": 0.6},
  "Nice": {"Rain": 0.2, "Snow": 0.3, "Nice": 0.5}
}
```
It is common to view this as a directed graph, as depicted in Figure \ref{fig:weather_mp}. The nodes are the states and the directed edges are the probabilistic state transitions, with the transition probabilities labeled on them.

<div style="text-align:center" markdown="1">
![Weather Markov Process \label{fig:weather_mp}](./chapter2/weather_mp.png "Weather Markov Process")
</div>

Now we are ready to write the code for the `FiniteMarkovProcess` class. We implement the `transition` method by utilizing the `Categorical` distribution type we learnt about in the previous chapter and by extracting probabilities from the `transition_map` attribute of the class. Note that along with the `transition` method, we have also implemented the `__repr__` method for a well-formatted display of `transition_map: S_TransType`.

```python
class FiniteMarkovProcess(MarkovProcess[S]):
    state_space: Sequence[S]
    transition_map: S_TransType

    def __init__(
        self,
        transition_map: S_TransType
    ):
        self.state_space = list(transition_map.keys())
        self.transition_map = transition_map

    def transition(self, state: S) -> FiniteDistribution[S]:
        return Categorical(self.transition_map[state].items())

    def __repr__(self) -> str:
        display = ""
        for s, d in self.transition_map.items():
            display += "From State %s:\n" % str(s)
            for s1, p in d.items():
                display += "  To State %s with Probability %.3f\n" %\
                    (str(s1), p)
        return display
```

## Simple Inventory Example
To help conceptualize Finite Markov Processes, let us consider a simple example of changes in inventory at a store. Assume you are the store manager and that you are tasked with controlling the ordering of inventory from a supplier. Let us focus on the inventory of a particular type of bicycle. Assume that each day there is random (non-negative integer) demand for the bicycle with the probabilities of demand following a Poisson distribution (with Poisson parameter $\lambda \in \mathbb{R}_{\geq 0}$), i.e. demand $i$ for each $i = 0, 1, 2, \ldots$ occurs with probability
$$f(i) = \frac {e^{-\lambda} \lambda^i} {i!}$$
Denote $F: \mathbb{Z}_{\geq 0} \rightarrow [0, 1]$ as the poisson cumulative distribution function, i.e.,
 $$F(i) = \sum_{j=0}^i f(j)$$


Assume you have storage capacity for at most $C \in \mathbb{Z}_{\geq 0}$ bicycles in your store. Each evening at 6pm when your store closes, you have the choice to order a certain number of bicycles from your supplier (including the option to not order any bicycles, on a given day). The ordered bicycles will arrive 36 hours later (at 6am the day after the day after you order - we refer to this as *delivery lead time* of 36 hours). Denote the *State* at 6pm store-closing each day as $(\alpha, \beta)$, where $\alpha$ is the inventory in the store (refered to as On-Hand Inventory at 6pm) and $\beta$ is the inventory on a truck from the supplier (that you had ordered the previous day) that will arrive in your store the next morning at 6am ($\beta$ is refered to as On-Order Inventory at 6pm). Due to your storage capacity constraint of at most $C$ bicycles, your ordering policy is to order $C-(\alpha + \beta)$ if $\alpha + \beta < C$ and to not order if $\alpha + \beta \geq C$. The precise sequence of events in a 24-hour cycle is:

* Observe the $(\alpha, \beta)$ *State* at 6pm store-closing (call this state $S_t$)
* Immediately order according to the ordering policy described above
* Receive bicycles at 6am if you had ordered 36 hours ago
* Open the store at 8am
* Experience random demand from customers according to demand probabilities stated above (number of bicycles sold for the day will be the minimum of demand on the day and inventory at store opening on the day)
* Close the store at 6pm and observe the state (this state is $S_{t+1}$)

If current state $S_t$ is $(\alpha, \beta)$, there are only $\alpha + \beta + 1$ possible next states $S_{t+1}$ as follows:
$$(\alpha + \beta - i, \max(C - (\alpha + \beta), 0)) \text{ for } i =0, 1, \ldots, \alpha + \beta$$
with transition probabilities governed by the Poisson probabilities of demand as follows:
$$\mathcal{P}((\alpha, \beta), (\alpha + \beta - i, \max(C - (\alpha + \beta), 0))) = f(i)\text{ for } 0 \leq i \leq \alpha + \beta - 1$$
$$\mathcal{P}((\alpha, \beta), (0, \max(C - (\alpha + \beta), 0))) = \sum_{j=\alpha+\beta}^{\infty} f(j) = 1 - F(\alpha + \beta - 1)$$
Note that the next state's ($S_{t+1}$) On-Hand can be zero resulting from any of infinite possible demand outcomes greater than or equal to $\alpha + \beta$.

So we are now ready to write code for this simple inventory example as a Markov Process. All we have to do is to create a derived class inherited from `FiniteMarkovProcess` and write a method to construct the `transition_map: S_TransType`. Note that the generic state `S` is replaced here with the type `Tuple[int, int]` to represent the pair of On-Hand and On-Order.

```python
IntPair = Tuple[int, int]
MPTransType = Mapping[IntPair, Mapping[IntPair, float]]

class SimpleInventoryMP(FiniteMarkovProcess[IntPair]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> MPTransType:
        d = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                ip = alpha + beta
                d1 = {}
                beta1 = max(self.capacity - ip, 0)
                for i in range(ip):
                    next_state = (ip - i, beta1)
                    probability = self.poisson_distr.pmf(i)
                    d1[next_state] = probability
                next_state = (0, beta1)
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                d1[next_state] = probability
                d[(alpha, beta)] = d1
        return d
```

Let us utilize the `__repr__` method written previously to view the transition probabilities for the simple case of $C=2$ and $\lambda = 1.0$ (this code is in the file [rl/chapter2/simple_inventory_mp.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/simple_inventory_mp.py))

```python
user_capacity = 2
user_poisson_lambda = 1.0

si_mp = SimpleInventoryMP(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda
)

print(si_mp)
```

The output we get is nicely displayed as:

```
From State (0, 0):
  To State (0, 2) with Probability 1.000
From State (0, 1):
  To State (1, 1) with Probability 0.368
  To State (0, 1) with Probability 0.632
From State (0, 2):
  To State (2, 0) with Probability 0.368
  To State (1, 0) with Probability 0.368
  To State (0, 0) with Probability 0.264
From State (1, 0):
  To State (1, 1) with Probability 0.368
  To State (0, 1) with Probability 0.632
From State (1, 1):
  To State (2, 0) with Probability 0.368
  To State (1, 0) with Probability 0.368
  To State (0, 0) with Probability 0.264
From State (2, 0):
  To State (2, 0) with Probability 0.368
  To State (1, 0) with Probability 0.368
  To State (0, 0) with Probability 0.264
```

For a graphical view of this Markov Process, see Figure \ref{fig:inventory_mp}. The nodes are the states, labeled with their corresponding $\alpha$ and $\beta$ values. The directed edges are the probabilistic state transitions from 6pm on a day to 6pm on the next day, with the transition probabilities labeled on them.

<div style="text-align:center" markdown="1">
![Simple Inventory Markov Process \label{fig:inventory_mp}](./chapter2/simple_inv_mp.png "Simple Inventory Markov Process")
</div>

We can perform a number of interesting experiments and calculations with this simple Markov Process and we encourage you to play with this code by changing values of the capacity $C$ and poisson mean $\lambda$, performing simulations and probabilistic calculations of natural curiosity for a store owner.

There is a rich and interesting theory for Markov Processes. However, we will not get into this theory as our coverage of Markov Processes so far is a sufficient building block to take us to the incremental topics of Markov Reward Processes and Markov Decision Processes. However, before we move on, we'd like to show just a glimpse of the rich theory with the calculation of *Stationary Probabilities* and apply it to the case of the above simple inventory Markov Process.

## Stationary Distribution of a Markov Process
\begin{definition} 
 The {\em Stationary Distribution} of a (Stationary) Markov Process with state space $\mathcal{S}$ and transition probability function $\mathcal{P}: \mathcal{S} \times \mathcal{S} \rightarrow [0, 1]$ is a probability distribution function $\pi: \mathcal{S} \rightarrow [0, 1]$ such that:
  $$\pi(s) = \sum_{s'\in \mathcal{S}} \pi(s) \cdot \mathcal{P}(s', s) \text{ for all } s \in \mathcal{S}$$
\end{definition}

The intuitive view of the stationary distribution $\pi$ is that (under specific conditions we are not listing here) if we let the Markov Process run forever, then in the long run the states occur at specific time steps with relative frequencies (probabilities) given by a distribution $\pi$ that is independent of the time step. The probability of occurrence of a specific state $s$ at a time step (asymptotically far out in the future) should be equal to the sum-product of probabilities of occurrence of all the states at the previous time step and the transition probabilities from those states to $s$. But since the states' occurrence probabilities are invariant in time, the $\pi$ distribution for the previous time step is the same as the $\pi$ distribution for the time step we considered. This argument holds for all states $s$, and that is exactly the statement of the definition of *Stationary Distribution* formalized above.

If we specialize this definition of *Stationary Distribution* to Finite-States Stationary Markov Processes with state space $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$, then we can express the Stationary Distribution $\mu$ as follows:
$$\pi(s_j) = \sum_{i=1}^n \pi(s_i) \cdot \mathcal{P}(s_i, s_j) \text{ for all } j = 1, 2, \ldots n$$
Abusing notation, let us refer to $\pi$ as a column vector of length $n$ and let us refer to $\mathcal{P}$ as the $n \times n$ transition probability matrix (rows are source states, columns are destination states with each row summing to 1).
Then, the statement of the above definition can be succinctly expressed as:
$$\pi^T = \pi^T \cdot \mathcal{P}$$
which can be re-written as:
$$\mathcal{P}^T \cdot \pi = \pi$$
But this is simply saying that $\pi$ is an eigenvector of $\mathcal{P}^T$ with eigenvalue of 1. So then, it should be easy to obtain the stationary distribution $\pi$ from an eigenvectors and eigenvalues calculation of $\mathcal{P}^T$. 

Let us write code to compute the stationary distribution. We shall add two methods in the `FiniteMarkovProcess` class, one for setting up the transition probability matrix $\mathcal{P}$ (`get_transition_matrix` method) and another to calculate the stationary distribution $\pi$ (`get_stationary_distribution`) from the transition probability matrix. Here's the code for the two methods (the full code for `FiniteMarkovProcess` is in the file [`rl/markov_process.py`](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py)):

```python
    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.state_space)
        mat = np.zeros((sz, sz))
        for i, s1 in enumerate(self.state_space):
            for j, s2 in enumerate(self.state_space):
                mat[i, j] = self.transition_map[s1].get(s2, 0.)
        return mat

    def get_stationary_distribution(self) -> FiniteDistribution[S]:
        eig_vals, eig_vecs = np.linalg.eig(self.get_transition_matrix().T)
        index_of_first_unit_eig_val = np.where(
            np.abs(eig_vals - 1) < 1e-8)[0][0]
        eig_vec_of_unit_eig_val = np.real(
            eig_vecs[:, index_of_first_unit_eig_val])
        return Categorical([
            (self.state_space[i], ev)
            for i, ev in enumerate(eig_vec_of_unit_eig_val /
                                   sum(eig_vec_of_unit_eig_val))
```

We will skip the theory that tells us about the conditions under which a stationary distribution is well-defined, or the conditions under which there is a unique stationary distribution. Instead, we will just go ahead with this calculation here assuming this Markov Process satisfies those conditions (it does!). So, we simply seek the index of the `eig_vals` vector with eigenvalue equal to 1 (accounting for floating-point error). Next, we pull out the column of the `eig_vecs` matrix at the `eig_vals` index calculated above, and convert it into a real-valued vector (eigenvectors/eigenvalues calculations are, in general, complex numbers calculations - see the reference for the `np.linalg.eig` function). So this gives us the real-valued eigenvector with eigenvalue equal to 1.  Finally, we have to normalize the eigenvector so it's values add up to 1 (since we want probabilities), and return the probabilities as a `Categorical` distribution).

Running this code for the simple case of capacity $C=2$ and poisson mean $\lambda = 1.0$ produces the following output for the stationary distribution $\mu$:

```
{(0, 0): 0.117,
 (0, 1): 0.279,
 (0, 2): 0.117,
 (1, 0): 0.162,
 (1, 1): 0.162,
 (2, 0): 0.162}
}
```

This tells us that On-Hand of 0 and On-Order of 1 is the state occurring most frequently (28% of the time) when the system is played out indefinitely.   

Let us summarize the 3 different representations we've covered:

* Functional Representation: as given by the `transition` method, i.e., given a state, the `transition` method returns a probability distribution of next states. This representation is valuable when performing simulations by sampling the next state from the returned probability distribution of the next state. This is applicable to the general case of Markov Processes (including infinite state spaces).
* Sparse Data Structure Representation: as given by `transition map: S_TransType`, which is convenient for compact storage and useful for visualization (eg: `__repr__` method display or as a directed graph figure). This is applicable only to Finite Markov Processes.
* Dense Data Structure Representation: as given by the `get_transition_matrix` 2D numpy array, which is useful for performing linear algebra that is often required to calculate mathematical properties of the process (eg: to calculate the stationary distribution). This is applicable only to Finite Markov Processes.

Now we are ready to move to our next topic of *Markov Reward Processes*. We'd like to finish this section by stating that the Markov Property owes its name to a mathematician from a century ago - [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov). Although the Markov Property seems like a simple enough concept, the concept has had profound implications on our ability to compute or reason with systems involving time-sequenced uncertainty in practice.

## Markov Reward Processes

As we've said earlier, the reason we covered Markov Processes is because we want to make our way to Markov Decision Processes (the framework for Reinforcement Learning algorithms) by adding incremental features to Markov Processes. This section covers an intermediate framework between Markov Processes and Markov Decision Processes, and is known as Markov Reward Processes. We essentially just include the notion of a numerical *reward* to a Markov Process each time we transition from one state to the next. These rewards will be random, and all we need is to specify the probability distributions of these rewards as we make state transitions. 

The main problem to solve regarding Markov Reward Processes is to calculate how much reward we would accumulate (in expectation, starting from each of the states) if we let the Process run indefinitely, bearing in mind that future rewards need to be discounted appropriately (otherwise the sum of rewards can blow up to $\infty$). In order to solve the problem of calculating expected accumulative rewards from each state, we will first set up some formalism for general Markov Reward Processes, develop some (elegant) theory on calculating rewards accumulation, write plenty of code (based on the theory), and apply the theory and code to the simple inventory example (which we will embellish with rewards equal to negative of the costs incurred at the store).

### Formalism of Markov Reward Processes

\begin{definition}
A {\em Discrete-Time Markov Reward Process} is a Discrete-Time Markov Process, along with:
\begin{itemize}
\item A time-indexed sequence of {\em Reward} random variables $R_t \in \mathbb{R}$ for each time $t=1, 2, \ldots$
\item Markov Property (including Rewards): $\mathbb{P}[(R_{t+1}, S_{t+1}) | S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[(R_{t+1}, S_{t+1}) | S_t]$ for all $t \geq 0$
\item Specification of a discount factor $\gamma \in [0,1]$
 \end{itemize}
\end{definition}

The role of $\gamma$ only comes in discounting future rewards when accumulating rewards from a given state (as mentioned earlier) - more on this later.

Since we commonly assume Stationarity of Discrete-Time Markov Processes, we shall also (by default) assume Stationarity for Discrete-Time Markov Reward Processes, i.e., $\mathbb{P}[(R_{t+1}, S_{t+1}) | S_t]$ is independent of $t$.

This means the transition probabilities of a Markov Reward Process can, in the most general case, be expressed as a transition probability function:
$$\mathcal{P}_R: \mathcal{S} \times \mathbb{R} \times \mathcal{S} \rightarrow [0,1]$$
defined as:
$$\mathcal{P}_R(s,r,s') = \mathbb{P}[(R_{t+1}=r, S_{t+1}=s') | S_t=s]$$
such that
$$\sum_{s'\in \mathcal{S}} \sum_{r \in \mathbb{R}} \mathcal{P}_R(s,r,s') = 1 \text{ for all } s \in \mathcal{S}$$

Let us now proceed to write some code that captures this formalism. We shall create a derived *abstract* class `MarkovRewardProcess` that inherits from the abstract class `MarkovProcess`. Analogous to `MarkovProcess`'s `@abstractmethod transition` (that represents $\mathcal{P}$), `MarkovRewardProcess` has an `@abstractmethod transition_reward` that represents $\mathcal{P}_R$. Also, analogous to `MarkovProcess`'s method `simulate`, `MarkovRewardProcess` has the mathod `simulate_reward`. These analogous methods extend the interface to return a pair: next state $S_{t+1}$ and reward $R_{t+1}$, given current state $S_t$ (versus the `MarkovProcess` methods whose interfaces return simply the next state $S_{t+1}$). Let's clarify this with actual code:

```python
class MarkovRewardProcess(MarkovProcess[S]):

    @abstractmethod
    def transition_reward(self, state: S) -> Distribution[Tuple[S, float]]:

    def simulate_reward(self, start_state: S) -> Iterable[Tuple[S, float]]:
        state: S = start_state
        reward: float = 0.

        while True:
            yield state, reward
            state, reward = self.transition_reward(state).sample()
```

So the idea is that if someone wants to model a Markov Reward Process, they'd simply have to create a derived class inheriting from `MarkovRewardProcess` and implement the `transition_reward` method in the derived class. For this derived class to be a concrete class, we'd need the `transition` method to be implemented. However, we don't have to implement it in the derived class - in fact, we can implement it in the `MarkovRewardProcess` class by tapping the method `transition_reward`. Here's the code for the `transition` method in `MarkovRewardProcess`:

```python
    def transition(self, state: S) -> Distribution[S]:

        def next_state(state=state):
            next_s, _ = self.transition_reward(state).sample()
            return next_s

        return SampledDistribution(next_state)
```

Note that since the `transition_reward` method is abstract in `MarkovRewardProcess`, the only thing the `transition` method can do is tap the `sample` method of the abstract `Distribution` object produced by `transition_reward`.

Now let us develop some more theory. Given a specification of $\mathcal{P}_R$, we can extract:
\begin{itemize}
\item The transition probability function $\mathcal{P}: \mathcal{S} \times \mathcal{S} \rightarrow [0,1]$ of the implicit Markov Process defined as:
$$\mathcal{P}(s, s') = \sum_{r\in \mathbb{R}} \mathcal{P}_R(s,r,s')$$
\item The reward transition function:
$$\mathcal{R}_T: \mathcal{S} \times \mathcal{S} \rightarrow \mathbb{R}$$
defined as:
$$\mathcal{R}_T(s,s') = \mathbb{E}[R_{t+1}|S_{t+1}=s',S_t=s] = \sum_{r\in \mathcal{R}} \frac {\mathcal{P}_R(s,r,s')} {\mathcal{P}(s,s')} \cdot r = \sum_{r\in \mathcal{R}} \frac {\mathcal{P}_R(s,r,s')} {\sum_{r\in \mathbb{R}} \mathcal{P}_R(s,r,s')} \cdot r$$
\end{itemize}

The Rewards specification of most Markov Reward Processes we encounter in practice can be directly expressed as the reward transition function $\mathcal{R}_T$ (versus the more general specification of $\mathcal{P}_R$). Lastly, we want to highlight that we can transform either of $\mathcal{P}_R$ or $\mathcal{R}_T$ into a "more compact" reward function that is sufficient to perform key calculations involving Markov Reward Processes. This reward function 
$$\mathcal{R}: \mathcal{S} \rightarrow \mathbb{R}$$
is defined as:
$$\mathcal{R}(s) = \mathbb{E}[R_{t+1}|S_t=s] = \sum_{s' \in \mathcal{S}} \mathcal{P}(s,s') \cdot \mathcal{R}_T(s,s') = \sum_{s'\in \mathcal{S}} \sum_{r\in\mathbb{R}} \mathcal{P}_R(s,r,s') \cdot r$$

### Finite Markov Reward Processes

The above calculations can be performed easily for the case of finite states (known as Finite Markov Reward Processes). So let us write some code for the case of $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$. We create a derived classs `FiniteMarkovRewardProcess` that primarily inherits from `FiniteMarkovProcess` (concrete class)and secondarily inherits from `MarkovRewardProcess` (abstract class). Our first task is to think about the data structure required to specify an instance of `FiniteMarkovRewardProcess` (i.e., the data structure we'd pass to the `__init__` method of `FiniteMarkovRewardProcess`). Analogous to how we curried $\mathcal{P}$ as $\mathcal{S} \rightarrow (\mathcal{S} \rightarrow [0,1])$ (where $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$), we curry $\mathcal{P}_R$ as:
$$\mathcal{S} \rightarrow (\mathcal{S} \times \mathbb{R} \rightarrow [0, 1])$$
This leads to the analog of the `S_TransType` data type for the case of Finite Markov Reward Processes as follows:
```python
SR_TransType = Mapping[S, Mapping[Tuple[S, float], float]]
```

With this as input to ``__init__`` (input named `transition_reward_map: SR_TransType`), the `FiniteMarkovRewardProcess` class has three responsibilities:

* It needs to implement the `transition_reward` method analogous to the implementation of the `transition` method in `FiniteMarkovProcess`
* It needs to create a `transition_map: S_TransType` (extracted from `transition_reward_map: SR_TransType`) in order to instantiate its concrete parent `FiniteMarkovProcess`.
* It needs to compute the reward fuction $\mathcal{R}: \mathcal{S} \rightarrow \mathbb{R}$ from the transition probability function $\mathcal{P}_R$ (i.e. from `transition_reward_map: SR_TransType`) based on the expectation calculation we specified above (as mentioned earlier, $\mathcal{R}$ is key to the relevant calculations we shall soon be performing on Finite Markov Reward Processes). To perform further calculations with the reward function $\mathcal{R}$, we need to produce it as a 1D numpy array (i.e., a vector) attribute of the class (we name it as `reward_function_vec`).

Here's the code that fulfils the above three responsibilities:

```python
class FiniteMarkovRewardProcess(
        FiniteMarkovProcess[S],
        MarkovRewardProcess[S]
):
    transition_reward_map: SR_TransType
    reward_function_vec: np.ndarray

    def __init__(self, transition_reward_map: SR_TransType):

        transition_map: Dict[S, Dict[S, float]] = {}

        for state, trans in transition_reward_map.items():
            transition_map[state] = defaultdict(float)
            for (next_state, _), probability in trans.items():
                transition_map[state][next_state] += probability

        super().__init__(transition_map)

        self.transition_reward_map = transition_reward_map

        self.reward_function_vec = np.array(
            [sum(probability * reward for (_, reward), probability in
                 transition_reward_map[state].items()) for state in
             self.state_space]

    def transition_reward(self, state: S) ->\
            FiniteDistribution[Tuple[S, float]]:
        return Categorical(self.transition_reward_map[state].items())
```

The above code for `FiniteMarkovRewardProcess` (and more) is in the file [rl/markov_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py).   

### Simple Inventory Example as a Finite Markov Reward Process
Now we return to the simple inventory example and embellish it with a reward structure to turn it into a Markov Reward Process (business costs will be modeled as negative rewards). Let us assume that your store business incurs two types of costs:

* Holding cost of $h$ for each bicycle that remains in your store overnight. Think of this as "interest on inventory" - each day your bicycle remains unsold, you lose the opportunity to gain interest on the cash you paid to buy the bicycle. Holding cost also includes the cost of upkeep of inventory.
*  Stockout cost of $p$ for each unit of "missed demand", i.e., for each customer wanting to buy a bicycle that you could not satisfy with available inventory, eg: if 3 customers show up during the day wanting to buy a bicycle each, and you have only 1 bicycle at 8am (store opening time), then you lost two units of demand, incurring a cost of $2p$. Think of the cost of $p$ per unit as the lost revenue plus disappointment for the customer. Typically $p \gg h$.

Let us go through the precise sequence of events, now with incorporation of rewards in each 24-hour cycle:

* Observe the $(\alpha, \beta)$ *State* at 6pm store-closing (call this state $S_t$)
* Immediately order according to the ordering policy we've described
* Record any overnight holding cost incurred as described above
* Receive bicycles at 6am if you had ordered 36 hours ago
* Open the store at 8am
* Experience random demand from customers according to the specified poisson probabilities
* Record any stockout cost due to missed demand as described above
* Close the store at 6pm, register the reward $R_{t+1}$ as the negative sum of overnight holding cost and the day's stockout cost, and observe the state (this state is $S_{t+1}$)

As mentioned previously, for most Markov Reward Processes we will encounter in practice, we can model $R_{t+1}$ in terms of $S_t$ and $S_{t+1}$. So it's convenient for us to express Markov Reward Processes by specifying $\mathcal{R}_T$, i.e. $\mathbb{E}[R_{t+1}|S_{t+1}, S_t]$. So now let us work out $\mathcal{R}_T$ for this simple inventory example based on the state transitions and rewards structure we have described.

When the next state's ($S_{t+1}$) On-Hand is greater than zero, it means all of the day's demand was satisfied with inventory that was available at store-opening ($=\alpha + \beta$), and hence, each of these next states $S_{t+1}$ correspond to no stockout cost and only an overnight holding cost of $h \alpha$. Therefore,
$$\mathcal{R}_T((\alpha, \beta), (\alpha + \beta - i, \max(C - (\alpha + \beta), 0))) = - h \alpha \text{ for } 0 \leq i \leq \alpha + \beta - 1$$
When next state's ($S_{t+1}$) On-Hand is equal to zero, there are two possibilities: 

1. The demand for the day was exactly $\alpha + \beta$, meaning all demand was satisifed with available store inventory (so no stockout cost and only overnight holding cost), or
2. The demand for the day was strictly greater than $\alpha + \beta$, meaning there's some stockout cost in addition to overnight holding cost. The exact stockout cost is an expectation calculation involving the number of units of missed demand under the corresponding poisson probabilities of demand exceeding $\alpha + \beta$.

This calculation is shown below:
$$\mathcal{R}_T((\alpha, \beta), (0, \max(C - (\alpha + \beta), 0))) = - h \alpha - p (\sum_{j=\alpha+\beta+1}^{\infty} f(j) \cdot (j - (\alpha + \beta)))$$
 $$= - h \alpha - p (\lambda (1 - F(\alpha + \beta - 1)) -  (\alpha + \beta)(1 - F(\alpha + \beta)))$$ 

So now we have a specification of $\mathcal{R}_T$ but really we were expected to specify $\mathcal{P}_R$ as that is the interface through which we create a `FiniteMarkovRewardProcess`. Fear not - a specification of $\mathcal{P}_R$ is easy once we have a specification of $\mathcal{R}_T$. We simply create 4-tuples $(s,r,s',p)$ for all $s,s' \in \mathcal{S}$ such that $r=\mathcal{R}_T(s, s')$ and $p=\mathcal{P}(s,s')$ (we know $\mathcal{P}$ along with $\mathcal{R}_T$), and the set of all these 4-tuples (for all $s,s' \in \mathcal{S}$) constitute the specification of $\mathcal{P}_R$, i.e., $\mathcal{P}_R(s,r,s') = p$. In fact, most Markov Processes you'd encounter in practice can be modeled as a combination of $\mathcal{R}_T$ and $\mathcal{P}$, and you'd simply follow the above routine to present this information in the form of $\mathcal{P}_R$ to instantiate a `FiniteMarkovRewardProcess`. We designed the interface to take in $\mathcal{P}_R$ as that is the most general interface for specifying Markov Reward Processes.

So now let's write some code for the simple inventory example as a Finite Markov Reward Process. All we have to do is to create a derived class inherited from `FiniteMarkovRewardProcess` and write a method to construct the `transition_reward_map: SR_TransType` (i.e., $\mathcal{P}_R$). Note that the generic state `S` is replaced here with the type `Tuple[int, int]` to represent the pair of On-Hand and On-Order.

```python
IntPair = Tuple[int, int]
MRPTransType = Mapping[IntPair, Mapping[Tuple[IntPair, float], float]]

class SimpleInventoryMRP(FiniteMarkovRewardProcess[IntPair]):

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

    def get_transition_reward_map(self) -> MRPTransType:
        d = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                ip = alpha + beta
                d1 = {}
                beta1 = max(self.capacity - ip, 0)
                for i in range(ip):
                    next_state = (ip - i, beta1)
                    reward = self.holding_cost * alpha
                    probability = self.poisson_distr.pmf(i)
                    d1[(next_state, reward)] = probability
                next_state = (0, beta1)
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = self.holding_cost * alpha + self.stockout_cost *\
                    (probability * (self.poisson_lambda - ip) +
                     ip * self.poisson_distr.pmf(ip))
                d1[(next_state, reward)] = probability
                d[(alpha, beta)] = d1
        return d
```

Let us view the transition probabilities of next states and rewards for the simple case of $C=2$ and $\lambda = 1.0$ (this code is in the file [rl/chapter2/simple_inventory_mrp.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter2/simple_inventory_mrp.py))

```python
user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = -1.0
user_stockout_cost = -10.0

si_mrp = SimpleInventoryMRP(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda,
    holding_cost=user_holding_cost,
    stockout_cost=user_stockout_cost
)
print(si_mrp)
```

The output we get (utilizing an analogously written `__repr__` method in `FiniteMarkovRewardProcess`) is nicely displayed as:

```
From State (0, 0):
  To [State (0, 2) and Reward -10.000] with Probability 1.000
From State (0, 1):
  To [State (1, 1) and Reward -0.000] with Probability 0.368
  To [State (0, 1) and Reward -3.679] with Probability 0.632
From State (0, 2):
  To [State (2, 0) and Reward -0.000] with Probability 0.368
  To [State (1, 0) and Reward -0.000] with Probability 0.368
  To [State (0, 0) and Reward -1.036] with Probability 0.264
From State (1, 0):
  To [State (1, 1) and Reward -1.000] with Probability 0.368
  To [State (0, 1) and Reward -4.679] with Probability 0.632
From State (1, 1):
  To [State (2, 0) and Reward -1.000] with Probability 0.368
  To [State (1, 0) and Reward -1.000] with Probability 0.368
  To [State (0, 0) and Reward -2.036] with Probability 0.264
From State (2, 0):
  To [State (2, 0) and Reward -2.000] with Probability 0.368
  To [State (1, 0) and Reward -2.000] with Probability 0.368
  To [State (0, 0) and Reward -3.036] with Probability 0.264
```


### Value Function of a Markov Reward Process

Now we are ready to formally define the main problem involving Markov Reward Processes. As we said earlier, we'd like to compute the "expected accumulated rewards" from any given state. However, if we simply add up the rewards in a simulation trace following time step $t$ as $\sum_{i=t+1}^{\infty} R_i = R_{t+1} + R_{t+2} + \ldots$, the sum would often diverge to infinity. This is where the discount factor $\gamma$ comes into play. We define the (random) *Return* $G_t$ as the "discounted accumulation of future rewards" following time step $t$. Formally,
$$G_t = \sum_{i=t+1}^{\infty} \gamma^{i-t-1} \cdot R_i = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots$$

Note that $\gamma$ can range from a value of 0 on one extreme (called "myopic") to a value of 1 on another extreme (called "far-sighted"). "Myopic" means the Return is the same as Reward (no accumulation of future Rewards in the Return). Note that "far-sighted" is indeed applicable if all random sequences of the Process end in an absorbing state AND the rewards associated with the infinite looping at the absorbing states are 0 (otherwise, the Return could diverge to infinity). 

Apart from the Return divergence consideration, $\gamma < 1$ helps algorithms become more tractable (as we shall see later when we get to Reinforcement Learning). We should also point out that the reason to have $\gamma < 1$ is not just for mathematical convenience or computational tractability - there are valid modeling reasons to discount Rewards when accumulating to a Return. When Reward is modeled as a financial quantity (revenues, costs, profits etc.), as will be the case in most financial applications, it makes sense to incorporate [time-value-of-money](https://en.wikipedia.org/wiki/Time_value_of_money) which is a fundamental concept in Economics/Finance that says there is greater benefit in receiving a dollar now versus later (which is the economic reason why interest is paid or earned). So it is common to set $\gamma$ to be the discounting based on the prevailing interest rate ($\gamma = \frac 1 {1+r}$ where $r$ is the interest rate over a single time step). Another technical reason for setting $\gamma < 1$ is that our models often don't fully capture future uncertainty and so, discounting with $\gamma$ acts to undermine future rewards that might not be accurate (due to future uncertainty modeling limitations). Lastly, from an AI perspective, if we want to build machines that acts like humans, psychologists have indeed demonstrated that human/animal behavior prefers immediate reward over future reward.

As you might imagine now, we'd want to identify states with large expected returns and states with small expected returns. This, in fact, is the main problem involving a Markov Reward Process - to compute the "Expected Return" associated with each state in the Markov Reward Process. Formally, we are interested in computing the *Value Function*
$$V: \mathcal{S} \rightarrow \mathbb{R}$$
defined as:
$$V(s) = \mathbb{E}[G_t|S_t=s] \text{ for all } s \in \mathcal{S} \text{ for all } t = 0, 1, 2, \ldots$$

Note that we are (as usual) assuming the fact that the Markov Reward Process is stationary (time-invariant probabilities of state transitions and rewards). Now we show a creative piece of mathematics due to [Richard Bellman](https://en.wikipedia.org/wiki/Richard_E._Bellman). Bellman noted that the Value Function has a recursive structure. Specifically, 

$$V(s) = \mathbb{E}[R_{t+1}|S_t=s] + \gamma \cdot \sum_{s' \in \mathcal{S}} \mathcal{P}(s, s') \cdot \mathbb{E}[G_{t+1}|S_{t+1}=s'] \text{ for all } s \in \mathcal{S} \text{ for all } t = 0, 1, 2, \ldots$$

This simplifies to:

$$V(s) = \mathcal{R}(s) + \gamma \cdot \sum_{s' \in \mathcal{S}} \mathcal{P}(s, s') \cdot V(s')$$

We will refer to this recursive equation for the Value Function as the Bellman Equation for a Markov Reward Process.

For the case of Finite Markov Reward Processes, $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$. Let us abuse notation and refer to $V$ as a column vector of length $n$, $\mathcal{P}$ as a $n \times n$ matrix, and $\mathcal{R}$ as a column vector of length $n$, so we can express the above equation in vector and matrix notation as follows:

$$V = \mathcal{R} + \gamma \mathcal{P} \cdot V$$
$$\Rightarrow V = (I_n - \gamma \mathcal{P})^{-1} \cdot \mathcal{R}$$
where $I_n$ is the $n \times n$ identity matrix.

Let us write some code to implement this calculation for Finite Markov Reward Processes. In the `FiniteMarkovRewardProcess` class, we implement the method `get_value_function_vec` that performs the above calculation for the Value Function $V$ in terms of the reward function $\mathcal{R}$ and the transition probability function $\mathcal{P}$ of the implicit Markov Process. The Value Function $V$ is produced as a 1D numpy array (i.e. a vector). Here's the code:

```python
    def get_value_function_vec(self, gamma) -> np.ndarray:
        return np.linalg.inv(
            np.eye(len(self.state_space)) - gamma * self.transition_matrix
        ).dot(self.reward_function_vec)

```

Invoking this `get_value_function_vec` method on `SimpleInventoryMRP` for the simple case of capacity $C=2$ and poisson mean $\lambda = 1.0$ yields the following result:

```
{(0, 0): -35.511,
 (0, 1): -27.932,
 (0, 2): -28.345,
 (1, 0): -28.932,
 (1, 1): -29.345,
 (2, 0): -30.345}
```

The corresponding values of the attribute `reward_function_vec` (i.e., $\mathcal{R}$) are:

```
{(0, 0): -10.0,
 (0, 1): -2.325,
 (0, 2): -0.274,
 (1, 0): -3.325,
 (1, 1): -1.274,
 (2, 0): -2.274}
```

This tells us that On-Hand of 0 and On-Order of 2 has the least expected cost (highest expected reward). However, the Value Function is highest for On-Hand of 0 and On-Order of 1.

This computation for the Value Function works if the state space is not too large (matrix to be inverted has size equal to state space size). When the state space is large, this direct matrix-inversion method doesn't work and we have to resort to numerical methods to solve the recursive Bellman equation. This is the topic of Dynamic Programming and Reinforcement Learning algorithms that we shall learn in this book. 

Before we end this chapter, we'd like to highlight the two highly important concepts we learnt in this chapter:

* Markov Property: A concept that enables us to reason effectively and compute efficiently in practical systems involving sequential uncertainty
* Bellman Equation: A mathematical insight that enables us to express the Value Function recursively - this equation (and its Optimality version covered in the next chapter) is in fact the core idea within all Dynamic Programming and Reinforcement Learning algorithms.
