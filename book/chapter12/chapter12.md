## Batch RL, Experience-Replay, DQN, LSPI, Gradient TD {#sec:batch-rl-chapter}

\index{reinforcement learning|(}

In Chapters [-@sec:rl-prediction-chapter] and [-@sec:rl-control-chapter], we covered the basic RL algorithms for Prediction and Control, respectively. Specifically, we covered the basic Monte-Carlo (MC) and Temporal-Difference (TD) techniques. We want to highlight two key aspects of these basic RL algorithms:

  1. The experiences data arrives in the form of a single unit of experience at a time (single unit is a *trace experience* for MC and an *atomic experience* for TD), the unit of experience is used by the algorithm for Value Function learning, and then that unit of experience is not used later in the algorithm (essentially, that unit of experience, once consumed, is *not re-consumed* for further learning later in the algorithm). It doesn't have to be this way—one can develop RL algorithms that re-use experience data—this approach is known as *Experience-Replay* (in fact, we saw a glimpse of Experience-Replay in Section [-@sec:experience-replay-section] of Chapter [-@sec:rl-prediction-chapter]).

  2. Learning occurs in a *granularly incremental* manner, by updating the Value Function after each unit of experience. It doesn't have to be this way—one can develop RL algorithms that take an entire batch of experiences (or in fact, all of the experiences that one could possibly get), and learn the Value Function directly for that entire batch of experiences. A key idea here is that if we know in advance what experiences data we have (or will have), and if we collect and organize all of that data, then we could directly (i.e., not incrementally) estimate the Value Function for *that* experiences data set. This approach to RL is known as *Batch RL* (versus the basic RL algorithms we covered in the previous chapters that can be termed as *Incremental RL*).

\index{reinforcement learning!experience replay|textbf}
\index{reinforcement learning!incremental|textbf}
\index{reinforcement learning!batch|textbf}

Thus, we have a choice or doing Experience-Replay or not, and we have a choice of doing Batch RL or Incremental RL. In fact, some of the interesting and practically effective algorithms combine both the ideas of Experience-Replay and Batch RL. This chapter starts with the coverage of Batch RL and Experience-Replay. Then, we cover some key algorithms (including Deep Q-Networks and Least Squares Policy Iteration) that effectively leverage Batch RL and/or Experience-Replay. Next, we look deeper into the issue of the *Deadly Triad* (that we had alluded to in Chapter [-@sec:rl-control-chapter]) by viewing Value Functions as Vectors (we had done this in Chapter [-@sec:dp-chapter]), understand Value Function Vector transformations with a balance of geometric intuition and mathematical rigor, providing insights into convergence issues for a variety of traditional loss functions used to develop RL algorithms. Finally, this treatment of Value Functions as Vectors leads us in the direction of overcoming the Deadly Triad by defining an appropriate loss function, calculating whose gradient provides a more robust set of RL algorithms known as Gradient Temporal Difference (abbreviated, as Gradient TD).

### Batch RL and Experience-Replay

\index{reinforcement learning!batch|(}

Let us understand Incremental RL versus Batch RL in the context of fixed finite experiences data. To make things simple and easy to understand, we first focus on understanding the difference for the case of MC Prediction (i.e., to calculate the Value Function of an MRP using Monte-Carlo). In fact, we had covered this setting in Section [-@sec:experience-replay-section] of Chapter [-@sec:rl-prediction-chapter].

\index{trace experience}
To refresh this setting, specifically we have access to a fixed finite sequence/stream of MRP trace experiences (i.e., `Iterable[Iterable[TransitionStep[S]]]`), which we know can be converted to returns-augmented data of the form `Iterable[Iterable[ReturnStep[S]]]` (using the `returns` function[^returns-file-2]). Flattening this data to `Iterable[ReturnStep[S]]` and extracting from it the (state, return) pairs gives us the fixed, finite training data for MC Prediction, that we denote as follows:

$$\mathcal{D} = [(S_i, G_i) | 1 \leq i \leq n]$$

[^returns-file-2]: `returns` is defined in the file [rl/returns.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/returns.py)

We've learnt in Chapter [-@sec:rl-prediction-chapter] that we can do an Incremental MC Prediction estimation $V(s;\bm{w})$ by updating $\bm{w}$ after each MRP trace experience with the gradient calculation $\nabla_{\bm{w}} \mathcal{L}(\bm{w})$ for each data pair $(S_i, G_i)$, as follows:
$$\mathcal{L}_{(S_i, G_i)}(\bm{w}) = \frac 1 2 \cdot (V(S_i; \bm{w}) - G_i)^2$$
$$\nabla_{\bm{w}} \mathcal{L}_{(S_i, G_i)}(\bm{w}) = (V(S_i; \bm{w}) - G_i) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$
$$\Delta \bm{w} = \alpha \cdot (G_i - V(S_i; \bm{w})) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$

The Incremental MC Prediction algorithm performs $n$ updates in sequence for data pairs $(S_i, G_i), i = 1, 2, \ldots, n$ using the `update` method of `FunctionApprox`. We note that Incremental RL makes inefficient use of available training data $\mathcal{D}$ because we essentially "discard" each of these units of training data after it's used to perform an update. We want to make efficient use of the given data with Batch RL. Batch MC Prediction aims to estimate the MRP Value Function $V(s;\bm{w^*})$ such that
\begin{align*}
\bm{w^*} & = \argmin_{\bm{w}} \frac 1 {2n} \cdot \sum_{i=1}^n (V(S_i;\bm{w}) - G_i)^2 \\
& = \argmin_{\bm{w}} \mathbb{E}_{(S,G) \sim \mathcal{D}} [\frac 1 2 \cdot (V(S; \bm{w}) - G)^2]
\end{align*}
This, in fact, is the `solve` method of `FunctionApprox` on training data $\mathcal{D}$. This approach is called Batch RL because we first collect and store the entire set (batch) of data $\mathcal{D}$ available to us, and then we find the best possible parameters $\bm{w^*}$ fitting this data $\mathcal{D}$. Note that unlike Incremental RL, here we are not updating the MRP Value Function estimate while the data arrives—we simply store the data as it arrives and start the MRP Value Function estimation procedure once we are ready with the entire (batch) data $\mathcal{D}$ in storage. As we know from the implementation of the `solve` method of `FunctionApprox`, finding the best possible parameters $\bm{w^*}$ from the batch $\mathcal{D}$ involves calling the `update` method of `FunctionApprox` with repeated use of the available data pairs $(S,G)$ in the stored data set $\mathcal{D}$. Each of these updates to the parameters $\bm{w}$ is as follows:
$$\Delta \bm{w} = \alpha \cdot \frac 1 n \cdot \sum_{i=1}^n (G_i - V(S_i; \bm{w})) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$

Note that unlike Incremental MC where each update to $\bm{w}$ uses data from a single trace experience, each update to $\bm{w}$ in Batch MC uses all of the trace experiences data (all of the batch data). If we keep doing these updates repeatedly, we will ultimately converge to the desired MRP Value Function $V(s;\bm{w^*})$. The repeated use of the available data in $\mathcal{D}$ means that we are doing Batch MC Prediction using *Experience-Replay*. So we see that this makes more efficient use of the available training data $\mathcal{D}$ due to the re-use of the data pairs in $\mathcal{D}$.

\index{reinforcement learning!experience replay}
\index{reinforcement learning!monte carlo!batch}

The code for this Batch MC Prediction algorithm (`batch_mc_prediction`) is shown below.[^batch-mc-file] From the input trace experiences (`traces` in the code below), we first create the set of `ReturnStep` transitions that span across the set of all input trace experiences (`return_steps` in the code below). This involves calculating the return associated with each state encountered in `traces` (across all trace experiences). From `return_steps`, we create the (state, return) pairs that constitute the fixed, finite training data $\mathcal{D}$, which is then passed to the `solve` method of `approx: ValueFunctionApprox[S]`.

\index{batch mc prediction@\texttt{batch\_mc\_prediction}}

```python
import rl.markov_process as mp
from rl.returns import returns
from rl.approximate_dynamic_programming import ValueFunctionApprox
import itertools

def batch_mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx: ValueFunctionApprox[S],
    gamma: float,
    episode_length_tolerance: float = 1e-6,
    convergence_tolerance: float = 1e-5
) -> ValueFunctionApprox[S]:
    '''traces is a finite iterable'''
    return_steps: Iterable[mp.ReturnStep[S]] = \
        itertools.chain.from_iterable(
            returns(trace, gamma, episode_length_tolerance) for trace in traces
        )
    return approx.solve(
        [(step.state, step.return_) for step in return_steps],
        convergence_tolerance
    )
```

[^batch-mc-file]: `batch_mc_prediction` is defined in the file [rl/monte_carlo.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/monte_carlo.py).

Now let's move on to Batch TD Prediction. Here we have fixed, finite experiences data $\mathcal{D}$ available as:
$$\mathcal{D} = [(S_i, R_i, S'_i) | 1 \leq i \leq n]$$
where $(R_i, S'_i)$ is the pair of reward and next state from a state $S_i$. So, Experiences Data $\mathcal{D}$ is presented in the form of a fixed, finite number of atomic experiences. This is represented in code as an `Iterable[TransitionStep[S]]`. 

\index{atomic experience}
\index{reinforcement learning!temporal difference!batch}

Just like Batch MC Prediction, here in Batch TD Prediction, we first collect and store the data as it arrives, and once we are ready with the batch of data $\mathcal{D}$ in storage, we start the MRP Value Function estimation procedure. The parameters $\bm{w}$ are updated with repeated use of the atomic experiences in the stored data $\mathcal{D}$. Each of these updates to the parameters $\bm{w}$ is as follows:
$$\Delta \bm{w} = \alpha \cdot \frac 1 n \cdot \sum_{i=1}^n (R_i + \gamma \cdot V(S'_i; \bm{w}) - V(S_i; \bm{w})) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$

Note that unlike Incremental TD where each update to $\bm{w}$ uses data from a single atomic experience, each update to $\bm{w}$ in Batch TD uses all of the atomic experiences data (all of the batch data). The repeated use of the available data in $\mathcal{D}$ means that we are doing Batch TD Prediction using *Experience-Replay*. So we see that this makes more efficient use of the available training data $\mathcal{D}$ due to the re-use of the data pairs in $\mathcal{D}$.

\index{reinforcement learning!experience replay}

The code for this Batch TD Prediction algorithm (`batch_td_prediction`) is shown below.[^batch-td-file] We create a `Sequence[TransitionStep]` from the fixed, finite-length input atomic experiences $\mathcal{D}$ (`transitions` in the code below), and call the `update` method of `FunctionApprox` repeatedly, passing the data $\mathcal{D}$ (now in the form of a `Sequence[TransitionStep]`) to each invocation of the `update` method (using the function `itertools.repeat`). This repeated invocation of the `update` method is done by using the function `iterate.accumulate`.  This is done until convergence (convergence based on the `done` function in the code below), at which point we return the converged `FunctionApprox`.

\index{batch td prediction@\texttt{batch\_td\_prediction}}

```python
import rl.markov_process as mp
from rl.approximate_dynamic_programming import ValueFunctionApprox, extended_vf
import rl.iterate as iterate
import itertools
import numpy as np

def batch_td_prediction(
    transitions: Iterable[mp.TransitionStep[S]],
    approx_0: ValueFunctionApprox[S],
    gamma: float,
    convergence_tolerance: float = 1e-5
) -> ValueFunctionApprox[S]:
    '''transitions is a finite iterable'''

    def step(
        v: ValueFunctionApprox[S],
        tr_seq: Sequence[mp.TransitionStep[S]]
    ) -> ValueFunctionApprox[S]:
        return v.update([(
            tr.state, tr.reward + gamma * extended_vf(v, tr.next_state)
        ) for tr in tr_seq])

    def done(
        a: ValueFunctionApprox[S],
        b: ValueFunctionApprox[S],
        convergence_tolerance=convergence_tolerance
    ) -> bool:
        return b.within(a, convergence_tolerance)

    return iterate.converged(
        iterate.accumulate(
            itertools.repeat(list(transitions)),
            step,
            initial=approx_0
        ),
        done=done
```

[^batch-td-file]: `batch_td_prediction` is defined in the file [rl/td.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/td.py).

Likewise, we can do Batch TD($\lambda$) Prediction. Here we are given a fixed, finite number of trace experiences
$$\mathcal{D} = [(S_{i,0}, R_{i,1}, S_{i,1}, R_{i,2}, S_{i,2}, \ldots, R_{i,T_i}, S_{i,T_i}) | 1 \leq i \leq n]$$
For trace experience $i$, for each time step $t$ in the trace experience, we calculate the eligibility traces as follows:
$$\bm{E}_{i,t} = \gamma \lambda \cdot \bm{E}_{i, t-1} + \nabla_{\bm{w}} V(S_{i,t};\bm{w}) \text{ for all } t = 1, 1, \ldots T_i - 1$$
with the eligiblity traces initialized at time 0 for trace experience $i$ as $\bm{E}_{i,0} = \nabla_{\bm{w}} V(S_{i,0}; \bm{w})$.

\index{eligibility traces}
\index{reinforcement learning!temporal difference$(\lambda)$!batch}

Then, each update to the parameters $\bm{w}$ is as follows:
\begin{equation}
\Delta \bm{w} = \alpha \cdot \frac 1 n \cdot \sum_{i=1}^n \frac 1 {T_i} \cdot \sum_{t=0}^{T_i - 1} (R_{i,t+1} + \gamma \cdot V(S_{i,t+1}; \bm{w}) - V(S_{i,t}; \bm{w})) \cdot \bm{E}_{i,t}
\label{eq:batch-td-lambda-update}
\end{equation}

\index{reinforcement learning!batch|)}

### A Generic Implementation of Experience-Replay

\index{experience replay|(}

Before we proceed to more algorithms involving Experience-Replay and/or Batch RL, it is vital to recognize that the concept of Experience-Replay stands on its own, independent of its use in Batch RL. In fact, Experience-Replay is a much broader concept, beyond its use in RL. The idea of Experience-Replay is that we have a stream of data coming in and instead of consuming it in an algorithm as soon as it arrives, we store each unit of incoming data in memory (which we shall call *Experience-Replay-Memory*, abbreviated as ER-Memory), and use samples of data from ER-Memory (with replacement) for our algorithm's needs. Thus, we are routing the incoming stream of data to ER-Memory and sourcing data needed for our algorithm from ER-Memory (by sampling with replacement). This enables re-use of the incoming data stream. It also gives us flexibility to sample an arbitrary number of data units at a time, so our algorithm doesn't need to be limited to using a single unit of data at a time. Lastly, we organize the data in ER-Memory in such a manner that we can assign different sampling weights to different units of data, depending on the arrival time of the data. This is quite useful for many algorithms that wish to give more importance to recently arrived data and de-emphasize/forget older data.

Let us now write some code to implement all of these ideas described above. The code below uses an arbitrary data type `T`, which means that the unit of data being handled with Experience-Replay could be any data structure (specifically, not limited to the `TransitionStep` data type that we care about for RL with Experience-Replay).

The attribute `saved_transitions: List[T]` is the data structure storing the incoming units of data, with the most recently arrived unit of data at the end of the list (since we `append` to the list). The attribute `time_weights_func` lets the user specify a function from the reverse-time-stamp of a unit of data to the sampling weight to assign to that unit of data ("reverse-time-stamp" means the most recently-arrived unit of data has a time-index of 0, although physically it is stored at the end of the list, rather than at the start). The attribute `weights` simply stores the sampling weights of all units of data in `saved_transitions`, and the attribute `weights_sum` stores the sum of the `weights` (the attributes `weights` and `weights_sum` are there purely for computational efficiency to avoid too many calls to `time_weights_func` and avoidance of summing a long list of weights, which is required to normalize the weights to sum up to 1).

`add_data` appends an incoming unit of data (`transition: T`) to `self.saved_transitions` and updates `self.weights` and `self.weights_sum`. `sample_mini_batches` returns a sample of specified size `mini_batch_size`, using the sampling weights in `self.weights`. We also have a method `replay` that takes as input an `Iterable` of `transitions` and a `mini_batch_size`, and returns an `Iterator` of `mini_batch_size`d data units. As long as the input `transitions:` `Iterable[T]` is not exhausted, `replay` appends each unit of data in `transitions` to `self.saved_transitions` and then `yield`s a `mini_batch_size`d sample of data. Once `transitions:` `Iterable[T]` is exhausted, it simply `yields` the samples of data. The `Iterator` generated by `replay` can be piped to any algorithm that expects an `Iterable` of the units of data as input, essentially enabling us to replace the pipe carrying an input data stream with a pipe carrying the data stream sourced from ER-Memory.

\index{ExperienceReplayMemory@\texttt{ExperienceReplayMemory}}

```python
T = TypeVar('T')

class ExperienceReplayMemory(Generic[T]):
    saved_transitions: List[T]
    time_weights_func: Callable[[int], float]
    weights: List[float]
    weights_sum: float

    def __init__(
        self,
        time_weights_func: Callable[[int], float] = lambda _: 1.0,
    ):
        self.saved_transitions = []
        self.time_weights_func = time_weights_func
        self.weights = []
        self.weights_sum = 0.0

    def add_data(self, transition: T) -> None:
        self.saved_transitions.append(transition)
        weight: float = self.time_weights_func(len(self.saved_transitions) - 1)
        self.weights.append(weight)
        self.weights_sum += weight

    def sample_mini_batch(self, mini_batch_size: int) -> Sequence[T]:
        num_transitions: int = len(self.saved_transitions)
        return Categorical(
            {tr: self.weights[num_transitions - 1 - i] / self.weights_sum
             for i, tr in enumerate(self.saved_transitions)}
        ).sample_n(min(mini_batch_size, num_transitions))

    def replay(
        self,
        transitions: Iterable[T],
        mini_batch_size: int
    ) -> Iterator[Sequence[T]]:

        for transition in transitions:
            self.add_data(transition)
            yield self.sample_mini_batch(mini_batch_size)

        while True:
            yield self.sample_mini_batch(mini_batch_size)
```

The code above is in the file [rl/experience_replay.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/experience_replay.py). We encourage you to implement Batch MC Prediction and Batch TD Prediction using this `ExperienceReplayMemory` class.

\index{experience replay|)}

### Least-Squares RL Prediction

\index{reinforcement learning!least squares|(}

We've seen how Batch RL Prediction is an iterative process of weight updates until convergence—the MRP Value Function is updated with repeated use of the fixed, finite (batch) data that is made available. However, if we assume that the MRP Value Function approximation $V(s; \bm{w})$ is a linear function approximation (linear in a set of feature functions of the state space), then we can solve for the MRP Value Function with direct and simple linear algebra operations (i.e., without the need for iterative weight updates until convergence). Let us see how.

\index{function approximation!linear}

We define a sequence of feature functions $\phi_j: \mathcal{N} \rightarrow \mathbb{R}, j = 1, 2, \ldots, m$ and we assume the parameters $\bm{w}$ is a weights vector $\bm{w} = (w_1, w_2, \ldots, w_m) \in \mathbb{R}^m$. Therefore, the MRP Value Function is approximated as:
$$V(s;\bm{w}) = \sum_{j=1}^m \phi_j(s) \cdot w_j = \bm{\phi}(s)^T \cdot \bm{w} \text{ for all } s \in \mathcal{N}$$
where $\bm{\phi}(s) \in \mathbb{R}^m$ is the feature vector for state $s$.

The direct solution of the MRP Value Function using simple linear algebra operations is known as Least-Squares (abbreviated as LS) solution. We start with Batch MC Prediction for the case of linear function approximation, which is known as Least-Squares Monte-Carlo (abbreviated as LSMC).

#### Least-Squares Monte-Carlo (LSMC)

\index{reinforcement learning!least squares!lsmc|(}

For the case of linear function approximation, the loss function for Batch MC Prediction with data $[(S_i, G_i) | 1 \leq i \leq n]$ is:
$$\mathcal{L}(\bm{w}) =  \frac 1 {2n} \cdot \sum_{i=1}^n (\sum_{j=1}^m \phi_j(S_i) \cdot w_j - G_i)^2 = \frac 1 {2n} \cdot \sum_{i=1}^n (\bm{\phi}(S_i)^T \cdot \bm{w} - G_i)^2$$
We set the gradient of this loss function to 0, and solve for $\bm{w}^*$. This yields:
$$\sum_{i=1}^n \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i)^T \cdot \bm{w^*} - G_i) = 0$$
We can calculate the solution $\bm{w^}*$ as $\bm{A}^{-1} \cdot \bm{b}$, where the $m \times m$ Matrix $\bm{A}$ is accumulated at each data pair $(S_i, G_i)$ as:
$$ \bm{A} \leftarrow \bm{A} + \bm{\phi}(S_i) \cdot \bm{\phi}(S_i)^T \text{ (i.e., outer-product of } \bm{\phi}(S_i) \text{ with itself})$$
and the $m$-Vector $\bm{b}$ is accumulated at each data pair $(S_i, G_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{\phi}(S_i) \cdot G_i$$

To implement this algorithm, we can simply call `batch_mc_prediction` that we had written earlier by setting the argument `approx` as `LinearFunctionApprox` and by setting the attribute `direct_solve` in `approx: LinearFunctionApprox[S]` as `True`. If you read the code under `direct_solve=True` branch in the `solve` method, you will see that it indeed performs the above-described linear algebra calculations. The inversion of the matrix $\bm{A}$ is $O(m^3)$ complexity. However, we can speed up the algorithm to be $O(m^2)$ with a different implementation—we can maintain the inverse of $\bm{A}$ after each $(S_i, G_i)$ update to $\bm{A}$ by applying the [Sherman-Morrison formula for incremental inverse](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula) [@10.1214/aoms/1177729893]. The Sherman-Morrison incremental inverse for $\bm{A}$ is as follows:

\index{Sherman-Morrison incremental inverse}

$$(\bm{A} + \bm{\phi}(S_i) \cdot \bm{\phi}(S_i)^T)^{-1} = \bm{A}^{-1} - \frac {\bm{A}^{-1} \cdot \bm{\phi}(S_i) \cdot \bm{\phi}(S_i)^T \cdot \bm{A}^{-1}} {1 + \bm{\phi}(S_i)^T \cdot \bm{A}^{-1} \cdot \bm{\phi}(S_i)}$$
with $\bm{A}^{-1}$ initialized to $\frac 1 {\epsilon} \cdot \bm{I}_m$, where $\bm{I}_m$ is the $m \times m$ identity matrix, and $\epsilon \in \mathbb{R}^+$ is a small number provided as a parameter to the algorithm. $\frac 1 \epsilon$ should be considered to be a proxy for the step-size $\alpha$ which is not required for least-squares algorithms. If $\epsilon$ is too small, the sequence of inverses of $\bm{A}$ can be quite unstable and if $\epsilon$ is too large, the learning is slowed.

This brings down the computational complexity of this algorithm to $O(m^2)$. We won't implement the Sherman-Morrison incremental inverse for LSMC, but in the next subsection we shall implement it for Least-Squares Temporal Difference (LSTD).

\index{reinforcement learning!least squares!lsmc|)}

#### Least-Squares Temporal-Difference (LSTD)

\index{reinforcement learning!least squares!lstd|(}

For the case of linear function approximation, the loss function for Batch TD Prediction with data $[(S_i, R_i, S'_i) | 1 \leq i \leq n]$ is:
$$\mathcal{L}(\bm{w}) = \frac 1 {2n} \cdot \sum_{i=1}^n (\bm{\phi}(S_i)^T \cdot \bm{w} - (R_i + \gamma \cdot \bm{\phi}(S'_i)^T \cdot \bm{w}))^2$$
We set the semi-gradient of this loss function to 0, and solve for $\bm{w}^*$. This yields:
$$\sum_{i=1}^n \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i)^T \cdot \bm{w^*} - (S_i + \gamma \cdot \bm{\phi}(S'_i)^T \cdot \bm{w}^*)) = 0$$
We can calculate the solution $\bm{w^}*$ as $\bm{A}^{-1} \cdot \bm{b}$, where the $m \times m$ Matrix $\bm{A}$ is accumulated at each atomic experience $(S_i, R_i, S'_i)$ as:
$$ \bm{A} \leftarrow \bm{A} + \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))^T \text{ (note the Outer-Product)}$$
and the $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(S_i, R_i, S'_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{\phi}(S_i) \cdot R_i$$

\index{Sherman-Morrison incremental inverse}

With Sherman-Morrison incremental inverse, we can reduce the computational complexity from $O(m^3)$ to $O(m^2)$, as follows:
$$(\bm{A} + \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S_i))^T)^{-1} = \bm{A}^{-1} - \frac {\bm{A}^{-1} \cdot \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))^T \cdot \bm{A}^{-1}} {1 + (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))^T \cdot \bm{A}^{-1} \cdot \bm{\phi}(S_i)}$$

with $\bm{A}^{-1}$ initialized to $\frac 1 {\epsilon} \cdot \bm{I}_m$, where $\bm{I}_m$ is the $m \times m$ identity matrix, and $\epsilon \in \mathbb{R}^+$ is a small number provided as a parameter to the algorithm.

This algorithm is known as the Least-Squares Temporal-Difference (LSTD) algorithm and is due to [Bradtke and Barto](http://incompleteideas.net/papers/bradtke-barto-96.pdf) [@journals/ml/BradtkeB96].

Now let's write some code to implement this LSTD algorithm. The arguments `transitions`, `feature_functions`, `gamma` and `epsilon` of the function `least_squares_td` below are quite self-explanatory. This is a batch method with direct calculation of the estimated Value Function from batch data (rather than iterative weight updates), so `least_squares_td` returns the estimated Value Function of type `LinearFunctionApprox[NonTerminal[S]]`, rather than an `Iterator` over the updated function approximations (as was the case in Incremental RL algorithms).

The code below should be fairly self-explanatory. `a_inv` refers to $\bm{A}^{-1}$ which is updated with the Sherman-Morrison incremental inverse method. `b_vec` refers to the $\bm{b}$ vector. `phi1` refers to $\bm{\phi}(S_i)$, `phi2` refers to $\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i)$ (except when $S'_i$ is a terminal state, in which case `phi2` is simply $\bm{\phi}(S_i)$). The temporary variable `temp` refers to $(\bm{A}^{-1})^T \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))$ and is used both in the numerator and denominator in the Sherman-Morrison formula to update $\bm{A}^{-1}$.

\index{least squares td@\texttt{least\_squares\_td}}

```python
from rl.function_approx import LinearFunctionApprox
import rl.markov_process as mp
import numpy as np

def least_squares_td(
    transitions: Iterable[mp.TransitionStep[S]],
    feature_functions: Sequence[Callable[[NonTerminal[S]], float]],
    gamma: float,
    epsilon: float
) -> LinearFunctionApprox[NonTerminal[S]]:
    ''' transitions is a finite iterable '''
    num_features: int = len(feature_functions)
    a_inv: np.ndarray = np.eye(num_features) / epsilon
    b_vec: np.ndarray = np.zeros(num_features)
    for tr in transitions:
        phi1: np.ndarray = np.array([f(tr.state) for f in feature_functions])
        if isinstance(tr.next_state, NonTerminal):
            phi2 = phi1 - gamma * np.array([f(tr.next_state)
                                        for f in feature_functions])
        else:
            phi2 = phi1
        temp: np.ndarray = a_inv.T.dot(phi2)
        a_inv = a_inv - np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
        b_vec += phi1 * tr.reward

    opt_wts: np.ndarray = a_inv.dot(b_vec)
    return LinearFunctionApprox.create(
        feature_functions=feature_functions,
        weights=Weights.create(opt_wts)
    )
```

The code above is in the file [rl/td.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/td.py).

Now let's test this on transitions data sampled from the `RandomWalkMRP` example we had constructed in Chapter [-@sec:rl-prediction-chapter]. As a reminder, this MRP consists of a random walk across states $\{0, 1, 2, \ldots, B\}$ with $0$ and $B$ as the terminal states (think of these as terminating barriers of a random walk) and the remaining states as the non-terminal states. From any non-terminal state $i$, we transition to state $i+1$ with probability $p$ and to state $i-1$ with probability $1-p$. The reward is 0 upon each transition, except if we transition from state $B-1$ to terminal state $B$ which results in a reward of 1. The code for `RandomWalkMRP` is in the file [rl/chapter10/random_walk_mrp.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter10/random_walk_mrp.py). 

First, we set up a `RandomWalkMRP` object with $B = 20, p = 0.55$ and calculate its true Value Function (so we can later compare against Incremental TD and LSTD methods).

```python
from rl.chapter10.random_walk_mrp import RandomWalkMRP
import nump as np

this_barrier: int = 20
this_p: float = 0.55
random_walk: RandomWalkMRP = RandomWalkMRP(
    barrier=this_barrier,
    p=this_p
)
gamma = 1.0
true_vf: np.ndarray = random_walk.get_value_function_vec(gamma=gamma)
```

Let's say we have access to only 10,000 transitions (each transition is an object of the type `TransitionStep`). First, we generate these 10,000 sampled transitions from the `RandomWalkMRP` object we created above.


```python
from rl.approximate_dynamic_programming import NTStateDistribution
from rl.markov_process import TransitionStep
import itertools

num_transitions: int = 10000
nt_states: Sequence[NonTerminal[int]] = random_walk.non_terminal_states
start_distribution: NTStateDistribution[int] = Choose(set(nt_states))
traces: Iterable[Iterable[TransitionStep[int]]] = \
    random_walk.reward_traces(start_distribution)
transitions: Iterable[TransitionStep[int]] = \
    itertools.chain.from_iterable(traces)
td_transitions: Iterable[TransitionStep[int]] = \
    itertools.islice(transitions, num_transitions)
```

Before running LSTD, let's run Incremental Tabular TD on the 10,000 transitions in `td_transitions` and obtain the resultant Value Function (`td_vf` in the code below). Since there are only 10,000 transitions, we use an aggressive initial learning rate of 0.5 to promote fast learning, but we let this high learning rate decay quickly so the learning stabilizes.

```python
from rl.function_approx import Tabular
import rl.iterate as iterate

initial_learning_rate: float = 0.5
half_life: float = 1000
exponent: float = 0.5
approx0: Tabular[NonTerminal[int]] = Tabular(
    count_to_weight_func=learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
)
td_func: Tabular[NonTerminal[int]] = \
    iterate.last(itertools.islice(
        td_prediction(
            transitions=td_transitions,
            approx_0=approx0,
            gamma=gamma
        ),
        num_transitions
    ))
td_vf: np.ndarray = td_func.evaluate(nt_states)
```

Finally, we run the LSTD algorithm on 10,000 transitions. Note that the Value Function of `RandomWalkMRP`, for $p \neq 0.5$, is non-linear as a function of the integer states. So we use non-linear features that can approximate arbitrary non-linear shapes—a good choice is the set of (orthogonal) [Laguerre Polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials). In the code below, we use the first 5 Laguerre Polynomials (i.e., upto degree 4 polynomial) as the feature functions for the linear function approximation of the Value Function. Then we invoke the LSTD algorithm we wrote above to calculate the `LinearFunctionApprox` based on this batch of 10,000 transitions.

```python
from rl.chapter12.laguerre import laguerre_state_features
from rl.function_approx import LinearFunctionApprox

num_polynomials: int = 5
features: Sequence[Callable[[NonTerminal[int]], float]] = \
    laguerre_state_features(num_polynomials)
lstd_transitions: Iterable[TransitionStep[int]] = \
    itertools.islice(transitions, num_transitions)
epsilon: float = 1e-4

lstd_func: LinearFunctionApprox[NonTerminal[int]] = \
    least_squares_td(
        transitions=lstd_transitions,
        feature_functions=features,
        gamma=gamma,
        epsilon=epsilon
    )
lstd_vf: np.ndarray = lstd_func.evaluate(nt_states)
```   

Figure \ref{fig:lstd_vf_comparison} depicts how the LSTD Value Function estimate (for 10,000 transitions) `lstd_vf` compares against Incremental Tabular TD Value Function estimate (for 10,000 transitions) `td_vf` and against the true value function `true_vf` (obtained using the linear-algebra-solver-based calculation of the MRP Value Function). We encourage you to modify the parameters used in the code above to see how it alters the results—specifically play around with `this_barrier`, `this_p`, `gamma`, `num_transitions`, the learning rate trajectory for Incremental Tabular TD, the number of Laguerre polynomials, and `epsilon`. The above code is in the file [rl/chapter12/random_walk_lstd.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/random_walk_lstd.py).

![LSTD and Tabular TD Value Functions \label{fig:lstd_vf_comparison}](./chapter12/lstd_vf_comparison.png "LSTD and Tabular TD Value Functions"){height=7cm}

\index{reinforcement learning!least squares!lstd|)}

#### LSTD($\lambda$)

\index{reinforcement learning!least squares!lstd$(\lambda)$}
\index{eligibility traces}

Likewise, we can do LSTD($\lambda$) using Eligibility Traces. Here we are given a fixed, finite number of trace experiences
$$\mathcal{D} = [(S_{i,0}, R_{i,1}, S_{i,1}, R_{i,2}, S_{i,2}, \ldots, R_{i,T_i}, S_{i,T_i}) | 1 \leq i \leq n]$$
Denote the Eligibility Traces of trace experience $i$ at time $t$ as $\bm{E}_{i,t}$. Note that the eligibility traces accumulate $\nabla_{\bm{w}} V(s;\bm{w}) = \bm{\phi}(s)$ in each trace experience. When accumulating, the previous time step's eligibility traces is discounted by $\lambda \gamma$. By setting the right-hand-side of Equation \eqref{eq:batch-td-lambda-update} to 0 (i.e., setting the update to $\bm{w}$ over all atomic experiences data to 0), we get:
$$\sum_{i=1}^n \frac 1 {T_i} \cdot \sum_{t=0}^{T_i - 1} \bm{E}_{i,t} \cdot (\bm{\phi}(S_{i,t})^T \cdot \bm{w^*} - (R_{i,t+1} + \gamma \cdot \bm{\phi}(S_{i,t+1})^T \cdot \bm{w}^*)) = 0$$
We can calculate the solution $\bm{w^}*$ as $\bm{A}^{-1} \cdot \bm{b}$, where the $m \times m$ Matrix $\bm{A}$ is accumulated at each atomic experience $(S_{i,t}, R_{i,t+1}, S_{i,t+1})$ as:
$$ \bm{A} \leftarrow \bm{A} + \frac 1 {T_i} \cdot \bm{E}_{i,t} \cdot (\bm{\phi}(S_{i,t}) - \gamma \cdot \bm{\phi}(S_{i,t+1}))^T \text{ (note the Outer-Product)}$$
and the $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(S_{i,t}, R_{i,t+1}, S_{i,t+1})$ as:
$$\bm{b} \leftarrow \bm{b} + \frac 1 {T_i} \cdot \bm{E}_{i,t} \cdot R_{i,t+1}$$
With Sherman-Morrison incremental inverse, we can reduce the computational complexity from $O(m^3)$ to $O(m^2)$.

#### Convergence of Least-Squares Prediction

\index{reinforcement learning!convergence|(}

Before we move on to Least-Squares for the Control problem, we want to point out that the convergence behavior of Least-Squares Prediction algorithms are identical to their counterpart Incremental RL Prediction algorithms, with the exception that Off-Policy LSMC does not have convergence guarantees. Figure \ref{fig:rl_prediction_with_ls_convergence} shows the updated summary table for convergence of RL Prediction algorithms (that we had displayed at the end of Chapter [-@sec:rl-control-chapter]) to now also include Least-Squares Prediction algorithms. 

\begin{figure}
\begin{center}
\begin{tabular}{ccccc}
\hline
On/Off Policy & Algorithm & Tabular & Linear & Non-Linear \\ \hline
& MC & \cmark & \cmark & \cmark \\
& \bfseries LSMC & \cmark & \cmark & - \\
On-Policy & TD & \cmark & \cmark & \xmark \\
& \bfseries LSTD & \cmark & \cmark & - \\
& \bfseries Gradient TD & \cmark & \cmark & \cmark \\ \hline
& MC & \cmark & \cmark & \cmark \\
& \bfseries LSMC & \cmark & \xmark & - \\
Off-Policy & TD & \cmark & \xmark & \xmark \\ 
& \bfseries LSTD & \cmark & \xmark & - \\ 
& Gradient TD & \cmark & \cmark & \cmark \\ \hline
\end{tabular}
\end{center}    
\caption{Convergence of RL Prediction Algorithms}
\label{fig:rl_prediction_with_ls_convergence}
\end{figure}

This ends our coverage of Least-Squares Prediction. Before we move on to Least-Squares Control, we need to cover Incremental RL Control with Experience-Replay as it serves as a stepping stone towards Least-Squares Control.

\index{reinforcement learning!convergence|)}
\index{reinforcement learning!least squares|)}

### Q-Learning with Experience-Replay

\index{reinforcement learning!temporal difference!q learning}
\index{reinforcement learning!experience replay}
\index{reinforcement learning!off-policy}
\index{reinforcement learning!incremental}

In this section, we cover Off-Policy Incremental TD Control with Experience-Replay. Specifically, we revisit the Q-Learning algorithm we covered in Chapter [-@sec:rl-control-chapter], but we tweak that algorithm such that the transitions used to make the Q-Learning updates are sourced from an experience-replay memory, rather than from a behavior policy derived from the current Q-Value estimate. While investigating the challenges with Off-Policy TD methods with deep learning function approximation, researchers identified two challenges:

1) The sequences of states made available to deep learning through trace experiences are highly correlated, whereas deep learning algorithms are premised on data samples being independent.
2) The data distribution changes as the RL algorithm learns new behaviors, whereas deep learning algorithms are premised on a fixed underlying distribution (i.e., stationary).

Experience-Replay serves to smooth the training data distribution over many past behaviors, effectively resolving the correlation issue as well as the non-stationary issue. Hence, Experience-Replay is a powerful idea for Off-Policy TD Control. The idea of using Experience-Replay for Off-Policy TD Control is due to the [Ph.D. thesis of Long Lin](https://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf) [@lin:phd].

To make this idea of Q-Learning with Experience-Replay clear, we make a few changes to the `q_learning` function we had written in Chapter [-@sec:rl-control-chapter] with the following function `q_learning_experience_replay`.

\index{q learning experience replay@\texttt{q\_learning\_experience\_replay}}

```python
from rl.markov_decision_process import TransitionStep
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import NTStateDistribution
from rl.experience_replay import ExperienceReplayMemory

PolicyFromQType = Callable[
    [QValueFunctionApprox[S, A], MarkovDecisionProcess[S, A]],
    Policy[S, A]
]

def q_learning_experience_replay(
    mdp: MarkovDecisionProcess[S, A],
    policy_from_q: PolicyFromQType,
    states: NTStateDistribution[S],
    approx_0: QValueFunctionApprox[S, A],
    gamma: float,
    max_episode_length: int,
    mini_batch_size: int,
    weights_decay_half_life: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    exp_replay: ExperienceReplayMemory[TransitionStep[S, A]] = \
        ExperienceReplayMemory(
            time_weights_func=lambda t: 0.5 ** (t / weights_decay_half_life),
        )
    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(q, mdp)
            action: A = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            exp_replay.add_data(TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            ))
            trs: Sequence[TransitionStep[S, A]] = \
                exp_replay.sample_mini_batch(mini_batch_size)
            q = q.update(
                [(
                    (tr.state, tr.action),
                    tr.reward + gamma * (
                        max(q((tr.next_state, a))
                            for a in mdp.actions(tr.next_state))
                        if isinstance(tr.next_state, NonTerminal) else 0.)
                ) for tr in trs],
            )
            yield q
            steps += 1
            state = next_state
```

The key difference between the `q_learning` algorithm we wrote in Chapter [-@sec:rl-control-chapter] and this `q_learning_experience_replay` algorithm is that here we have an experience-replay memory (using the `ExperienceReplayMemory` class we had implemented earlier). In the `q_learning` algorithm, the (`state`, `action`, `next_state`, `reward`) 4-tuple comprising `TransitionStep` (that is used to perform the Q-Learning update) was the result of `action` being sampled from the behavior policy (derived from the current estimate of the Q-Value Function, e.g., $\epsilon$-greedy), and then the `next_state` and `reward` being generated from the (`state`, `action`) pair using the `step` method of `mdp`. Here in `q_learning_experience_replay`, we don't use this 4-tuple `TransitionStep` to perform the update—rather, we append this 4-tuple to the `ExperienceReplayMemory` (using the `add_data` method), then we sample `mini_batch_size`d `TransitionStep`s from the `ExperienceReplayMemory` (giving more sampling weightage to the more recently added `TransitionStep`s), and use those 4-tuple `TransitionStep`s to perform the Q-Learning update. Note that these sampled `TransitionStep`s might be from old behavior policies (derived from old estimates of the Q-Value estimate). The key is that this algorithm re-uses atomic experiences that were previously prepared by the algorithm, which also means that it re-uses behavior policies that were previously constructed by the algorithm.

The argument `mini_batch_size` refers to the number of `TransitionStep`s to be drawn from the `ExperienceReplayMemory` at each step. The argument `weights_decay_half_life` refers to the half life of an exponential decay function for the weights used in the sampling of the `TransitionStep`s (the most recently added `TransitionStep` has the highest weight). With this understanding, the code should be self-explanatory.

The above code is in the file [rl/td.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/td.py).

#### Deep Q-Networks (DQN) Algorithm

\index{reinforcement learning!dqn}
\index{function approximation!neural network}
\index{experience replay}
\index{reinforcement learning!temporal difference!q learning}
\index{reinforcement learning!deep reinforcement learning}

[DeepMind](https://deepmind.com/) developed an innovative and practically effective RL Control algorithm based on Q-Learning with Experience-Replay—an algorithm they named as Deep Q-Networks (abberviated as DQN). Apart from reaping the above-mentioned benefits of Experience-Replay for Q-Learning with a Deep Neural Network approximating the Q-Value function, they also benefited from employing a second Deep Neural Network (let us call the main DNN as the Q-Network, referring to its parameters at $\bm{w}$, and the second DNN as the target network, referring to its parameters as $\bm{w}^-$). The parameters $\bm{w}^-$ of the target network are infrequently updated to be made equal to the parameters $\bm{w}$ of the Q-network. The purpose of the Q-Network is to evaluate the Q-Value of the current state $s$ and the purpose of the target network is to evaluate the Q-Value of the next state $s'$, which in turn is used to obtain the Q-Learning target (note that the Q-Value of the current state is $Q(s,a;\bm{w})$ and the Q-Learning target is $r + \gamma \cdot \max_{a'} Q(s', a'; \bm{w}^-)$ for a given atomic experience $(s,a,r,s')$).

Deep Learning is premised on the fact that the supervised learning targets (response values $y$ corresponding to predictor values $x$) are pre-generated fixed values. This is not the case in TD learning where the targets are dependent on the Q-Values. As Q-Values are updated at each step, the targets also get updated, and this correlation between the current state's Q-Value estimate and the target value typically leads to oscillations or divergence of the Q-Value estimate. By infrequently updating the parameters $\bm{w}^-$ of the target network (providing the target values) to be made equal to the parameters $\bm{w}$ of the Q-network (which are updated at each iteration), the targets in the Q-Learning update are essentially kept fixed. This goes a long way in resolving the core issue of correlation between the current state's Q-Value estimate and the target values, helping considerably with convergence of the Q-Learning algorithm. Thus, DQN reaps the benefits of not just Experience-Replay in Q-Learning (which we articulated earlier), but also the benefits of having "fixed" targets. DNN utilizes a parameter $C$ such that the updating of $\bm{w}^-$ to be made equal to $\bm{w}$ is done once every $C$ updates to $\bm{w}$ (updates to $\bm{w}$ are based on the usual Q-Learning update equation).

We won't implement the DQN algorithm in Python code—however, we sketch the outline of the algorithm, as follows:

At each time $t$ for each episode:

* Given state $S_t$, take action $A_t$ according to $\epsilon$-greedy policy extracted from Q-network values $Q(S_t,a;\bm{w})$.
* Given state $S_t$ and action $A_t$, obtain reward $R_{t+1}$ and next state $S_{t+1}$ from the environment.
* Append atomic experience $(S_t, A_t, R_{t+1}, S_{t+1})$ in experience-replay memory $\mathcal{D}$.
* Sample a random mini-batch of atomic experiences $(s_i,a_i,r_i,s'_i) \sim \mathcal{D}$.
* Using this mini-batch of atomic experiences, update the Q-network parameters $\bm{w}$ with the Q-learning targets based on "frozen" parameters $\bm{w}^-$ of the target network.
$$\Delta \bm{w} = \alpha \cdot \sum_i (r_i + \gamma \cdot \max_{a'_i} Q(s'_i, a'_i; \bm{w}^-) - Q(s_i,a_i;\bm{w})) \cdot \nabla_{\bm{w}} Q(s_i,a_i;\bm{w})$$
* $S_t \leftarrow S_{t+1}$
* Once every $C$ time steps, set $\bm{w}^- \leftarrow \bm{w}$.

To learn more about the effectiveness of DQN for Atari games, see the [Original DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) [@mnih2013atari] and the [DQN Nature Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) [@mnih2015humanlevel] that DeepMind has published.

Now we are ready to cover Batch RL Control (specifically Least-Squares TD Control), which combines the ideas of Least-Squares TD Prediction and Q-Learning with Experience-Replay.

### Least-Squares Policy Iteration (LSPI)

\index{reinforcement learning!least squares!lspi|(}

Having seen Least-Squares Prediction, the natural question is whether we can extend the Least-Squares (batch with linear function approximation) methodology to solve the Control problem. For On-Policy MC Control and On-Policy TD Control, we take the usual route of Generalized Policy Iteration (GPI) with:

1. Policy Evaluation as Least-Squares $Q$-Value Prediction. Specifically, the $Q$-Value for a policy $\pi$ is approximated as:
$$Q^{\pi}(s,a) \approx Q(s,a;\bm{w}) = \bm{\phi}(s, a)^T \cdot \bm{w} \text{ for all } s \in \mathcal{N}, \text{ for all } a \in \mathcal{A}$$
with a direct linear-algebraic solve for the linear function approximation weights $\bm{w}$ using batch experiences data generated using policy $\pi$.
2. $\epsilon$-Greedy Policy Improvement.

\index{reinforcement learning!off-policy}

In this section, we focus on Off-Policy Control with Least-Squares TD. This algorithm is known as Least-Squares Policy Iteration, abbreviated as LSPI, developed by [Lagoudakis and Parr](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf) [@journals/jmlr/LagoudakisP03]. LSPI has been an important go-to algorithm in the history of RL Control because of its simplicity and effectiveness. The basic idea of LSPI is that it does Generalized Policy Iteration (GPI) in the form of *Q-Learning with Experience-Replay*, with the key being that instead of doing the usual Q-Learning update after each atomic experience, we do *batch Q-Learning* for the Policy Evaluation phase of GPI. We spend the rest of this section describing LSPI in detail and then implementing it in Python code.

The input to LSPI is a fixed finite data set $\mathcal{D}$, consisting of a set of $(s,a,r,s')$ atomic experiences, i.e., a set of `rl.markov_decision_process.TransitionStep` objects, and the task of LSPI is to determine the Optimal Q-Value Function (and hence, Optimal Policy) based on this experiences data set $\mathcal{D}$ using an experience-replayed, batch Q-Learning technique described below. Assume $\mathcal{D}$ consists of $n$ atomic experiences, indexed as $i = 1, 2, \ldots n$, with atomic experience $i$ denoted as $(s_i, a_i, r_i, s'_i)$.

In LSPI, each iteration of GPI involves access to:

* The experiences data set $\mathcal{D}$.
* A *Deterministic Target Policy* (call it $\pi_D$), that is made available from the previous iteration of GPI.

Given $\mathcal{D}$ and $\pi_D$, the goal of each iteration of GPI is to solve for weights $\bm{w}^*$ that minimizes:
\begin{align*}
\mathcal{L}(\bm{w}) & = \sum_{i=1}^n (Q(s_i,a_i; \bm{w}) - (r_i + \gamma \cdot Q(s'_i,\pi_D(s'_i); \bm{w})))^2\\
& = \sum_{i=1}^n (\bm{\phi}(s_i,a_i)^T \cdot \bm{w} - (r_i + \gamma \cdot \bm{\phi}(s'_i, \pi_D(s'_i))^T \cdot \bm{w}))^2
\end{align*}

The solution for the weights $\bm{w}^*$ is attained by setting the semi-gradient of $\mathcal{L}(\bm{w})$ to 0, i.e.,
\begin{equation}
\sum_{i=1}^n \phi(s_i,a_i) \cdot (\bm{\phi}(s_i,a_i)^T \cdot \bm{w}^* - (r_i + \gamma \cdot \bm{\phi}(s'_i, \pi_D(s'_i))^T \cdot \bm{w}^*)) = 0
\label{eq:lspi-loss-semi-gradient}
\end{equation}
We can calculate the solution $\bm{w^}*$ as $\bm{A}^{-1} \cdot \bm{b}$, where the $m \times m$ Matrix $\bm{A}$ is accumulated for each `TransitionStep` $(s_i, a_i, r_i, s'_i)$ as:
$$ \bm{A} \leftarrow \bm{A} + \bm{\phi}(s_i, a_i) \cdot (\bm{\phi}(s_i, a_i) - \gamma \cdot \bm{\phi}(s'_i, \pi_D(s'_i)))^T $$
and the $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(s_i,a_i,r_i,s'_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{\phi}(s_i, a_i) \cdot r_i$$
With Sherman-Morrison incremental inverse, we can reduce the computational complexity from $O(m^3)$ to $O(m^2)$.

\index{Sherman-Morrison incremental inverse}

This solved $\bm{w}^*$ defines an updated $Q$-Value Function as follows:
$$Q(s,a; \bm{w}^*) = \bm{\phi}(s,a)^T \cdot \bm{w}^* = \sum_{j=1}^m \phi_j(s,a) \cdot w_j^*$$
This defines an updated, improved deterministic policy $\pi'_D$ (serving as the *Deterministic Target Policy* for the next iteration of GPI):
$$\pi'_D(s) = \argmax_a Q(s,a; \bm{w}^*) \text{ for all } s \in \mathcal{N}$$

\index{reinforcement learning!least squares!lstdq}

This least-squares solution of $\bm{w}^*$ (Prediction) is known as Least-Squares Temporal Difference for Q-Value, abbreviated as *LSTDQ*. Thus, LSPI is GPI with LSTDQ and greedy policy improvements. Note how LSTDQ in each iteration re-uses the same data $\mathcal{D}$, i.e., LSPI does experience-replay.

We should point out here that the LSPI algorithm we described above should be considered as the *standard variant* of LSPI. However, we can design several other variants of LSPI, in terms of how the experiences data is sourced and used. Firstly, we should note that the experiences data $\mathcal{D}$ essentially provides the behavior policy for Q-Learning (along with the consequent reward and next state transition). In the *standard variant* we described above, since $\mathcal{D}$ is provided from an external source, the behavior policy that generates this data $\mathcal{D}$ must come from an external source. It doesn't have to be this way—we could generate the experiences data from a behavior policy derived from the Q-Value estimates produced by LSTDQ (e.g., $\epsilon$-greedy policy). This would mean the experiences data used in the algorithm is not a fixed, finite data set, rather a variable, incrementally-produced data set. Even if the behavior policy was external, the data set $\mathcal{D}$ might not be a fixed finite data set—rather, it could be made available as an on-demand, variable data stream. Furthermore, in each iteration of GPI, we could use a subset of the experiences data made available until that point of time (rather than the approach of the standard variant of LSPI that uses all of the available experiences data). If we choose to sample a subset of the available experiences data, we might give more sampling-weightage to the more recently generated data. This would especially be the case if the experiences data was being generated from a policy derived from the Q-Value estimates produced by LSTDQ. In this case, we would leverage the `ExperienceReplayMemory` class we'd written earlier.

Next, we write code to implement the *standard variant* of LSPI we described above. First, we write a function to implement LSTDQ. As described above, the inputs to LSTDQ are the experiences data $\mathcal{D}$ (`transitions` in the code below) and a deterministic target policy $\pi_D$ (`target_policy` in the code below). Since we are doing a linear function approximation, the input also includes a set of features, described as functions of state and action (`feature_functions` in the code below). Lastly, the inputs also include the discount factor $\gamma$ and the numerical control parameter $\epsilon$. The code below should be fairly self-explanatory, as it is a straightforward extension of LSTD (implemented in function `least_squares_td` earlier). The key differences are that this is an estimate of the Action-Value (Q-Value) function, rather than the State-Value Function, and the target used in the least-squares calculation is the Q-Learning target (produced by the `target_policy`).

\index{least squares tdq@\texttt{least\_squares\_tdq}}

```python
def least_squares_tdq(
    transitions: Iterable[TransitionStep[S, A]],
    feature_functions: Sequence[Callable[[Tuple[NonTerminal[S], A]], float]],
    target_policy: DeterministicPolicy[S, A],
    gamma: float,
    epsilon: float
) -> LinearFunctionApprox[Tuple[NonTerminal[S], A]]:
    '''transitions is a finite iterable'''
    num_features: int = len(feature_functions)
    a_inv: np.ndarray = np.eye(num_features) / epsilon
    b_vec: np.ndarray = np.zeros(num_features)
    for tr in transitions:
        phi1: np.ndarray = np.array([f((tr.state, tr.action))
                                     for f in feature_functions])
        if isinstance(tr.next_state, NonTerminal):
            phi2 = phi1 - gamma * np.array([
                f((tr.next_state, target_policy.action_for(tr.next_state.state)))
                for f in feature_functions])
        else:
            phi2 = phi1
        temp: np.ndarray = a_inv.T.dot(phi2)
        a_inv = a_inv - np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
        b_vec += phi1 * tr.reward

    opt_wts: np.ndarray = a_inv.dot(b_vec)
    return LinearFunctionApprox.create(
        feature_functions=feature_functions,
        weights=Weights.create(opt_wts)
    )
```

Now we are ready to write the standard variant of LSPI. The code below is a straightforward implementation of our description above, looping through the iterations of GPI, `yield`ing the Q-Value `LinearFunctionApprox` after each iteration of GPI.

\index{least squares policy iteration@\texttt{least\_squares\_policy\_iteration}}

```python
def least_squares_policy_iteration(
    transitions: Iterable[TransitionStep[S, A]],
    actions: Callable[[NonTerminal[S]], Iterable[A]],
    feature_functions: Sequence[Callable[[Tuple[NonTerminal[S], A]], float]],
    initial_target_policy: DeterministicPolicy[S, A],
    gamma: float,
    epsilon: float
) -> Iterator[LinearFunctionApprox[Tuple[NonTerminal[S], A]]]:
    '''transitions is a finite iterable'''
    target_policy: DeterministicPolicy[S, A] = initial_target_policy
    transitions_seq: Sequence[TransitionStep[S, A]] = list(transitions)
    while True:
        q: LinearFunctionApprox[Tuple[NonTerminal[S], A]] = \
            least_squares_tdq(
                transitions=transitions_seq,
                feature_functions=feature_functions,
                target_policy=target_policy,
                gamma=gamma,
                epsilon=epsilon,
            )
        target_policy = greedy_policy_from_qvf(q, actions)
        yield q
```

The above code is in the file [rl/td.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/td.py).

\index{reinforcement learning!least squares!lspi|)}

#### Saving Your Village from a Vampire   

Now we consider a Control problem we'd like to test the above LSPI algorithm on. We call it the Vampire problem that can be described as a good old-fashioned bedtime story, as follows:

> *A village is visited by a vampire every morning who uniform-randomly eats 1 villager upon entering the village, then retreats to the hills, planning to come back the next morning. The villagers come up with a plan. They will poison a certain number of villagers each night until the vampire eats a poisoned villager the next morning, after which the vampire dies immediately (due to the poison in the villager the vampire ate). Unfortunately, all villagers who get poisoned also die the day after they are given the poison. If the goal of the villagers is to maximize the expected number of villagers at termination (termination is when either the vampire dies or all villagers die), what should be the optimal poisoning strategy? In other words, if there are $n$ villagers on any day, how many villagers should be poisoned (as a function of $n$)?*

It is straightforward to model this problem as an MDP. The *State* is the number of villagers at risk on any given night (if the vampire is still alive, the *State* is the number of villagers and if the vampire is dead, the *State* is 0, which is the only *Terminal State*). The *Action* is the number of villagers poisoned on any given night. The *Reward* is zero as long as the vampire is alive, and is equal to the number of villagers remaining if the vampire dies. Let us refer to the initial number of villagers as $I$. Thus,

$$\mathcal{S} = \{0, 1, \ldots, I\}, \mathcal{T} = \{0\}$$
$$\mathcal{A}(s) = \{0, 1, \ldots, s - 1\} \text{ where } s \in \mathcal{N}$$
For all $s \in \mathcal{N}$, for all $a \in \mathcal{A}(s)$,
$$
 \mathcal{P}_R(s, a, r, s') =
 \begin{cases}
 \frac {s-a} {s} & \text{ if } r = 0 \text{ and } s' = s - a - 1\\
 \frac a s & \text{ if } r = s - a \text{ and } s' = 0\\
 0 & \text{ otherwise }
 \end{cases}
$$

It is rather straightforward to solve this with Dynamic Programming (say, Value Iteration) since we know the transition probabilities and rewards function and since the state and action spaces are finite. However, in a situation where we don't know the exact probabilities with which the vampire operates, and we only had access to observations on specific days, we can attempt to solve this problem with Reinforcement Learning (assuming we had access to observations of many vampires operating on many villages). In any case, our goal here is to test LSPI using this vampire problem as an example. So we write some code to first model this MDP as described above, solve it with value iteration (to obtain the benchmark, i.e., true Optimal Value Function and true Optimal Policy to compare against), then generate atomic experiences data from the MDP, and then solve this problem with LSPI using this stream of generated atomic experiences.

\index{VampireMDP@\texttt{VampireMDP}}

```python
from rl.markov_decision_process import TransitionStep
from rl.distribution import Categorical, Choose
from rl.function_approx import LinearFunctionApprox
from rl.policy import DeterministicPolicy, FiniteDeterministicPolicy
from rl.dynamic_programming import value_iteration_result, V
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.td import least_squares_policy_iteration
from numpy.polynomial.laguerre import lagval
import itertools
import rl.iterate as iterate
import numpy as np

class VampireMDP(FiniteMarkovDecisionProcess[int, int]):

    initial_villagers: int

    def __init__(self, initial_villagers: int):
        self.initial_villagers = initial_villagers
        super().__init__(self.mdp_map())

    def mdp_map(self) -> \
            Mapping[int, Mapping[int, Categorical[Tuple[int, float]]]]:
        return {s: {a: Categorical(
            {(s - a - 1, 0.): 1 - a / s, (0, float(s - a)): a / s}
        ) for a in range(s)} for s in range(1, self.initial_villagers + 1)}

    def vi_vf_and_policy(self) -> \
            Tuple[V[int], FiniteDeterministicPolicy[int, int]]:
        return value_iteration_result(self, 1.0)

    def lspi_features(
        self,
        factor1_features: int,
        factor2_features: int
    ) -> Sequence[Callable[[Tuple[NonTerminal[int], int]], float]]:
        ret: List[Callable[[Tuple[NonTerminal[int], int]], float]] = []
        ident1: np.ndarray = np.eye(factor1_features)
        ident2: np.ndarray = np.eye(factor2_features)
        for i in range(factor1_features):
            def factor1_ff(x: Tuple[NonTerminal[int], int], i=i) -> float:
                return lagval(
                    float((x[0].state - x[1]) ** 2 / x[0].state),
                    ident1[i]
                )
            ret.append(factor1_ff)
        for j in range(factor2_features):
            def factor2_ff(x: Tuple[NonTerminal[int], int], j=j) -> float:
                return lagval(
                    float((x[0].state - x[1]) * x[1] / x[0].state),
                    ident2[j]
                )
            ret.append(factor2_ff)
        return ret

    def lspi_transitions(self) -> Iterator[TransitionStep[int, int]]:
        states_distribution: Choose[NonTerminal[int]] = \
            Choose(self.non_terminal_states)
        while True:
            state: NonTerminal[int] = states_distribution.sample()
            action: int = Choose(range(state.state)). sample()
            next_state, reward = self.step(state, action).sample()
            transition: TransitionStep[int, int] = TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            )
            yield transition

    def lspi_vf_and_policy(self) -> \
            Tuple[V[int], FiniteDeterministicPolicy[int, int]]:
        transitions: Iterable[TransitionStep[int, int]] = itertools.islice(
            self.lspi_transitions(),
            20000
        )
        qvf_iter: Iterator[LinearFunctionApprox[Tuple[
            NonTerminal[int], int]]] = least_squares_policy_iteration(
                transitions=transitions,
                actions=self.actions,
                feature_functions=self.lspi_features(4, 4),
                initial_target_policy=DeterministicPolicy(
                    lambda s: int(s / 2)
                ),
                gamma=1.0,
                epsilon=1e-5
            )
        qvf: LinearFunctionApprox[Tuple[NonTerminal[int], int]] = \
            iterate.last(
                itertools.islice(
                    qvf_iter,
                    20
                )
            )
        return get_vf_and_policy_from_qvf(self, qvf)
```

The above code should be self-explanatory. The main challenge with LSPI is that we need to construct features function of the state and action such that the Q-Value Function is linear in those features. In this case, since we simply want to test the correctness of our LSPI implementation, we define feature functions (in method `lspi_feature` above) based on our knowledge of the true optimal Q-Value Function from the Dynamic Programming solution. The atomic experiences comprising the experiences data $\mathcal{D}$ for LSPI to use is generated with a uniform distribution of non-terminal states and a uniform distribution of actions for a given state (in method `lspi_transitions` above).

Figure \ref{fig:lspi_opt_vf_comparison} shows the plot of the True Optimal Value Function (from Value Iteration) versus the LSPI-estimated Optimal Value Function. 

![True versus LSPI Optimal Value Function \label{fig:lspi_opt_vf_comparison}](./chapter12/vampire_lspi_opt_vf.png "True versus LSPI Optimal Value Function"){height=7cm}

Figure \ref{fig:lspi_opt_policy_comparison} shows the plot of the True Optimal Policy (from Value Iteration) versus the LSPI-estimated Optimal Policy. 

![True versus LSPI Optimal Policy \label{fig:lspi_opt_policy_comparison}](./chapter12/vampire_lspi_opt_policy.png "True versus LSPI Optimal Policy"){height=7cm}

The above code is in the file [rl/chapter12/vampire.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter12/vampire.py). As ever, we encourage you to modify some of the parameters in this code (including choices of feature functions, nature and number of atomic transitions used, number of GPI iterations, choice of $\epsilon$, and perhaps even a different dynamic for the vampire behavior), and see how the results change.

#### Least-Squares Control Convergence

\index{reinforcement learning!convergence}

We wrap up this section by including the convergence behavior of LSPI in the summary table for convergence of RL Control algorithms (that we had displayed at the end of Chapter [-@sec:rl-control-chapter]). Figure \ref{fig:rl_control_with_lspi_convergence} shows the updated summary table for convergence of RL Control algorithms to now also include LSPI. Note that \(\cmark) means it doesn't quite hit the Optimal Value Function, but bounces around near the Optimal Value Function. But this is better than Q-Learning in the case of linear function approximation.

\begin{figure}
\begin{center}
\begin{tabular}{cccc}
\hline
Algorithm & Tabular & Linear & Non-Linear \\ \hline
MC Control & \cmark & ( \cmark ) & \xmark \\
SARSA & \cmark & ( \cmark ) & \xmark \\
Q-Learning & \cmark & \xmark & \xmark \\
\bfseries LSPI & \cmark & ( \cmark ) & - \\ 
Gradient Q-Learning & \cmark & \cmark & \xmark \\ \hline
\end{tabular}
\end{center}
\caption{Convergence of RL Control Algorithms}
\label{fig:rl_control_with_lspi_convergence}
\end{figure}


### RL for Optimal Exercise of American Options

\index{finance!derivative!american option}
\index{finance!derivative!optimal exercise|(}

We learnt in Chapter [-@sec:derivatives-pricing-chapter] that the American Options Pricing problem is an Optimal Stopping problem and can be modeled as an MDP so that solving the Control problem of the MDP gives us the fair price of an American Option. We can solve it with Dynamic Programming or Reinforcement Learning, as appropriate. 

\index{finance!derivative!underlying}
In the financial trading industry, it has traditionally not been a common practice to explicitly view the American Options Pricing problem as an MDP. Specialized algorithms have been developed to price American Options. We now provide a quick overview of the common practice in pricing American Options in the financial trading industry. Firstly, we should note that the price of some American Options is equal to the price of the corresponding European Option, for which we have a closed-form solution under the assumption of a lognormal process for the underlying—this is the case for a plain-vanilla American call option whose price (as we proved in Chapter [-@sec:derivatives-pricing-chapter]) is equal to the price of a plain-vanilla European call option. However, this is not the case for a plain-vanilla American put option. Secondly, we should note that if the payoff of an American option is dependent on only the current price of the underlying (and not on the past prices of the underlying)—in which case, we say that the option payoff is not "history-dependent"—and if the dimension of the state space is not large, then we can do a simple backward induction on a binomial tree (as we showed in Chapter [-@sec:derivatives-pricing-chapter]). In practice, a more detailed data structure such as a [trinomial tree](https://en.wikipedia.org/wiki/Trinomial_tree) or a lattice is often used for more accurate backward-induction calculations. However, if the payoff is history-dependent (i.e., payoff depends on past prices of the underlying) or if the payoff depends on the prices of several underlying assets, then the state space is too large for backward induction to handle. In such cases, the standard approach in the financial trading industry is to use the [Longstaff-Schwartz pricing algorithm](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf) [@LongstaffSchwartz2001]. We won't cover the Longstaff-Schwartz pricing algorithm in detail in this book—it suffices to share here that the Longstaff-Schwartz pricing algorithm combines 3 ideas:

* The Pricing is based on a set of sampling traces of the underlying prices.
* Function approximation of the continuation value for in-the-money states.
* Backward-recursive determination of early exercise states.

The goal of this section is to explain how to price American Options with Reinforcement Learning, as an alternative to the Longstaff-Schwartz algorithm.

#### LSPI for American Options Pricing

\index{reinforcement learning!least squares!lspi}

[A paper by Li, Szepesvari, Schuurmans](http://proceedings.mlr.press/v5/li09d/li09d.pdf) [@li2009] showed that LSPI can be an attractive alternative to the Longstaff-Schwartz algorithm in pricing American Options. Before we dive into the details of pricing American Options with LSPI, let's review the MDP model for American Options Pricing.

\index{Markov decision process!state}
\index{Markov decision process!action}
\index{Markov decision process!reward}

* *State* is [Current Time, Relevant History of Underlying Security Prices].
* *Action* is Boolean: Exercise (i.e., Stop) or Continue.
* *Reward* is always 0, except upon Exercise (when the *Reward* is equal to the Payoff).
* *State*-transitions are based on the Underlying Securities' Risk-Neutral Process.

\index{function approximation!linear}
The key is to create a linear function approximation of the state-conditioned *continuation value* of the American Option (*continuation value* is the price of the American Option at the current state, conditional on not exercising the option at the current state, i.e., continuing to hold the option). Knowing the continuation value in any state enables us to compare the continuation value against the exercise value (i.e., payoff), thus providing us with the Optimal Stopping criteria (as a function of the state), which in turn enables us to determine the Price of the American Option. Furthermore, we can customize the LSPI algorithm to the nuances of the American Option Pricing problem, yielding a specialized version of LSPI. The key customization comes from the fact that there are only two actions. The action to exercise produces a (state-conditioned) reward (i.e., option payoff) and transition to a terminal state. The action to continue produces no reward and transitions to a new state at the next time step. Let us refer to these 2 actions as: $a=c$ (continue the option) and $a=e$ (exercise the option).

Since we know the exercise value in any state, we only need to create a linear function approximation for the continuation value, i.e., for the Q-Value $Q(s, c)$ for all non-terminal states $s$. If we denote the payoff in non-terminal state $s$ as $g(s)$, then $Q(s,e) = g(s)$. So we write
$$
\hat{Q}(s,a; \bm{w}) =
\begin{cases}
\bm{\phi}(s)^T \cdot \bm{w} & \text{ if } a = c \\
g(s) & \text{ if } a = e
\end{cases}
\text{ for all } s \in \mathcal{N}
$$
for feature functions $\bm{\phi}(\cdot) = [\phi_i(\cdot)|i = 1, \ldots, m]$, which are feature functions of only state (and not action).

Each iteration of GPI in the LSPI algorithm starts with a deterministic target policy $\pi_D(\cdot)$ that is made available as a greedy policy derived from the previous iteration's LSTDQ-solved $\hat{Q}(s;a;\bm{w}^*)$. The LSTDQ solution for $\bm{w}^*$ is based on pre-generated training data and with the Q-Learning target policy set to be $\pi_D$. Since we learn the Q-Value function for only $a=c$, the behavior policy $\mu$ generating experiences data for training is a constant function $\mu(s) = c$. Note also that for American Options, the reward for $a=c$ is 0. So each atomic experience for training is of the form $(s,c,0,s')$. This means we can represent each atomic experience for training as a 2-tuple $(s,s')$. This reduces the LSPI Semi-Gradient Equation \eqref{eq:lspi-loss-semi-gradient} to:
\begin{equation}
\sum_i \bm{\phi}(s_i) \cdot (\bm{\phi}(s_i)^T \cdot \bm{w}^* - \gamma \cdot \hat{Q}(s'_i, \pi_D(s'_i); \bm{w}^*)) = 0
\label{eq:customized-lspi-loss-semi-gradient}
\end{equation}
We need to consider two cases for the term $\hat{Q}(s'_i, \pi_D(s'_i); \bm{w}^*)$:

* $C1$: If $s'_i$ is non-terminal and $\pi_D(s'_i) = c$ (i.e., $\bm{\phi}(s'_i)^T \cdot \bm{w} \geq g(s'_i)$):
Substitute $\bm{\phi}(s'_i)^T \cdot \bm{w}^*$ for $\hat{Q}(s'_i,\pi_D(s'_i); \bm{w}^*)$ in Equation \eqref{eq:customized-lspi-loss-semi-gradient}
* $C2$: If $s'_i$ is a terminal state or $\pi_D(s'_i) = e$ (i.e., $g(s'_i) > \bm{\phi}(s'_i)^T \cdot \bm{w}$):
Substitute $g(s'_i)$ for $\hat{Q}(s'_i,\pi_D(s'_i); \bm{w}^*)$ in Equation \eqref{eq:customized-lspi-loss-semi-gradient}

So we can rewrite Equation \eqref{eq:customized-lspi-loss-semi-gradient} using indicator notation $\mathbb{I}$ for cases $C1, C2$ as:
$$\sum_i \bm{\phi}(s_i) \cdot (\bm{\phi}(s_i)^T \cdot \bm{w}^* - \mathbb{I}_{C1} \cdot \gamma \cdot \bm{\phi}(s'_i)^T \cdot \bm{w}^*  -  \mathbb{I}_{C2} \cdot \gamma \cdot g(s'_i)) = 0$$

Factoring out $\bm{w}^*$, we get:

$$(\sum_i \bm{\phi}(s_i) \cdot (\bm{\phi}(s_i) - \mathbb{I}_{C1} \cdot \gamma \cdot \bm{\phi}(s'_i))^T) \cdot \bm{w}^*= \gamma \cdot \sum_i  \mathbb{I}_{C2} \cdot \bm{\phi}(s_i) \cdot g(s'_i)$$

This can be written in the familiar vector-matrix notation as: $\bm{A} \cdot \bm{w}^* = \bm{b}$

$$\bm{A} = \sum_i \bm{\phi}(s_i) \cdot (\bm{\phi}(s_i) - \mathbb{I}_{C1} \cdot \gamma \cdot \bm{\phi}(s'_i))^T$$
$$\bm{b} = \gamma \cdot \sum_i \mathbb{I}_{C2} \cdot \bm{\phi}(s_i) \cdot  g(s'_i)$$

The $m \times m$ Matrix $\bm{A}$ is accumulated at each atomic experience $(s_i,s'_i)$ as:

$$\bm{A} \leftarrow \bm{A} + \bm{\phi}(s_i) \cdot (\bm{\phi}(s_i) -  \mathbb{I}_{C1} \cdot \gamma \cdot \bm{\phi}(s'_i))^T$$

The $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(s_i, s'_i)$ as:

$$\bm{b} \leftarrow \bm{b} + \gamma  \cdot \mathbb{I}_{C2} \cdot \bm{\phi}(s_i) \cdot g(s'_i)$$

With Sherman-Morrison incremental inverse of $\bm{A}$, we can reduce the time-complexity from $O(m^3)$ to $O(m^2)$.

\index{Sherman-Morrison incremental inverse}

This solved $\bm{w}^*$ updates the $Q$-Value Function Approximation to $\hat{Q}(s,a;\bm{w}^*)$. This defines an updated, improved deterministic policy $\pi'_D$ (serving as the *Deterministic Target Policy* for the next iteration of GPI):
$$\pi'_D(s) = \argmax_a \hat{Q}(s,a; \bm{w}^*) \text{ for all } s \in \mathcal{N}$$

[Li, Szepesvari, Schuurmans](http://proceedings.mlr.press/v5/li09d/li09d.pdf) [@li2009] recommend in their paper to use 7 feature functions, the first 4 Laguerre polynomials that are functions of the underlying price and 3 functions of time. Precisely, the feature functions they recommend are:
\index{finance!derivative!underlying}
\index{function approximation!feature functions}

* $\phi_0(S_t) = 1$
* $\phi_1(S_t) = e^{-\frac {M_t} 2}$
* $\phi_2(S_t) = e^{-\frac{M_t} 2} \cdot (1-M_t)$
* $\phi_3(S_t) = e^{-\frac{M_t} 2} \cdot (1-2M_t+M_t^2/2)$
* $\phi_0^{(t)}(t) = sin(\frac {\pi(T-t)} {2T})$
* $\phi_1^{(t)}(t) = \log(T-t)$
* $\phi_2^{(t)}(t) = (\frac t T)^2$

where $M_t = \frac {S_t} {K}$ ($S_t$ is the current underlying price and $K$ is the American Option strike), $t$ is the current time, and $T$ is the expiration time (i.e., $0 \leq t < T$).

#### Deep Q-Learning for American Options Pricing

\index{reinforcement learning!temporal difference!q learning}
\index{reinforcement learning!deep reinforcement learning}

LSPI is data-efficient and compute-efficient, but linearity is a limitation in the function approximation. The alternative is (incremental) Q-Learning with neural network function approximation, which we cover in this subsection. We employ the same set up as LSPI (including Experience Replay)—specifically, the function approximation is required only for continuation value. Precisely,

$$
\hat{Q}(s,a; \bm{w}) =
\begin{cases}
f(s;\bm{w}) & \text{ if } a = c \\
g(s) & \text{ if } a = e
\end{cases}
\text{ for all } s \in \mathcal{N}
$$
where $f(s; \bm{w})$ is the deep neural network function approximation.

The Q-Learning update for each atomic experience $(s_i,s'_i)$ is:
$$\Delta \bm{w} = \alpha \cdot (\gamma \cdot \hat{Q}(s'_i, \pi(s'_i); \bm{w}) - f(s_i;\bm{w})) \cdot \nabla_{\bm{w}} f(s_i;\bm{w})$$

When $s'_i$ is a non-terminal state, the update is:
$$\Delta \bm{w} =  \alpha \cdot (\gamma \cdot \max(g(s'_i), f(s'_i;\bm{w})) - f(s_i;\bm{w})) \cdot \nabla_{\bm{w}} f(s_i;\bm{w})$$
When $s'_i$ is a terminal state, the update is:
$$\Delta \bm{w} = \alpha \cdot (\gamma \cdot g(s'_i) - f(s_i;\bm{w})) \cdot \nabla_{\bm{w}} f(s_i;\bm{w})$$

\index{finance!derivative!optimal exercise|)}

### Value Function Geometry

\index{value function!value function geometry|(}
\index{value function!value function as vector}
\index{reinforcement learning!deadly triad}

Now we look deeper into the issue of the *Deadly Triad* (that we had alluded to in Chapter [-@sec:rl-control-chapter]) by viewing Value Functions as Vectors (we had done this in Chapter [-@sec:dp-chapter]), understand Value Function Vector transformations with a balance of geometric intuition and mathematical rigor, providing insights into convergence issues for a variety of traditional loss functions used to develop RL algorithms. As ever, the best way to understand Vector transformations is to visualize it and so, we loosely refer to this topic as Value Function Geometry. The geometric intuition is particularly useful for linear function approximations. To promote intuition, we shall present this content for linear function approximations of the Value Function and stick to Prediction (rather than Control) although many of the concepts covered in this section are well-extensible to non-linear function approximations and to the Control problem.

This treatment was originally presented in [the LSPI paper by Lagoudakis and Parr](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf) [@journals/jmlr/LagoudakisP03] and has been covered in detail in the [RL book by Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html) [@Sutton1998]. This treatment of Value Functions as Vectors leads us in the direction of overcoming the Deadly Triad by defining an appropriate loss function, calculating whose gradient provides a more robust set of RL algorithms known as Gradient Temporal-Difference (abbreviated, as Gradient TD), which we shall cover in the next section. 

Along with visual intuition, it is important to write precise notation for Value Function transformations and approximations. So we start with a set of formal definitions, keeping the setting fairly simple and basic for ease of understanding.

#### Notation and Definitions

\index{Markov decision process!finite}
\index{Markov decision process!state space}
\index{Markov decision process!action space}
\index{Markov decision process!prediction}

Assume our state space is finite without any terminal states, i.e. $\mathcal{S} = \mathcal{N} = \{s_1, s_2, \ldots, s_n\}$. Assume our action space $\mathcal{A}$ consists of a finite number of actions. This coverage can be extended to infinite/continuous spaces, but we shall stick to this simple setting in this section. Also, as mentioned above, we restrict this coverage to the case of a fixed (potentially stochastic) policy denoted as $\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$. This means we are restricting to the case of the Prediction problem (although it's possible to extend some of this coverage to the case of Control).

\index{value function!value function for fixed policy}
We denote the Value Function for a policy $\pi$ as $\bvpi: \mathcal{S} \rightarrow \mathbb{R}$. Consider the $n$-dimensional vector space $\mathbb{R}^n$, with each dimension corresponding to a state in $\mathcal{S}$.  Think of a Value Function (typically denoted $\bv$): $\mathcal{S} \rightarrow \mathbb{R}$ as a vector in the $\mathbb{R}^n$ vector space. Each dimension's coordinate is the evaluation of the Value Function for that dimension's state. The coordinates of vector $\bvpi$ for policy $\pi$ are: $[\bvpi(s_1), \bvpi(s_2), \ldots, \bvpi(s_n)]$. Note that this treatment is the same as the treatment in our coverage of Dynamic Programming in Chapter [-@sec:dp-chapter].

\index{function approximation!feature functions}
\index{function approximation!linear}
Our interest is in identifying an appropriate function approximation of the Value Function $\bvpi$. For the function approximation, assume there are $m$ feature functions $\phi_1, \phi_2, \ldots, \phi_m : \mathcal{S} \rightarrow \mathbb{R}$, with $\bm{\phi}(s) \in \mathbb{R}^m$ denoting the feature vector for any state $s \in \mathcal{S}$. To keep things simple and to promote understanding of the concepts, we limit ourselves to linear function approximations. For linear function approximation of the Value Function with weights $\bw = (w_1, w_2, \ldots, w_m)$,  we use the notation $\bvw: \mathcal{S} \rightarrow \mathbb{R}$, defined as:
$$\bvw(s) = \bm{\phi}(s)^T \cdot \bw = \sum_{j=1}^m \phi_j(s) \cdot w_j \mbox{ for all } s \in \mathcal{S}$$. 

Assuming independence of the feature functions, the $m$ feature functions give us $m$ independent vectors in the vector space $\mathbb{R}^n$. Feature function $\phi_j$ gives us the vector $[\phi_j(s_1), \phi_j(s_2), \ldots, \phi_j(s_n)] \in \mathbb{R}^n$. These $m$ vectors are the $m$ columns of the $n \times m$ matrix $\bphi = [\phi_j(s_i)], 1 \leq i \leq n, 1 \leq j \leq m$. The span of these $m$ independent vectors is an $m$-dimensional vector subspace within this $n$-dimensional vector space, spanned by the set of all $\bw = (w_1, w_2, \ldots, w_m) \in \mathbb{R}^m$. The vector $\bvw = \bphi \cdot \bw$ in this vector subspace has coordinates $[\bvw(s_1), \bvw(s_2), \ldots , \bvw(s_n)]$. The vector $\bvw$ is fully specified by $\bw$ (so we often say $\bw$ to mean $\bvw$). Our interest is in identifying an appropriate $\bw \in \mathbb{R}^m$ that represents an adequate linear function approximation $\bvw = \bphi \cdot \bw$ of the Value Function $\bvpi$.

We denote the probability distribution of occurrence of states under policy $\pi$ as $\bmu : \mathcal{S} \rightarrow [0, 1]$.
In accordance with the notation we used in Chapter [-@sec:mdp-chapter], $\mathcal{R}(s,a)$ refers to the Expected Reward upon taking action $a$ in state $s$, and $\mathcal{P}(s,a,s')$ refers to the probability of transition from state $s$ to state $s'$ upon taking action $a$. Define
$$\brew(s) = \sum_{a \in \mathcal{A}} \pi(s, a) \cdot \mathcal{R}(s,a) \text{ for all } s \in \mathcal{S}$$
$$\bprob(s,s') = \sum_{a \in \mathcal{A}} \pi(s, a) \cdot \mathcal{P}(s,a,s') \text{ for all } s, s' in \mathcal{S}$$
to denote the Expected Reward and state transition probabilities respectively of the $\pi$-implied MRP.

\index{Markov process!transition probability function}
\index{Markov reward process!reward function}

$\brew$ refers to vector $[\brew(s_1), \brew(s_2), \ldots, \brew(s_n)]$ and $\bprob$ refers to matrix $[\bprob(s_i, s_{i'})], 1 \leq i, i' \leq n$. Denote $\gamma < 1$ (since there are no terminal states) as the MDP discount factor.

#### Bellman Policy Operator and Projection Operator

\index{Bellman policy operator}

In Chapter [-@sec:mdp-chapter], we introduced the Bellman Policy Operator $\bb$ for policy $\pi$ operating on any Value Function vector $\bv$. As a reminder,
$$\bb (\bv) = \bm{\mathcal{R}}^{\pi} + \gamma \bm{\mathcal{P}}^{\pi} \cdot \bv \text{ for any VF vector } \bv \in \mathbb{R}^n$$
Note that $\bb$ is an [affine transformation](https://en.wikipedia.org/wiki/Affine_transformation) on vectors in $\mathbb{R}^n$. We lighten notation for application of the $\bb$ operator on any vector $\bv \in \mathbb{R}^n$ as simply $\bb \cdot \bv$ (with $\cdot$ conveying the notion of operator application). We've learnt in Chapter [-@sec:mdp-chapter] that $\bvpi$ is the fixed point of $\bb$. Therefore, we can write:
$$\bb \cdot \bvpi = \bvpi$$
\index{fixed-point theory!Banach fixed-point theorem}
This means, if we start with an arbitrary Value Function vector $\bv$ and repeatedly apply $\bb$, by Banach Fixed-Point Theorem \ref{th:banach_fixed_point_theorem}, we will reach the fixed point $\bvpi$. We've learnt in Chapter [-@sec:mdp-chapter] that this is, in fact, the Dynamic Programming Policy Evaluation algorithm. Note that Tabular Monte Carlo also converges to $\bvpi$ (albeit slowly).

\index{dynamic programming!policy evaluation}
\index{projection operator}

Next, we introduce the Projection Operator $\bpi$ for the subspace spanned by the column vectors (feature functions) of $\bphi$. We define $\bpi (\bv)$ as the vector in the subspace spanned by the column vectors of $\bphi$ that represents the orthogonal projection of Value Function vector $\bv$ on the $\bphi$ subspace. To make this precise, we first define "distance" $d(\bm{V_1}, \bm{V_2})$ between Value Function vectors $\bm{V_1}, \bm{V_2}$, weighted by $\bmu$ across the $n$ dimensions of $\bm{V_1}, \bm{V_2}$. Specifically,
$$d(\bm{V_1}, \bm{V_2}) = \sum_{i=1}^n \bmu(s_i) \cdot  (\bm{V_1}(s_i) - \bm{V_2}(s_i))^2 =  (\bm{V_1} - \bm{V_2})^T \cdot \bd \cdot (\bm{V_1} - \bm{V_2})$$
where $\bd$ is the square diagonal matrix consisting of the diagonal elements $\bmu(s_i), 1 \leq i \leq n$.

With this "distance" metric, we define $\bpi (\bv)$ as the Value Function vector in the subspace spanned by the column vectors of $\bphi$ that is given by $\argmin_{\bw} d(\bv, \bvw)$. This is a weighted least squares regression with solution:
$$\bw^* = (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd \cdot \bv$$
Since $\bpi (\bv) = \bphi \cdot \bw^*$, we henceforth denote and treat Projection Operator $\bpi$ as the following $n \times n$ matrix:
$$\bpi = \bphi \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd$$

#### Vectors of Interest in the $\bphi$ Subspace

In this section, we cover 4 Value Function vectors of interest in the $\bphi$ subspace, as candidate linear function approximations of the Value Function $\bvpi$. To lighten notation, we will refer to the $\bphi$-subspace Value Function vectors by their corresponding weights $\bw$. All 4 of these Value Function vectors are depicted in Figure \ref{fig:vf_geometry}, an image we are borrowing from [Sutton and Barto's RL book](http://incompleteideas.net/book/the-book-2nd.html) [@Sutton1998]. We spend the rest of this section going over these 4 Value Function vectors in detail.

![Value Function Geometry (Image Credit: Sutton-Barto's RL Book) \label{fig:vf_geometry}](./chapter12/vf_geometry.pdf "Value Function Geometry (Image Credit: Sutton-Barto's RL Book)"){height=8cm}

The first Value Function vector of interest in the $\bphi$ subspace is the Projection $\bpi \cdot \bvpi$, denoted as $\bm{w}_{\pi} = \argmin_{\bw} d(\bvpi, \bvw)$. This is the linear function approximation of the Value Function $\bvpi$ we seek because it is the Value Function vector in the $\bphi$ subspace that is "closest" to $\bvpi$. Monte-Carlo with linear function approximation will (slowly) converge to $\bw_{\pi}$. Figure \ref{fig:vf_geometry} provides the visualization. We've learnt that Monte-Carlo can be slow to converge, so we seek function approximations in the $\bphi$ subspace that are based on Temporal-Difference (TD), i.e., bootstrapped methods. The remaining three Value Function vectors in the $\bphi$ subspace are based on TD methods.

\index{Bellman error|textbf}
\index{Bellman policy operator}

We denote the second Value Function vector of interest in the $\bphi$ subspace as $\bm{w}_{BE}$. 
The acronym $BE$ stands for *Bellman Error*. To understand this, consider the application of the Bellman Policy Operator $\bb$ on a Value Function vector $\bvw$ in the $\bphi$ subspace. Applying $\bb$ on $\bvw$ typically throws $\bvw$ out of the $\bphi$ subspace. The idea is to find a Value Function vector $\bvw$ in the $\bphi$ subspace such that the "distance" between $\bvw$ and $\bb \cdot \bvw$ is minimized, i.e. we minimize the "error vector" $BE = \bb \cdot \bvw - \bvw$ (Figure \ref{fig:vf_geometry} provides the visualization). Hence, we say we are minimizing the *Bellman Error* (or simply that we are minimizing $BE$), and we refer to $w_{BE}$ as the Value Function vector in the $\bphi$ subspace for which $BE$ is minimized.  Formally, we define it as:
\begin{align*}
\bm{w}_{BE} & = \argmin_{\bw} d(\bb \cdot \bvw, \bvw) \\
& = \argmin_{\bw} d(\bvw, \brew + \gamma \bprob \cdot \bvw) \\
& = \argmin_{\bw} d(\bphi \cdot \bw, \brew + \gamma \bprob \cdot \bphi \cdot \bw)\\
& = \argmin_{\bw} d(\bphi \cdot \bw - \gamma \bprob \cdot \bphi \cdot \bw, \brew)\\
& = \argmin_{\bw} d((\bphi - \gamma \bprob \cdot \bphi) \cdot \bw, \brew )\\
\end{align*}
This is a weighted least-squares linear regression of $\brew$ against $\bphi - \gamma \bprob \cdot \bphi$
with weights $\bmu$, whose solution is:
$$\bm{w}_{BE} = ((\bphi - \gamma \bprob \cdot \bphi)^T \cdot \bd \cdot (\bphi - \gamma \bprob \cdot \bphi))^{-1} \cdot (\bphi - \gamma \bprob \cdot \bphi)^T \cdot \bd \cdot \brew$$

\index{linear regression}

The above formulation can be used to compute $\bm{w}_{BE}$ if we know the model probabilities $\bprob$ and reward function $\brew$. But often, in practice, we don't know $\bprob$ and $\brew$, in which case we seek model-free learning of $\bm{w}_{BE}$, specifically with a TD (bootstrapped) algorithm.

Let us refer to
$$(\bphi - \gamma \bprob \cdot \bphi)^T \cdot \bd \cdot (\bphi - \gamma \bprob \cdot \bphi)$$
as matrix $\bm{A}$ and let us refer to
$$(\bphi - \gamma \bprob \cdot \bphi)^T \cdot \bd \cdot \brew$$ as vector $\bm{b}$ so that $\bm{w}_{BE} = \bm{A}^{-1} \cdot \bm{b}$.

Following policy $\pi$, each time we perform an individual transition from $s$ to $s'$ getting reward $r$, we get a sample estimate of $\bm{A}$ and $\bm{b}$. The sample estimate of $\bm{A}$ is the outer-product of vector $\bm{\phi}(s) - \gamma \cdot \bm{\phi}(s')$ with itself. The sample estimate of $\bm{b}$ is scalar $r$ times vector $\bm{\phi}(s) - \gamma \cdot \bm{\phi}(s')$. We average these sample estimates across many such individual transitions. However, this requires $m$ (the number of features) to be not too large.

\index{outer product}

\index{function approximation!non-linear}
\index{reinforcement learning!off-policy}

If $m$ is large or if we are doing non-linear function approximation or off-policy, then we seek a gradient-based TD algorithm. We defined $\bm{w}_{BE}$ as the vector in the $\bphi$ subspace for which the Bellman Error is minimized. But Bellman Error for a state is the expectation of the TD error $\delta$ for that state when following policy $\pi$. So we want to do Stochastic Gradient Descent with the gradient of the square of expected TD error, as follows:
\begin{align*}
\Delta \bw & = - \alpha \cdot \frac{1}{2} \cdot \nabla_{\bw} (\mathbb{E}_{\pi}[\delta])^2\\
& = - \alpha \cdot \mathbb{E}_{\pi}[r + \gamma \cdot \bm{\phi}(s')^T \cdot \bw - \bm{\phi}(s)^T \cdot \bw] \cdot \nabla_{\bw} \mathbb{E}_{\pi}[\delta]\\
& = \alpha \cdot (\mathbb{E}_{\pi}[r + \gamma \cdot \bm{\phi}(s')^T \cdot \bw] - \bm{\phi}(s)^T \cdot \bw) \cdot (\bm{\phi}(s) - \gamma \cdot \mathbb{E}_{\pi}[\bm{\phi}(s')])\\
\end{align*}
This is called the [*Residual Gradient* algorithm, due to Leemon Baird](http://www.cs.utsa.edu/~bylander/cs6243/baird95residual.pdf) [@Baird:95]. It requires two independent samples of $s'$ transitioning from $s$. If we do have that, it converges to $\bm{w}_{BE}$ robustly (even for non-linear function approximations). But this algorithm is slow, and doesn't converge to a desirable place. Another issue is that $\bm{w}_{BE}$ is not learnable if we can only access the features, and not underlying states. These issues led researchers to consider alternative TD algorithms. 

\index{reinforcement learning!residual gradient|textbf}

We denote the third Value Function vector of interest in the $\bphi$ subspace as $\bm{w}_{TDE}$ and define it as the vector in the $\bphi$ subspace for which the expected square of the TD error $\delta$ (when following policy $\pi$) is minimized. Formally,
$$\bm{w}_{TDE} = \argmin_{\bw} \sum_{s \in \mathcal{S}} \bmu(s) \sum_{r,s'} \mathbb{P}_{\pi}(r, s'|s) \cdot (r + \gamma \cdot \bm{\phi}(s')^T \cdot \bw - \bm{\phi}(s)^T \cdot \bw)^2$$
To perform Stochastic Gradient Descent, we have to estimate the gradient of the expected square of TD error by sampling. The weight update for each gradient sample in the Stochastic Gradient Descent is:
\begin{align*}
\Delta \bw & = - \alpha \cdot \frac{1}{2} \cdot \nabla_{\bw} (r + \gamma \cdot \bm{\phi}(s')^T \cdot \bw - \bm{\phi}(s)^T \cdot \bw)^2\\
& = \alpha \cdot (r + \gamma \cdot \bm{\phi}(s')^T \cdot \bw - \bm{\phi}(s)^T \cdot \bw) \cdot (\bm{\phi}(s) - \gamma \cdot \bm{\phi}(s'))\\
\end{align*}
\index{reinforcement learning!naive residual gradient|textbf}
This algorithm is called [*Naive Residual Gradient*, due to Leemon Baird](http://www.cs.utsa.edu/~bylander/cs6243/baird95residual.pdf)  [@Baird:95]. Naive Residual Gradient converges robustly, but again, not to a desirable place. So researchers had to look even further.

\index{projected Bellman error}
\index{Bellman policy operator}

This brings us to the fourth (and final) Value Function vector of interest in the $\bphi$ subspace. We denote this Value Function vector as $w_{PBE}$. The acronym $PBE$ stands for *Projected Bellman Error*. To understand this, first consider the composition of the Projection Operator $\bpi$ and the Bellman Policy Operator $\bb$, i.e., $\bpi \cdot \bb$ (we call this composed operator as the *Projected Bellman* operator). Visualize the application of this *Projected Bellman* operator on a Value Function vector $\bvw$ in the $\bphi$ subspace. Applying $\bb$ on $\bvw$ typically throws $\bvw$ out of the $\bphi$ subspace and then further applying $\bpi$ brings it back to the $\bphi$ subspace (call this resultant Value Function vector $\bm{V}_{\bm{w}'}$). The idea is to find a Value Function vector $\bvw$ in the $\bphi$ subspace for which the "distance" between $\bvw$ and $\bm{V}_{\bm{w}'}$ is minimized, i.e. we minimize the "error vector" $PBE = \bpi \cdot \bb \cdot \bvw - \bvw$ (Figure \ref{fig:vf_geometry} provides the visualization). Hence, we say we are minimizing the *Projected Bellman Error* (or simply that we are minimizing $PBE$), and we refer to $w_{PBE}$ as the Value Function vector in the $\bphi$ subspace for which $PBE$ is minimized. It turns out that the minimum of PBE is actually zero, i.e., $\bphi \cdot \bm{w}_{PBE}$ is a fixed point of operator $\bpi \cdot \bb$. Let us write out this statement formally. We know:
$$\bpi = \bphi \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd$$
$$\bb \cdot \bv = \brew + \gamma \bprob \cdot \bv$$
Therefore, the statement that $\bphi \cdot \bm{w}_{PBE}$ is a fixed point of operator $\bpi \cdot \bb$ can be written as follows:
$$\bphi \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd \cdot (\brew + \gamma \bprob \cdot \bphi \cdot \bm{w}_{PBE}) = \bphi \cdot \bm{w}_{PBE}$$
Since the columns of $\bphi$ are assumed to be independent (full rank),
\begin{align}
(\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd \cdot (\brew + \gamma \bprob \cdot \bphi \cdot \bm{w}_{PBE}) & = \bm{w}_{PBE}i \notag \\
\bphi^T \cdot \bd \cdot (\brew + \gamma \bprob \cdot \bphi \cdot \bm{w}_{PBE}) &= \bphi^T \cdot \bd \cdot \bphi \cdot \bm{w}_{PBE} \notag \\
\bphi^T \cdot \bd \cdot (\bphi - \gamma \bprob \cdot \bphi) \cdot \bm{w}_{PBE} &= \bphi^T \cdot \bd \cdot \brew \label{eq:w-pbe-equation}
\end{align}
This is a square linear system of the form $\bm{A} \cdot \bm{w}_{PBE} = \bm{b}$ whose solution is:
$$\bm{w}_{PBE} = \bm{A}^{-1} \cdot \bm{b} = (\bphi^T \cdot \bd \cdot (\bphi - \gamma \bprob \cdot \bphi))^{-1} \cdot \bphi^T \cdot \bd \cdot \brew$$

The above formulation can be used to compute $\bm{w}_{PBE}$ if we know the model probabilities $\bprob$ and reward function $\brew$. But often, in practice, we don't know $\bprob$ and $\brew$, in which case we seek model-free learning of $\bm{w}_{PBE}$, specifically with a TD (bootstrapped) algorithm.

The question is how do we construct matrix

$$\bm{A} = \bphi^T \cdot \bd \cdot (\bphi - \gamma \bprob \cdot \bphi)$$
and vector
$$\bm{b} = \bphi^T \cdot \bd \cdot \brew$$
without a model?

\index{reinforcement learning!least squares!lstd}
\index{function approximation!semi-gradient}

Following policy $\pi$, each time we perform an individual transition from $s$ to $s'$ getting reward $r$, we get a sample estimate of $\bm{A}$ and $\bm{b}$. The sample estimate of $\bm{A}$ is the outer-product of vectors $\bm{\phi}(s)$ and $\bm{\phi}(s) - \gamma \cdot \bm{\phi}(s')$. The sample estimate of $\bm{b}$ is scalar $r$ times vector $\bm{\phi}(s)$. We average these sample estimates across many such individual transitions. Note that this algorithm is exactly the Least Squares Temporal Difference (LSTD) algorithm we've covered earlier in this chapter. Thus, we now know that LSTD converges to $w_{PBE}$, i.e., minimizes (in fact, takes down to 0) $PBE$. If the number of features $m$ is large or if we are doing non-linear function approximation or Off-Policy, then we seek a gradient-based TD algorithm. It turns out that our usual Semi-Gradient TD algorithm converges to $\bm{w}_{PBE}$ in the case of on-policy linear function approximation. Note that the update for the usual Semi-Gradient TD algorithm in the case of on-policy linear function approximation is as follows:
$$\Delta \bw = \alpha \cdot (r + \gamma \cdot \bm{\phi}(s')^T \cdot \bw - \bm{\phi}(s)^T \cdot \bw) \cdot \bm{\phi}(s)$$
This converges to $\bm{w}_{PBE}$ because at convergence, we have: $\mathbb{E}_{\pi}[\Delta \bw] = 0$, which can be expressed as:
$$ \bphi^T \cdot \bd \cdot (\brew + \gamma \bprob \cdot \bphi \cdot \bw - \bphi \cdot \bw) = 0$$
$$ \Rightarrow \bphi^T \cdot \bd \cdot (\bphi - \gamma \bprob \cdot \bphi) \cdot \bw = \bphi^T \cdot \bd \cdot \brew$$ 
which is satisfied for $\bw = \bm{w}_{PBE}$ (as seen from Equation \eqref{eq:w-pbe-equation}).

\index{value function!value function geometry|)}

### Gradient Temporal-Difference (Gradient TD)

\index{reinforcement learning!temporal difference!gradient td|(}
\index{function approximation!non-linear}
\index{reinforcement learning!off-policy}
For on-policy linear function approximation, the semi-gradient TD algorithm gives us $w_{PBE}$. But to obtain $w_{PBE}$ in the case of non-linear function approximation or in the case of Off-Policy, we need a different approach. The different approach is Gradient Temporal-Difference (abbreviated, Gradient TD), the subject of this section.

The [original Gradient TD algorithm, due to Sutton, Szepesvari, Maei](https://proceedings.neurips.cc/paper/2008/file/e0c641195b27425bb056ac56f8953d24-Paper.pdf) [@sutton2008] is typically abbreviated as GTD. [Researchers then came up with a second-generation Gradient TD algorithm](https://cseweb.ucsd.edu//~gary/190-RL/SMPBSSW-09.pdf) [@sutton2009], which is typically abbreviated as GTD-2. [The same researchers also came up with a TD algorithm with Gradient Correction](https://cseweb.ucsd.edu/~gary/190-RL/SMPBSSW-09.pdf) [@sutton2009], which is typically abbreviated as TDC.

We now cover the TDC algorithm. For simplicity of articulation and ease of understanding, we restrict to the case of linear function approximation in our coverage of the TDC algorithm below. However, do bear in mind that much of the concepts below extend to non-linear function approximation (which is where we reap the benefits of Gradient TD).

\index{function approximation!loss function}
\index{function approximation!gradient descent}

Our first task is to set up the appropriate loss function whose gradient will drive the Stochastic Gradient Descent. 
$$\bm{w}_{PBE} = \argmin_{\bw} d(\bpi \cdot \bb \cdot \bvw, \bvw) = \argmin_{\bw} d(\bpi \cdot \bb \cdot \bvw, \bpi \cdot \bvw)$$
So we define the loss function (denoting $\bb \cdot \bvw - \bvw$ as $\bdel$) as:
$$\mathcal{L}({\bw})  = (\bpi \cdot \bdel)^T \cdot \bd \cdot (\bpi \cdot \bdel) = \bdel^T \cdot \bpi^T \cdot \bd \cdot \bpi \cdot \bdel$$
$$=  \bdel^T \cdot (\bphi \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd)^T \cdot \bd \cdot  (\bphi \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd) \cdot \bdel$$
$$= \bdel^T \cdot (\bd \cdot \bphi \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T) \cdot \bd \cdot  (\bphi \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot \bphi^T \cdot \bd) \cdot \bdel$$
$$= (\bdel^T \cdot \bd \cdot \bphi) \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot (\bphi^T \cdot \bd \cdot  \bphi) \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot (\bphi^T \cdot \bd \cdot \bdel)$$
$$= (\bphi^T \cdot \bd \cdot \bdel)^T \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot (\bphi^T \cdot \bd \cdot \bdel)$$

We derive the TDC Algorithm based on $\nabla_{\bw} \mathcal{L}({\bw})$.
$$\nabla_{\bw} \mathcal{L}({\bw}) = 2 \cdot (\nabla_{\bw} (\bphi^T \cdot \bd \cdot \bdel)^T) \cdot (\bphi^T \cdot \bd \cdot \bphi)^{-1} \cdot (\bphi^T \cdot \bd \cdot \bdel)$$

We want to estimate this gradient from individual transitions data. So we express each of the 3 terms forming the product in the gradient expression above as expectations of functions of individual transitions $s \stackrel{\pi}\longrightarrow (r,s')$. Denoting $r + \gamma \cdot \bm{\phi}(s')^T \cdot \bw - \bm{\phi}(s)^T \cdot \bw$ as $\delta$, we get:

$$\bphi^T \cdot \bd \cdot \bdel = \mathbb{E}[\delta \cdot \bm{\phi}(s)]$$
$$\nabla_{\bw} (\bphi^T \cdot \bd \cdot \bdel)^T = \mathbb{E}[(\nabla_{\bw} \delta) \cdot \bm{\phi}(s)^T] = \mathbb{E}[(\gamma \cdot \bm{\phi}(s') - \bm{\phi}(s)) \cdot \bm{\phi}(s)^T]$$
$$\bphi^T \cdot \bd \cdot \bphi = \mathbb{E}[\bm{\phi}(s) \cdot \bm{\phi}(s)^T]$$

Substituting, we get:
$$\nabla_{\bw} \mathcal{L}({\bw}) = 2 \cdot  \mathbb{E}[(\gamma \cdot \bm{\phi}(s') - \bm{\phi}(s)) \cdot \bm{\phi}(s)^T] \cdot (\mathbb{E}[\bm{\phi}(s) \cdot \bm{\phi}(s)^T])^{-1} \cdot \mathbb{E}[\delta \cdot \bm{\phi}(s)]$$
$$\Delta \bw  = - \alpha \cdot \frac 1 2 \cdot \nabla_{\bw} \mathcal{L}({\bw})$$
$$ = \alpha \cdot \mathbb{E}[(\bm{\phi}(s) - \gamma \cdot \bm{\phi}(s')) \cdot \bm{\phi}(s)^T] \cdot (\mathbb{E}[\bm{\phi}(s) \cdot \bm{\phi}(s)^T])^{-1} \cdot \mathbb{E}[\delta \cdot \bm{\phi}(s)]$$
$$ = \alpha \cdot (\mathbb{E}[\bm{\phi}(s) \cdot \bm{\phi}(s)^T] - \gamma \cdot \mathbb{E}[\bm{\phi}(s') \cdot \bm{\phi}(s)^T]) \cdot (\mathbb{E}[\bm{\phi}(s) \cdot \bm{\phi}(s)^T])^{-1} \cdot \mathbb{E}[\delta \cdot \bm{\phi}(s)]$$
$$ = \alpha \cdot (\mathbb{E}[\delta \cdot \bm{\phi}(s)] - \gamma \cdot \mathbb{E}[\bm{\phi}(s') \cdot \bm{\phi}(s)^T] \cdot(\mathbb{E}[\bm{\phi}(s) \cdot \bm{\phi}(s)^T])^{-1} \cdot \mathbb{E}[\delta \cdot \bm{\phi}(s)])$$
$$ = \alpha \cdot (\mathbb{E}[\delta \cdot \bm{\phi}(s)] - \gamma \cdot \mathbb{E}[\bm{\phi}(s') \cdot \bm{\phi}(s)^T] \cdot \btheta)$$
where $\btheta = (\mathbb{E}[\bm{\phi}(s) \cdot \bm{\phi}(s)^T])^{-1} \cdot \mathbb{E}[\delta \cdot \bm{\phi}(s)]$ is the solution to the weighted least-squares linear regression of $\bb \cdot \bv - \bv$ against $\bphi$, with weights as $\mu_{\pi}$.

We can perform this gradient descent with a technique known as *Cascade Learning*, which involves simultaneously updating both $\bw$ and $\btheta$ (with $\btheta$ converging faster). The updates are as follows:

\index{cascade learning}

$$\Delta \bw = \alpha \cdot \delta \cdot \bm{\phi}(s)  - \alpha \cdot \gamma \cdot \bm{\phi}(s') \cdot (\bm{\phi}(s)^T \cdot \btheta)$$
$$\Delta \btheta = \beta \cdot (\delta - \bm{\phi}(s)^T \cdot \btheta) \cdot \bm{\phi}(s)$$
where $\beta$ is the learning rate for $\btheta$. Note that $\bm{\phi}(s)^T \cdot \btheta$ operates as an estimate of the TD error $\delta$ for current state $s$.

\index{reinforcement learning!deadly triad}
\index{reinforcement learning!off-policy}
\index{function approximation!non-linear}
\index{bootstrapping}
Repeating what we had said in Chapter [-@sec:rl-control-chapter], Gradient TD converges reliably for the Prediction problem even when we are faced with the Deadly Triad of [Bootstrapping, Off-Policy, Non-Linear Function Approximation]. The picture is less rosy for Control. Gradient Q-Learning (Gradient TD for Off-Policy Control) converges reliably for both on-policy and off-policy linear function approximations, but there are divergence issues for non-linear function approximations. For Control problems with non-linear function approximations (especially, neural network approximations with off-policy learning), one can leverage the approach of the DQN algorithm (Experience Replay with fixed Target Network helps overcome the Deadly Triad).
\index{reinforcement learning!temporal difference!dqn}

\index{reinforcement learning|)}
\index{reinforcement learning!temporal difference!gradient td|)}

### Key Takeaways from This Chapter

* Batch RL makes efficient use of data.
* DQN uses Experience-Replay and fixed Q-learning targets, avoiding the pitfalls of time-correlation and varying TD Target.
* LSTD is a direct (gradient-free) solution of Batch TD Prediction.
* LSPI is an Off-Policy, Experience-Replay Control Algorithm using LSTDQ for Policy Evaluation.
* Optimal Exercise of American Options can be tackled with LSPI and Deep Q-Learning algorithms.
* For Prediction, the 4 Value Function vectors of interest in the $\bphi$ subspace are $\bw_{\pi}, \bw_{BE}, \bw_{TDE}, \bw_{PBE}$ with $\bw_{PBE}$ as the key sought-after function approximation for Value Function $\bvpi$.
* For Prediction, Gradient TD solves for $\bw_{PBE}$ efficiently and robustly in the case of non-linear function approximation and in the case of Off-Policy.
