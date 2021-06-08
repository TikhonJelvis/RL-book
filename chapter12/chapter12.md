## Experience-Replay, Least-Squares Policy Iteration, and Gradient TD {#sec:batch-rl-chapter}

In Chapters [-@sec:rl-prediction-chapter] and [-@sec:rl-control-chapter], we covered the basic RL algorithms for Prediction and Control respectively. Specifically, we covered the basic Monte-Carlo (MC) and Temporal-Difference (TD) techniques. We want to highlight two key aspects of these basic RL algorithms:

1. The experiences data arrives in the form of a single unit of experience at a time (single unit is a *trace experience* for MC and an *atomic experience* for TD), the unit of experience is used by the algorithm for Value Function learning, and then that unit of experience is not used later in the algorithm (essentially, that unit of experience, once consumed, is *not re-consumed* for further learning later in the algorithm). It doesn't have to be this way - one can develop RL algorithms that re-use experience data - this approach is known as *Experience-Replay* (in fact, we saw a glimpse of Experience-Replay in Section [-@sec:experience-replay-section] of Chapter [-@sec:rl-prediction-chapter]).
2. Learning occurs in an *incremental* manner, by updating the Value Function after each unit of experience. It doesn't have to be this way - one can develop RL algorithms that take an entire batch of experiences (or in fact, all of the experiences that one could possibly get), and learn the Value Function directly for that entire batch of experiences. The idea here is that we know in advance what experiences data we have (or will have), and if we collect and organize all of that data, then we could directly (i.e., not incrementally) estimate the Value Function for *that* experiences data set. This approach to RL is known as *Batch RL* (versus the basic RL algorithms we covered in the previous chapters that can be termed as *Incremental RL*).

Thus, we have a choice or doing Experience-Replay or not, and we have a choice of doing Batch RL or Incremental RL. In fact, some of the interesting and practically effective algorithms combine both the ideas of Experience-Replay and Batch RL. This chapter starts with the coverage of Experience-Replay and Batch RL. Then we look deeper into the issue of the *Deadly Triad* (that we had alluded to in Chapter [-@sec:rl-control-chapter]) by viewing Value Functions as Vectors (we had done this in Chapter [-@sec:dp-chapter]), understand Value Function Vector transformations with a balance of geometric intuition and mathematical rigor, providing insights into convergence issues for a variety of traditional loss functions used to develop RL algorithms. Finally this treatment of Value Functions as Vectors leads us in the direction of overcoming the Deadly Triad by defining an appropriate loss function, calculating whose gradient provides a more robust set of RL algorithms known as Gradient Temporal Difference (abbreviated, as Gradient TD).

### Batch RL and Experience-Replay

Let us understand Incremental RL versus Batch RL in the context of fixed finite experiences data. To make things simple and easy to understand, we first focus on understanding the difference for the case of MC Prediction (i.e., to calculate the Value Function of an MRP using Monte-Carlo). In fact, we had covered this setting in Section [-@sec:experience-replay-section] of Chapter [-@sec:rl-prediction-chapter]. To refresh this setting, specifically we have access to a fixed finite sequence/stream of MRP trace experiences (i.e., `Iterable[Iterable[rl.markov_process.TransitionStep[S]]]`), which we know can be converted to returns-augmented data of the form `Iterable[Iterable[rl.markov_process.ReturnStep[S]]]` (using the `returns` function in [rl/returns.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/returns.py)). Flattening this data to `Iterable[rl.markov_process.ReturnStep[S]]` and extracting from it the (state, return) pairs gives us the fixed, finite training data for MC Prediction, that we denote as follows:

$$\mathcal{D} = [(S_i, G_i) | 1 \leq i \leq n]$$

We've learnt in Chapter [-@sec:rl-prediction-chapter] that we can do an Incremental MC Prediction estimation $V(s;\bm{w})$ by updating $\bm{w}$ after each MRP trace experience with the gradient calculation $\nabla_{\bm{w}} \mathcal{L}(\bm{w})$ for each data pair $(S_i, G_i)$, as follows:
$$\mathcal{L}_{(S_i, G_i)}(\bm{w}) = \frac 1 2 \cdot (V(S_i; \bm{w}) - G_i)^2$$
$$\nabla_{\bm{w}} \mathcal{L}_{(S_i, G_i)}(\bm{w}) = (V(S_i; \bm{w}) - G_i) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$
$$\Delta \bm{w} = \alpha \cdot (G_i - V(S_i; \bm{w})) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$

The Incremental MC Prediction algorithm performs $n$ updates in sequence for data pairs $(S_i, G_i), i = 1, 2, \ldots, n$ using the `update` method of `FunctionApprox`. We note that Incremental RL makes inefficient use of available training data $\mathcal{D}$ because we essentially "discard" each of these units of training data after it's used to perform an update. We want to make efficient use of the given data with Batch RL. Batch MC Prediction aims to estimate the MRP Value Function $V(s;\bm{w^*})$ such that
\begin{align*}
\bm{w^*} & = \argmin_{\bm{w}} \frac 1 n \cdot \sum_{i=1}^n \frac 1 2 \cdot (V(S_i;\bm{w}) - G_i)^2 \\
& = \argmin_{\bm{w}} \mathbb{E}_{(S,G) \sim \mathcal{D}} [\frac 1 2 \cdot (V(S; \bm{w}) - G)^2]
\end{align*}
This in fact is the `solve` method of `FunctionApprox` on training data $\mathcal{D}$. This approach is called Batch RL because we first collect and store the entire set (batch) of data $\mathcal{D}$ available to us, and then we find the best possible parameters $\bm{w^*}$ fitting this data $\mathcal{D}$. Note that unlike Incremental RL, here we are not updating the MRP Value Function estimate while the data arrives - we simply store the data as it arrives and start the MRP Value Function estimation procedure once we are ready with the entire (batch) data $\mathcal{D}$ in storage. As we know from the implementation of the `solve` method of `FunctionApprox`, finding the best possible parameters $\bm{w^*}$ from the batch $\mathcal{D}$ involves calling the `update` method of `FunctionApprox` with repeated use of the available data pairs $(S,G)$ in the stored data set $\mathcal{D}$. Each of these updates to the parameters $\bm{w}$ is as follows:
$$\Delta \bm{w} = \alpha \cdot \frac 1 n \cdot \sum_{i=1}^n (G_i - V(S_i; \bm{w})) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$
If we keep doing these updates repeatedly, we will ultimately converge to the desired MRP Value Function $V(s;\bm{w^*})$. The repeated use of the available data in $\mathcal{D}$ means that we are doing Batch MC Prediction using *Experience-Replay*. So we see that this makes more efficient use of the available training data $\mathcal{D}$ due to the re-use of the data pairs in $\mathcal{D}$.

The code for this Batch MC Prediction algorithm is shown below (function `batch_mc_prediction`). From the input trace experiences (`traces` in the code below), we first create the set of `ReturnStep` transitions that span across the set of all input trace experiences (`return_steps` in the code below). This involves calculating the return associated with each state encountered in `traces` (across all trace experiences). From `return_steps`, we create the (state, return) pairs that constitute the fixed, finite training data $\mathcal{D}$, which is then passed to the `solve` method of `approx: ValueFunctionApprox[S]`.

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
    return_steps: Iterable[mp.ReturnStep[S]] = \
        itertools.chain.from_iterable(
            returns(trace, gamma, episode_length_tolerance) for trace in traces
        )
    return approx.solve(
        [(step.state, step.return_) for step in return_steps],
        convergence_tolerance
    )
```
The code above is in the file [rl/monte_carlo.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/monte_carlo.py).

Now let's move on to Batch TD Prediction. Here we have fixed, finite experiences data $\mathcal{D}$ available as:
$$\mathcal{D} = [(S_i, R_i, S'_i) | 1 \leq i \leq n]$$
where $(R_i, S'_i)$ is the pair of reward and next state from a state $S_i$. So, Experiences Data $\mathcal{D}$ is presented in the form of a fixed, finite number of atomic experiences. This is represented in code as an `Iterable[rl.markov_process.TransitionStep[S]]`.

Just like Batch MC Prediction, here in Batch TD Prediction, we first collect and store the data as it arrives, and once we are ready with the batch of data $\mathcal{D}$ in storage, we start the MRP Value Function estimation procedure. The parameters $\bm{w}$ are updated with repeated use of the unit experiences in the stored data $\mathcal{D}$. Each update is done using a random data point $(S,R,S') \sim \mathcal{D}$, as follows:
$$\Delta \bm{w} = \alpha \cdot (R + \gamma \cdot V(S'; \bm{w}) - V(S; \bm{w})) \cdot \nabla_{\bm{w}} V(S; \bm{w})$$

We keep performing these updates by repeatedly randomly sampling $(S,G) \sim \mathcal{D}$ until convergence. Thus, Batch TD Prediction also does Experience-Replay, hence making efficient use of the available training data $\mathcal{D}$. Specifically, this algorithm does TD Prediction with Experience-Replay on a fixed finite set of atomic experiences presented in the form of $\mathcal{D} = [(S_i, R_i, S'_i) | 1 \leq i \leq n]$.

The code for this Batch TD Prediction algorithm is shown below (function `batch_td_prediction`). From the input atomic experiences $\mathcal{D}$ (`transitions` in the code below), we first store it as a list (`tr_seq` in the code below). Then we create an infinite stream (`Iterator`) of transitions using the function `transitions_stream` that repeatedly randomly chooses a `TransitionStep` from the stored `tr_seq`. Finally, we pass this infinite stream of transitions to `td_prediction` (that we had written in Chapter [-@sec:rl-prediction-chapter]), and let it run until convergence (up to `convergence_tolerance`).

```python
import rl.markov_process as mp
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.td import td_prediction
import rl.iterate as iterate
import itertools
import numpy as np

def batch_td_prediction(
    transitions: Iterable[mp.TransitionStep[S]],
    approx: ValueFunctionApprox[S],
    gamma: float,
    convergence_tolerance: float = 1e-5
) -> ValueFunctionApprox[S]:
    tr_seq: Sequence[mp.TransitionStep[S]] = list(transitions)

    def transitions_stream(
        tr_seq=tr_seq
    ) -> Iterator[mp.TransitionStep[S]]:
        while True:
            yield tr_seq[np.random.randint(len(tr_seq))]

    def done(
        a: ValueFunctionApprox[S],
        b: ValueFunctionApprox[S],
        convergence_tolerance=convergence_tolerance
    ) -> bool:
        return b.within(a, convergence_tolerance)

    return iterate.converged(
        td_prediction(transitions_stream(), approx, gamma),
        done=done
    )
```

The code above is in the file [rl/td.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/td.py).

Likewise, we can do Batch TD($\lambda$) Prediction. Here we are given a fixed, finite number of trace experiences
$$\mathcal{D} = [(S_{i,0}, R_{i,1}, S_{i,1}, R_{i,2}, S_{i,2}, \ldots, R_{i,T_i}, S_{i,T_i}) | 1 \leq i \leq n]$$
In each iteration, we randomly pick a trace experience (say indexed $i$) from the stored data $\mathcal{D}$. For trace experience $i$, the parameters $\bm{w}$ are updated at each time step $t$ in the trace experience as follows:
$$\bm{E}_t = \gamma \lambda \cdot \bm{E}_{t-1} + \nabla_{\bm{w}} V(S_{i,t};\bm{w})$$
\begin{equation}
\Delta \bm{w} = \alpha \cdot (R_{i,t+1} + \gamma \cdot V(S_{i,t+1}; \bm{w}) - V(S_{i,t}; \bm{w})) \cdot \bm{E}_t
\label{eq:batch-td-lambda-update}
\end{equation}
where $\bm{E}_t$ denotes the eligibility trace at time step $t$, and $\bm{E}_0$ is initialized to 0 at the start of each trace experience.

### Least-Squares RL Prediction

In the previous section, we saw how Batch RL Prediction is an iterative process until convergence -  the MRP Value Function is updated with repeated use of the fixed, finite (batch) data that is made available. However, if we assume that the MRP Value Function approximation $V(s; \bm{w})$ is a linear function approximation (linear in a set of feature functions of the state space), then we can solve for the MRP Value Function with direct and simple linear algebra operations (ie., without the need for iterations until convergence). Let us see how.

We define a sequence of feature functions $\phi_j: \mathcal{S} \rightarrow \mathbb{R}, j = 1, 2, \ldots, m$ and we assume the parameters $\bm{w}$ is a weights vector $\bm{w} = (w_1, w_2, \ldots, w_m) \in \mathbb{R}^m$. Therefore, the MRP Value Function is approximated as:
$$V(s;\bm{w}) = \sum_{j=1}^m \phi_j(s) \cdot w_j = \bm{\phi}(s)^T \cdot \bm{w}$$
where $\bm{\phi}(s) \in \mathbb{R}^m$ is the feature vector for state $s$

The direct solution of the MRP Value Function using simple linear algebra operations is known as Least-Squares (abbreviated as LS) solution. We start with Batch MC Prediction for the case of linear function approximation, which is known as Least-Squares Monte-Carlo (abbreviated as LSMC).

#### Least-Squares Monte-Carlo (LSMC)

For the case of linear function approximation, the loss function for Batch MC Prediction with data $[(S_i, G_i) | 1 \leq i \leq n]$ is:
$$\mathcal{L}(\bm{w}) =  \frac 1 {2n} \cdot \sum_{i=1}^n (\sum_{j=1}^m \phi_j(S_i) \cdot w_j - G_i)^2 = \frac 1 {2n} \cdot \sum_{i=1}^n (\bm{\phi}(S_i)^T \cdot \bm{w} - G_i)^2$$
We set the gradient of this loss function to 0, and solve for $\bm{w}^*$. This yields:
$$\sum_{i=1}^n \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i)^T \cdot \bm{w^*} - G_i) = 0$$
We can calculate the solution $\bm{w^}*$ as $\bm{A}^{-1} \cdot \bm{b}$, where the $m \times m$ Matrix $\bm{A}$ is accumulated at each data pair $(S_i, G_i)$ as:
$$ \bm{A} \leftarrow \bm{A} + \bm{\phi}(S_i) \cdot \bm{\phi}(S_i)^T \text{ (i.e., outer-product of } \bm{\phi}(S_i) \text{ with itself})$$
and the $m$-Vector $\bm{b}$ is accumulated at each data pair $(S_i, G_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{\phi}(S_i) \cdot G_i$$

To implement this algorithm, we can simply call `batch_mc_prediction` that we had written earlier by setting the argument `approx` as `LinearFunctionApprox` and by setting the attribute `direct_solve` in `approx: LinearFunctionApprox[S]` as `True`. If you read the code under `direct_solve=True` branch in the `solve` method, you will see that it will indeed perform the above-described linear algebra calculations. The inversion of the matrix $\bm{A}$ is $O(m^3)$ complexity. However, we can speed up the algorithm to be $O(n^2)$ with a different implementation - we can maintain the inverse of $\bm{A}$ after each $(S_i, G_i)$ update to $\bm{A}$ by applying the [Sherman-Morrison formula for incremental inverse](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula). The Sherman-Morrison incremental inverse for $\bm{A}$ is as follows:

$$(\bm{A} + \bm{\phi}(S_i) \cdot \bm{\phi}(S_i)^T)^{-1} = \bm{A}^{-1} - \frac {\bm{A}^{-1} \cdot \bm{\phi}(S_i) \cdot \bm{\phi}(S_i)^T \cdot \bm{A}^{-1}} {1 + \bm{\phi}(S_i)^T \cdot \bm{A}^{-1} \cdot \bm{\phi}(S_i)}$$

with $\bm{A}^{-1}$ initialized to $\frac 1 {\epsilon} \cdot \bm{I}_m$, where $\bm{I}_m$ is the $m \times m$ identity matrix, and $\epsilon \in \mathbb{R}^+$ is a small number provided as a parameter to the algorithm. $\frac 1 \epsilon$ should be considered to be a proxy for the step-size $\alpha$ which is not required for least-squares algorithms. If $\epsilon$ is too small, the sequence of inverses of $\bm{A}$ can be quite unstable and if $\epsilon$ is too large, the learning is slowed.

This brings down the computational complexity of this algorithm to $O(m^2)$. We won't implement the Sherman-Morrison incremental inverse for LSMC, but in the next subsection we shall implement it for Least-Squares Temporal Difference (LSTD).

#### Least-Squares Temporal-Difference (LSTD)

For the case of linear function approximation, the loss function for Batch TD Prediction with data $[(S_i, R_i, S'_i) | 1 \leq i \leq n]$ is:
$$\mathcal{L}(\bm{w}) = \frac 1 {2n} \cdot \sum_{i=1}^n (\bm{\phi}(S_i)^T \cdot \bm{w} - (R_i + \gamma \cdot \bm{\phi}(S'_i)^T \cdot \bm{w}))^2$$
We set the semi-gradient of this loss function to 0, and solve for $\bm{w}^*$. This yields:
$$\sum_{i=1}^n \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i)^T \cdot \bm{w^*} - (S_i + \gamma \cdot \bm{\phi}(S'_i)^T \cdot \bm{w}^*)) = 0$$
We can calculate the solution $\bm{w^}*$ as $\bm{A}^{-1} \cdot \bm{b}$, where the $m \times m$ Matrix $\bm{A}$ is accumulated at each each atomic experience $(S_i, R_i, S'_i)$ as:
$$ \bm{A} \leftarrow \bm{A} + \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))^T \text{ (note the Outer-Product)}$$
and the $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(S_i, R_i, S'_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{\phi}(S_i) \cdot R_i$$

With Sherman-Morrison incremental inverse, we can reduce the computational complexity from $O(m^3)$ to $O(m^2)$.
$$(\bm{A} + \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S_i))^T)^{-1} = \bm{A}^{-1} - \frac {\bm{A}^{-1} \cdot \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))^T \cdot \bm{A}^{-1}} {1 + (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))^T \cdot \bm{A}^{-1} \cdot \bm{\phi}(S_i)}$$

with $\bm{A}^{-1}$ initialized to $\frac 1 {\epsilon} \cdot \bm{I}_m$, where $\bm{I}_m$ is the $m \times m$ identity matrix, and $\epsilon \in \mathbb{R}^+$ is a small number provided as a parameter to the algorithm.

Now let's write some code to implement this LSTD algorithm. The arguments `transitions`, `feature_functions`, `gamma` and `epsilon` of the function `least_squares_td` below are quite self-explanatory. Since this is a batch method with direct calculation of the estimated Value Function from batch data (rather than iterative updates), `least_squares_td` returns the estimated Value Function of type `LinearFunctionApprox[NonTerminal[S]]`, rather than an `Iterator` over the updated function approximations (as was the case in Incremental RL algorithms). The code below should be fairly self-explanatory. `a_inv` refers to $\bm{A}^{-1}$ which is updated with the Sherman-Morrison incremental inverse method. `b_vec` refers to the $\bm{b}$ vector. `phi1` refers to $\bm{\phi}(S_i)$, `phi2` refers to $\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i)$ (except when $S'_i$ is a terminal state, in which case `phi2` is simply $\bm{\phi}(S_i)$). The temporary variable `temp` refers to $(\bm{A}^{-1})^T \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))$ and is used both in the numerator and denominator in the Sherman-Morrison formula to update $\bm{A}^{-1}$.

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

First we set up a `RandomWalkMRP` object with $B = 20, p = 0.55$ and calculate it's true Value Function (so we can later compare against Incremental TD and LSTD methods).

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

Let's say we have access to only 10,000 transitions (each transition is to type `rl.markov_process.TransitionStep`). First we generate these 10,000 sampled transitions from the `RandomWalkMRP` object we created above.


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

Finally, we run the LSTD algorithm on 10,000 transitions. Note that the Value Function of `RandomWalkMRP`, for $p \neq 0.5$, is non-linear as a function of the integer states. So we use non-linear features that can approximate arbitrary non-linear shapes - a good choice is the set of (orthogonal) [Laguerre Polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials). In the code below, we use the first 5 Laguerre Polynomials (i.e. upto degree 4 polynomial) as the feature functions for the linear function approximation of the Value Function. Then we invoke the LSTD algorithm we wrote above to calculate the `LinearFunctionApprox` based on this batch of 10,000 transitions.

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

Figure \ref{fig:lstd_vf_comparison} depicts how the LSTD Value Function estimate (for 10,000 transitions) `lstd_vf` compares against Incremental Tabular TD Value Function estimate (for 10,000 transitions) `td_vf` and against the true value function `true_vf` (obtained using the linear-algebraic formula for MRP Value Function calculation). We encourage you to modify the parameters used in the code above to see how it alters the results - specifically play around with `this_barrier`, `this_p`, `gamma`, `num_transitions`, the learning rate trajectory for Incremental Tabular TD, the number of Laguerre polynomials, and `epsilon`. The above code is in the file [rl/chapter12/random_walk_lstd.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/random_walk_lstd.py).

![LSTD and Tabular TD Value Functions \label{fig:lstd_vf_comparison}](./chapter12/lstd_vf_comparison.png "LSTD and Tabular TD Value Functions")

#### LSTD($\lambda$)

Likewise, we can do LSTD($\lambda$) using Eligibility Traces. Denote the Eligibility Trace of atomic experience $i$ as $\bm{E}_i$. Note that $\bm{E}_i$ accumulates $\nabla_{\bm{w}} V(s;\bm{w}) = \bm{\phi}(s)$ in each trace experience. When accumulating, the previous step's eligibility trace is discounted by $\lambda \gamma$. By summing up the right-hand-side of Equation \eqref{eq:batch-td-lambda-update} over all atomic experiences and setting it to 0 (i.e., setting the update to $\bm{w}$ over all atomic experiences data to 0), we get:
$$\sum_{i=1}^n \bm{E_i} \cdot (\bm{\phi}(S_i)^T \cdot \bm{w^*} - (R_i + \gamma \cdot \bm{\phi}(S'_i)^T \cdot \bm{w}^*)) = 0$$
We can calculate the solution $\bm{w^}*$ as $\bm{A}^{-1} \cdot \bm{b}$, where the $m \times m$ Matrix $\bm{A}$ is accumulated at each each atomic experience $(S_i, R_i, S'_i)$ as:
$$ \bm{A} \leftarrow \bm{A} + \bm{E_i} \cdot (\bm{\phi}(S_i) - \gamma \cdot \bm{\phi}(S'_i))^T \text{ (note the Outer-Product)}$$
and the $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(S_i, R_i, S'_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{E_i} \cdot R_i$$
With Sherman-Morrison incremental inverse, we can reduce the computational complexity from $O(m^3)$ to $O(m^2)$.

#### Least-Squares Prediction Convergence

Before we move on to Least-Squares for the Control problem, we want to point out that the convergence behavior of Least-Squares Prediction algorithms are identical to their counterpart Incremental RL Prediction algorithms, with the exception that Off-Policy LSMC does not have convergence guarantees. Figure \ref{fig:rl_prediction_with_ls_convergence} shows the updated summary table for convergence of RL Prediction algorithms (that we had displayed at the end of Chapter [-@sec:rl-control-chapter]) to now also include Least-Squares Prediction algorithms. 

\begin{figure}
\begin{center}
\begin{tabular}{ccccc}
\hline
On/Off Policy & Algorithm & Tabular & Linear & Non-Linear \\ \hline
& MC & \cmark & \cmark & \cmark \\
On-Policy & \bfseries LSMC & \cmark & \cmark & - \\
& TD & \cmark & \cmark & \xmark \\
& \bfseries LSTD & \cmark & \cmark & - \\ \hline
& MC & \cmark & \cmark & \cmark \\
Off-Policy & \bfseries LSMC & \cmark & \xmark & - \\
& TD & \cmark & \xmark & \xmark \\ 
& \bfseries LSTD & \cmark & \xmark & - \\ \hline
\end{tabular}
\end{center}    
\caption{Convergence of RL Prediction Algorithms}
\label{fig:rl_prediction_with_ls_convergence}
\end{figure}

This ends our coverage of Least-Squares Prediction. Before we move on to Least-Squares Control, we need to cover Incremental RL Control with Experience-Replay as it serves as a stepping stone towards Least-Squares Control.

### Q-Learning with Experience-Replay

In this subsection, we cover Off-Policy Incremental TD Control with Experience-Replay. Specifically, we revisit the Q-Learning algorithm we covered in Chapter [-@sec:rl-control-chapter], but we tweak that algorithm such that the transitions used to make the Q-Learning updates are sourced from an experience replay memory, rather than from a behavior policy derived from the current Q-Value estimate. While investigating the challenges with Off-Policy TD methods with deep learning function approximation, researchers identified two challenges:

1) The sequences of states made available to deep learning through trace experiences are highly correlated, whereas deep learning algorithms are premised on data samples being independent.
2) The data distribution changes as the RL algorithm learns new behaviors, whereas deep learning algorithms are premised on a fixed underlying distribution (i.e., stationary).

Experience-Replay serves to smooth the training data distribution over many past behaviors, effectively resolving the correlation issue as well as the non-stationarity issue. Hence, Experience-Replay is a powerful idea for Off-Policy TD Control.

To make this idea of Q-Learning with Experience-Replay clear, we make a few changes to the `q_learning` function we had written in Chapter [-@sec:rl-control-chapter] with the following function `q_learning_experience_replay`.

```python
from rl.markov_decision_process import TransitionStep
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import NTStateDistribution

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
    batch_size: int,
    weights_decay_half_life: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    replay_memory: List[TransitionStep[S, A]] = []
    decay_weights: List[float] = []
    factor: float = np.exp(-1.0 / weights_decay_half_life)
    random_gen = np.random.default_rng()
    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(q, mdp)
            action: A = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            replay_memory.append(TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            ))
            replay_len: int = len(replay_memory)
            decay_weights.append(factor ** (replay_len - 1))
            norm_factor: float = (1 - factor ** replay_len) / (1 - factor)
            norm_decay_weights: Sequence[float] = [w * norm_factor for w in
                                                   reversed(decay_weights)]
            trs: Sequence[TransitionStep[S, A]] = \
                [replay_memory[i] for i in random_gen.choice(
                    replay_len,
                    min(batch_size, replay_len),
                    replace=False,
                    p=norm_decay_weights
                )]
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

The key difference between the `q_learning` algorithm we wrote in Chapter [-@sec:rl-control-chapter] and this `q_learning_experience_replay` algorithm is that here we have an experience-replay memory (`replay_memory` in the code above). In the `q_learning` algorithm, the (`state`, `action`, `next_state`, `reward`) 4-tuple comprising `TransitionStep` (that is used to perform the Q-Learning update) was the result of `action` being sampled from the behavior policy (derived from the current estimate of the Q-Value Function, eg: $\epsilon$-greedy), and then the `next_state` and `reward` being generated from the (`state`, `action`) pair using the `step` method of `mdp`. Here in `q_learning_experience_replay`, we don't use this 4-tuple `TransitionStep` to perform the update - rather we append this 4-tuple to the list of `TransitionStep`s comprising the experience-replay memory (`replay_memory` in the code), then we randomly draw a set of `TransitionStep`s from `replay_memory` (giving more drawing weightage to the more recently added `TransitionSteps`), and use those 4-tuple `TransitionStep`s to perform the Q-Learning update. Note that these randomly picked `TransitionStep`s might be from old behavior policies (derived from old estimates of the Q-Value estimate). The key is that this algorithm re-uses atomic experiences that were previously generated, which also means that it re-uses behavior policies that were previously constructed in the course of the algorithm execution.

The argument `batch_size` refers to the number of `TransitionStep`s to be drawn from the experience-replay memory at each step (the language used for this set of drawn `TransitionStep`s used to perform the update is *mini-batch*). The argument `weights_decay_half_life` refers to the half life of an exponential decay function for the weights used in the random draw of the `TransitionStep`s (the most recently added `TransitionStep` has the highest weight). With this understanding, the code should be self-explanatory.

The above code is in the file [rl/td.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/td.py).

#### Deep Q-Networks (DQN) Algorithm

[DeepMind](https://deepmind.com/) developed an innovative and practically effective RL Control Control algorithm based on Q-Learning with Experience-Replay - an algorithm they named as Deep Q-Networks (abberviated as DQN). Apart from reaping the above-mentioned benefits of Experience-Replay for Q-Learning with a Deep Neural Network approximating the Q-Value function, they also benefited from employing a second Deep Neural Network (let us call the main DNN as the Q-Network, refering to it's parameters at $\bm{w}$, and the second DNN as the target network, refering to it's parameters as $\bm{w}^-$). The parameters $\bm{w}^-$ of the target network are infrequently updated to be made equal to the parameters $\bm{w}$ of the Q-network. The purpose of the target network is to evaluate the Q-Value simply to calculate the Q-Learning target (note that the target is $r + \gamma \cdot \max_{a'} Q(s', a'; \bm{w}^-)$ for a given atomic experience $(s,a,r,s')$).

Deep Learning is premised on the fact that the supervised learning targets (response values $y$ corresponding to predictor values $x$) are pre-generated fixed values. This is not the case in TD learning where the targets are dependent on the Q-Values. As Q-Values are updated at each step, the targets also get updated, and this correlation between the Q-Value estimate and the target vale typically leads to oscillations or divergence of the Q-Value estimate. By infrequently updating the parameters $\bm{w}^-$ of the target network (providing the target values) to be made equal to the parameters $\bm{w}$ of the Q-network (which are updated at each iteration, the targets in the Q-Learning update are essentially fixed. This goes a long way in resolving the core issue of correlation between the Q-Value estimate and the target values, helping considerably with convergence of the Q-Learning algorithm. Thus, DQN reaps the benefits of not just Experience-Replay in Q-Learning (which we articulated earlier), but also the benefits of having "fixed" targets. DNN utilizes a parameter $C$ such that the updating of $\bm{w}^-$ to be made equal to $\bm{w}$ is done once every $C$ updates to $\bm{w}$ (based on the usual Q-Learning update equation).

We won't implement the DQN algorithm in Python code - however, we sketch the outline of the algorithm, as follows:

At each time $t$ for each episode:

* Given state $S_t$, take action $A_t$ according to $\epsilon$-greedy policy extracted from Q-network values $Q(S_t,a;\bm{w})$.
* Given state $S_t$ and action $A_t$, obtain reward $R_{t+1}$ and next state $S_{t+1}$ from the environment.
* Append atomic experience $(S_t, A_t, R_{t+1}, S_{t+1})$ in experience-replay memory $\mathcal{D}$.
* Sample a random mini-batch of atomic experiences $(s_i,a_i,r_i,s'_i) \sim \mathcal{D}$.
* Using this mini-batch of atomic experiences, update the Q-network parameters $\bm{w}$ with the Q-learning targets based on "frozen" parameters $\bm{w}^-$ of the target network.
$$\Delta \bm{w} = \alpha \cdot \sum_i (r_i + \gamma \cdot \max_{a'_i} Q(s'_i, a'_i; \bm{w}^-) - Q(s_i,a_i;\bm{w})) \cdot \nabla_{\bm{w}} Q(s_i,a_i;\bm{w})$$
* $S_t \leftarrow S_{t+1}$
* Once every $C$ time steps, set $\bm{w}^- \leftarrow \bm{w}$.

To learn more about the effectiveness of DQN for Atari games, see the [Original DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and the [DQN Nature Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) that DeepMind has published.

Now we are ready to cover Batch RL Control (specifically Least-Squares TD Control), which combines the ideas of Least-Squares TD Prediction and Q-Learning with Experience-Replay.

### Least-Squares Policy Iteration (LSPI)

Having seen Least-Squares Prediction, the natural question is whether we can extend the Least-Squares (batch with linear function approximation) methodology to solve the Control problem. For On-Policy MC and On-Policy TD Control, we take the usual route of Generalized Policy Iteration (GPI) with:

1. Policy Evaluation as Least-Squares $Q$-Value Prediction. Specifically, the $Q$-Value for a policy $\pi$ is approximated as:
$$Q^{\pi}(s,a) \approxeq Q(s,a;\bm{w}) = \bm{\phi}(s)^T \cdot \bm{w}$$
with a direct linear-algebraic solve for the linear function approximation weights $\bm{w}$ using batch experiences data generated using policy $\pi$.
2. $\epsilon$-Greedy Policy Improvement.

In this section, we focus on Off-Policy Control with Least-Squares TD. This algorithm is known as Least-Squares Policy Iteration, abbreviated as LSPI, developed by [Lagoudakis and Parr](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf). LSPI has been an important go-to algorithm in the history of RL Control because of it's simplicity and effectiveness. The basic idea of LSPI is that it's Generalized Policy Iteration (GPI) is essentially *Q-Learning with Experience-Replay*, with the key being that instead of doing the usual Q-Learning update after each atomic experience, we do *batch Q-Learning* for the Policy Evaluation phase of GPI. We spend the rest of this section describing LSPI in detail and then implementing it in Python code.

The input to LSPI is a fixed finite data set $\mathcal{D}$, consisting of set of $(s,a,r,s')$ atomic experiences, i.e., a set of `rl.markov_decision_process.TransitionStep` objects, and the task of LSPI is to determine the Optimal Q-Value Function (and hence, Optimal Policy) based on this experiences data set $\mathcal{D}$ using an experience-replayed, batch Q-Learning technique described below. Assume $\mathcal{D}$ consists of $n$ atomic experiences, indexed as $i = 1, 2, \ldots n$, with the atomic experience $i$ denoted as $(s_i, a_i, r_i, s'_i)$.

In LSPI, each iteration of GPI involves access to:

* The experiences data set $\mathcal{D}$.
* A *Deterministic Target Policy* (call it $\pi_D$), that is made available from the previous iteration of GPI.

Given $\mathcal{D}$ and $\pi_D$, the goal of each iteration of GPI is to solve for weights $\bm{w}^*$ that minimizes:
\begin{align*}
\mathcal{L}(\bm{w}) & = \sum_{i=1}^n (Q(s_i,a_i; \bm{w}) - (r_i + \gamma \cdot Q(s'_i,\pi_D(s'_i); \bm{w})))^2\\
& = \sum_{i=1}^n (\bm{\phi}(s_i,a_i)^T \cdot \bm{w} - (r_i + \gamma \cdot \bm{\phi}(s'_i, \pi_D(s'_i))^T \cdot \bm{w}))^2
\end{align*}

The solved $\bm{w}^*$ defines a $Q$-Value Function as follows:
$$Q(s,a; \bm{w}^*) = \bm{\phi}(s,a)^T \cdot \bm{w}^* = \sum_{j=1}^m \phi_j(s,a) \cdot w_j^*$$
This defines a deterministic policy $\pi_D$ (serving as the *Target Policy* for the next iteration of GPI):
$$\pi_D(s) = \argmax_a Q(s,a; \bm{w}^*)$$

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

This least-squares solution of $\bm{w}^*$ (Prediction) is known as Least-Squares Temporal Difference for Q-Value, abbreviated as *LSTDQ*. Thus, LSPI is GPI with LSTDQ and greedy policy improvements. Note how LSTDQ in each iteration re-uses the same data $\mathcal{D}$, i.e., LSPI does experience-replay.

We should point out here that the LSPI algorithm we described above should be considered as the *standard variant* of LSPI. However, we can design several other variants of LSPI, in terms of how the experiences data is sourced and used. Firstly, we should note that the experiences data $\mathcal{D}$ essentially provides the behavior policy for Q-Learning (along with the consequent reward and next state transition). In the *standard variant* we described above, since $\mathcal{D}$ is provided from an external source, the behavior policy that generates this data $\mathcal{D}$ must come from an external source. It doesn't have to be this way - we could generate the experiences data from a behavior policy derived from the Q-Value estimates produced by LSTDQi (eg: $\epsilon$-greedy policy). This would mean the experiences data used in the algorithm is not a fixed, finite data set, rather a variable, incrementally-produced data set. Even if the behavior policy was external, the data set $\mathcal{D}$ might not be a fixed finite data set - rather, it could be made available as an on-demand, variable data stream. Furthermore, in each iteration of GPI, we could use a subset of the experiences data made available until that point of time (rather than the approach of the standard version that uses all of the available experiences data). If we choose to sample a subset of the available experiences data, we might give more sampling-weightage to the more recently generated data. This would be especially the case if the experiences data was being generated from a policy derived the Q-Value estimates produced by LSTDQ. In this case, we would leverage the `ExperienceReplayMemory` class we'd written earlier.

Next, we write code to implement the *standard variant* of LSPI we described above. First, we write a function to implement LSTDQ. As described above, the inputs to LSTQD are the experiences data $\mathcal{D}$ (`transitions` in the code below) and a deterministic target policy $\pi_D$ (`target_policy` in the code below). Since we are doing a linear function approximation, the input also includes a set of features, described as functions of state and action (`feature_functions` in the code below). Lastly, the inputs also include the discount factor $\gamma$ and the numerical control parameter $\epsilon$. The code below should be fairly self-explanatory, as it is a straightforward extenion of LSTD (implemented in function `least_squares_td` above). The key differences are that this is an estimate of the Action-Value (Q-Value) function, rather than the State-Value Function, and the target used in the least-squares calculation is the Q-Learning target (produced by the `target_policy`).

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

Now we are ready to write the standard version of LSPI. The code below is a straightforward implementation of our description above, looping through the iterations of GPI, `yield`ing the Q-Value `LinearFunctionApprox` after each iteration of GPI.

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

#### Saving your Village from a Vampire   

Now we consider a Control problem we'd like to test the above LSPI algorithm on. We call it the Vampire problem that can be described as a good, old-fashioned bedtime story, as follows:

> A village is visited by a vampire every morning who randomly eats 1 villager upon entering the village, then retreats to the hills, planning to come back the next morning. The villagers come up with a plan. They will poison a certain number of villagers (picked at random uniformly) each night until the vampire eats a poisoned villager the next morning, after which the vampire dies (due to the poison in the villager the vampire ate). Unfortunately, all villagers who get poisoned also die the day after they are given the poison. If the goal of the villagers is to maximize the expected number of villagers at termination (termination is when either the vampire dies or all villagers are dead), what should be the optimal poisoning policy? In other words, if there are $n$ villagers on any day, how many villagers should be poisoned (as a function of $n$)?

It is straightforward to model this problem as an MDP. The *State* is the number of villagers at risk on any given night (if the vampire is still alive, the *State* is the number of villagers and if the vampire is dead, the *State* is 0, which is the only *Terminal State*). The *Action* is the number of villagers poisoned on any given night. The *Reward* is zero as long as the vampire is alive, and is equal to the number of villagers remaining if the vampire dies. Therefore. Let us refer to the initial number of villagers as $I$.

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
            50000
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
                    100
                )
            )
        return get_vf_and_policy_from_qvf(self, qvf)
```

The above code should be self-explanatory. The main challenge with LSPI is that we need to construct features function of the state and action such that the Q-Value Function is linear in those features. In this case, since we simply want to test the correctness of our LSPI implementation, we define feature functions (in method `lspi_feature` above) based on our knowledge of the true optimal Q-Value Function from the Dynamic Programming solution. The atomic experiences comprising the experiences data $\mathcal{D}$ for LSPI to use is generated with a simple uniform distribution of non-terminal states and action (in method `lspi_transitions` above).

Figure \ref{fig:lspi_opt_vf_comparison} shows the plot of the True Optimal Value Function (from Value Iteration) versus the LSPI-estimated Optimal Value Function. 

![True versus LSPI Optimal Value Function \label{fig:lspi_opt_vf_comparison}](./chapter12/vampire_lspi_opt_vf.png "True versus LSPI Optimal Value Function")

Figure \ref{fig:lspi_opt_policy_comparison} shows the plot of the True Optimal Policy (from Value Iteration) versus the LSPI-estimated Optimal Policy. 

![True versus LSPI Optimal Policy \label{fig:lspi_opt_policy_comparison}](./chapter12/vampire_lspi_opt_policy.png "True versus LSPI Optimal Policy")

The above code is in the file [rl/chapter12/vampire.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter12/vampire.py). As ever, we encourage you to modify some of the parameters in this code (including choices of feature functions, nature and number of atomic transitions used, number of GPI iterations, choice of $\epsilon$, and perhaps even a different dynamic for the vampire behavior), and see how the results change.
