## Experience-Replay, Least-Squares Policy Iteration, and Gradient TD {#sec:batch-rl-chapter}

In Chapters [-@sec:rl-prediction-chapter] and [-@sec:rl-control-chapter], we covered the basic RL algorithms for Prediction and Control respectively. Specifically, we covered the basic Monte-Carlo (MC) and Temporal-Difference (TD) techniques. We want to highlight two key aspects of these basic RL algorithms:

1. The experiences data arrives in the form of a single unit of experience at a time (single unit is a *trace experience* for MC and an *atomic experience* for TD), the unit of experience is used by the algorithm for Value Function learning, and then that unit of experience is not used later in the algorithm (essentially, that unit of experience, once consumed, is *not re-consumed* for further learning later in the algorithm). It doesn't have to be this way - one can develop RL algorithms that re-use experience data - this approach is known as *Experience Replay* (in fact, we saw a glimpse of Experience Replay in Section [-@sec:experience-replay-section] of Chapter [-@sec:rl-prediction-chapter]).
2. Learning occurs in an *incremental* manner, by updating the Value Function after each unit of experience. It doesn't have to be this way - one can develop RL algorithms that take an entire batch of experiences (or in fact, all of the experiences that one could possibly get), and learn the Value Function directly for that entire batch of experiences. The idea here is that we know in advance what experiences data we have (or will have), and if we collect and organize all of that data, then we could directly (i.e., not incrementally) estimate the Value Function for *that* experiences data set. This approach to RL is known as *Batch RL* (versus the basic RL algorithms we covered in the previous chapters that can be termed as *Incremental RL*).

Thus, we have a choice or doing Experience Replay or not, and we have a choice of doing Batch RL or Incremental RL. In fact, some of the interesting and practically effective algorithms combine both the ideas of Experience Replay and Batch RL. This chapter starts with the coverage of Experience Replay and Batch RL. Then we look deeper into the issue of the *Deadly Triad* (that we had alluded to in Chapter [-@sec:rl-control-chapter]) by viewing Value Functions as Vectors (we had done this in Chapter [-@sec:dp-chapter]), understand Value Function Vector transformations with a balance of geometric intuition and mathematical rigor, providing insights into convergence issues for a variety of traditional loss functions used to develop RL algorithms. Finally this treatment of Value Functions as Vectors leads us in the direction of overcoming the Deadly Triad by defining an appropriate loss function, calculating whose gradient provides a more robust set of RL algorithms known as Gradient Temporal Difference (abbreviated, as Gradient TD).

### Batch RL and Experience-Replay

Let us understand Incremental RL versus Batch RL in the context of fixed finite experiences data. To make things simple and easy to understand, we first focus on understanding the difference for the case of MC Prediction. In fact, we had covered this setting in Section [-@sec:experience-replay-section] of Chapter [-@sec:rl-prediction-chapter]. To refresh this setting, specifically we have access to a fixed finite sequence/stream of trace experiences (i.e., `Iterable[Iterable[TransitionStep[S]]]`), which we know can be converted to returns-augmented data of the form `Iterable[Iterable[ReturnStep[S]]]` (using the `returns` function in [rl/returns.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/returns.py)). Flattening this data to `Iterable[ReturnStep[S]]` and extracting from it the (state, return) pairs gives us the fixed, finite training data for MC Prediction, that we denote as follows:

$$\mathcal{D} = [(S_i, G_i) | 1 \leq i \leq n]$$

We've learnt in Chapter [-@sec:rl-prediction-chapter] that we can do an Incremental MC Prediction estimation $V(s;\bm{w})$ by updating $\bm{w}$ after each trace experience with the gradient calculation $\nabla_{\bm{w}} \mathcal{L}(\bm{w})$ for each data pair $(S_i, G_i)$, as follows:
$$\mathcal{L}_{(S_i, G_i)}(\bm{w}) = \frac 1 2 \cdot (V(S_i; \bm{w}) - G_i)^2$$
$$\nabla_{\bm{w}} \mathcal{L}_{(S_i, G_i)}(\bm{w}) = (V(S_i; \bm{w}) - G_i) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$
$$\Delta \bm{w} = \alpha \cdot (G_i - V(S_i; \bm{w})) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$

The Incremental MC Prediction algorithm performs $n$ updates in sequence for data pairs $(S_i, G_i), i = 1, 2, \ldots, n$ using the `update` method of `FunctionApprox`$. We note that Incremental RL makes inefficient use of available training data $\mathcal{D}$ because we essentially "discard" each of these units of training data after it's used to perform an update. We want to make efficient use of the given data with Batch RL. Batch MC Prediction aims to estimate the Value Function $V(s;\bm{w^*})$ such that
\begin{align*}
\bm{w^*} & = \argmin_{\bm{w}} \frac 1 n \cdot \sum_{i=1}^n \frac 1 2 \cdot (V(S_i;\bm{w}) - G_i)^2 \\
& = \argmin_{\bm{w}} \mathbb{E}_{(S,G) \sim \mathcal{D}} [\frac 1 2 \cdot (V(S; \bm{w}) - G)^2]
\end{align*}
This in fact is the `solve` method of `FunctionApprox` on training data $\mathcal{D}$. This approach is called Batch RL because we first collect and store the entire set (batch) of data $\mathcal{D}$ available to us, and then we find the best possible parameters $\bm{w^*}$ fitting this data $\mathcal{D}$. Note that unlike Incremental RL, here we are not updating the Value Function estimate while the data arrives - we simply store the data as it arrives and start the Value Function estimation procedure once we are ready with the entire (batch) data $\mathcal{D}$ in storage. As we know from the implementation of the `solve` method of `FunctionApprox`, finding the best possible parameters $\bm{w^*}$ from the batch $\mathcal{D}$ involves calling the `update` method of `FunctionApprox` with repeated use of the available data pairs $(S,G)$ in the stored data set $\mathcal{D}$. Each of these updates to the parameters $\bm{w}$ is as follows:
$$\Delta \bm{w} = \alpha \cdot \frac 1 n \cdot \sum_{i=1}^n (G_i - V(S_i; \bm{w})) \cdot \nabla_{\bm{w}} V(S_i; \bm{w})$$
If we keep doing these updates repeatedly, we will ultimately converge to the desired value function $V(s;\bm{w^*})$. The repeated use of the available data in $\mathcal{D}$ means that we are doing Batch MC Prediction using *Experience Replay*. So we see that this makes more efficient use of the available training data $\mathcal{D}$ due to the re-use of the data pairs in $\mathcal{D}$.

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
The above code is in the file [rl/monte_carlo.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/monte_carlo.py).

Now let's move on to Batch TD Prediction. Here we have fixed, finite experiences data $\mathcal{D}$ available as:
$$\mathcal{D} = [(S_i, R_i, S'_i) | 1 \leq i \leq n]$$
where $(R_i, S'_i)$ is the pair of reward and next state from a state $S_i$. So, Experiences Data $\mathcal{D}$ is presented in the form of a fixed, finite number of atomic experiences. This is represented in code as an `Iterable[TransitionStep[S]]`.

Just like Batch MC Prediction, here in Batch TD Prediction, we first collect and store the data as it arrives, and once we are ready with the batch of data $\mathcal{D}$ in storage, we start the Value Function estimation procedure. The parameters $\bm{w}$ are updated with repeated use of the unit experiences in the stored data $\mathcal{D}$. Each update is done using a random data point $(S,R,S') \sim \mathcal{D}$, as follows:
$$\Delta \bm{w} = \alpha \cdot (R + \gamma \cdot V(S'; \bm{w}) - V(S; \bm{w})) \cdot \nabla_{\bm{w}} V(S; \bm{w})$$

We keep performing these updates by repeatedly randomly sampling $(S,G) \sim \mathcal{D}$ until convergence. Thus, Batch TD Prediction also does Experience Replay, hence making efficient use of the available training data $\mathcal{D}$. Specifically, this algorithm does TD Prediction with Experience Replay on a fixed finite set of atomic experiences presented in the form of $\mathcal{D} = [(S_i, R_i, S'_i) | 1 \leq i \leq n]$.

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

Likewise, we can do Batch TD($\lambda$) Prediction. Here we are given a fixed, finite number of trace experiences
$$\mathcal{D} = [(S_{i,0}, R_{i,1}, S_{i,1}, R_{i,2}, S_{i,2}, \ldots, R_{i,T_i}, S_{i,T_i}) | 1 \leq i \leq n]$$
In each iteration, we randomly pick a trace experience (say indexed $i$) from the stored data $\mathcal{D}$. For trace experience $i$, the parameters $\bm{w}$ are updated at each time step $t$ in the trace experience as follows:
$$\bm{E}_t = \gamma \lambda \cdot \bm{E}_{t-1} + \nabla_{\bm{w}} V(S_{i,t};\bm{w})$$
$$\Delta \bm{w} = \alpha \cdot (R_{i,t+1} + \gamma \cdot V(S_{i,t+1}; \bm{w}) - V(S_{i,t}; \bm{w})) \cdot \bm{E}_t$$
where $\bm{E}_t$ denotes the eligibility trace at time step $t$, and $\bm{E}_0$ is initialized to 0 at the start of each trace experience.

### Least-Squares RL Prediction

* Batch RL Prediction for general function approximation is iterative
* Uses experience replay and gradient descent
* We can solve directly (without gradient) for linear function approx
* Define a sequence of feature functions $\phi_j: \mathcal{X} \rightarrow \mathbb{R}, j = 1, 2, \ldots, m$
* Parameters $w$ is a weights vector $\bm{w} = (w_1, w_2, \ldots, w_m) \in \mathbb{R}^m$
* Value Function is approximated as:
$$V(s;\bm{w}) = \sum_{j=1}^m \phi_j(s) \cdot w_j = \bm{\phi}(s)^T \cdot \bm{w}$$
where $\bm{\phi}(s) \in \mathbb{R}^m$ is the feature vector for state $s$

#### Least-Squares Monte-Carlo (LSMC)
* Loss function for Batch MC Prediction with data $[(S_i, G_i) | 1 \leq i \leq n]$:
$$\mathcal{L}(\bm{w}) =  \frac 1 {2n} \cdot \sum_{i=1}^n (\sum_{j=1}^m \phi_j(S_i) \cdot w_j - G_i)^2 = \frac 1 {2n} \cdot \sum_{i=1}^n (\bm{\phi}(S_i)^T \cdot \bm{w} - G_i)^2$$
* The gradient of this Loss function is set to 0 to solve for $\bm{w}^*$
$$\sum_{i=1}^n \bm{\phi}(S_i) \cdot (\bm{\phi}(S_i)^T \cdot \bm{w^*} - G_i) = 0$$
* $\bm{w^}*$ is solved as $\bm{A}^{-1} \cdot \bm{b}$
* $m \times m$ Matrix $\bm{A}$ is accumulated at each data pair $(S_i, G_i)$ as:
$$ \bm{A} \leftarrow \bm{A} + \bm{\phi}(S_i) \cdot \bm{\phi}(S_i)^T \text{ (i.e., outer-product of } \bm{\phi}(S_i) \text{ with itself})$$
* $m$-Vector $\bm{b}$ is accumulated at each data pair $(S_i, G_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{\phi}(S_i) \cdot G_i$$
* Shermann-Morrison incremental inverse can be done in $O(m^2)$

#### Least-Squares Temporal-Difference (LSTD)
* Loss function for Batch TD Prediction with data $[(s_i, r_i, s'_i) | 1 \leq i \leq n]$:
$$\mathcal{L}(\bm{w}) = \frac 1 {2n} \cdot \sum_{i=1}^n (\bm{\phi}(s_i) \cdot \bm{w} - (r_i + \gamma \cdot \bm{\phi}(s'_i)^T \cdot \bm{w}))^2$$
* The semi-gradient of this Loss function is set to 0 to solve for $\bm{w}^*$
$$\sum_{i=1}^n \bm{\phi}(s_i) \cdot (\bm{\phi}(s_i)^T \cdot \bm{w^*} - (r_i + \gamma \cdot \bm{\phi}(s'_i)^T \cdot \bm{w}^*)) = 0$$
* $\bm{w}^*$ is solved as $\bm{A}^{-1} \cdot \bm{b}$
* $m \times m$ Matrix $\bm{A}$ is accumulated at each atomic experience $(s_i, r_i, s'_i)$:
$$ \bm{A} \leftarrow \bm{A} + \bm{\phi}(s_i) \cdot (\bm{\phi}(s_i) - \gamma \cdot \bm{\phi}(s'_i))^T \text{ (note the Outer-Product)}$$
* $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(s_i, r_i, s'_i)$:
$$\bm{b} \leftarrow \bm{b} + \bm{\phi}(s_i) \cdot r_i$$
* Shermann-Morrison incremental inverse can be done in $O(m^2)$

#### LSTD($\lambda$)

* Likewise, we can do LSTD($\lambda$) using Eligibility Traces
* Denote the Eligibility Trace of atomic experience $i$ as $\bm{E}_i$
* Note: $\bm{E}_i$ accumulates $\nabla_{\bm{w}} V(s;\bm{w}) = \bm{\phi}(s)$ in each trace experience
* When accumulating, previous step's eligibility trace discounted by $\lambda \gamma$
$$\sum_{i=1}^n \bm{E_i} \cdot (\bm{\phi}(s_i)^T \cdot \bm{w^*} - (r_i + \gamma \cdot \bm{\phi}(s'_i)^T \cdot \bm{w}^*)) = 0$$
* $\bm{w}^*$ is solved as $\bm{A}^{-1} \cdot \bm{b}$
* $m \times m$ Matrix $\bm{A}$ is accumulated at each atomic experience $i$:
$$ \bm{A} \leftarrow \bm{A} + \bm{E_i} \cdot (\bm{\phi}(s_i) - \gamma \cdot \bm{\phi}(s'_i))^T \text{ (note the Outer-Product)}$$
* $m$-Vector $\bm{b}$ is accumulated at each atomic experience $(s_i, r_i, s'_i)$ as:
$$\bm{b} \leftarrow \bm{b} + \bm{E_i} \cdot r_i$$
* Shermann-Morrison incremental inverse can be done in $O(m^2)$

### Q-Learning with Experience Replay

```python
from rl.markov_decision_process import TransitionStep
from rl.approximate_dynamic_programming import QValueFunctionApprox

def q_learning_experience_replay(
        transitions: Iterable[TransitionStep[S, A]],
        actions: Callable[[NonTerminal[S]], Iterable[A]],
        approx_0: QValueFunctionApprox[S, A],
        gamma: float
) -> Iterator[QValueFunctionApprox[S, A]]:

    def step(
            q: QValueFunctionApprox[S, A],
            transition: TransitionStep[S, A]
    ) -> QValueFunctionApprox[S, A]:
        next_return: float = max(
            q((transition.next_state, a))
            for a in actions(transition.next_state)
        ) if isinstance(transition.next_state, NonTerminal) else 0.
        return q.update([
            ((transition.state, transition.action),
             transition.reward + gamma * next_return)
        ])

    return iterate.accumulate(transitions, step, initial=approx_0)
```

#### Deep Q-Networks (DQN) Control Algorithm

DeepMind took the idea of Q-Learning with Experience Replay one step further with an RL Control algorithm they named as Deep Q-Networks (abberviated as DQN).

DQN uses *Experience-Replay* and *Fixed Q-learning targets*. At each time $t$ for each episode:
* Given state $S_t$, take action $A_t$ according to $\epsilon$-greedy policy extracted from Q-network values $Q(S_t,a;\bm{w})$
* Given state $S_t$ and action $A_t$, obtain reward $R_{t+1}$ and next state $S_{t+1}$
* Store atomic experience $(S_t, A_t, R_{t+1}, S_{t+1})$ in replay memory $\mathcal{D}$
* Sample random mini-batch  of atomic experiences $(s_i,a_i,r_i,s'_i) \sim \mathcal{D}$
* Update Q-network parameters $\bm{w}$ using Q-learning targets based on ``frozen'' parameters $\bm{w}^-$ of {\em target network}
$$\Delta \bm{w} = \alpha \cdot \sum_i (r_i + \gamma \cdot \max_{a'_i} Q(s'_i, a'_i; \bm{w}^-) - Q(s_i,a_i;\bm{w})) \cdot \nabla_{\bm{w}} Q(s_i,a_i;\bm{w})$$
* $S_t \leftarrow S_{t+1}$
*
Parameters $\bm{w}^-$ of target network infrequently updated to values of Q-network parameters $\bm{w}$ (hence, Q-learning targets treated as "frozen")

