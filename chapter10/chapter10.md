# Reinforcement Learning Algorithms

## Monte-Carlo (MC) and Temporal-Difference (TD) for Prediction {#sec:rl-prediction-chapter}

### Overview of the Reinforcement Learning approach

In Module I, we covered Dynamic Programming (DP) and Approximate Dynamic Programming (ADP) algorithms to solve the problems of Prediction and Control. DP and ADP algorithms assume that we have access to a *model* of the MDP environment (by *model*, we mean the transitions defined by $\mathcal{P}_R$ - notation from Chapter [-@sec:mdp-chapter] - refering to probabilities of next state and reward, given current state and action). However, in real-world situations, we often do not have access to a model of the MDP environment and so, we'd need to access the actual MDP environment directly. As an example, a robotics application might not have access to a model of a certain type of terrain to learn to walk on, and so we'd need to access the actual (physical) terrain. This means we'd need to interact with the actual MDP environment. Note that the actual MDP environment doesn't give us transition probabilities - it simple serves up a new state and reward when we take an action in a certain state. In other words, it gives us sample transitions of next state and reward, rather than the actual probabilities of occurrence of next states and rewards. So, the natural question to ask is whether we can infer the Optimal Value Function/Optimal Policy without access to a model (in the case of Prediction - the question is whether we can infer the Value Function for a given policy). The answer to this question is *Yes* and the algorithms that achieve this are known as Reinforcement Learning algorithms.

It's also important to recognize that even if we had access to a model, a typical real-world environment is non-stationary (meaning the probabilities $\mathcal{P}_R$ change over time) and so, the model would need to be re-estimated periodically. Moreover, real-world models typically have large state spaces and complex transitions structure, and so transition probabilities are either hard to compute or impossible to store/compute (within current storage/compute constraints). This means even if we could theoretically run a DP/ADP algorithm (by estimating a model from interactions with the actual environment), it's typically not feasible in a real-world situation. However, sometimes it's possible to construct a sampling model (a model that serves up samples of next state and reward) even when it's hard/impossible to construct a model of explicit transition probabilities. This means practically there are only two options:

1. The Agent interacts with the actual environment and doesn't bother with either a model of explicit transition probabilities or a model of transition samples.
2. We create a model (from interaction with the actual environment) of transition samples, treating this model as a simulated environment, and hence, the agent interacts with this simulated environment.

From the perspective of the agent, either way there is an environment interface that will serve up (at each time step) a single instance of (next state, reward) pair when the agent performs a certain action in a given state. So essentially, either way, our access is simply to samples rather than explicit probabilities. So, then the question is - at a conceptual level, how does RL go about solving Prediction and Control problems with just this limited access (access to only samples and not explicit probabilities)? This will become clearer and clearer as we make our way through Module III, but it would be a good idea now for us to briefly sketch an intuitive overview of the RL approach (before we dive into the actual RL algorithms).

To understand the core idea of how RL works, we take you back to the start of the book where we went over how a baby learns to walk. Specifically, we'd like you to develop intuition for how humans and other animals learn to perform requisite tasks and behave in appropriate ways, and get trained to make suitable decisions. We (i.e., humans/animals) don't build a model of explicit probabilities in our minds in a way that a DP/ADP algorithm would require. Rather, our learning is essentially a sort of "trial and error" method - we try an action, receive an experience (i.e., next state and reward), take a new action, receive another experience, and so on, and over a period of time, we figure out which actions might be leading to good outcomes (producing good rewards) and which actions might be leading to poor outcomes (poor rewards). This learning process involves raising the priority of actions we perceive as good, and  lowering the priority of actions we perceive as bad. We don't quite link our actions to the immediate reward - we link our actions to the cumulative rewards (*Return*s) obtained after performing an action. Linking actions to cumulative rewards is indeed challenging because multiple actions have significantly overlapping rewards sequences, and often rewards show up in a delayed manner. Indeed, attributing specific actions to good versus bad outcomes is the powerful part of human/animal learning. Humans/animals are essentially estimating a Q-Value Function and are updating their Q-Value function each time they receive a new experience (of essentially a pair of next state and reward). Exactly how humans/animals manage to estimate Q-Value functions efficiently is unclear (a big area of ongoing research), but RL algorithms have specific techniques to estimate the Q-Value function in an incremental manner by updating the Q-Value function in subtle (and sometimes not so subtle) ways after each experience (i.e., after every sample of next state and reward received from either the actual environment or simulated environment).

We should also point out another feature of human/animal learning - it is the fact that humans/animals are good at generalizing their inferences from experiences, i.e., they can interpolate and extrapolate the linkages between their actions and outcomes. Technically, this translates to a suitable function approximation of the Q-Value function. So before we embark on studying the details of various RL algorithms, it's important to recognize that RL overcomes complexity (specifically, the Curse of Dimensionality and Curse of Modeling, as we have alluded to in previous chapters) with a combination of:

1. Learning incrementally by updating the Q-Value function from samples of next state and reward received after performing actions in specific states.
2. Good generalization ability of the Q-Value function with a suitable function approximation (indeed, recent progress in capabilities of deep neural networks have helped considerably).

Lastly, as mentioned in previous chapters, most RL algorithms are founded on the Bellman Equations and all RL Control algorithms are based on the fundamental idea of *Generalized Policy Iteration* that we have explained in Chapter [-@sec:mdp-chapter]. But the exact ways in which the Bellman Equations and Generalized Policy Iteration idea are utilized in RL algorithms differ from one algorithm to another, and they differ significantly from how the Bellman Equations/Generalized Policy Iteration idea is utilized in DP algorithms.

As has been our practice, we start with the Prediction problem (this chapter) and then move to the Control problem (next chapter). 

### RL for Prediction

We re-use a lot of the notation we had developed in Module I. As a reminder, Prediction is the problem of estimating the Value Function of an MDP for a given policy $\pi$. We know from Chapter [-@sec:mdp-chapter] that this is equivalent to estimating the Value Function of the $\pi$-implied MRP. So in this chapter, we assume that we are working with an MRP (rather than an MDP) and we assume that the MRP is available in the form of an interface that serves up a sample of (next state, reward) pair, given a current state. Running this sampling interface in succession gives us a trace consisting of alternating state and reward (which we call an episode):

$$S_0, R_1, S_1, R_2, S_2, \ldots$$

for some starting state $S_0$ for the episode.

Given a sufficient set of such episodes, the RL Prediction problem is to estimate the *Value Function* $V: \mathcal{N} \rightarrow \mathbb{R}$ of the MRP defined as:

$$V(s) = \mathbb{E}[G_t|S_t = s] \text{ for all } s \in \mathcal{N}, \text{ for all } t = 0, 1, 2, \ldots$$

where the *Return* $G_t$ for each $t = 0, 1, 2, \ldots$ is defined as:

$$G_t = \sum_{i=t+1}^{\infty} \gamma^{i-t-1} \cdot R_i = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots = R_{t+1} + \gamma \cdot G_{t+1}$$

We use the above definition of *Return* even for a terminating sequence (say terminating at $t=T$, i.e., $S_T \in \mathcal{T}$), by treating $R_i = 0$ for all $i > T$.

We take you back to the code in Chapter [-@sec:mrp-chapter] where we had set up a `@dataclass TransitionStep` that serves as a building block in the method `simulate_reward` in the abstract class `MarkovRewardProcess`. Let's add a method called `add_return` to `TransitionStep` so we can augment the triple (state, reward, next state) with a return attribute that is comprised of the reward plus gamma times the return from the next state. The `ReturnStep` class (derived from `TransitionStep`) includes the additional attribute named `return_` 

```python
@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: S
    next_state: S
    reward: float

    def add_return(self, gamma: float, return_: float) -> ReturnStep[S]:
        return ReturnStep(
            self.state,
            self.next_state,
            self.reward,
            return_=self.reward + gamma * return_
        )


@dataclass(frozen=True)
class ReturnStep(TransitionStep[S]):
    return_: float
```

Note that `simulate_reward` produces an `Iterator` (i.e. stream) of `TransitionStep` objects (representing a single episode). Let's add a method `reward_traces` to `MarkovRewardProcess` that produces an `Iterator` (stream) of the episodes produced by `simulate_reward`. The RL algorithms we will develop will consume this stream of episodes to learn the requisite Value Function. Note that the input `start_state_distribution` is the specification of the probability distribution of start states (state to start an episode) to draw from for each episode.

```python
    def reward_traces(
            self,
            start_state_distribution: Distribution[S]
    ) -> Iterable[Iterable[TransitionStep[S]]]:
        while True:
            yield self.simulate_reward(start_state_distribution)
```

The code above is in the file [rl/markov_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_process.py).

### Monte-Carlo (MC) Prediction

Monte-Carlo (MC) Prediction is a very simple RL algorithm that performs supervised learning to predict the expected return from a given state of an MRP (i.e., estimates the Value Function of an MRP). Note that we wrote the abstract class `FunctionApprox` in Chapter [-@sec:funcapprox-chapter] for supervised learning that takes data in the form of $(x,y)$ pairs. For this Monte-Carlo prediction problem, the $x$-values are the samples of states across a set of episodes and the $y$-values are the associated returns on the episode (from each state). The following function (in the file [rl/monte_carlo.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/monte_carlo.py)) `evaluate_mrp` performs the requisite supervised learning in an incremental manner, by calling the method `update` of `approx_0: FunctionApprox[S]` on an `Iterator` of (state, return) pairs of each episode. The sample of states across episodes are made available by calling the `reward_traces` method (that we wrote above) of `mrp: MarkovRewardProcess[S]`. Note that `states:Distribution[S]` is the probability distribution of start states of the episodes. `evaluate_mrp` produces an `Iterator` of `FunctionApprox[S]`, i.e., an updated function approximation of the Value Function at the end of each episode.

```python
def evaluate_mrp(
        mrp: MarkovRewardProcess[S],
        states: Distribution[S],
        approx_0: FunctionApprox[S],
        gamma: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[S]]:
    v = approx_0

    for trace in mrp.reward_traces(states):
        steps = returns(trace, gamma, tolerance)
        v = v.update((step.state, step.return_) for step in steps)
        yield v
```

The core of the `evaluate_mrp` function above is the call to the `returns` function (detailed below and available in the file [rl/returns.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/returns.py)). `returns` takes as input `trace` representing an episode (`Iterable` of `TransitionStep`), the discount factor `gamma`, and a `tolerance` that determines how many steps to cover in each episode (as many steps as until $\gamma^{steps} \leq tolerance$ when $\gamma < 1$). `returns` calculates the returns $G_t$ (accumulated rewards) from each sampled state $S_t$ in the episode. The key is to walk backwards from the end of the episode to the start (so as to reuse the calculated returns $G_t$ while walking backwards: $G_t = R_{t+1} + \gamma \cdot G_{t+1}$). Note the use of `itertools.accumulate` to perform this backwards-walk calculation, which in turn uses the `add_return` method in `TransitionStep` to create an instance of `ReturnStep` (that we had written earlier).
   

```python
import itertools
import rl.markov_process as mp

def returns(
        trace: Iterable[mp.TransitionStep[S]],
        gamma: float,
        tolerance: float
) -> Iterator[mp.ReturnStep[S]]:
    trace = iter(trace)

    max_steps = round(math.log(tolerance) / math.log(gamma))
        if gamma < 1 else None
    if max_steps is not None:
        trace = itertools.islice(trace, max_steps * 2)

    *transitions, last_transition = list(trace)

    return_steps = itertools.accumulate(
        reversed(transitions),
        func=lambda next, curr: curr.add_return(gamma, next.return_),
        initial=last_transition.add_return(gamma, 0)
    )
    return_steps = reversed(list(return_steps))

    if max_steps is not None:
        return_steps = itertools.islice(return_steps, max_steps)

    return return_steps
```   


Now we consider a simple case of Monte-Carlo Prediction where the MRP consists of a finite state space with the non-terminal states $\mathcal{N} = \{s_1, s_2, \ldots, s_m\}$. In this case, we represent the Value Function of the MRP in a data structure (dictionary) of (state, expected return) pairs. This is known as "Tabular" Monte-Carlo (more generally as Tabular RL to reflect the fact that we represent the calculated Value Function in a "table", i.e., dictionary). Note that in this case, Monte-Carlo Prediction reduces to a very simple calculation wherein for each state, we simply maintain the average of the episode returns obtained from that state (averaged across episodes), and the average is updated in an incremental manner. Recall from Section [-@sec:tabular-funcapprox-section] of Chapter [-@sec:funcapprox-chapter] that this is exactly what's done in the `Tabular` class (in file [rl/func_approx.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/func_approx.py)). We also recall from Section[-@sec:tabular-funcapprox-section] of Chapter [-@sec:funcapprox-chapter] that `Tabular` implements the interface of the abstract class `FunctionApprox` and so, we can perform Tabular Monte-Carlo Prediction by passing a `Tabular` instance as the `approx0: FunctionApprox` argument to the `evaluate_mrp` function above. The implementation of the `update` method in Tabular is exactly as we desire: it performs an incremental averaging of the episode returns obtained from each state (over a stream of episodes). 

Let us denote $V_n(s_i)$ as the estimate of the Value Function for a state $s_i$ after the $n$-th occurrence of the state $s_i$ (when doing Tabular Monte-Carlo Prediction) and let $Y^{(1)}_i, Y^{(2)}_i, \ldots, Y^{(n)}_i$ be the episode returns associated with the $n$ occurrences of state $s_i$. Let us denote `count_to_weights_func` by $f$, Then, the `Tabular` update at the $n$-th occurrence of state $s_i$ (with it's associated return $Y^{(n)}_i$) is as follows:

\begin{equation}
V_n(s_i) = (1 - f(n)) \cdot V_{n-1}(s_i) + f(n) \cdot Y^{(n)}_i = V_{n-1}(s_i) + f(n) \cdot (Y^{(n)}_i - V_{n-1}(s_i))
\label{eq:tabular-mc-update}
\end{equation}

Thus, we see that the update (change) to the Value Function for a state $s_i$ is equal to $f(n)$ (weight for the latest episode return from state $s_i$) times the difference between the latest episode return ($Y^{(n)}_i$) and the current Value Function estimate $V_{n-1}(s_i)$. This is a good perspective as it tells us how to adjust the Value Function estimate. In the case of the default setting of `count_to_weight_func` as $f(n) = \frac 1 n$, we get:

\begin{equation}
V_n(s_i) = \frac {n-1} n \cdot V_{n-1}(s_i) + \frac 1 n \cdot Y^{(n)}_i = V_{n-1}(s_i) + \frac 1 n \cdot (Y^{(n)}_i - V_{n-1}(s_i))
\label{eq:tabular-mc-update-equal-wt}
\end{equation}

So if we have 9 occurrences of a state with an average episode return of 50 and if the 10th occurrence of the state gives an episode return of 60, then we consider $\frac 1 {10}$ of $60-50$ (equal to 1) and increase the Value Function estimate for the state from 50 to 50+1 = 51.

Opening up the incremental updates in Equation \eqref{eq:tabular-mc-update}, we get:

\begin{align}
V_n(s_i) = & f(n) \cdot Y^{(n)}_i + (1 - f(n)) \cdot f(n-1) \cdot Y^{(n-1)}_i + \ldots \nonumber \\
& + (1-f(n)) \cdot (1-f(n-1)) \cdots (1-f(2)) \cdot f(1) \cdot Y^{(1)}_i \label{eq:tabular-mc-estimate}
\end{align}

In the case of the default setting of `count_to_weights_func` as $f(n) = \frac 1 n$, we get:

\begin{equation}
V_n(s_i) = \frac 1 n \cdot Y^{(n)}_i + \frac {n-1} n\cdot \frac 1 {n-1} \cdot Y^{(n-1)}_i + \ldots + \frac {n-1} n \cdot \frac {n-2} {n-1} \cdots \frac 1 2 \cdot \frac 1 1 \cdot Y^{(1)}_i = \frac {\sum_{k=1}^n Y^{(k)}_i} n
\label{eq:tabular-mc-estimate-equal-wt}
\end{equation}

which is an equally-weighted average of the episode returns from the state.

Note that the `Tabular` class as an implementation of the abstract class `FunctionApprox` is not just a software design happenstance - there is a formal mathematical specialization here that is vital to recognize. This tabular representation is actually a special case of linear function approximation by setting a feature function $\phi_i(\cdot)$ for each $x_i$ as: $\phi_i(x) = 1$ for $x=x_i$ and $\phi_(x) = 0$ for each $x \neq x_i$ (i.e., $\phi_i(x)$ is the indicator function for $x_i$, and the $\bm{\Phi}$ matrix we had refered to in Chapter [-@sec:funcapprox-chapter] reduces to the identity matrix). In using `Tabular` for Monte-Carlo Prediction, the feature functions are the indicator functions for each of the non-terminal states and the linear-approximation parameters $w_i$ are the Value Function estimates for the corresponding non-terminal states.

With this understanding, we can view Tabular RL as a special case of RL with Linear Function Approximation. Moreover, the `count_to_weights_func` attribute of `Tabular` plays the role of the learning rate (as a function of the number of iterations in stochastic gradient descent). This becomes clear if we write Equation \eqref{eq:tabular-mc-update} in terms of parameter updates: write $V_n(s_i)$ as parameter value $w^{(n)}_i$ to denote the $n$-th update to parameter $w_i$ corresponding to state $s_i$, and write $f(n)$ as learning rate $\alpha_n$ for the $n$-th update to $w_i$.

$$w^{(n)}_i = w^{(n-1)}_i + \alpha_n \cdot (Y^{(n)}_i - w^{(n-1)}_i)$$

So, the change in parameter $w_i$ for state $s_i$ is $\alpha_n$ times $Y^{(n)}_i - w^{(n-1)}_i$. We observe that $Y^{(n)}_i - w^{(n-1)}_i$ represents the gradient of the loss function for the data point $(s_i, Y^{(n)}_i)$ in the case of linear function approximation with features as indicator variables (for each state). This is because the loss function for the data point $(s_i, Y^{(n)}_i)$ is $\frac 1 2 \cdot (Y^{(n)}_i - \sum_{j=1}^m \phi_j(s_i) \cdot w_j)^2$ which reduces to $\frac 1 2 \cdot (Y^{(n)}_i - w^{(n-1)}_i)^2$, whose gradient in the direction of $w_i$ is $Y^{(n)}_i - w^{(n-1)}_i$ and 0 in the other directions (for $j \neq i$). So we see that `Tabular` updates are basically a special case of `LinearFunctionApprox` updates if we set the features to be indicator functions for each of the states (with `count_to_weights_func` playing the role of the learning rate).

Now that you recognize that `count_to_weights_func` essentially plays the role of the learning rate and governs the importance given to the latest episode return relative to past episode returns, we want to point out that real-world situations are non-stationary in the sense that the environment typically evolves over a period of time and so, RL algorithms have to appropriately adapt to the changing environment. The way to adapt effectively is to have an element of "forgetfulness" of the past because if one learns about the distant past far too strongly in a changing environment, our predictions (and eventually control) would not be effective. So, how does an RL algorithm "forget"? Well, one can "forget" through an appropriate time-decay of the weights when averaging episode returns. If we set a constant learning rate $\alpha$ (in `Tabular`, this would correspond to `count_to_weights_func=lambda _: alpha`), we'd obtain "forgetfulness" with lower weights for old data points and higher weights for recent data points. This is because with a constant learning rate $\alpha$, Equation \eqref{eq:tabular-mc-estimate} reduces to:

\begin{align*}
V_n(s_i) & = \alpha \cdot Y^{(n)}_i + (1 - \alpha) \cdot \alpha \cdot Y^{(n-1)}_i + \ldots + (1-\alpha)^{n-1} \cdot \alpha \cdot Y^{(1)}_i \\
& = \sum_{j=1}^n \alpha \cdot (1 - \alpha)^{n-j} \cdot Y^{(j)}_i
\end{align*}

which means we have exponentially-decaying weights in the weighted average of the episode returns for any given state.

Note that for $0 < \alpha \leq 1$, the weights sum up to 1 as $n$ tends to infinity, i.e.,

$$\lim_{n\rightarrow \infty} \sum_{j=1}^n \alpha \cdot (1 - \alpha)^{n-j} = \lim_{n\rightarrow \infty} 1 - (1 - \alpha)^n = 1$$

It's worthwhile pointing out that the Monte-Carlo algorithm we've implemented above is known as Each-Visit Monte-Carlo to refer to the fact that we include each occurrence of a state in an episode. So if a particular state appears 10 times in a given episode, we have 10 (state, episode return) pairs that are used to make the update (for just that state)  at the end of that episode. This is in contrast to First-Visit Monte-Carlo in which only the first occurrence of a state is included in the set of (state, episode return) pairs to make an update at the end of the episode. So First-Visit Monte-Carlo needs to keep track of whether a state has already been visited in an episode (repeat occurrences of states in an episode are ignored). We won't implement First-Visit Monte-Carlo in this book, and leave it to you as an exercise.

Now let's write some code to test out our implementation of Monte-Carlo Prediction. To do so, we go back to a simple finite MRP example from Chapter [-@sec:mrp-chapter] - `SimpleInventoryMRPFinite`. The following code creates an instance of the MRP and computes it's Value Function with an exact calculation.

```python
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite

user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0
user_gamma = 0.9

si_mrp = SimpleInventoryMRPFinite(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda,
    holding_cost=user_holding_cost,
    stockout_cost=user_stockout_cost
)
si_mrp.display_value_function(gamma=user_gamma)
```

This prints the following:   

```
{InventoryState(on_hand=0, on_order=0): -35.511,
 InventoryState(on_hand=1, on_order=0): -28.932,
 InventoryState(on_hand=0, on_order=1): -27.932,
 InventoryState(on_hand=0, on_order=2): -28.345,
 InventoryState(on_hand=2, on_order=0): -30.345,
 InventoryState(on_hand=1, on_order=1): -29.345}
 ```
    
 Next, we run Monte-Carlo Prediction by calling `evaluate_mrp` using `Tabular`. Note that Monte-Carlo Prediction is a highly inefficient RL algorithm, so we run a large number of traces (`num_traces = 100000`).

```python
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.function_approx import Tabular, FunctionApprox
from rl.distribution import Choose
from rl.iterate import last
from rl.monte_carlo import evaluate_mrp
from itertools import islice
from pprint import pprint

it: Iterator[FunctionApprox[InventoryState]] = evaluate_mrp(
    mrp=si_mrp,
    states=Choose(set(si_mrp.states())),
    approx_0=Tabular(),
    gamma=user_gamma,
    tolerance=1e-6
)

num_traces = 100000

last_func: FunctionApprox[InventoryState] = last(islice(it, num_traces))
pprint({s: round(last_func.evaluate([s])[0], 3)
        for s in si_mrp.non_terminal_states})
```   
 
This prints the following:

``` 
{InventoryState(on_hand=0, on_order=0): -35.506,
 InventoryState(on_hand=1, on_order=0): -28.933,
 InventoryState(on_hand=0, on_order=1): -27.931,
 InventoryState(on_hand=0, on_order=2): -28.340,
 InventoryState(on_hand=2, on_order=0): -30.343,
 InventoryState(on_hand=1, on_order=1): -29.343}
```   
     
We see that the Value Function computed by Tabular Monte-Carlo Prediction with 100000 episodes is within 0.005 of the exact Value Function.     
     
Chapter in Progress …      
