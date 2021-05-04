## Monte-Carlo (MC) and Temporal-Difference (TD) for Control {#sec:rl-control-chapter}

In chapter [-@sec:rl-prediction-chapter], we covered MC and TD algorithms to solve the *Prediction* problem. In this chapter, we cover MC and TD algorithms to solve the *Control* problem. As a reminder, MC and TD algorithms are Reinforcement Learning algorithms that only have access to an individual experience (at a time) of next state and reward when the agent performs an action in a given state. The individual experience could be the result of an interaction with an actual environment or could be served by a simulated environment (as explained at the state of Chapter [-@sec:rl-prediction-chapter]). It also pays to remind that RL algorithms overcome the Curse of Dimensionality and the Curse of Modeling by incrementally updating (learning) an appropriate function approximation of the Value Function from a stream of individual experiences. Hence, large-scale Control problems that are typically seen in the real-world are often tackled by RL.

### Refresher on *Generalized Policy Iteration* (GPI)

We shall soon see that all RL Control algorithms are based on the fundamental idea of *Generalized Policy Iteration* (introduced initially in Chapter [-@sec:dp-chapter]), henceforth abbreviated as GPI. The exact ways in which the GPI idea is utilized in RL algorithms differs from one algorithm to another, and they differ significantly from how the GPI idea is utilized in DP algorithms. So before we get into RL Control algorithms, it's important to ground on the abstract concept of GPI. We now ask you to re-read Section [-@sec:gpi] in Chapter [-@sec:dp-chapter].

To summarize, the key concept in GPI is that we can evaluate the Value Function for a policy with *any* Policy Evaluation method, and we can improve a policy with *any* Policy Improvement method (not necessarily the methods used in the classical Policy Iteration DP algorithm). The word *any* does not simply mean alternative algorithms for Policy Evaluation and/or Policy Improvements - the word *any* also refers to the fact that we can do a "partial" Policy Evaluation or a "partial" Policy Improvement. The word "partial" is used quite generically here - any set of calculations that simply take us *towards* a complete Policy Evaluation qualify. This means GPI allows us to switch from Policy Evaluation to Policy Improvements without doing a complete Policy Evaluation (for instance, we don't have to take Policy Evaluation calculations all the way to convergence). Figure \ref{fig:generalized_policy_iteration_lines_repeat} illustrates Generalized Policy Iteration as the red arrows (versus the black arrows which correspond to usual Policy Iteration algorithm). Note how the red arrows don't go all the way to either the "value function line" or the "policy line" but the red arrows do go some part of the way towards the line they are meant to go towards at that stage in the algorithm.

![Progression Lines of Value Function and Policy in Generalized Policy Iteration \label{fig:generalized_policy_iteration_lines_repeat}](./chapter4/gpi.png "Progression Lines of Value Function and Policy in Policy Iteration and Generalized Policy Iteration")

As has been our norm in the book so far, our approach to RL Control algorithms is by first covering the simple case of Tabular RL Control algorithms to illustrate the core concepts in a simple and intuitive manner. In many Tabular RL Control algorithms (especially Tabular TD Control), GPI consists of the Policy Evaluation step for just a single state (versus for all states in usual Policy Iteration) and the Policy Improvement step is also done for just a single state. So essentially these RL Control algorithms are an alternating sequence of single-state policy evaluation and single-state policy improvement (where the single-state is the state produced by sampling or the state that is encountered in a real-world environment interaction). Similar to the case of Prediction, we first cover Monte-Carlo (MC) Control and then move on to Temporal-Difference (TD) Control. 

### GPI with Evaluation as Monte-Carlo

Let us think about how to do MC Control based on the GPI idea. The natural thought that emerges is to do the Policy Evaluation step with MC (this is basically MC Prediction), followed by greedy Policy Improvement, then MC Policy Evaluation with the improved policy, and so on … This is indeed a valid MC Control algorithm. However, this algorithm is not practical as each Policy Evaluation step typically takes very long to converge (as we have noted in Chapter [-@sec:rl-prediction-chapter]) and the number of iterations of Evaluation and Improvement will also be large. More importantly, this algorithm simply modifies the Policy Iteration DP/ADP algorithm by replacing DP/ADP Policy Evaluation with MC Policy Evaluation - hence, we simply end up with a slower version of the Policy Iteration DP/ADP algorithm. Instead, we seek an MC Algorithm that switches from Policy Evaluation to Policy Improvement without requiring Policy Evaluation to converge (this is essentially the GPI idea). 

MC Policy Evaluation is essentially MC Prediction. So the natural idea for GPI here would be to do the usual MC Prediction updates at the end of an episode, then improve the policy, then perform MC Prediction updates (with improved policy) at the end of the next episode, and so on … This seems like a reasonable idea but there are two reasons this won't quite work and we need to tweak this idea a bit to make it work.

The first problem is that the Greedy Policy Improvement calculation (Equation \ref{eq:greedy-policy-improvement-mc}) requires a model of the state transition probability function $\mathcal{P}$ and the reward function $\mathcal{R}$), which is not available in an RL interface. 

\begin{equation}
\pi_D'(s) = \argmax_{a\in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{N}} \mathcal{P}(s,a,s') \cdot V^{\pi}(s') \} \text{ for all } s \in \mathcal{N}
\label{eq:greedy-policy-improvement-mc}
\end{equation}

However, we note that Equation \ref{eq:greedy-policy-improvement-mc} can be written more succinctly as:

\begin{equation}
\pi_D'(s) = \argmax_{a\in \mathcal{A}} Q^{\pi}(s,a) \text{ for all } s \in \mathcal{N}
\label{eq:greedy-policy-improvement-mc-q}
\end{equation}

This view of Greedy Policy Improvement is valuable to us here because instead of having Policy Evaluation estimate $V^{\pi}$, we instead have Policy Evaluation estimate $Q^{\pi}$. This would mean that we don't need a transition probability model of the MDP and we can easily extract the improved policy $\pi_D'$ from $Q^{\pi}$. In fact, Equation \ref{eq:greedy-policy-improvement-mc-q} tells us that all we need at any time step in any episode is an estimate of the Q-Value Function - the requisite greedy action from any state immediately follows from Equation \ref{eq:greedy-policy-improvement-mc-q}. For ease of understanding, for now, let us just restrict ourselves to the case of Tabular Every-Visit MC Control. In this case, we can simply perform the following two updates at the end of each episode for each $(S_t, A_t)$ pair encountered in the episode (note that at each time step $t$, $A_t$ is based on the greedy policy derived from the current estimate of the Q-Value function):

\begin{equation}
\begin{split}
Count(S_t, A_t) & \leftarrow Count(S_t,A_t) + 1 \\
Q(S_t,A_t) & \leftarrow Q(S_t,A_t) + \frac 1 {Count(S_t,A_t)} \cdot (G_t - Q(S_t,A_t))
\end{split}
\label{eq:tabular-mc-control-updates}
\end{equation}

It's important to note that $Count(S_t, A_t)$ is accumulated over the set of all episodes seen thus far. This means the estimate $Q(S_t,A_t)$ is not an estimate of the Q-Value Function for a single policy - rather it keeps updating as we encounter new greedy policies across the set of episodes.

So is this now our first Tabular RL Control algorithm? Not quite - there is a second problem that we need to understand (and resolve). This problem is more subtle and we illustrate the problem with a simple example. Let's consider a specific state (call it $s$) and assume that there are only two allowable actions $a_1$ and $a_2$ for state $s$. Let's say the true Q-Value Function for state $s$ is: $Q_{true}(s,a_1) = 2, Q_{true}(s,a_2) = 5$. Let's say we initialize the Q-Value Function estimate as: $Q(s,a_1) = Q(s,a_2) = 0$. When we encounter state $s$ for the first time, the action to be taken is arbitrary between $a_1$ and $a_2$ since they both have the same Q-Value estimate (meaning both $a_1$ and $a_2$ yield the same max value for $Q(s,a)$ among the two choices for $a$). Let's say we arbitrarily pick $a_1$ as the action choice and let's say for this first encounter of action $s$ (with the arbitrarily picked action $a_1$), the return obtained is 3. So $Q(s,a_1)$ updates to the value 3. So when the state $s$ is encountered for the second time, we see that $Q(s,a_1) = 3$ and $Q(s,a_2) = 0$ and so, action $a_1$ will be taken according to the greedy policy implied by the estimate of the Q-Value Function. Let's say we now obtain a return of -1, updating $Q(s,a_1)$ to $\frac {3 - 1} 2 = 1$. When $s$ is encountered for the third time, yet again action $a_1$ will be taken according to the greedy policy implied by the estimate of the Q-Value Function. Let's say we now obtain a return of 2, updating $Q(s,a_1)$ to $\frac {3 - 1 + 2} 3 = \frac 4 3$. We see that as long as the returns associated with $a_1$ are not negative enough to make the estimate $Q(s,a_1)$ negative, $a_2$ is "locked out" by $a_1$ because the first few occurrences of $a_1$ happen to yield an average return greater than the initialization of $Q(s,a_2)$. Even if $a_2$ was chosen, it is possible that the first few occurrences of $a_2$ yield an average return smaller than the average return obtained on the first few occurrences of $a_1$, in which case $a_2$ could still get locked-out prematurely. 

We did not encounter this problem with greedy policy improvement in DP Control algorithms because the updates were not based on individual transitions - rather the updates were based on expected values (using transition probabilities). However, when it comes to RL Control, updates can get biased by initial random occurrences of returns, which in turn could prevent certain actions from being sufficiently chosen (thus, disallowing accurate estimates of the Q-Values for those actions). While we do want to *exploit* actions that are fetching higher episode returns, we also want to adequately *explore* all possible actions so we can obtain an accurate-enough estimate of their Q-Values. This is in fact the Explore-Exploit dilemma of the famous [Multi-Armed Bandit Problem](https://en.wikipedia.org/wiki/Multi-armed_bandit). In Chapter [-@sec:multi-armed-bandits-chapter], we will cover the Multi-Armed Bandit problem in detail, along with a variety of techniques to solve the Multi-Armed Bandit problem (which are essentially creative ways of resolving the Explore-Exploit dilemma). We will see in Chapter [-@sec:multi-armed-bandits-chapter] that a simple way of resolving the Explore-Exploit dilemma is with a method known as $\epsilon$-greedy, which essentially means we must be greedy ("exploit") a certain fraction of the time and for the remaining fraction of the time, we explore all possible actions. The term "certain fraction of the time" refers to probabilities of choosing actions, which means an $\epsilon$-greedy policy (generated from a Q-Value Function estimate) will be a stochastic policy. For the sake of simplicity, in this book, we will employ the $\epsilon$-greedy method to resolve the Explore-Exploit dilemma in all RL Control algorithms involving the Explore-Exploit dilemma (although you must understand that we can replace the $\epsilon$-greedy method by the other methods we shall cover in Chapter [-@sec:multi-armed-bandits-chapter] in any of the RL Control algorithms where we run into the Explore-Exploit dilemma). So we need to tweak the Tabular MC Control algorithm described above to perform Policy Improvement with the $\epsilon$-greedy method. The formal definition of the $\epsilon$-greedy stochastic policy $\pi'$ (obtained from the current estimate of the Q-Value Function) is as follows:

$$\text{Improved Stochastic Policy } \pi'(s,a) =
\begin{cases}
\frac {\epsilon} {|\mathcal{A}|} + 1 - \epsilon & \text{ if } a = \argmax_{b \in \mathcal{A}} Q(s, b) \\
\frac {\epsilon} {|\mathcal{A}|} & \text{ otherwise}
\end{cases}
$$

where $\mathcal{A}$ denotes the set of allowable actions.

This says that with probability $1 - \epsilon$, we select the action that maximizes the Q-Value Function estimate for a given state, and with probability $\epsilon$, we uniform-randomly select each of the allowable actions (including the maximizing action). Hence, the maximizing action is chosen with probability $\frac {\epsilon} {|\mathcal{A}|} + 1 - \epsilon$. Note that if $\epsilon$ is zero, $\pi'$ reduces to the deterministic greedy policy $\pi_D'$ that we had defined earlier. So the greedy policy can be considered to be a special case of $\epsilon$-greedy policy with $\epsilon = 0$.

But we haven't yet actually proved that an $\epsilon$-greedy policy is indeed an improved policy. We do this in the theorem below. Note that in the following theorem's proof, we re-use the notation and inductive-proof approach used in the Policy Improvement Theorem (Theorem \ref{th:policy_improvement_theorem}) in Chapter [-@sec:dp-chapter]. So it would be a good idea to re-read the proof of Theorem \ref{th:policy_improvement_theorem} in Chapter [-@sec:dp-chapter] before reading the following theorem's proof.

\begin{theorem}
For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi'$ obtained from $Q^{\pi}$ is an improvement over $\pi$, i.e., $\bm{V}^{\pi'}(s) \geq \bm{V}^{\pi}(s)$ for all $s \in \mathcal{N}$.
\end{theorem}

\begin{proof}
We've previously learnt that for any policy $\pi'$, if we apply the Bellman Policy Operator $\bm{B}^{\pi'}$ repeatedly (starting with $\bvpi$), we converge to $\bm{V}^{\pi'}$. In other words,
$$\lim_{i\rightarrow \infty} (\bm{B}^{\pi'})^i(\bvpi) = \bm{V}^{\pi'}$$
So the proof is complete if we prove that:
$$(\bm{B}^{\pi'})^{i+1}(\bvpi) \geq (\bm{B}^{\pi'})^i(\bvpi) \text{ for all } i = 0, 1, 2, \ldots$$
In plain English, this says we need to prove that repeated application of $\bm{B}^{\pi'}$ produces an increasing tower of Value Functions $[(\bm{B}^{\pi'})^i(\bvpi)|i = 0, 1, 2, \ldots]$.

We prove this by induction. The base case of the proof by induction is to show that $\bm{B}^{\pi'}(\bvpi)  \geq  \bm{V}^{\pi}$

\begin{align*}
\bm{B}^{\pi'}(\bvpi)(s) & = (\bm{\mathcal{R}}^{\pi'} + \gamma \cdot \bm{\mathcal{P}}^{\pi'} \cdot \bvpi)(s)\\
& = \bm{\mathcal{R}}^{\pi'}(s) + \gamma \cdot \sum_{s' \in \mathcal{S}} \bm{\mathcal{P}}^{\pi'}(s,s') \cdot \bvpi(s') \\
& = \sum_{a \in \mathcal{A}} \pi'(s, a) \cdot (\mathcal{R}(s,a) + \gamma \cdot \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bvpi(s')) \\
& = \sum_{a \in \mathcal{A}} \pi'(s, a) \cdot Q^{\pi}(s,a) \\
& = \sum_{a \in \mathcal{A}} \frac {\epsilon} {|\mathcal{A}|} \cdot Q^{\pi}(s,a) + (1 - \epsilon) \cdot \max_{a \in \mathcal{A}} Q^{\pi}(s,a) \\
& \geq \sum_{a \in \mathcal{A}} \frac {\epsilon} {|\mathcal{A}|} \cdot Q^{\pi}(s,a) + (1 - \epsilon) \cdot Q^{\pi}(s,\argmax_{a \in \mathcal{A}} \pi(s,a)) \\
& = \sum_{a \in \mathcal{A}} \pi(s,a) \cdot Q^{\pi}(s,a) \text{ (since $\pi$ is an $\epsilon$-greedy policy)} \\
& = \bm{V}^{\pi}(s) \text{ for all } s \in \mathcal{N}
\end{align*}

This completes the base case of the proof by induction.

The induction step is easy and is proved as a consequence of the monotonicity of the $\bm{B}^{\pi}$ operator (for any $\pi$), which is defined as follows:
$$\text{Monotonicity Property of } \bm{B}^{\pi}: \bm{X} \geq \bm{Y} \Rightarrow \bm{B}^{\pi}(\bm{X}) \geq \bm{B}^{\pi}(\bm{Y})$$
A straightforward application of this monotonicity property provides the induction step of the proof:
$$(\bm{B}^{\pi'})^{i+1}(\bvpi) \geq (\bm{B}^{\pi'})^i(\bvpi) \Rightarrow (\bm{B}^{\pi'})^{i+2}(\bvpi) \geq (\bm{B}^{\pi'})^{i+1}(\bvpi) \text{ for all } i = 0, 1, 2, \ldots $$
This completes the proof.
\end{proof}


### GLIE Monte-Control Control

So to summarize, we've resolved two problems - firstly, we replaced the state-value function estimate with the action-value function estimate and secondly, we replaced greedy policy improvement with $\epsilon$-greedy policy improvement. So our MC Control algorithm will do GPI as follows:

* Do Policy Evaluation with the Q-Value Function with Q-Value updates at the end of each episode.
* Do Policy Improvement with an $\epsilon$-greedy Policy (readily obtained from the Q-Value Function estimate at any step in the algorithm).

So now we are ready to develop the details of the Monte-Control algorithm that we've been seeking. For ease of understanding, we first cover the Tabular version and then we will implement the generalized version with function approximation. Note that an $\epsilon$-greedy policy enables adequate exploration of actions, but we will also need to do adequate exploration of states in order to achieve a suitable estimate of the Q-Value Function. Moreover, as our Control algorithm proceeds and the Q-Value Function estimate gets better and better, we reduce the amount of exploration and eventually (as the number of episodes tend to infinity), we want to have $\epsilon$ (degree of exploration) tend to zero. In fact, this behavior has a catchy acronym associated with it, which we define below:

\begin{definition}
We refer to {\em Greedy In The Limit with Infinite Exploration} (abbreviated as GLIE) as the behavior that has the following two properties:

\begin{enumerate}
\item  All state-action pairs are explored infinitely many times, i.e., for all $s \in \mathcal{N}$, for all $a \in \mathcal{A}$, and $Count_k(s,a)$ denoting the number of occurrences of $(s,a)$ pairs after $k$ episodes:
$$\lim_{k \rightarrow \infty} Count_k(s,a) = \infty$$
\item The policy converges to a greedy policy, i.e., for all $s \in \mathcal{N}$, for all $a \in \mathcal{A}$, and $\pi_k(s,a)$ denoting the $\epsilon$-greedy policy obtained from the Q-Value Function estimate after $k$ episodes:
$$\lim_{k\rightarrow \infty} \pi_k(s,a) = \mathbb{I}_{a = \argmax_{b \in \mathcal{A}} Q(s,b)}$$
\end{enumerate}

\end{definition}

A simple way by which our method of using the $\epsilon$-greedy policy (for policy improvement) can be made GLIE is by reducing $\epsilon$ as a function of number of episodes $k$ as follows:

$$\epsilon_k = \frac 1 k$$

So now we are ready to describe the Tabular MC Control algorithm we've been seeking. We ensure that this algorithm has GLIE behavior and so, we refer to it as *GLIE Tabular Monte-Carlo Control*. The following is the outline of the procedure for each episode (terminating trace experience) in the algorithm:

* Generate the episode with actions sampled from the $\epsilon$-greedy policy $\pi$ obtained from the estimate of the Q-Value Function that is available at the start of the episode. Also, sample the first state of the episode from a uniform distribution of states in $\mathcal{N}$. This ensures infinite exploration of both states and actions. Let's denote the contents of this episode as:
$$S_0, A_0, R_1, S_1, A_1, \ldots, R_T, S_T$$
and define the trace return $G_t$ associated with $(S_t, A_t)$ as:
$$G_t = \sum_{i=t+1}^{T} \gamma^{i-t-1} \cdot R_i = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots \gamma^{T-t-1} \cdot R_T$$
* For each state $S_t$ and action $A_t$ in the episode, perform the following updates at the end of the episode:
$$Count(S_t,A_t) \leftarrow Count(S_t,A_t) + 1$$
$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \frac 1 {Count(S_t, A_t)} \cdot (G_t - Q(S_t,A_t))$$
* Let's say this episode is the $k$-th episode in the sequence of episodes. Then, at the end of the episode, set:
$$\epsilon \leftarrow \frac 1 k$$

We state the following important theorem without proof.

\begin{theorem}
The above-described GLIE Tabular Monte-Carlo Control algorithm converges to the Optimal Action-Value function: $Q(s,a) \rightarrow Q^*(s,a)$ for all $s \in \mathcal{N}$, for all $a \in \mathcal{A}$. Hence, GLIE Tabular Monte-Carlo Control converges to an Optimal (Deterministic) Policy $\pi^*$.
\end{theorem}

The extension from Tabular to Function Approximation of the Q-Value Function is straightforward. The update (change) in the parameters $\bm{w}$ of the Q-Value Function Approximation $Q(s, a; \bm{w})$ is as follows:

\begin{equation}
\Delta \bm{w} = \alpha \cdot (G_t - Q(S_t, A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t, A_t;\bm{w})
\label{eq:mc-control-funcapprox-params-adj}
\end{equation}

where $\alpha$ is the learning rate in the gradient descent and $G_t$ is the trace return from state $S_t$ upon taking action $A_t$ at time $t$ on a trace experience.

Now let us write some code to implement the above description of GLIE Monte-Carlo Control, generalized to handle Function Approximation of the Q-Value Function. As you shall see in the code below, there are a couple of other generalizations from the algorithm outline described above. Let us start by understanding the various arguments to the below function `glie_mc_control`.

* `mdp: MarkovDecisionProcess[S, A]` - This represents the interface to an abstract Markov Decision Process. Note that this interface doesn't provide any access to the transition probabilities or reward function. The core functionality available through this interface are the two `@abstractmethods` `step` and `actions`. The `step` method only allows us to access a sample of the next state and reward pair given the current state and action (since it returns an abstract `Distribution` object). The `actions` method gives us the allowable actions for a given state.
* `states: Distribution[S]` - This represents an arbitrary distribution of the non-terminal states, which in turn allows us to sample the starting state (from this distribution) for each episode.
* `approx_0: FunctionApprox[Tuple[S, A]]` - This represents the initial function approximation of the Q-Value function (that is meant to be updated, in an immutable manner, through the course of the algorithm).
* `gamma: float` - This represents the discount factor to be used in estimating the Q-Value Function.
* `epsilon_as_func_of_episodes: Callable[[int], float]` - This represents the extent of exploration ($\epsilon$) as a function of the number of episodes (allowing us to generalize from our default choice of $\epsilon(k) = \frac 1 k$).
* `episode_length_tolerance: float` - This represents the $tolerance$ that determines the episode length $T$ (the minimum $T$ such that $\gamma^T < tolerance$).

`glie_mc_control` produces a generator (`Iterator`) of Q-Value Function estimates at the end of each episode. The code is fairly self-explanatory. The method `simulate_actions` of `mdp: MarkovDecisionProcess` creates a single sampling trace (i.e., an episode). At the end of each episode, the `update` method of `FunctionApprox` updates the Q-Value Function (creates a new Q-Value Function without mutating the currrent Q-Value Function) using each of the trace returns (and associated state-actions pairs) from the episode. The $\epsilon$-greedy policy is derived from the Q-Value Function estimate by using the function `epsilon_greedy_policy` that is shown below and is quite self-explanatory (`epsilon_greedy_policy` is in the file [rl/markov_decision_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_decision_process.py)).

```python
from rl.markov_decision_process import epsilon_greedy_policy, TransitionStep

def glie_mc_control(
    mdp: MarkovDecisionProcess[S, A],
    states: Distribution[S],
    approx_0: FunctionApprox[Tuple[S, A]],
    gamma: float,
    epislon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    q: FunctionApprox[Tuple[S, A]] = approx_0
    p: Policy[S, A] = epsilon_greedy_policy(q, mdp)
    yield q

    num_episodes: int = 0
    while True:
        trace: Iterable[TransitionStep[S, A]] = \
            mdp.simulate_actions(states, p)
        num_episodes += 1
        q = q.update(
            ((step.state, step.action), step.return_)
            for step in returns(trace, gamma, episode_length_tolerance)
        )
        p = epsilon_greedy_policy(q, mdp, epsilon_as_func_of_episodes(num_episodes))
        yield q
```

The above code is in the file [rl/monte_carlo.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/monte_carlo.py).

```python
from rl.distribution import Bernoulli, Choose, Constant

def epsilon_greedy_policy(
        q: FunctionApprox[Tuple[S, A]],
        mdp: MarkovDecisionProcess[S, A],
        epsilon: float = 0.0
) -> Policy[S, A]:
    explore = Bernoulli(epsilon)

    class QPolicy(Policy[S, A]):
        def act(self, s: S) -> Optional[Distribution[A]]:
            if mdp.is_terminal(s):
                return None

            if explore.sample():
                return Choose(set(mdp.actions(s)))

            _, action = q.argmax((s, a) for a in mdp.actions(s))
            return Constant(action)

    return QPolicy()
```

Let us test this on the simple inventory MDP we wrote in Chapter [-@sec:mdp-chapter].

```python
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap

capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0
gamma: float = 0.9

si_mdp: SimpleInventoryMDPCap = SimpleInventoryMDPCap(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)
```

First let's run Value Iteration so we can determine the true Optimal Value Function and Optimal Policy   


```python
from rl.dynamic_programming import value_iteration_result
true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)
print("True Optimal Value Function")
pprint(true_opt_vf)
print("True Optimal Policy")
print(true_opt_policy)
```

This prints:

```
True Optimal Value Function
{InventoryState(on_hand=0, on_order=0): -34.89484576629397,
 InventoryState(on_hand=1, on_order=0): -28.660950216301437,
 InventoryState(on_hand=0, on_order=1): -27.66095021630144,
 InventoryState(on_hand=0, on_order=2): -27.991890076067463,
 InventoryState(on_hand=2, on_order=0): -29.991890076067463,
 InventoryState(on_hand=1, on_order=1): -28.991890076067467}
True Optimal Policy
For State InventoryState(on_hand=0, on_order=0):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=0, on_order=1):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=0, on_order=2):
  Do Action 0 with Probability 1.000
For State InventoryState(on_hand=1, on_order=0):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=1, on_order=1):
  Do Action 0 with Probability 1.000
For State InventoryState(on_hand=2, on_order=0):
  Do Action 0 with Probability 1.000
```

Now let's run GLIE MC Control with the following parameters:

```python
from rl.function_approx import Tabular
from rl.distribution import Choose

episode_length_tolerance: float = 1e-5
epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: k ** -0.5
initial_learning_rate: float = 0.1
half_life: float = 10000.0
exponent: float = 1.0

initial_qvf_dict: Mapping[Tuple[S, A], float] = {
    (s, a): 0. for s in si_mdp.non_terminal_states for a in si_mdp.actions(s)
}
learning_rate_func: Callable[[int], float] = learning_rate_schedule(
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent
)
qvfs: Iterator[FunctionApprox[Tuple[S, A]]] = glie_mc_control(
    mdp=si_mdp,
    states=Choose(set(si_mdp.non_terminal_states)),
    approx_0=Tabular(
        values_map=initial_qvf_dict,
        count_to_weight_func=learning_rate_func
    ),
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    episode_length_tolerance=episode_length_tolerance
)
```

Now let's fetch the final estimate of the Optimal Q-Value Function after `num_episodes` have run, and extract from it the estimate of the Optimal State-Value Function and the Optimal Policy.

```python
from rl.distribution import Constant
from rl.dynamic_programming import V
import itertools
import rl.iterate as iterate

num_episodes = 10000
final_qvf: FunctionApprox[Tuple[S, A]] = \
    iterate.last(itertools.islice(qvfs, num_episodes))

def get_vf_and_policy_from_qvf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    qvf: FunctionApprox[Tuple[S, A]]
) -> Tuple[V[S], FinitePolicy[S, A]]:
    opt_vf: V[S] = {
        s: max(qvf((s, a)) for a in mdp.actions(s))
        for s in mdp.non_terminal_states
    }
    opt_policy: FinitePolicy[S, A] = FinitePolicy({
        s: Constant(qvf.argmax((s, a) for a in mdp.actions(s))[1])
        for s in mdp.non_terminal_states
    })
    return opt_vf, opt_policy

opt_vf, opt_policy = get_vf_and_policy_from_qvf(
    mdp=si_mdp,
    qvf=final_qvf
)
print(f"GLIE MC Optimal Value Function with {num_episodes:d} episodes")
pprint(opt_vf)
print(f"GLIE MC Optimal Policy with {num_episodes:d} episodes")
print(opt_policy)
```

This prints:

```
GLIE MC Optimal Value Function with 10000 episodes
{InventoryState(on_hand=0, on_order=0): -35.264313848274746,
 InventoryState(on_hand=1, on_order=0): -28.976909203198172,
 InventoryState(on_hand=0, on_order=1): -27.919371014970242,
 InventoryState(on_hand=0, on_order=2): -28.3136884351702,
 InventoryState(on_hand=2, on_order=0): -30.228723325193638,
 InventoryState(on_hand=1, on_order=1): -29.465071124981524}
GLIE MC Optimal Policy with 10000 episodes
For State InventoryState(on_hand=0, on_order=0):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=0, on_order=1):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=0, on_order=2):
  Do Action 0 with Probability 1.000
For State InventoryState(on_hand=1, on_order=0):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=1, on_order=1):
  Do Action 0 with Probability 1.000
For State InventoryState(on_hand=2, on_order=0):
  Do Action 0 with Probability 1.000
```

The code above is in the file [rl/chapter11/simple_inventory_mdp_cap.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter11/simple_inventory.py). Also see the helper functions in [rl/chapter11/control_utils.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter11/control_utils.py) which you can use to run your own experiments and tests for RL Control algorithms.

### SARSA

Just like in the case of RL Prediction, the natural idea is to replace MC Control with TD Control using the TD Target $R_{t+1}  + \gamma \cdot Q(S_{t+1}, A_{t+1}; \bm{w})$ as a biased estimate of $G_t$ when updating $Q(S_t, A_t; \bm{w})$. This means the parameters update in Equation \eqref{eq:mc-control-funcapprox-params-adj} gets modified to the following parameters update:

\begin{equation}
\Delta \bm{w} = \alpha \cdot (R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1}; \bm{w}) - Q(S_t, A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t, A_t;\bm{w})
\label{eq:td-control-funcapprox-params-adj}
\end{equation}

Unlike MC Control where updates are made at the end of each trace experience (i.e., episode), a TD control algorithm can update at the end of each atomic experience. This means the Q-Value Function Approximation is updated after each atomic experience (*continuous learning*), which in turn means that the $\epsilon$-greedy policy will be (automatically) updated at the end of each atomic experience. At each time step $t$ in a trace experience, the current $\epsilon$-greedy policy is used to sample $A_t$ from $S_t$ and is also used to sample $A_{t+1}$ from $S_{t+1}$. Note that in MC Control, the same $\epsilon$-greedy policy is used to sample all the actions from their corresponding states in the trace experience, and so in MC Control, we were able to generate the entire trace experience with the currently available $\epsilon$-greedy policy. However, here in TD Control, we need to generate a trace experience incrementally since the action to be taken from a state depends on the just-updated $\epsilon$-greedy policy (that is derived from the just-updated Q-Value Function).

Just like in the case of RL Prediction, the disadvantage of the TD Target being a biased estimate of the return is compensated by a reduction in the variance of the return estimate. Also, TD Control offers a better speed of convergence (as we shall soon illustrate). Most importantly, TD Control offers the ability to use in situations where we have incomplete trace experiences (happens often in real-world situations where experiments gets curtailed/disrupted) and also, we can use it in situations where there are no terminal states (*continuing traces*). 

Note that Equation \eqref{eq:td-control-funcapprox-params-adj} has the entities

* **S**tate $S_t$
* **A**ction $A_t$
* **R**eward $R_t$
* **S**tate $S_{t+1}$
* **A**ction $A_{t+1}$

which prompted this TD Control algorithm to be named SARSA (for **S**tate-**A**ction-**R**eward-**S**tate-**A**ction). Following our convention from Chapter [-@sec:mdp-chapter], we depict the SARSA algorithm in Figure \ref{fig:sarsa_figure} with states as elliptical-shaped nodes, actions as rectangular-shaped nodes, and the edges as samples from transition probability distribution and $\epsilon$-greedy policy distribution. 

<div style="text-align:center" markdown="1">
![Visualization of SARSA Algorithm \label{fig:sarsa_figure}](./chapter11/sarsa.png "Visualization of SARSA Algorithm")
</div>

Now let us write some code to implement the above-described SARSA algorithm. Let us start by understanding the various arguments to the below function `glie_sarsa`.

* `mdp: MarkovDecisionProcess[S, A]` - This represents the interface to an abstract Markov Decision Process. We want to remind that this interface doesn't provide any access to the transition probabilities or reward function. The core functionality available through this interface are the two `@abstractmethods` `step` and `actions`. The `step` method only allows us to access a sample of the next state and reward pair given the current state and action (since it returns an abstract `Distribution` object). The `actions` method gives us the allowable actions for a given state.
* `states: Distribution[S]` - This represents an arbitrary distribution of the non-terminal states, which in turn allows us to sample the starting state (from this distribution) for each trace experience.
* `approx_0: FunctionApprox[Tuple[S, A]]` - This represents the initial function approximation of the Q-Value function (that is meant to be updated, after each atomic experience, in an immutable manner, through the course of the algorithm).
* `gamma: float` - This represents the discount factor to be used in estimating the Q-Value Function.
* `epsilon_as_func_of_episodes: Callable[[int], float]` - This represents the extent of exploration ($\epsilon$) as a function of the number of episodes.
* `max_episode_length: int` - This represents the number of time steps at which we would curtail a trace experience and start a new one. As we've explained, TD Control doesn't require complete trace experiences, and so we can do as little or as large a number of time steps in a trace experience (`max_episode_length` gives us that control).

`glie_sarsa` produces a generator (`Iterator`) of Q-Value Function estimates at the end of each atomic experience. The `while True` loops over trace experiences. The inner `while` loops over time steps - each of these steps involves the following:

* Given the current `state` and `action`, we obtain a sample of the pair of `next_state` and `reward` (using the `sample` method of the `Distribution` obtained from `mdp.step(state, action)`.
* Obtain the `next_action`  from `next_state` using the function `epsilon_greedy_action` which utilizes the $\epsilon$-greedy policy derived from the current Q-Value Function estimate (referenced by `q`).
* Update the Q-Value Function based on Equation \eqref{eq:td-control-funcapprox-params-adj} (using the `update` method of `q: FunctionApprox[Tuple[S, A]]`). Note that this is an immutable update since we produce an `Iterable` (generator) of the Q-Value Function estimate after each time step.

Before the code for `glie_sarsa`, let's understand the code for `epsilon_greedy_action` which returns an action sampled from the $\epsilon$-greedy policy probability distribution that is derived from the Q-Value Function estimate, given as input a non-terminal state, a Q-Value Function estimate, and $\epsilon$.

```python
from operator import itemgetter
from Distribution import Categorical

def epsilon_greedy_action(
    q: FunctionApprox[Tuple[S, A]],
    nt_state: S,
    actions: Set[A],
    epsilon: float
) -> A:
    greedy_action: A = max(
        ((a, q((nt_state, a))) for a in actions),
        key=itemgetter(1)
    )[0]
    return Categorical(
        {a: epsilon / len(actions) +
         (1 - epsilon if a == greedy_action else 0.) for a in actions}
    ).sample()

def glie_sarsa(
    mdp: MarkovDecisionProcess[S, A],
    states: Distribution[S],
    approx_0: FunctionApprox[Tuple[S, A]],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    q: FunctionApprox[Tuple[S, A]] = approx_0
    yield q
    num_episodes: int = 0
    while True:
        num_episodes += 1
        epsilon: float = epsilon_as_func_of_episodes(num_episodes)
        state: S = states.sample()
        action: A = epsilon_greedy_action(
            q=q,
            nt_state=state,
            actions=set(mdp.actions(state)),
            epsilon=epsilon
        )
        steps: int = 0
        while not mdp.is_terminal(state) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if mdp.is_terminal(next_state):
                q = q.update([((state, action), reward)])
            else:
                next_action: A = epsilon_greedy_action(
                    q=q,
                    nt_state=next_state,
                    actions=set(mdp.actions(next_state)),
                    epsilon=epsilon,
                )
                q = q.update([(
                    (state, action),
                    reward + gamma * q((next_state, next_action))
                )])
                action = next_action
            yield q
            steps += 1
            state = next_state
```

The above code is in the file [rl/td.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/td.py).

Let us test this on the simple inventory MDP we tested GLIE MC Control on (we use the same `si_mdp: SimpleInventoryMDPCap` object and the same parameter values that were set up earlier when testing GLIE MC Control).

```python
max_episode_length: int = 100
epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: k ** -0.5
initial_learning_rate: float = 0.1
half_life: float = 10000.0
exponent: float = 1.0
gamma: float = 0.9

initial_qvf_dict: Mapping[Tuple[S, A], float] = {
    (s, a): 0. for s in si_mdp.non_terminal_states for a in si_mdp.actions(s)
}
learning_rate_func: Callable[[int], float] = learning_rate_schedule(
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent
)
qvfs: Iterator[FunctionApprox[Tuple[S, A]]] = glie_sarsa(
    mdp=si_mdp,
    states=Choose(set(si_mdp.non_terminal_states)),
    approx_0=Tabular(
        values_map=initial_qvf_dict,
        count_to_weight_func=learning_rate_func
    ),
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    max_episode_length=max_episode_length
)
```
Now let's fetch the final estimate of the Optimal Q-Value Function after `num_episodes * max_episode_length` updates of the Q-Value Function, and extract from it the estimate of the Optimal State-Value Function and the Optimal Policy (using the function `get_vf_and_policy_from_qvf` that we had written earlier).

```python
num_updates = num_episodes * max_episode_length

final_qvf: FunctionApprox[Tuple[S, A]] = \
    iterate.last(itertools.islice(qvfs, num_updates))
opt_vf, opt_policy = get_vf_and_policy_from_qvf(
    mdp=si_mdp,
    qvf=final_qvf
)

print(f"GLIE SARSA Optimal Value Function with {num_updates:d} updates")
pprint(opt_vf)
print(f"GLIE SARSA Optimal Policy with {num_updates:d} updates")
print(opt_policy)
```

This prints:

```
GLIE SARSA Optimal Value Function with 1000000 updates
{InventoryState(on_hand=0, on_order=0): -35.08738000125797,
 InventoryState(on_hand=1, on_order=0): -28.86993224244749,
 InventoryState(on_hand=0, on_order=1): -27.824025125495023,
 InventoryState(on_hand=0, on_order=2): -27.93572998295015,
 InventoryState(on_hand=2, on_order=0): -30.25590685806991,
 InventoryState(on_hand=1, on_order=1): -29.2470416465806}
GLIE SARSA Optimal Policy with 1000000 updates
For State InventoryState(on_hand=0, on_order=0):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=0, on_order=1):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=0, on_order=2):
  Do Action 0 with Probability 1.000
For State InventoryState(on_hand=1, on_order=0):
  Do Action 1 with Probability 1.000
For State InventoryState(on_hand=1, on_order=1):
  Do Action 0 with Probability 1.000
For State InventoryState(on_hand=2, on_order=0):
  Do Action 0 with Probability 1.000`
```

We see that this reasonably converges to the true Value Function (and reaches the true Optimal Policy) as produced by Value Iteration (whose results were displayed when we tested GLIE MC Control).

The code above is in the file [rl/chapter11/simple_inventory_mdp_cap.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter11/simple_inventory.py). Also see the helper functions in [rl/chapter11/control_utils.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter11/control_utils.py) which you can use to run your own experiments and tests for RL Control algorithms.

For Tabular GLIE MC Control, we stated a theorem for theoretical guarantee of convergence to the true Optimal Value Function (and hence, true Optimal Policy). Is there something analogous for Tabular GLIE SARSA? This answers in the affirmative with the added condition that we reduce the learning rate according to Robbin-Monro schedule. We state the following theorem without proof.

\begin{theorem}
Tabular SARSA converges to the Optimal Action-Value function, $Q(s,a) \rightarrow Q^*(s,a)$ (hence, converges to an Optimal Deterministic Policy $\pi^*$), under the following conditions:
\begin{itemize}
\item GLIE schedule of policies $\pi_t(s,a)$
\item Robbins-Monro schedule of step-sizes $\alpha_t$
$$\sum_{t=1}^{\infty} \alpha_t = \infty$$
$$\sum_{t=1}^{\infty} \alpha_t^2 < \infty$$
\end{itemize}
\end{theorem}

Now let's compare GLIE MC Control and GLIE SARSA. This comparison is analogous to the comparison in Section [-@sec:bias-variance-convergence] in Chapter [-@sec:rl-prediction-chapter] regarding their bias, variance and convergence properties. Thus, GLIE SARSA carries a biased estimate of the Q-Value Function compared to the unbiased estimate of GLIE MC Control. On the flip side, the TD Target $R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1}; \bm{w})$ has much lower variance than $G_t$ because $G_t$ depends on many random state transitions and random rewards (on the remainder of the trace experience) whose variances accumulate, whereas the TD Target depends on only the next random state transition $S_{t+1}$ and the next random reward $R_{t+1}$. The bad news with GLIE SARSA (due to the bias in it's update) is that with function approximation, it does not always converge to the Optimal Value Function/Policy.

As mentioned in Chapter [-@sec:rl-prediction-chapter], because MC and TD have significant differences in their usage of data, nature of updates, and frequency of updates, it is not even clear how to create a level-playing field when comparing MC and TD for speed of convergence or for efficiency in usage of limited experiences data. The typical comparisons between MC and TD are done with constant learning rates, and it's been determined that practically GLIE SARSA learns faster than GLIE MC Control with constant learning rates. We illustrate this by running GLIE MC Control and GLIE SARSA on `SimpleInventoryMDPCap`, and plot the root-mean-squared-errors (RMSE) of the Q-Value Function estimate averaged across the non-terminal states as a function of batches of episodes (i.e., visualize how the RMSE of the Q-Value Function evolves as the two algorithms progress). This is done by calling the function `compare_mc_sarsa_ql` which is in the file [rl/chapter11/control_utils.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter11/control_utils.py).

![GLIE MC Control and GLIE SARSA Convergence for SimpleInventoryMDPCap \label{fig:mc_sarsa_convergence}](./chapter11/mc_sarsa_convergence.png "GLIE MC Control and GLIE SARSA Convergence for SimpleInventoryMDPCap")

Figure \ref{fig:mc_sarsa_convergence} depicts the convergence for our implementations of GLIE MC Control and GLIE SARSA for a constant learning rate of $\alpha = 0.05$. We produced this Figure by using data from 500 episodes generated from the same `SimpleInventoryMDPCap` object we had created earlier (with same discount factor $\gamma = 0.9$). We plotted the RMSE after each batch of 10 episodes, hence both curves shown in the Figure have 50 RMSE data points plotted. Firstly, we clearly see that MC Control has significantly more variance as evidenced by the choppy MC Control RMSE progression curve. Secondly, we note that MC Control RMSE curve progresses quite quickly in the first few episode batches but is slow to converge after the first few episode batches (relative to the progression of SARSA). This results in SARSA reaching fairly small RMSE quicker than MC Control. This behavior of GLIE SARSA outperforming the comparable GLIE MC Control (with constant learning rate) is typical in most MDP Control problems.

Lastly, it's important to recognize that MC Control is not very sensitive to the initial Value Function while SARSA is more sensitive to the initial Value Function. We encourage you to play with the initial Value Function for this `SimpleInventoryMDPCap` example and evaluate how it affects the convergence speeds.

More generally, we encourage you to play with the `compare_mc_sarsa_ql` function on other MDP choices (ones we have created earlier in this book, or make up your own MDPs) so you can develop good intuition for how GLIE MC Control and GLIE SARSA algorithms converge for a variety of choices of learning rate schedules, initial Value Function choices, choices of discount factor etc.

### SARSA($\lambda$)

Much like how we extended TD Prediction to TD($\lambda$) Prediction, we can extend SARSA to SARSA($\lambda$), which gives us a way to tune the spectrum from MC Control to SARSA using the $\lambda$ parameter. Recall that in order to develop TD($\lambda$) Prediction from TD Prediction, we first developed the $n$-step TD Prediction Algorithm, then the Offline $\lambda$-Return TD Algorithm, and finally the Online TD($\lambda$) Algorithm. We develop an analogous progression from SARSA to SARSA($\lambda$).

So the first thing to do is to extend SARSA to 2-step-bootstrapped SARSA, whose update is as follows:

$$\Delta \bm{w} = \alpha \cdot (R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot Q(S_{t+2}, A_{t+2}; \bm{w}) - Q(S_t, A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t, A_t;\bm{w})$$

Generalizing this to $n$-step-bootstrapped SARSA, the update would then be as follows:

$$\Delta \bm{w} = \alpha \cdot (G_{t,n} - Q(S_t, A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t, A_t;\bm{w})$$
where the $n$-step-bootstrapped Return $G_{t,n}$ is defined as:
\begin{align*}
G_{t,n} & = \sum_{i=t+1}^{t+n} \gamma^{i-t-1} \cdot R_i  + \gamma^n \cdot Q(S_{t+n}, A_{t+n}; \bm{w}) \\
& = R_{t+1} + \gamma \cdot R_{t+2} + \ldots + \gamma^{n-1} \cdot R_{t+n} + \gamma^n \cdot Q(S_{t+n}, A_{t+n}; \bm{w})
\end{align*}

Instead of $G_{t,n}$, a valid target is a weighted-average target:
$$\sum_{n=1}^N u_n \cdot G_{t,n} + u \cdot G_t \text{ where } u + \sum_{n=1}^N u_n = 1$$
Any of the $u_n$ or $u$ can be 0, as long as they all sum up to 1. The $\lambda$-Return target is a special case of weights $u_n$ and $u$, defined as follows:
$$u_n = (1 - \lambda) \cdot \lambda^{n-1} \text{ for all } n = 1, \ldots, T-t-1$$
$$u_n = 0 \text{ for all } n \geq T-t \text{ and } u = \lambda^{T-t-1}$$
We denote the $\lambda$-Return target as $G_t^{(\lambda)}$, defined as:
$$G_t^{(\lambda)} = (1-\lambda) \cdot \sum_{n=1}^{T-t-1} \lambda^{n-1} \cdot G_{t,n} + \lambda^{T-t-1} \cdot G_t$$

Then, the Offline $\lambda$-Return SARSA Algorithm makes the following updates (performed at the end of each trace experience) for each $(S_t,A_t)$ encountered in the episode:
$$\Delta \bm{w} = \alpha \cdot (G_t^{(\lambda)} - Q(S_t, A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t, A_t;\bm{w})$$

Finally, we create the SARSA($\lambda)$ Algorithm, which is the online "version" of the above $\lambda$-Return SARSA Algorithm. The calculations/updates at each time step $t$ are as follows:

$$\delta_t = R_{t+1} + \gamma \cdot Q(S_{t+1},A_{t+1};\bm{w}) - Q(S_t,A_t;\bm{w})$$
$$\bm{E}_t = \gamma \lambda \cdot \bm{E}_{t-1} + \nabla_{\bm{w}} Q(S_t,A_t;\bm{w})$$
$$\Delta \bm{w} = \alpha \cdot \delta_t \cdot \bm{E}_t$$

The eligibility trace is reset to 0 at the start of each trace experience, i.e., $\bm{E}_0 = 0$.
Note that just like in SARSA, the $\epsilon$-greedy policy improvement is automatic from updated Q-Value Function estimate after each time step.

We leave the implementation of SARSA($\lambda$) in Python code as an exercise for you to do.

### Off-Policy Control

All control algorithms face a tension between wanting to learn Q-Values contingent on *subsequent optimal behavior* and wanting to explore all actions. This almost seems contradictory because the quest for exploration deters one from optimal behavior. Our approach so far of pursuing an $\epsilon$-greedy policy (to be thought of as an *almost optimal* policy) is a hack to resolve this tension. A cleaner approach is to use two separate policies for the two separate goals of wanting to be optimal and wanting to explore. The first policy is the one that we learn about and that becomes the optimal policy - we call this policy the *Target Policy* (to signify the "target" of Control). The second policy is the one that behaves in an exploratory manner, so we can obtain sufficient data for all actions, enabling us to adequately estimate the Q-Value Function - we call this policy the *Behavior Policy*.

In SARSA, at a given time step, we are in a current state $S$, take action $A$, after which we obtain the reward $R$ and next state $S'$, upon which we take the next action $A'$. The action $A$ taken from the current state $S$ is meant to come from an exploratory policy (behavior policy) so that for each state $S$, we have adequate occurrences of all actions in order to accurately estimate the Q-Value Function. The action $A'$ taken from the next state $S'$ is meant to come from the target policy as we aim for *subsequent optimal behavior* ($Q^*(S, A)$ requires optimal behavior subsequent to taking action $A$). However, in the SARSA algorithm, the behavior policy producing $A$ from $S$ and the target policy producing $A'$ from $S'$ are in fact the same policy - the $\epsilon$-greedy policy. Algorithms such as SARSA in which the behavior policy is the same as the target policy are refered to as On-Policy Algorithms to indicate the fact that the behavior used to generate data (experiences) does not deviate from the policy we are aiming for (target policy, which drives towards the optimal policy). 

The separation of behavior policy and target policy as two separate policies gives us algorithms that are known as Off-Policy Algorithms to indicate the fact that the behavior policy is allowed to "deviate off" from the target policy. This separation enables us to construct more general and more powerful RL algorithms. We will use the notation $\pi$ for the target policy and the notation $\mu$ for the behavior policy - therefore, we say that Off-Policy algorithms estimate the Value Function for target policy $\pi$ while following behavior policy $\mu$. Off-Policy algorithms can be very valuable in real-world situations where we can learn the target policy $\pi$ by observing humans or other AI agents following a behavior policy $\mu$. Another great practical benefit is to be able to re-use prior experiences that were generated from old policies, say $\pi_1, \pi_2, \ldots$. Yet another powerful benefit is that we can learn multiple policies $\mu_1, \mu_2, \ldots$ while following one policy $\pi$. Let's now make the concept of Off-Policy Learning concrete by covering the most basic (and most famous) Off-Policy Control Algorithm, which goes by the name of Q-Learning.

#### Q-Learning

The best way to understand the (Off-Policy) Q-Learning algorithm is to tweak SARSA to make it Off-Policy. Instead of having both the action $A$ and the next action $A'$ being generated by the same $\epsilon$-greedy policy, we generate (i.e., sample) action $A$ (from state $S$) using an exploratory behavior policy $\mu$ and we generate the next action $A'$ (from next state $S'$) using the target policy $\pi$. The behavior policy can be any policy as long as it is exploratory enough to be able to obtain sufficient data for all actions (in order to obtain an adequate estimate of the Q-Value Function). Note that in SARSA, when we roll over to the next time step, the new time step's state $S$ is set to be equal to the previous time step's next state $S'$ and the new time step's action $A$ is set to be equal to the previous time step's next action $A'$. However, in Q-Learning, we simply set the new time step's state $S$ to be equal to the previous time step's next state $S'$. The action $A$ for the next time step will be generated using the behavior policy $\mu$, and won't be equal to the previous time step's next action $A'$ (that would have been generated using the target policy $\pi$).

This Q-Learning idea of two separate policies - behavior policy and target policy - is fairly generic, and can be used in algorithms beyond solving the Control problem. However, here we are interested in Q-Learning for Control and so, we want to ensure that the target policy eventually becomes the optimal policy. One straightfoward way to accomplish this is to make the target policy equal to the deterministic greedy policy derived from the Q-Value Function estimate at every step. Thus, the update for Q-Learning Control algorithm is as follows:

$$\Delta \bm{w} = \alpha \cdot \delta_t \cdot \nabla_{\bm{w}} Q(S_t, A_t; \bm{w})$$
where

\begin{align*}
\delta_t & = R_{t+1} + \gamma \cdot Q(S_{t+1}, \argmax_{a \in \mathcal{A}} Q(S_{t+1}, a; \bm{w}); \bm{w}) - Q(S_t, A_t;\bm{w}) \\
& = R_{t+1} + \gamma \cdot \max_{a \in \mathcal{A}} Q(S_{t+1}, a; \bm{w}) - Q(S_t, A_t;\bm{w})
\end{align*}

Although we have highlighted some attractive features of Q-Learning (on account of being Off-Policy), it turns out that Q-Learning when combined with function approximation of the Q-Value Function leads to convergence issues (more on this later). However, Tabular Q-Learning Control converges under the usual appropriate conditions. There is considerable literature on convergence of Tabular Q-Learning Control and we won't go over those convergence theorems in this book - here it suffices to say that the convergence proofs for Tabular Q-Learning Control require infinite exploration of all (state, action) pairs and appropriate stochastic approximation conditions for step sizes.

#### Importance Sampling
