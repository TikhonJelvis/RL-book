## Policy Gradient Algorithms {#sec:policy-gradient-chapter}

It's time to take stock of what we have learnt so far to set up context for this chapter. So far, we have covered a range of RL Control algorithms, all of which are based on Generalized Policy Iteration (GPI). All of these algorithms perform GPI by learning the Q-Value Function and improving the policy by identifying the action that fetches the best Q-Value (i.e., action value) for each state. Notice that the way we implemented this *best action identification* is by sweeping through all the actions for each state. This works well only if the set of actions for each state is reasonably small. But if the action space is large/continuous, we have to resort to some sort of optimization method to identify the best action for each state.

In this chapter, we cover RL Control algorithms that take a vastly different approach. These Control algorithms are still based on GPI, but the Policy Improvement of their GPI is not based on consulting the Q-Value Function, as has been the case with Control algorithms we covered in the previous two chapters. Rather, the approach in the class of algorithms we cover in this chapter is to directly find the Policy that fetches the "Best Expected Returns". Specifically, the algorithms of this chapter perform a Gradient Ascent on "Expected Returns" with the gradient defined with respect to the parameters of a Policy function approximation. We shall work with a stochastic policy of the form $\pi(s,a; \bm{\theta})$, with $\bm{\theta}$ denoting the parameters of the policy function approximation $\pi$. So we are basically learning this parameterized policy that selects actions without consulting a Value Function. Note that we might still engage a Value Function approximation (call it $Q(s;a; \bm{w})$) in our algorithm, but it's role is to only help learn the policy parameters $\bm{\theta}$ and not to identify the action with the best action-value for each state. So the two function approximations $\pi(s,a;\bm{\theta})$ and $Q(s,a;\bm{w})$ are collaborating to improve the policy using gradient ascent (based on gradient of "expected returns" with respect to $\bm{\theta}$). $\pi(s,a;\bm{\theta})$ is the primary worker here (known as *Actor*) and $Q(s,a;\bm{w})$ is the support worker (known as *Critic*). The Critic parameters $\bm{w}$ are optimized by minimizing a suitable loss function defined in terms of $Q(s, a; \bm{w})$ while the Actor parameters $\bm{\theta}$ are optimized by maximizing a suitable "Expected Returns" function". Note that we still haven't defined what this "Expected Returns" function is (we will do so shortly), but we already see that this idea is appealing for large/continuous action spaces where sweeping through actions is infeasible. We will soon dig into the details of this new approach to RL Control (known as *Policy Gradient*, abbreviated as PG) - for now, it's important to recognize the big picture that PG is basically GPI with Policy Improvement done as a *Policy Gradient Ascent*.

The contrast between the RL Control algorithms covered in the previous two chapters and the algorithms of this chapter actually is part of the following bigger-picture classification of learning algorithms for Control:

* Value Function-based: Here we learn the Value Function (typically with a function approximation for the Value Function) and the Policy is implicit, readily derived from the Value Function (eg: $\epsilon$-greedy).
* Policy-based: Here we learn the Policy (with a function approximation for the Policy), and there is no need to learn a Value Function.
* Actor-Critic: Here we primarily learn the Policy (with a function approximation for the Policy, known as *Actor*), and secondarily learn the Value Function (with a function approximation for the Value Function, known as *Critic*).

PG Algorithms can be Policy-based or Actor-Critic, whereas the Control algorithms we covered in the previous two chapters are Value Function-based.

In this chapter, we start by enumerating the advantages and disadvantages of Policy Gradient Algorithms, state and prove the Policy Gradient Theorem (which provides the fundamental calculation underpinning Policy Gradient Algorithms), then go on to address how to lower the bias and variance in these algorithms, and finally finish with special cases of Policy Gradient algorithms that have found success in practical applications.

### Advantages and Disadvantages of Policy Gradient Algorithms

Let us start by enumerating the advantages of PG algorithms. We've already said that PG algorithms are effective in large action spaces, high-dimensional or continuous action spaces because in such spaces selecting an action by deriving an improved policy from an updating Q-Value function is intractable. A key advantage of PG is that it naturally *explores* because the policy function approximation is configured as a stochastic policy. Moreover, PG finds the best Stochastic Policy. This is not a factor for MDPs since we know that there exists an optimal Deterministic Policy for any MDP but we often deal with Partially-Observable MDPs (POMDPs) in the real-world, for which the set of optimal policies might all be stochastic policies. We have an advantage in the case of MDPs as well since PG algorithms naturally converge to the deterministic policy (the variance in the policy distribution will automatically converge to 0) whereas in Value Function-based algorithms, we have to reduce the $\epsilon$ of the $\epsilon$-greedy policy by-hand and the appropriate declining trajectory of $\epsilon$ is typically hard to figure out by manual tuning. In situations where the policy function is a simpler function compared to the Value Function, we naturally benefit from pursuing Policy-based algorithms than Value Function-based algorithms. Perhaps the biggest advantage of PG algorithms is that prior knowledge of the functional form of the Optimal Policy enables us to structure the known functional form in the function approximation for the policy. Lastly, PG offers numerical benefits as small changes in $\bm{\theta}$ yield small changes in $\pi$, and consequently small changes in the distribution of occurrences of states. This results in stronger convergence guarantees for PG algorithms relative to Value Function-based algorithms.

Now let's understand the disadvantages of PG Algorithms. The main disadvantage of PG Algorithms is that because they are based on gradient ascent, they typically converge to a local optimum whereas Value Function-based algorithms converge to a global optimum. Furthermore, the Policy Evaluation of PG is typically inefficient and can have high variance. Lastly, the Policy Improvements of PG happens in small steps and so, PG algorithms are slow to converge.

### Policy Gradient Theorem

In this section, we start by setting up some notation, and then state and prove the Policy Gradient Theorem (abbreviated as PGT). The PGT provides the key calculation for PG Algorithms.

#### Notation and Definitions

Denoting the discount factor as $\gamma$, we shall assume either episodic sequences with $0 \leq \gamma \leq 1$ or non-episodic (continuing) sequences  with $0 \leq \gamma < 1$. We shall use our usual notation of discrete-time, countable-spaces, stationary MDPs although we can indeed extend PGT and PG Algorithms to more general settings as well. We lighten $\mathcal{P}(s,a,s')$ notation to $\mathcal{P}_{s,s'}^a$ and $\mathcal{R}(s,a)$ notation to $\mathcal{R}_s^a$ because we want to save some space in the very long equations in the derivation of PGT. 

We denote the probability distribution of the starting state as $p_0 : \mathcal{N} \rightarrow [0,1]$. The policy function approximation is denoted as $\pi(s,a;\bm{\theta}) = \mathbb{P}[A_t=a | S_t=s; \bm{\theta}]$.

The PG coverage is quite similar for non-discounted, non-episodic MDPs, by considering the average-reward objective, but we won't cover it in this book.

Now we formalize the "Expected Returns" Objective $J(\bm{\theta})$.

$$J(\bm{\theta}) = \mathbb{E}_{\pi}[\sum_{t=0}^\infty \gamma^t \cdot R_{t+1}]$$

Value Function $V^{\pi}(s)$ and Action Value function $Q^{\pi}(s,a)$ are defined as:
$$V^{\pi}(s) = \mathbb{E}_{\pi}[\sum_{k=t}^\infty \gamma^{k-t} \cdot R_{k+1} | S_t=s] \text{ for all } t = 0, 1, 2, \ldots$$
$$Q^{\pi}(s,a) = \mathbb{E}_{\pi}[\sum_{k=t}^\infty \gamma^{k-t} \cdot R_{k+1} | S_t=s, A_t=a] \text{ for all } t = 0, 1, 2, \ldots$$

$J(\bm{\theta}), V^{\pi}, Q^{\pi}$ are all measures of Expected Returns, so it pays to specify exactly how they differ. $J(\bm{\theta})$ is the Expected Return when following policy $\pi$ (that is parameterized by $\theta$), *averaged over all states $s \in \mathcal{N}$ and all actions $a \in \mathcal{A}$*. The idea is to perform a gradient ascent with $J(\theta)$ as the objective function, with each step in the gradient ascent essentially pushing $\bm{\theta}$ (and hence, $\pi$) in a desirable direction, until $J(\bm{\theta})$ is maximized. $V^{\pi}(s)$ is the Expected Return for a specific state $s \in \mathcal{N}$ when following policy $\pi$ $Q^{\pi}(s,a)$ is the Expected Return for a specific state $s \in \mathcal{N}$ and specific action $a \in \mathcal{A}$ when following policy $\pi$.

We define the *Advantage Function* as:

$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$

The advantage function captures how much more value does a particular action provide relative to the average value across actions (for a given state). The advantage function plays an important role in reducing the variance in PG Algorithms.

Also, $p(s \rightarrow s', t, \pi)$ will be a key function for us in the PGT proof - it denotes the probability of going from state $s$ to $s'$ in $t$ steps by following policy $\pi$.

We express the "Expected Returns" Objective $J(\bm{\theta})$ as follows:

$$J(\bm{\theta}) = \mathbb{E}_{\pi}[\sum_{t=0}^\infty \gamma^t \cdot R_{t+1}] = \sum_{t=0}^\infty \gamma^t \cdot \mathbb{E}_{\pi}[R_{t+1}]$$
$$ = \sum_{t=0}^\infty \gamma^t \cdot \sum_{s \in \mathcal{N}} (\sum_{S_0 \in \mathcal{N}}  p_0(S_0) \cdot p(S_0 \rightarrow s, t, \pi)) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot \mathcal{R}_s^a$$
$$ =  \sum_{s \in \mathcal{N}} (\sum_{S_0 \in \mathcal{N}}  \sum_{t=0}^\infty \gamma^t \cdot p_0(S_0) \cdot p(S_0 \rightarrow s, t, \pi)) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot \mathcal{R}_s^a$$

\begin{definition}
$$J(\bm{\theta}) =  \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot \mathcal{R}_s^a$$
\end{definition}
where
$$\rho^{\pi}(s) = \sum_{S_0 \in \mathcal{N}}  \sum_{t=0}^\infty \gamma^t \cdot p_0(S_0) \cdot p(S_0 \rightarrow s, t, \pi)$$
is the key function (for PG) that we shall refer to as *Discounted-Aggregate State-Visitation Measure*. Note that $\rho^{\pi}(s)$ is a [measure](https://en.wikipedia.org/wiki/Measure_(mathematics)) over the set of non-terminal states, but is not a [probability measure](https://en.wikipedia.org/wiki/Probability_measure). Think of $\rho^{\pi}(s)$ as weights reflecting the relative likelihood of occurrence of states on a trace experience (adjusted for discounting, i.e, lesser importance to reaching a state later on a trace experience).

#### Statement of the Policy Gradient Theorem

The Policy Gradient Theorem (PGT) provides a powerful formulas for the gradient of $J_{\bm{\theta}}$ with respect to $\bm{\theta}$ so we can perform Gradient Ascent. The challenge with this gradient is that $J_{\bm{\theta}}$ depends not only on the selection of actions through policy $\pi$ (parameterized by $\bm{\theta}$), but also on the distribution of occurrence of states (also affected by $\pi$, and hence by $\bm{\theta}$). With knowledge of the functional form of $\pi$ on $\theta$, it is not difficult to evaluate the dependency of actions selection on $\bm{\theta}$, but evaluating the dependency of distribution of occurrence of states on $\bm{\theta}$ is difficult since the environment only provides atomic experiences at a time (and not probabilities of transitions). However, the PGT (below) comes to our rescue because the gradient of $J_{\bm{\theta}}$ with respect to $\bm{\theta}$ involves only the gradient of $\pi$ with respect to $\bm{\theta}$, and not the gradient of distribution of occurrence of states with respect to $\bm{\theta}$. Precisely, we have:


\begin{theorem}[Policy Gradient Theorem]
$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s,a; \bm{\theta}) \cdot Q^{\pi}(s,a)$$
\end{theorem}

As mentioned above, note that $\rho^{\pi}(s)$ (representing the distribution of occurrence of states) depends on $\bm{\theta}$ but there's no $\nabla_{\bm{\theta}} \rho^{\pi}(s)$ term in $\nabla_{\bm{\theta}} J(\bm{\theta})$.

Also note that:
$$\nabla_{\bm{\theta}} \pi(s,a; \bm{\theta}) = \pi(s,a;\bm{\theta}) \cdot \nabla_{\bm{\theta}} \log{\pi(s,a; \bm{\theta})}$$
$\nabla_{\bm{\theta}} \log{\pi(s,a; \bm{\theta})}$ is the [Score function](https://en.wikipedia.org/wiki/Score_(statistics)) (Gradient of log-likelihood) that is commonly-used in Statistics.

Since $\rho^{\pi}$ is the *Discounted-Aggregate State-Visitation Measure*, we can estimate $\nabla_{\bm{\theta}} J(\bm{\theta})$ by calculating $\gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(s,a; \bm{\theta})}) \cdot Q^{\pi}(s,a)$ at each time step in each trace experience, and sum them up over a set of trace experiences (noting that the state occurrence probabilities and action occurrence probabilities are implicit in the trace experiences).

In many PG Algorithms, we estimate $Q^{\pi}(s,a)$ with a function approximation $Q(s,a;\bm{w})$. We will later show how to avoid the estimate bias of $Q(s,a;\bm{w})$.

This numerical estimate of $\nabla_{\bm{\theta}} J(\bm{\theta})$ enables *Policy Gradient Ascent*. Before we prove the PGT, let us look at the score function of some canonical policy function approximations $\pi(s,a; \bm{\theta})$.

#### Canonical $\pi(s,a; \bm{\theta})$ for Finite Action Spaces

For finite action spaces, we often use the Softmax Policy.

Assume $\bm{\theta}$ is an $m$-vector $(\theta_1, \ldots, \theta_m)$ and assume features vector $\bm{\phi}(s,a)$ is given by: $(\phi_1(s,a), \ldots, \phi_m(s,a))$ for all $s \in \mathcal{N}, a \in \mathcal{A}$.

We weight actions using a linear combinations of features, i.e., $\bm{\phi}(s,a)^T \cdot \bm{\theta}$ and we set the action probabilities to be proportional to exponentiated weights, as follows:
$$\pi(s,a; \bm{\theta}) = \frac {e^{\bm{\phi}(s,a)^T \cdot \bm{\theta}}} {\sum_{b \in \mathcal{A}} e^{\bm{\phi}(s,b)^T \cdot \bm{\theta}}} \mbox{ for all } s \in \mathcal{N}, a \in \mathcal{A}$$
Then the score function is given by:
$$\nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) = \bm{\phi}(s,a) - \sum_{b \in \mathcal{A}} \pi(s,b; \bm{\theta}) \cdot \bm{\phi}(s,b) = \bm{\phi}(s,a) - \mathbb{E}_{\pi}[\bm{\phi}(s,\cdot)]$$

The intuitive interpretation is that the score function for an action $a$ represents the "advantage" of the feature value for action $a$ over the mean feature value (across all actions), for a given state $s$.

#### Canonical $\pi(s,a; \bm{\theta})$ for Continuous Action Spaces

For continuous action spaces, we often use a Gaussian distribution for the Policy.

For simplicity, assume the action space is $\mathbb{R}$. Assume $\bm{\theta}$ is an $m$-vector $(\theta_1, \ldots, \theta_m)$ and assume the state features vector $\bm{\phi}(s)$ is given by $(\phi_1(s), \ldots, \phi_m(s))$ for all $s \in \mathcal{N}$.

We set the mean of the gaussian distribution for the Policy as a linear combination of state features, i.e.,  $\bm{\phi}(s)^T \cdot \bm{\theta}$ and we set the variance to be a fixed value, say $\sigma^2$ (or we can make the variance parameterized as well).

The Gaussian policy selects an action $a$ as follows:

$$a \sim \mathcal{N}(\bm{\phi}(s)^T \cdot \bm{\theta}, \sigma^2) \mbox{ for all } s \in \mathcal{N}$$

Then The score function is given by:
$$\nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) = \frac {(a - \bm{\phi}(s)^T \cdot \bm{\theta}) \cdot \bm{\phi}(s)} {\sigma^2}$$

The intuitive interpretation is that the score function for an action $a$ is proportional to the "advantage" of the action $a$ over the mean action (note: each $a \in \mathbb{R}$).

For each of the above two examples (finite action spaces and continuous action spaces), think of the "advantage" of an action as the compass for the Gradient Ascent (i.e., *Score* drives the Ascent) so as to ultimately get to a point where the optimal action is selected for each state.

#### Proof of the Policy Gradient Theorem

We begin the proof by noting that:
$$J(\bm{\theta}) = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot V^{\pi}(S_0)  = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0)$$
Calculate $\nabla_{\bm{\theta}} J(\bm{\theta})$ by it's product parts $\pi(S_0, A_0; \bm{\theta})$ and $Q^{\pi}(S_0, A_0)$.
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot  \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) \\
& + \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \pi(S_0, A_0; \bm{\theta}) \cdot \nabla_{\bm{\theta}} Q^{\pi}(S_0, A_0)
\end{align*}
Now expand $Q^{\pi}(S_0, A_0)$ as:
$$\mathcal{R}_{S_0}^{A_0} + \sum_{S_1 \in \mathcal{N}} \gamma \cdot \mathcal{P}_{S_0, S_1}^{A_0} \cdot V^{\pi}(S_1) \mbox{ (Bellman Policy Equation)}$$
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) \\
& + \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \pi(S_0, A_0; \bm{\theta}) \cdot \nabla_{\bm{\theta}} (\mathcal{R}_{S_0}^{A_0} + \sum_{S_1 \in \mathcal{N}} \gamma \cdot  \mathcal{P}_{S_0,S_1}^{A_0} \cdot V^{\pi}(S_1))
\end{align*}
Note: $\nabla_{\theta} \mathcal{R}_{S_0}^{A_0} = 0$, so remove that term.
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) \\
& + \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \pi(S_0, A_0; \bm{\theta}) \cdot \nabla_{\bm{\theta}} (\sum_{S_1 \in \mathcal{N}} \gamma \cdot \mathcal{P}_{S_0,S_1}^{A_0} \cdot V^{\pi}(S_1))
\end{align*}
Now bring the $\nabla_{\bm{\theta}}$ inside the $\sum_{S_1 \in \mathcal{N}}$ to apply only on $V^{\pi}(S_1)$.
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) \\
& + \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \pi(S_0, A_0; \bm{\theta}) \cdot \sum_{S_1 \in \mathcal{N}} \gamma \cdot \mathcal{P}_{S_0,S_1}^{A_0} \cdot \nabla_{\bm{\theta}} V^{\pi}(S_1)
\end{align*}
Now bring $\sum_{S_0 \in \mathcal{N}}$ and $\sum_{A_0 \in \mathcal{A}}$ inside the $\sum_{S_1 \in \mathcal{N}}$
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) \\
& + \sum_{S_1 \in \mathcal{N}}  \sum_{S_0 \in \mathcal{N}} \gamma \cdot p_0(S_0) \cdot (\sum_{A_0 \in \mathcal{A}} \pi(S_0, A_0; \bm{\theta}) \cdot \mathcal{P}_{S_0,S_1}^{A_0}) \cdot \nabla_{\bm{\theta}}V^{\pi}(S_1)
\end{align*}
$$\text{Note that } \sum_{A_0 \in \mathcal{A}} \pi(S_0, A_0; \bm{\theta}) \cdot \mathcal{P}_{S_0,S_1}^{A_0} = p(S_0 \rightarrow S_1, 1, \pi)$$
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) \\
& + \sum_{S_1 \in \mathcal{N}}  \sum_{S_0 \in \mathcal{N}} \gamma \cdot p_0(S_0) \cdot p(S_0 \rightarrow S_1, 1, \pi) \cdot \nabla_{\bm{\theta}}V^{\pi}(S_1)
\end{align*}
$$\text{Now expand } V^{\pi}(S_1) \text{ to } \sum_{A_1 \in \mathcal{A}} \pi(S_1, A_1; \bm{\theta}) \cdot Q^{\pi}(S_1,A_1)$$.
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) \\
& + \sum_{S_1 \in \mathcal{N}}  \sum_{S_0 \in \mathcal{N}} \gamma \cdot p_0(S_0) \cdot p(S_0 \rightarrow S_1, 1, \pi) \cdot \nabla_{\bm{\theta}} (\sum_{A_1 \in \mathcal{A}} \pi(S_1, A_1; \bm{\theta}) \cdot Q^{\pi}(S_1,A_1))
\end{align*}
We are now back to when we started calculating gradient of $\sum_a \pi \cdot Q^{\pi}$. Follow the same process of splitting $\pi \cdot Q^{\pi}$, then Bellman-expanding $Q^{\pi}$ (to calculate its gradient), and iterate.
$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{S_0 \in \mathcal{N}} p_0(S_0) \cdot \sum_{A_0 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_0, A_0; \bm{\theta}) \cdot Q^{\pi}(S_0, A_0) + $$
$$\sum_{S_1 \in \mathcal{N}} \sum_{S_0 \in \mathcal{N}} \gamma \cdot p_0(S_0) \cdot p(S_0 \rightarrow S_1, 1, \pi) \cdot  (\sum_{A_1 \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_1, A_1; \bm{\theta}) \cdot Q^{\pi}(S_1,A_1) + \ldots)$$
This iterative process leads us to:
$$ \nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{t=0}^\infty \sum_{S_t \in \mathcal{N}} \sum_{S_0 \in \mathcal{N}} \gamma^t \cdot p_0(S_0) \cdot p(S_0 \rightarrow S_t, t, \pi) \cdot \sum_{A_t \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_t, A_t; \bm{\theta}) \cdot Q^{\pi}(S_t,A_t)$$

$$\text{Bring } \sum_{t=0}^{\infty} \text{ inside } \sum_{S_t \in \mathcal{N}} \sum_{S_0 \in \mathcal{N}} \text{ and note that}$$
$$\sum_{A_t \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(S_t, A_t; \bm{\theta}) \cdot Q^{\pi}(S_t,A_t) \text{ is independent of } t$$
$$ \nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{s \in \mathcal{N}} \sum_{S_0 \in \mathcal{N}} \sum_{t=0}^{\infty} \gamma^t \cdot p_0(S_0) \cdot p(S_0 \rightarrow s, t, \pi) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s, a; \bm{\theta}) \cdot Q^{\pi}(s,a)$$
$$\text{Remember that } \sum_{S_0 \in \mathcal{N}} \sum_{t=0}^{\infty} \gamma^t \cdot p_0(S_0) \cdot p(S_0 \rightarrow s, t, \pi) \overset{\mathrm{def}}{=} \rho^{\pi}(s) \mbox{. So,}$$
$$ \nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s, a; \bm{\theta}) \cdot Q^{\pi}(s,a) $$
$$\mathbb{Q.E.D.}$$
