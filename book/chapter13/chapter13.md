## Policy Gradient Algorithms {#sec:policy-gradient-chapter}

It's time to take stock of what we have learnt so far to set up context for this chapter. So far, we have covered a range of RL Control algorithms, all of which are based on Generalized Policy Iteration (GPI). All of these algorithms perform GPI by learning the Q-Value Function and improving the policy by identifying the action that fetches the best Q-Value (i.e., action value) for each state. Notice that the way we implemented this *best action identification* is by sweeping through all the actions for each state. This works well only if the set of actions for each state is reasonably small. But if the action space is large/continuous, we have to resort to some sort of optimization method to identify the best action for each state.

In this chapter, we cover RL Control algorithms that take a vastly different approach. These Control algorithms are still based on GPI, but the Policy Improvement of their GPI is not based on consulting the Q-Value Function, as has been the case with Control algorithms we covered in the previous two chapters. Rather, the approach in the class of algorithms we cover in this chapter is to directly find the Policy that fetches the "Best Expected Returns". Specifically, the algorithms of this chapter perform a Gradient Ascent on "Expected Returns" with the gradient defined with respect to the parameters of a Policy function approximation. We shall work with a stochastic policy of the form $\pi(s,a; \bm{\theta})$, with $\bm{\theta}$ denoting the parameters of the policy function approximation $\pi$. So we are basically learning this parameterized policy that selects actions without consulting a Value Function. Note that we might still engage a Value Function approximation (call it $Q(s;a; \bm{w})$) in our algorithm, but it's role is to only help learn the policy parameters $\bm{\theta}$ and not to identify the action with the best action-value for each state. So the two function approximations $\pi(s,a;\bm{\theta})$ and $Q(s,a;\bm{w})$ are collaborating to improve the policy using gradient ascent (based on gradient of "expected returns" with respect to $\bm{\theta}$). $\pi(s,a;\bm{\theta})$ is the primary worker here (known as *Actor*) and $Q(s,a;\bm{w})$ is the support worker (known as *Critic*). The Critic parameters $\bm{w}$ are optimized by minimizing a suitable loss function defined in terms of $Q(s, a; \bm{w})$ while the Actor parameters $\bm{\theta}$ are optimized by maximizing a suitable "Expected Returns" function". Note that we still haven't defined what this "Expected Returns" function is (we will do so shortly), but we already see that this idea is appealing for large/continuous action spaces where sweeping through actions is infeasible. We will soon dig into the details of this new approach to RL Control (known as *Policy Gradient*, abbreviated as PG) - for now, it's important to recognize the big picture that PG is basically GPI with Policy Improvement done as a *Policy Gradient Ascent*.

The contrast between the RL Control algorithms covered in the previous two chapters and the algorithms of this chapter actually is part of the following bigger-picture classification of learning algorithms for Control:

* Value Function-based: Here we learn the Value Function (typically with a function approximation for the Value Function) and the Policy is implicit, readily derived from the Value Function (eg: $\epsilon$-greedy).
* Policy-based: Here we learn the Policy (with a function approximation for the Policy), and there is no need to learn a Value Function.
* Actor-Critic: Here we primarily learn the Policy (with a function approximation for the Policy, known as *Actor*), and secondarily learn the Value Function (with a function approximation for the Value Function, known as *Critic*).

PG Algorithms can be Policy-based or Actor-Critic, whereas the Control algorithms we covered in the previous two chapters are Value Function-based.

In this chapter, we start by enumerating the advantages and disadvantages of Policy Gradient Algorithms, state and prove the Policy Gradient Theorem (which provides the fundamental calculation underpinning Policy Gradient Algorithms), then go on to address how to lower the bias and variance in these algorithms, give an overview of special cases of Policy Gradient algorithms that have found success in practical applications, and finish with a description of Evolutionary Strategies (that although technically not RL) resemble Policy Gradient algorithms and are quite effective in solving certain Control problems.

### Advantages and Disadvantages of Policy Gradient Algorithms

Let us start by enumerating the advantages of PG algorithms. We've already said that PG algorithms are effective in large action spaces, high-dimensional or continuous action spaces because in such spaces selecting an action by deriving an improved policy from an updating Q-Value function is intractable. A key advantage of PG is that it naturally *explores* because the policy function approximation is configured as a stochastic policy. Moreover, PG finds the best Stochastic Policy. This is not a factor for MDPs since we know that there exists an optimal Deterministic Policy for any MDP but we often deal with Partially-Observable MDPs (POMDPs) in the real-world, for which the set of optimal policies might all be stochastic policies. We have an advantage in the case of MDPs as well since PG algorithms naturally converge to the deterministic policy (the variance in the policy distribution will automatically converge to 0) whereas in Value Function-based algorithms, we have to reduce the $\epsilon$ of the $\epsilon$-greedy policy by-hand and the appropriate declining trajectory of $\epsilon$ is typically hard to figure out by manual tuning. In situations where the policy function is a simpler function compared to the Value Function, we naturally benefit from pursuing Policy-based algorithms than Value Function-based algorithms. Perhaps the biggest advantage of PG algorithms is that prior knowledge of the functional form of the Optimal Policy enables us to structure the known functional form in the function approximation for the policy. Lastly, PG offers numerical benefits as small changes in $\bm{\theta}$ yield small changes in $\pi$, and consequently small changes in the distribution of occurrences of states. This results in stronger convergence guarantees for PG algorithms relative to Value Function-based algorithms.

Now let's understand the disadvantages of PG Algorithms. The main disadvantage of PG Algorithms is that because they are based on gradient ascent, they typically converge to a local optimum whereas Value Function-based algorithms converge to a global optimum. Furthermore, the Policy Evaluation of PG is typically inefficient and can have high variance. Lastly, the Policy Improvements of PG happen in small steps and so, PG algorithms are slow to converge.

### Policy Gradient Theorem

In this section, we start by setting up some notation, and then state and prove the Policy Gradient Theorem (abbreviated as PGT). The PGT provides the key calculation for PG Algorithms.

#### Notation and Definitions

Denoting the discount factor as $\gamma$, we shall assume either episodic sequences with $0 \leq \gamma \leq 1$ or non-episodic (continuing) sequences  with $0 \leq \gamma < 1$. We shall use our usual notation of discrete-time, countable-spaces, time-homogeneous MDPs although we can indeed extend PGT and PG Algorithms to more general settings as well. We lighten $\mathcal{P}(s,a,s')$ notation to $\mathcal{P}_{s,s'}^a$ and $\mathcal{R}(s,a)$ notation to $\mathcal{R}_s^a$ because we want to save some space in the very long equations in the derivation of PGT. 

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
is the key function (for PG) that we shall refer to as *Discounted-Aggregate State-Visitation Measure*. Note that $\rho^{\pi}(s)$ is a [measure](https://en.wikipedia.org/wiki/Measure_(mathematics)) over the set of non-terminal states, but is not a [probability measure](https://en.wikipedia.org/wiki/Probability_measure). Think of $\rho^{\pi}(s)$ as weights reflecting the relative likelihood of occurrence of states on a trace experience (adjusted for discounting, i.e, lesser importance to reaching a state later on a trace experience). We can still talk about the distribution of states under the measure $\rho^{\pi}$, but we say that this distribution is *improper* to convey the fact that $\sum_{s \in \mathcal{N}} \rho^{\pi}(s) \neq 1$ (i.e., the distribution is not normalized). We talk about this improper distribution of states under the measure $\rho^{\pi}$ so we can use (as a convenience) the "expected value" notation for any random variable $f: \mathcal{N} \rightarrow \mathbb{R}$ under the improper distribution, i.e., we use the notation:
$$\mathbb{E}_{s \sim \rho^{\pi}} [f(s)] = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot f(s)$$
Using this notation, we can re-write the above definition of $J(\bm{\theta})$ as:
$$J(\bm{\theta}) =  \mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi} [\mathcal{R}_s^a]$$

#### Statement of the Policy Gradient Theorem

The Policy Gradient Theorem (PGT) provides a powerful formula for the gradient of $J({\bm{\theta}})$ with respect to $\bm{\theta}$ so we can perform Gradient Ascent. The key challenge is that $J({\bm{\theta}})$ depends not only on the selection of actions through policy $\pi$ (parameterized by $\bm{\theta}$), but also on the probability distribution of occurrence of states (also affected by $\pi$, and hence by $\bm{\theta}$). With knowledge of the functional form of $\pi$ on $\theta$, it is not difficult to evaluate the dependency of actions selection on $\bm{\theta}$, but evaluating the dependency of the probability distribution of occurrence of states on $\bm{\theta}$ is difficult since the environment only provides atomic experiences at a time (and not probabilities of transitions). However, the PGT (below) comes to our rescue because the gradient of $J({\bm{\theta}})$ with respect to $\bm{\theta}$ involves only the gradient of $\pi$ with respect to $\bm{\theta}$, and not the gradient of the probability distribution of occurrence of states with respect to $\bm{\theta}$. Precisely, we have:


\begin{theorem}[Policy Gradient Theorem]
$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s,a; \bm{\theta}) \cdot Q^{\pi}(s,a)$$
\label{th:policy-gradient-theorem}
\end{theorem}

As mentioned above, note that $\rho^{\pi}(s)$ (representing the discounting-adjusted probability distribution of occurrence of states, ignoring normalizing factor turning the $\rho^{\pi}$ measure into a probability measure) depends on $\bm{\theta}$ but there's no $\nabla_{\bm{\theta}} \rho^{\pi}(s)$ term in $\nabla_{\bm{\theta}} J(\bm{\theta})$.

Also note that:
$$\nabla_{\bm{\theta}} \pi(s,a; \bm{\theta}) = \pi(s,a;\bm{\theta}) \cdot \nabla_{\bm{\theta}} \log{\pi(s,a; \bm{\theta})}$$
$\nabla_{\bm{\theta}} \log{\pi(s,a; \bm{\theta})}$ is the [Score function](https://en.wikipedia.org/wiki/Score_(statistics)) (Gradient of log-likelihood) that is commonly used in Statistics.

Since $\rho^{\pi}$ is the *Discounted-Aggregate State-Visitation Measure*, we can estimate $\nabla_{\bm{\theta}} J(\bm{\theta})$ by calculating $\gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t,A_t; \bm{\theta})}) \cdot Q^{\pi}(S_t,A_t)$ at each time step in each trace experience (noting that the state occurrence probabilities and action occurrence probabilities are implicit in the trace experiences, and ignoring the probability measure-normalizing factor), and update the parameters $\bm{\theta}$ (according to Stochastic Gradient Ascent) using each atomic experience's $\nabla_{\bm{\theta}} J(\bm{\theta})$ estimate.

We typically calculate the Score $\nabla_{\bm{\theta}} \log{\pi(s,a; \bm{\theta})}$ using an analytically-convenient functional form for the conditional probability distribution $a|s$ (in terms of $\bm{\theta}$) so that the derivative of the logarithm of this functional form is analytically tractable (this will be clear in the next section when we consider a couple of examples of canonical functional forms for $a|s$). In many PG Algorithms, we estimate $Q^{\pi}(s,a)$ with a function approximation $Q(s,a;\bm{w})$. We will later show how to avoid the estimate bias of $Q(s,a;\bm{w})$.

Thus, the PGT enables a numerical estimate of $\nabla_{\bm{\theta}} J(\bm{\theta})$ which in turn enables *Policy Gradient Ascent*.

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

This proof is borrowed from the Appendix of [the famous paper by Sutton, McAllester, Singh, Mansour on Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) [@sutton2001policy].

Note that using the "Expected Value"" notation under the improper distribution implied by the Discounted-Aggregate State-Visitation Measure $\rho^{\pi}$, we can write the statement of PGT as:


\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a;\bm{\theta}) \cdot (\nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta})) \cdot Q^{\pi}(s,a) \\
& =  \mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi} [(\nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta})) \cdot Q^{\pi}(s,a)]
\end{align*}

As explained earlier, since the state occurrence probabilities and action occurrence probabilities are implicit in the trace experiences, we can estimate $\nabla_{\bm{\theta}} J(\bm{\theta})$ by calculating $\gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t,A_t; \bm{\theta})}) \cdot Q^{\pi}(S_t,A_t)$ at each time step in each trace experience, and update the parameters $\bm{\theta}$ (according to Stochastic Gradient Ascent) with this calculation.

### Score function for Canonical Policy Functions {#sec:canonical-policy-functions}

Now we illustrate how the Score function $\nabla_{\bm{\theta}} \pi(s,a;\bm{\theta})$ is calculated using an analytically-convenient functional form for the conditional probability distribution $a|s$ (in terms of $\bm{\theta}$) so that the derivative of the logarithm of this functional form is analytically tractable. We do this with a couple of canonical functional forms for $a|s$, one for finite action spaces and one for single-dimensional continuous action spaces.

#### Canonical $\pi(s,a; \bm{\theta})$ for Finite Action Spaces

For finite action spaces, we often use the Softmax Policy. Assume $\bm{\theta}$ is an $m$-vector $(\theta_1, \ldots, \theta_m)$ and assume features vector $\bm{\phi}(s,a)$ is given by: $(\phi_1(s,a), \ldots, \phi_m(s,a))$ for all $s \in \mathcal{N}, a \in \mathcal{A}$.

We weight actions using a linear combinations of features, i.e., $\bm{\phi}(s,a)^T \cdot \bm{\theta}$ and we set the action probabilities to be proportional to exponentiated weights, as follows:
$$\pi(s,a; \bm{\theta}) = \frac {e^{\bm{\phi}(s,a)^T \cdot \bm{\theta}}} {\sum_{b \in \mathcal{A}} e^{\bm{\phi}(s,b)^T \cdot \bm{\theta}}} \mbox{ for all } s \in \mathcal{N}, a \in \mathcal{A}$$
Then the score function is given by:
$$\nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) = \bm{\phi}(s,a) - \sum_{b \in \mathcal{A}} \pi(s,b; \bm{\theta}) \cdot \bm{\phi}(s,b) = \bm{\phi}(s,a) - \mathbb{E}_{\pi}[\bm{\phi}(s,\cdot)]$$

The intuitive interpretation is that the score function for an action $a$ represents the "advantage" of the feature value for action $a$ over the mean feature value (across all actions), for a given state $s$.

#### Canonical $\pi(s,a; \bm{\theta})$ for Single-Dimensional Continuous Action Spaces

For single-dimensional continuous action spaces (i.e., $\mathcal{A} = \mathbb{R}$), we often use a Gaussian distribution for the Policy. Assume $\bm{\theta}$ is an $m$-vector $(\theta_1, \ldots, \theta_m)$ and assume the state features vector $\bm{\phi}(s)$ is given by $(\phi_1(s), \ldots, \phi_m(s))$ for all $s \in \mathcal{N}$.

We set the mean of the gaussian distribution for the Policy as a linear combination of state features, i.e.,  $\bm{\phi}(s)^T \cdot \bm{\theta}$, and we set the variance to be a fixed value, say $\sigma^2$. We could make the variance parameterized as well, but let's work with fixed variance to keep things simple.

The Gaussian policy selects an action $a$ as follows:

$$a \sim \mathcal{N}(\bm{\phi}(s)^T \cdot \bm{\theta}, \sigma^2) \mbox{ for all } s \in \mathcal{N}$$

Then the score function is given by:
$$\nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) = \frac {(a - \bm{\phi}(s)^T \cdot \bm{\theta}) \cdot \bm{\phi}(s)} {\sigma^2}$$

This is easily extensible to multi-dimensional continuous action spaces by considering a multi-dimensional gaussian distribution for the Policy.

The intuitive interpretation is that the score function for an action $a$ is proportional to the "advantage" of the action $a$ over the mean action (note: each $a \in \mathbb{R}$), for a given state $s$.

For each of the above two examples (finite action spaces and continuous action spaces), think of the "advantage" of an action as the compass for the Gradient Ascent. The gradient estimate for an encountered action is proportional to the action's "advantage" scaled by the action's Value Function. The intuition is that the Gradient Ascent encourages picking actions that are yielding more favorable outcomes (*Policy Improvement*) so as to ultimately get to a point where the optimal action is selected for each state.

### REINFORCE Algorithm (Monte-Carlo Policy Gradient)

Now we are ready to write our first Policy Gradient algorithm. As ever, the simplest algorithm is a Monte-Carlo algorithm. In the case of Policy Gradient, a simple Monte-Carlo calculation provides us with an [algorithm known as REINFORCE, due to R.J.Williams](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) [@Williams:92], which we cover in this section.

We've already explained that we can calculate the Score function using an analytical derivative of a specified functional form for $\pi(S_t,A_t;\bm{\theta})$ for each atomic experience $(S_t, A_t, R_t, S'_t)$. What remains is to obtain an estimate of $Q^{\pi}(S_t,A_t)$ for each atomic experience $(S_t, A_t, R_t, S'_t)$. REINFORCE uses the trace experience return $G_t$ for $(S_t, A_t)$, while following policy $\pi$, as an unbiased sample of $Q^{\pi}(S_t,A_t)$. Thus, at every time step (i.e., at every atomic experience) in each episode, we estimate $\nabla_{\bm{\theta}} J(\bm{\theta})$ by calculating $\gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t, A_t; \bm{\theta})}) \cdot G_t$ (noting that the state occurrence probabilities and action occurrence probabilities are implicit in the trace experiences), and update the parameters $\bm{\theta}$ at the end of each episode (using each atomic experience's $\nabla_{\bm{\theta}} J(\bm{\theta})$ estimate) according to Stochastic Gradient Ascent as follows:

$$\Delta \bm{\theta} = \alpha \cdot \gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t,A_t;\bm{\theta})}) \cdot G_t$$

where $\alpha$ is the learning rate.


This Policy Gradient algorithm is Monte-Carlo because it is not bootstrapped (complete returns are used as an unbiased sample of $Q^{\pi}$, rather than a bootstrapped estimate). In terms of our previously-described classification of RL algorithms as Value Function-based or Policy-based or Actor-Critic, REINFORCE is a Policy-based algorithm since REINFORCE does not involve learning a Value Function.

Now let's write some code to implement the REINFORCE algorithm. In this chapter, we will focus our Python code implementation of Policy Gradient algorithms to continuous action spaces, although it should be clear based on the discussion so far that the Policy Gradient approach applies to arbitrary action spaces (we've already seen an example of the policy function parameterization for discrete action spaces). To keep things simple, the function `reinforce_gaussian` below implements REINFORCE for the simple case of single-dimensional continuous action spaces (i.e. $\mathcal{A} = \mathbb{R}$), although this can be easily extended to multi-dimensional continuous action spaces. So in the code below, we work with a generic state space given by `TypeVar('S')` and the action space is specialized to `float` (representing $\mathbb{R}$).

As seen earlier in the canonical example for single-dimensional continuous action space, we assume a Gaussian distribution for the policy. Specifically, the policy is represented by an arbitrary parameterized function approximation using the class `FunctionApprox`. As a reminder, an instance of `FunctionApprox` represents a probability distribution function $f$ of the conditional random variable variable $y|x$ where $x$ belongs to an arbitrary domain $\mathcal{X}$ and $y \in \mathbb{R}$ (probability of $y$ conditional on $x$ denoted as $f(x; \bm{\theta})(y)$ where $\bm{\theta}$ denotes the parameters of the `FunctionApprox`). Note that the `evaluate` method of `FunctionApprox` takes as input an `Iterable` of $x$ values and calculates $g(x; \bm{\theta}) = \mathbb{E}_{f(x;\bm{\theta})}[y]$ for each of the $x$ values. In our case here, $x$ represents non-terminal states in $\mathcal{N}$ and $y$ represents actions in $\mathbb{R}$, so $f(s;\bm{\theta})$ denotes the probability distribution of actions, conditional on state $s \in \mathcal{N}$, and $g(s; \bm{\theta})$ represents the *Expected Value* of (real-numbered) actions, conditional on state $s \in \mathcal{N}$. Since we have assumed the policy to be Gaussian, 

$$\pi(s,a; \bm{\theta}) = \frac 1 {\sqrt{2 \pi \sigma^2}} \cdot e^{-\frac {(a - g(s; \bm{\theta}))^2} {2 \sigma^2}}$$

To be clear, our code below works with the `@abstractclass FunctionApprox` (meaning it is an arbitrary parameterized function approximation) with the assumption that the probability distribution of actions given a state is Gaussian whose variance $\sigma^2$ is assumed to be a constant.  Assume we have $m$ features for our function approximation, denoted as $\bm{\phi}(s) = (\phi_1(s), \ldots, \phi_m(s))$ for all $s \in \mathcal{N}$.

$\sigma$ is specified in the code below with the input `policy_stdev`. The input `policy_mean_approx0: FunctionApprox[NonTerminal[S]]` specifies the function approximation we initialize the algorithm with (it is up to the user of `reinforce_gaussian` to configure `policy_mean_approx0` with the appropriate functional form for the function approximation, the hyper-parameter values, and the initial values of the parameters $\bm{\theta}$ that we want to solve for). 

The Gaussian policy (of the type `GaussianPolicyFromApprox`) selects an action $a$ (given state $s$) by sampling from the gaussian distribution defined by mean $g(s;\bm{\theta})$ and variance $\sigma^2$.

The score function is given by:
$$\nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) = \frac {(a - g(s;\bm{\theta})) \cdot \nabla_{\bm{\theta}} g(s;\bm{\theta})} {\sigma^2}$$

The outer loop of `while True:` loops over trace experiences produced by the method `simulate_actions` of the input `mdp` for a given input `start_states_distribution` (specifying the initial states distribution $p_0: \mathcal{N} \rightarrow [0, 1]$), and the current policy $\pi$ (that is parameterized by $\bm{\theta}$, which update after each trace experience). The inner loop loops over an `Iterator` of `step: ReturnStep[S, float]` objects produced by the `returns` method for each trace experience. 

The variable `grad` is assigned the value of the negative score for an encountered $(S_t, A_t)$ in a trace experience, i.e., it is assigned the value:

$$- \nabla_{\bm{\theta}} \log \pi(S_t,A_t; \bm{\theta}) = \frac {(g(S_t;\bm{\theta}) - A_t) \cdot \nabla_{\bm{\theta}}g(S_t; \bm{\theta})} {\sigma^2}$$

We negate the sign of the score because we are performing Gradient Ascent rather than Gradient Descent (the `FunctionApprox` class has been written for Gradient Descent). The variable `scaled_grad` multiplies the negative of score (`grad`) with $\gamma^t$ (`gamma_prod`) and return $G_t$ (`step.return_`). The rest of the code should be self-explanatory.

`reinforce_gaussian` returns an `Iterable` of `FunctionApprox` representing the stream of updated policies $\pi(s,\cdot; \bm{\theta})$, with each of these `FunctionApprox` being generated (using `yield`) at the end of each trace experience.

```python
import numpy as np
from rl.distribution import Distribution, Gaussian
from rl.policy import Policy
from rl.markov_process import NonTerminal
from rl.markov_decision_process import MarkovDecisionProcess, TransitionStep
from rl.function_approx import FunctionApprox, Gradient

S = TypeVar('S')

@dataclass(frozen=True)
class GaussianPolicyFromApprox(Policy[S, float]):
    function_approx: FunctionApprox[NonTerminal[S]]
    stdev: float

    def act(self, state: NonTerminal[S]) -> Gaussian:
        return Gaussian(
            mu=self.function_approx(state),
            sigma=self.stdev
        )

def reinforce_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    start_states_distribution: Distribution[NonTerminal[S]],
    policy_stdev: float,
    gamma: float,
    episode_length_tolerance: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    while True:
        policy: Policy[S, float] = GaussianPolicyFromApprox(
            function_approx=policy_mean_approx,
            stdev=policy_stdev
        )
        trace: Iterable[TransitionStep[S, float]] = mdp.simulate_actions(
            start_states=start_states_distribution,
            policy=policy
        )
        gamma_prod: float = 1.0
        for step in returns(trace, gamma, episode_length_tolerance):
            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)
            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(step.state, step.action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * step.return_
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            gamma_prod *= gamma
        yield policy_mean_approx
```

The above code is in the file [rl/policy_gradient.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/policy_gradient.py).

### Optimal Asset Allocation (Revisited)

In this chapter, we will test the PG algorithms we implement on the Optimal Asset Allocation problem of Chapter [-@sec:portfolio-chapter], specifically the setting of the class `AssetAllocDiscrete` covered in Section [-@sec:asset-alloc-discrete-code]. As a reminder, in this setting, we have a single risky asset and at each of a fixed finite number of time steps, one has to make a choice of the quantity of current wealth to invest in the risky asset (remainder in the riskless asset) with the goal of maximizing the expected utility of wealth at the end of the finite horizon. Thus, this finite-horizon MDP's state at any time $t$ is the pair $(t, W_t)$ where $W_t \in \mathbb{R}$ denotes the wealth at time $t$, and the action at time $t$ is the investment $x_t \in \mathbb{R}$ in the risky asset.

We provided an approximate DP backward-induction solution to this problem in Chapter [-@sec:funcapprox-chapter], implemented with `AssetAllocDiscrete` (code in [rl/chapter7/asset_alloc_discrete.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/policy_gradient.py)). Now we want to solve it with PG algorithms, starting with REINFORCE. So we require a new interface and hence, we implement a new class `AssetAllocPG` with appropriate tweaks to `AssetAllocDiscrete`. The key change in the interface is that we have inputs `policy_feature_funcs`, `policy_mean_dnn_spec` and `policy_stdev` (see code below). `policy_feature_funcs` represents the sequence of feature functions for the `FunctionApprox` representing the mean action for a given state (i.e., $g(s;\bm{\theta}) = \mathbb{E}_{f(s;\bm{\theta})}[a]$ where $f$ represents the policy probability distribution of actions for a given state). `policy_mean_dnn_spec` specifies the architecture of a deep neural network a user would like to use for the `FunctionApprox`. `policy_stdev` represents the fixed standard deviation $\sigma$ of the policy probability distribution of actions for any state. `AssetAllocState = Tuple[float, int]` is the data type for the state $(t, W_t)$. 

```python
from rl.function_approx import DNNSpec

AssetAllocState = Tuple[int, float]

@dataclass(frozen=True)
class AssetAllocPG:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]]
    policy_mean_dnn_spec: DNNSpec
    policy_stdev: float
    policy_mean
    initial_wealth_distribution: Distribution[float]
```

Unlike the backward-induction solution of `AssetAllocDiscrete` where we had to model a separate MDP for each time step in the finite horizon (where the state for each time step's MDP is the wealth), here we model a single MDP across all time steps with the state as the pair of time step index $t$ and the wealth $W_t$. The method `get_mdp` below sets up this MDP (should be self-explanatory as the constructions is very similar to the construction of the single-step MDPs in `AssetAllocDiscrete`).


```python
from rl.distribution import SampledDistribution

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def get_mdp(self) -> MarkovDecisionProcess[AssetAllocState, float]:
        steps: int = self.time_steps()
        distrs: Sequence[Distribution[float]] = self.risky_return_distributions
        rates: Sequence[float] = self.riskless_returns
        utility_f: Callable[[float], float] = self.utility_func

        class AssetAllocMDP(MarkovDecisionProcess[AssetAllocState, float]):

            def step(
                self,
                state: NonTerminal[AssetAllocState],
                action: float
            ) -> SampledDistribution[Tuple[State[AssetAllocState], float]]:

                def sr_sampler_func(
                    state=state,
                    action=action
                ) -> Tuple[State[AssetAllocState], float]:
                    time, wealth = state.state
                    next_wealth: float = action * (1 + distrs[time].sample()) \
                        + (wealth - action) * (1 + rates[time])
                    reward: float = utility_f(next_wealth) \
                        if time == steps - 1 else 0.
                    next_pair: AssetAllocState = (time + 1, next_wealth)
                    next_state: State[AssetAllocState] = \
                        Terminal(next_pair) if time == steps - 1 \
                        else NonTerminal(next_pair)
                    return (next_state, reward)

                return SampledDistribution(sampler=sr_sampler_func)

            def actions(self, state: NonTerminal[AssetAllocState]) \
                    -> Sequence[float]:
                return []

        return AssetAllocMDP()
```

The methods `start_states_distribution` and `policy_mean_approx` below create the `SampledDistribution` of start states and the `DNNApprox` representing the mean action for a given state respectively. Finally, the `reinforce` method below simply collects all the ingredients and passes along to the `reinforce_gaussian` to solve this asset allocation problem.

```python
from rl.function_approx import AdamGradient, DNNApprox

    def start_states_distribution(self) -> \
            SampledDistribution[NonTerminal[AssetAllocState]]:

        def start_states_distribution_func() -> NonTerminal[AssetAllocState]:
            wealth: float = self.initial_wealth_distribution.sample()
            return NonTerminal((0, wealth))

        return SampledDistribution(sampler=start_states_distribution_func)

    def policy_mean_approx(self) -> \
            FunctionApprox[NonTerminal[AssetAllocState]]:
        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.003,
            decay1=0.9,
            decay2=0.999
        )
        ffs: List[Callable[[NonTerminal[AssetAllocState]], float]] = []
        for f in self.policy_feature_funcs:
            def this_f(st: NonTerminal[AssetAllocState], f=f) -> float:
                return f(st.state)
            ffs.append(this_f)
        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=self.policy_mean_dnn_spec,
            adam_gradient=adam_gradient
        )

    def reinforce(self) -> \
            Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        return reinforce_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            episode_length_tolerance=1e-5
        )
```   

The above code is in the file [rl/chapter13/asset_alloc_pg.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter13/asset_alloc_pg.py).

Let's now test this out on an instance of the problem for which we have a closed-form solution (so we can verify the REINFORCE solution against the closed-form solution). The special instance is the setting covered in Section [-@sec:discrete-asset-alloc] of Chapter [-@sec:portfolio-chapter]. From Equation \eqref{eq:pi-star-solution-discrete}, we know that the optimal action in state $(t, W_t)$ is linear in a single feature defined as $(1+r)^t$ where $r$ is the constant risk-free rate across time steps. So we should set up the function approximation as `LinearFunctionApprox` with this single feature and check if the solved coefficient of this single feature matches up with the close form solution of Equation \eqref{eq:pi-star-solution-discrete}. 

Let us use similar settings that we had used in Chapter [-@sec:portfolio-chapter] to test `AssetAllocDiscrete`. In the code below, we create an instance of `AssetAllocPG` with time steps $T=5$, $\mu = 13\%, \sigma = 20\%, r = 7\%$, coefficient of CARA $a = 1.0$. We set up `risky_return_distributions` as a sequence of identical `Gaussian` distributions, `riskless_returns` as a sequence of identical riskless rate of returns, and `utility_func` as a `lambda` parameterized by the coefficient of CARA $a$.  We set the probability distribution of wealth at time $t=0$ (start of each trace experience) as $\mathcal{N}(1.0, 0.1)$, and we set the constant standard deviation $\sigma$ of the policy probability distribution of actions for a given state as $0.5$.

```python
steps: int = 5
mu: float = 0.13
sigma: float = 0.2
r: float = 0.07
a: float = 1.0
init_wealth: float = 1.0
init_wealth_stdev: float = 0.1
policy_stdev: float = 0.5
```

Next, we print the analytical solution of the optimal action for states at each time step (note: analytical solution for optimal action is independent of wealth $W_t$, and is only dependent on $t$).

```python
base_alloc: float = (mu - r) / (a * sigma * sigma)
for t in range(steps):
    alloc: float = base_alloc / (1 + r) ** (steps - t - 1)
    print(f"Time {t:d}: Optimal Risky Allocation = {alloc:.3f}")
```

This prints:

```
Time 0: Optimal Risky Allocation = 1.144
Time 1: Optimal Risky Allocation = 1.224
Time 2: Optimal Risky Allocation = 1.310
Time 3: Optimal Risky Allocation = 1.402
Time 4: Optimal Risky Allocation = 1.500
```

Next we set up an instance of `AssetAllocPG` with the above parameters. Note that the `policy_mean_dnn_spec` argument to the constructor of `AssetAllocPG` is set up as a trivial neural network with no hidden layers and the identity function as the output layer activation function. Note also that the `policy_feature_funcs` argument to the constructor is set up with the single feature function $(1+r)^t$.

```python
from rl.distribution import Gaussian
from rl.function_approx import 

risky_ret: Sequence[Gaussian] = [Gaussian(mu=mu, sigma=sigma)
                                 for _ in range(steps)]
riskless_ret: Sequence[float] = [r for _ in range(steps)]
utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]] = \
    [
        lambda w_t: (1 + r) ** w_t[1]
    ]
init_wealth_distr: Gaussian = Gaussian(mu=init_wealth, sigma=init_wealth_stdev)
policy_mean_dnn_spec: DNNSpec = DNNSpec(
    neurons=[],
    bias=False,
    hidden_activation=lambda x: x,
    hidden_activation_deriv=lambda y: np.ones_like(y),
    output_activation=lambda x: x,
    output_activation_deriv=lambda y: np.ones_like(y)
)

aad: AssetAllocPG = AssetAllocPG(
    risky_return_distributions=risky_ret,
    riskless_returns=riskless_ret,
    utility_func=utility_function,
    policy_feature_funcs=policy_feature_funcs,
    policy_mean_dnn_spec=policy_mean_dnn_spec,
    policy_stdev=policy_stdev,
    initial_wealth_distribution=init_wealth_distr
)
```

Next, we invoke the method `reinforce` of this `AssetAllocPG` instance. In practice, we'd have parameterized the standard deviation of the policy probability distribution just like we parameterized the mean of the policy probability distribution, and we'd have updated those parameters in a similar manner (the standard deviation would converge to 0, i.e., the policy would converge to the optimal deterministic policy given by the closed-form solution). As an exercise, extend the function `reinforce_gaussian` to include a second `FunctionApprox` for the standard deviation of the policy probability distribution and update this `FunctionApprox` along with the updates to the mean `FunctionApprox`. However, since we set the standard deviation of the policy probability distribution to be a constant $\sigma$ and since we use a Monte-Carlo method, the variance of the mean estimate of the policy probability distribution is significantly high. So we take the average of the mean estimate over several iterations (below we average the estimate from iteration 10000 to iteration 20000).

```python
reinforce_policies: Iterator[FunctionApprox[
    NonTerminal[AssetAllocState]]] = aad.reinforce()

num_episodes: int = 10000
averaging_episodes: int = 10000

policies: Sequence[FunctionApprox[NonTerminal[AssetAllocState]]] = \
    list(itertools.islice(
        reinforce_policies,
        num_episodes,
        num_episodes + averaging_episodes
    ))
for t in range(steps):
    opt_alloc: float = np.mean([p(NonTerminal((init_wealth, t)))
                               for p in policies])
    print(f"Time {t:d}: Optimal Risky Allocation = {opt_alloc:.3f}")
```

This prints:

```
Time 0: Optimal Risky Allocation = 1.215
Time 1: Optimal Risky Allocation = 1.300
Time 2: Optimal Risky Allocation = 1.392
Time 3: Optimal Risky Allocation = 1.489
Time 4: Optimal Risky Allocation = 1.593
```

So we see that the estimate of the mean action for the 5 time steps from our implementation of the REINFORCE method gets fairly close to the closed-form solution.

The above code is in the file [rl/chapter13/asset_alloc_reinforce.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter13/asset_alloc_reinforce.py). As ever, we encourage you to tweak the parameters and explore how the results vary.

As an exercise, we encourage you to implement an extension of this problem. Along with the risky asset allocation choice as the action at each time step, also include a consumption quantity (wealth to be extracted at each time step, along the lines of Merton's Dynamic Portfolio Allocation and Consumption problem) as part of the action at each time step. So the action at each time step would be a pair $(c, a)$ where $c$ is the quantity to consume and $a$ is the quantity to allocate to the risky asset. Note that the consumption is constrained to be non-negative and at most the amount of wealth at any time step ($a$ is unconstrained). The reward at each time step is the Utility of Consumption.

### Actor-Critic and Variance Reduction

As we've mentioned in the previous section, REINFORCE has high variance since it's a Monte-Carlo method. So it can take quite long for REINFORCE to converge. A simple way to reduce the variance is to use a function approximation for the Q-Value Function instead of using the trace experience return as an unbiased sample of the Q-Value Function. Variance reduction happens from the simple fact that a function approximation of the Q-Value Function updates gradually (using gradient descent) and so, does not vary enormously like the trace experience returns would. Let us denote the function approximation of the Q-Value Function as $Q(s,a;\bm{w})$ where $\bm{w}$ denotes the parameters of the function approximation. We refer to $Q(s,a;\bm{w})$ as the *Critic* and we refer to the $\pi(s,a;\bm{\theta})$ function approximation as the *Actor*. The two function approximations $\pi(s,a;\bm{\theta})$ and $Q(s,a;\bm{w})$ collaborate to improve the policy using gradient ascent (guided by the PGT, using $Q(s,a;\bm{w})$ in place of the true Q-Value Function $Q^{\pi}(s,a)$). $\pi(s,a;\bm{\theta})$ is called *Actor* because it is the primary worker and $Q(s,a;\bm{w})$ is called *Critic* because it is the support worker. The intuitive way to think about this is that the Actor updates policy parameters in a direction that is suggested by the Critic.

After each atomic experience, both $\bm{\theta}$ and $\bm{w}$ are updated. $\bm{w}$ is updated such that a suitable loss function is minimized. This can be done using any of the usual Value Function approximation methods we have covered previously, including:

* Monte-Carlo, i.e., $\bm{w}$ updated using trace experience returns $G_t$.
* Temporal-Difference, i.e., $\bm{w}$ updated using TD Targets.
* TD($\lambda$), i.e., $\bm{w}$ updated using targets based on eligibility traces.
* It could even be LSTD if we assume a linear function approximation for the critic $Q(s,a;\bm{w})$.

This method of calculating the gradient of $J(\bm{\theta})$ can be thought of as *Approximate Policy Gradient* due to the bias of the Critic $Q(s,a;\bm{w})$ (serving as an approximation of $Q^{\pi}(s,a)$), i.e.,

$$\nabla_{\bm{\theta}} J(\bm{\theta}) \approx \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s,a; \bm{\theta}) \cdot Q(s,a; \bm{w})$$

Now let's implement some code to perform Policy Gradient with the Critic updated using Temporal-Difference (again, for the simple case of single-dimensional continuous action spaces). In the function `actor_critic_gaussian` below, the key changes (from the code in `reinforce_gaussian`) are:

* The Q-Value function approximation parameters $\bm{w}$ are updated after each atomic experience as:
$$\Delta \bm{w} = \beta \cdot (R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1}; \bm{w}) - Q(S_t,A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t,A_t; \bm{w})$$
where $\beta$ is the learning rate for the Q-Value function approximation.
* The policy mean parameters $\bm{\theta}$ are updated after each atomic experience as:
$$\Delta \bm{\theta} = \alpha \cdot \gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t;A_t;\bm{\theta})}) \cdot Q(S_t,A_t;\bm{w})$$
(instead of $\alpha \cdot \gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t,A_t;\bm{\theta})}) \cdot G_t$).

```python
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import NTStateDistribution

def actor_critic_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    q_value_func_approx0: QValueFunctionApprox[S, float],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    q: QValueFunctionApprox[S, float] = q_value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        action: float = Gaussian(
            mu=policy_mean_approx(state),
            sigma=policy_stdev
        ).sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                next_action: float = Gaussian(
                    mu=policy_mean_approx(next_state),
                    sigma=policy_stdev
                ).sample()
                q = q.update([(
                    (state, action),
                    reward + gamma * q((next_state, next_action))
                )])
                action = next_action
            else:
                q = q.update([((state, action), reward)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * q((state, action))
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state
```

The above code is in the file [rl/policy_gradient.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/policy_gradient.py). We leave it to you as an exercise to implement the update of $Q(s,a;\bm{w})$ with TD($\lambda$), i.e., with eligibility traces.

We can reduce the variance of this Actor-Critic method by subtracting a Baseline Function $B(s)$ from $Q(s,a;\bm{w})$ in the Policy Gradient estimate. This means we update the parameters $\bm{\theta}$ as:

$$\Delta \bm{\theta} = \alpha \cdot \gamma^t \cdot \nabla_{\bm{\theta}} \log \pi(S_t,A_t; \bm{\theta}) \cdot (Q(S_t,A_t; \bm{w}) - B(S_t))$$

Note that the Baseline Function $B(s)$ is only a function of state $s$ (and not of action $a$). This ensures that subtracting the Baseline Function $B(s)$ does not add bias. This is because:
\begin{align*}
& \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s,a; \bm{\theta}) \cdot B(s)\\
 = & \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot B(s) \cdot \nabla_{\bm{\theta}} (\sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta})) \\
 = & \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot B(s) \cdot \nabla_{\bm{\theta}} 1 \\
  = & 0
\end{align*}

A good Baseline Function $B(s)$ is a function approximation $V(s; \bm{v})$ of the State-Value Function $V^{\pi}(s)$. So then we can rewrite the Actor-Critic Policy Gradient algorithm using an estimate of the Advantage Function, as follows:
$$A(s,a;\bm{w},\bm{v}) = Q(s,a;\bm{w}) - V(s; \bm{v})$$
With this, the approximation for $\nabla_{\bm{\theta}} J(\bm{\theta})$ is given by:
$$J(\bm{\theta}) \approx \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s, a; \bm{\theta}) \cdot A(s,a; \bm{w}, \bm{v})$$

The function `actor_critic_advantage_gaussian` in the file [rl/policy_gradient.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/policy_gradient.py) implements this algorithm, i.e., Policy Gradient with two Critics $Q(s,a;\bm{w})$ and $V(s;\bm{v})$, each updated using Temporal-Difference (again, for the simple case of single-dimensional continuous action spaces). Specifically, in the code of `actor_critic_advantage_gaussian`:

* The Q-Value function approximation parameters $\bm{w}$ are updated after each atomic experience as:
$$\Delta \bm{w} = \beta_{\bm{w}} \cdot (R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1}; \bm{w}) - Q(S_t,A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t,A_t; \bm{w})$$
where $\beta_{\bm{w}}$ is the learning rate for the Q-Value function approximation.
* The State-Value function approximation parameters $\bm{v}$ are updated after each atomic experience as:
$$\Delta \bm{v} = \beta_{\bm{v}} \cdot (R_{t+1} + \gamma \cdot V(S_{t+1}; \bm{v}) - V(S_t;\bm{v})) \cdot \nabla_{\bm{v}} V(S_t; \bm{v})$$
where $\beta_{\bm{v}}$ is the learning rate for the State-Value function approximation.
* The policy mean parameters $\bm{\theta}$ are updated after each atomic experience as:
$$\Delta \bm{\theta} = \alpha \cdot \gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t;A_t;\bm{\theta})}) \cdot (Q(S_t,A_t;\bm{w}) - V(S_t; \bm{v}))$$

A simpler way is to use the TD Error of the State-Value Function as an estimate of the Advantage Function. To understand this idea, let $\delta^{\pi}$ denote the TD Error for the *true* State-Value Function $V^{\pi}(s)$. Then,
$$\delta^{\pi} = r + \gamma \cdot V^{\pi}(s') - V^{\pi}(s)$$
Note that $\delta^{\pi}$ is an unbiased estimate of the Advantage function $A^{\pi}(s,a)$. This is because
$$\mathbb{E}_{\pi}[\delta^{\pi} | s,a] = \mathbb{E}_{\pi}[r + \gamma \cdot V^{\pi}(s') | s, a] - V^{\pi}(s) = Q^{\pi}(s,a) - V^{\pi}(s) = A^{\pi}(s,a)$$
So we can write Policy Gradient in terms of $\mathbb{E}_{\pi}[\delta^{\pi} | s,a]$:
$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s, a; \bm{\theta}) \cdot \mathbb{E}_{\pi}[\delta^{\pi} | s,a]$$
In practice, we use a function approximation for the TD error, and sample:
$$\delta(s,r,s';\bm{v}) = r + \gamma \cdot V(s';\bm{v}) - V(s;\bm{v})$$
This approach requires only one set of critic parameters $\bm{v}$, and we don't have to worry about the Action-Value Function $Q$.

Now let's implement some code for this TD Error-based PG Algorithm (again, for the simple case of single-dimensional continuous action spaces). In the function `actor_critic_td_error_gaussian` below:

* The State-Value function approximation parameters $\bm{v}$ are updated after each atomic experience as:
$$\Delta \bm{v} = \alpha_{\bm{v}} \cdot (R_{t+1} + \gamma \cdot V(S_{t+1}; \bm{v}) - V(S_t;\bm{v})) \cdot \nabla_{\bm{v}} V(S_t; \bm{v})$$
where $\alpha_{\bm{v}}$ is the learning rate for the State-Value function approximation.
* The policy mean parameters $\bm{\theta}$ are updated after each atomic experience as:
$$\Delta \bm{\theta} = \alpha_{\bm{\theta}} \cdot \gamma^t \cdot (\nabla_{\bm{\theta}} \log{\pi(S_t;A_t;\bm{\theta})}) \cdot (R_{t+1} + \gamma \cdot V(S_{t+1}; \bm{v}) -  V(S_t; \bm{v}))$$
where $\alpha_{\bm{\theta}}$ is the learning rate for the Policy Mean function approximation.

```python
from rl.approximate_dynamic_programming import ValueFunctionApprox

def actor_critic_td_error_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    value_func_approx0: ValueFunctionApprox[S],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    vf: ValueFunctionApprox[S] = value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            action: float = Gaussian(
                mu=policy_mean_approx(state),
                sigma=policy_stdev
            ).sample()
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                td_target: float = reward + gamma * vf(next_state)
            else:
                td_target = reward
            td_error: float = td_target - vf(state)
            vf = vf.update([(state, td_target)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * td_error
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state
```

The above code is in the file [rl/policy_gradient.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/policy_gradient.py).

Likewise, we can implement an Actor-Critic algorithm using Eligibility Traces (i.e., TD($\lambda$)) for the State-Value Function Approximation and also for the Policy Mean Function Approximation. The updates after each atomic experience to parameters $\bm{v}$ of the State-Value function approximation and parameters $\bm{\theta}$ of the policy mean function approximation are given by:

$$\Delta \bm{v} = \alpha_{\bm{v}} \cdot (R_{t+1} + \gamma \cdot V(S_{t+1}; \bm{v}) - V(S_t; \bm{v})) \cdot \bm{E_v}$$
$$\Delta \bm{\theta} = \alpha_{\bm{\theta}} \cdot (R_{t+1} + \gamma \cdot V(S_{t+1}; \bm{v}) - V(S_t; \bm{v})) \cdot \bm{E_{\theta}}$$
where the Eligibility Traces $\bm{E_v}$ and $\bm{E_{\theta}}$ are updated after each atomic experience as follows:
$$\bm{E_v} \leftarrow \gamma \cdot \lambda_{\bm{v}} \cdot \bm{E_v} + \nabla_{\bm{v}} V(S_t;\bm{v})$$
$$\bm{E_{\theta}} \leftarrow \gamma \cdot \lambda_{\bm{\theta}} \cdot \bm{E_{\theta}} + \gamma^t \cdot \nabla_{\bm{\theta}} \log \pi(S_t,A_t;\bm{\theta})$$
where $\lambda_{\bm{v}}$ and $\lambda_{\bm{\theta}}$ are the TD($\lambda$) parameters respectively for the State-Value Function Approximation and the Policy Mean Function Approximation.

We encourage you to implement in code this Actor-Critic algorithm using Eligibility Traces.

Now let's compare these methods on the `AssetAllocPG` instance we had created earlier to test REINFORCE, i.e., for time steps $T=5$, $\mu = 13\%, \sigma = 20\%, r = 7\%$, coefficient of CARA $a = 1.0$, probability distribution of wealth at the start of each trace experience as $\mathcal{N}(1.0, 0.1)$, and constant standard deviation $\sigma$ of the policy probability distribution of actions for a given state as $0.5$. The `__main__` code in [rl/chapter13/asset_alloc_pg.py](https://github.com/TikhonJelvis/RL-book/blob/master/chapter13/asset_alloc_pg.py) evaluates the mean action for the start state of $(t=0, W_0 = 1.0)$ after each episode (over 20,000 episodes) for each of the above-implemented PG algorithms' function approximation for the policy mean. It then plots the progress of the evaluated mean action for the start state over the 20,000 episodes, along with the benchmark of the optimal action for the start state from the known closed-form solution. Figure \ref{fig:pg_convergence} shows the graph, validating the points we have made above on bias and variance of these algorithms.

![Progress of PG Algorithms \label{fig:pg_convergence}](./chapter13/pg_convergence.png "Progress of PG Algorithms")

Actor-Critic methods were developed in the late 1970s and 1980s, but not paid attention to in the 1990s. In the past two decades, there has been a revival of Actor-Critic methods. For a more detailed coverage of Actor-Critic methods, see the [paper by Degris, White, Sutton](https://arxiv.org/abs/1205.4839) [@journals/corr/abs-1205-4839].


### Overcoming Bias with Compatible Function Approximation

We've talked a lot about reducing variance for faster convergence of PG Algorithms. Specifically, we've talked about the following proxies for $Q^{\pi}(s,a)$ in the form of Actor-Critic algorithms in order to reduce variance.

* $Q(s,a;\bm{w})$
* $A(s,a;\bm{w}, \bm{v}) = Q(s,a;\bm{w}) - V(s;\bm{v})$
* $\delta(s,s',r;\bm{v}) = r + \gamma \cdot V(s';\bm{v}) - V(s;\bm{v})$

However, each of the above proxies for $Q^{\pi}(s,a)$ in PG algorithms have a bias. In this section, we talk about how to overcome bias. The basis for overcoming bias is an important Theorem known as the *Compatible Function Approximation Theorem*. We state and prove this theorem, and then explain how we could use it in a PG algorithm.


\begin{theorem}[Compatible Function Approximation Theorem]
If the following two conditions are satisfied:
\begin{enumerate}
\item Critic gradient is {\em compatible} with the Actor score function
$$\nabla_{\bm{w}} Q(s,a;\bm{w}) = \nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta})$$ 
\item Critic parameters $\bm{w}$ minimize the following mean-squared error:
$$\epsilon = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot (Q^{\pi}(s,a) - Q(s,a;\bm{w}))^2$$
\end{enumerate}
Then the Policy Gradient using critic $Q(s,a;\bm{w})$ is exact:
$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s, a; \bm{\theta}) \cdot Q(s,a; \bm{w})$$
\label{th:cfa-theorem}
\end{theorem}

\begin{proof}
For $\bm{w}$ that minimizes
$$\epsilon = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot (Q^{\pi}(s,a) - Q(s,a;\bm{w}))^2,$$
$$\sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot (Q^{\pi}(s,a) - Q(s,a;\bm{w})) \cdot \nabla_{\bm{w}} Q(s,a;\bm{w}) = 0$$
But since $\nabla_{\bm{w}} Q(s,a;\bm{w}) = \nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta})$, we have:
$$\sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot (Q^{\pi}(s,a) - Q(s,a;\bm{w})) \cdot \nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) = 0$$
\begin{align*}
\mbox{Therefore, } & \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot Q^{\pi}(s,a) \cdot \nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) \\
= & \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot Q(s,a; \bm{w}) \cdot \nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta})\\
\end{align*}
$$\mbox{But } \nabla_{\bm{\theta}} J(\bm{\theta}) = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot  \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot Q^{\pi}(s,a) \cdot \nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta})$$
\begin{align*}
\mbox{So, } \nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}) \cdot Q(s,a; \bm{w}) \cdot \nabla_{\bm{\theta}} \log \pi(s,a; \bm{\theta}) \\
& = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \nabla_{\bm{\theta}} \pi(s,a; \bm{\theta}) \cdot Q(s,a; \bm{w})\\
\end{align*}
$$\mathbb{Q.E.D.}$$
\end{proof}

This proof originally appeared in [the famous paper by Sutton, McAllester, Singh, Mansour on Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) [@sutton2001policy].

This means with conditions (1) and (2) of Compatible Function Approximation Theorem, we can use the critic function approximation $Q(s,a;\bm{w})$ and still have exact Policy Gradient (i.e., no bias due to using a function approximation for the Q-Value Function).

A simple way to enable Compatible Function Approximation is to make $Q(s,a;\bm{w})$ a linear function approximation, with the features of the linear function approximation equal to the Score of the policy function approximation. Let us write the linear function approximation $Q(s,a;\bm{w})$ as:
$$Q(s,a;\bm{w}) = \sum_{i=1}^m \eta_i(s,a) \cdot w_i$$
Compatible Function Approximation Theorem requires
$$\pdv{Q(s,a;\bm{w})}{w_i} = \eta_i(s,a) = \pdv{\log \pi(s,a; \bm{\theta})}{\theta_i} \text{ for all } i = 1, \ldots, m$$
Therefore, to ensure Compatible Function Approximation Theorem, we must have:
$$Q(s,a;\bm{w}) = \sum_{i=1}^m \pdv{\log \pi(s,a;\bm{\theta})}{\theta_i} \cdot w_i$$
which means:
$$\eta_i(s,a) = \pdv{\log \pi(s,a;\bm{\theta})}{\theta_i} \text{ for all } i = 1, \ldots, m$$

Note that although here we assume $Q(s,a;\bm{w})$ to be a linear function approximation, the policy function approximation $\pi(s,a;\bm{\theta})$ doesn't need to be linear. All that is required is that $\bm{\theta}$ consists of exactly $m$ parameters (matching the number of number of parameters $m$ of $\bm{w}$) and that each of the partial derivatives $\pdv{\log \pi(s,a;\bm{\theta})} {\theta_i}$ lines up with a corresponding feature $\eta_i(s,a)$ of the linear function approximation $Q(s,a;\bm{w})$. This means that as $\bm{\theta}$ updates (as a consequence of Stochastic Gradient Ascent), $\pi(s,a;\bm{\theta})$ updates, and consequently the features $\eta_i(s,a) = \pdv{\log \pi(s,a;\bm{\theta})} {\theta_i}$ update. This means the feature vector $[\eta_i(s,a)|i = 1, \ldots, m]$ is not constant for a given $(s,a)$ pair. Rather, the feature vector $[\eta_i(s,a)|i = 1, \ldots, m]$ for a given $(s,a)$ pair varies in accordance with $\bm{\theta}$ varying.

If we assume the canonical function approximation for $\pi(s,a;\theta)$ for finite action spaces that we had described in Section [-@sec:canonical-policy-functions], then:

$$\eta_i(s,a) = \bm{\phi}(s,a) - \sum_{b \in \mathcal{A}} \pi(s,b; \bm{\theta}) \cdot \bm{\phi}(s,b) \text{ for all } s \in \mathcal{N} \text{ for all } a \in \mathcal{A}$$

Note the dependency of $\eta_i(s,a)$ on $\bm{\theta}$.

If we assume the canonical function approximation for $\pi(s,a;\theta)$ for single-dimensional continuous action spaces that we had described in Section [-@sec:canonical-policy-functions], then:

$$\eta_i(s,a) = \frac {(a - \bm{\phi}(s)^T \cdot \bm{\theta}) \cdot \bm{\phi}(s)} {\sigma^2} \text{ for all } s \in \mathcal{N} \text{ for all } a \in \mathcal{A}$$

Note the dependency of $\eta_i(s,a)$ on $\bm{\theta}$.

We note that any compatible linear function approximation $Q(s,a;\bm{w})$ serves as an approximation of the advantage function because:
$$\sum_{a \in \mathcal{A}} \pi(s,a;\bm{\theta}) \cdot Q(s,a;\bm{w}) = \sum_{a \in \mathcal{A}} \pi(s,a;\bm{\theta}) \cdot (\sum_{i=1}^m \pdv{\log \pi(s,a;\bm{\theta})}{\theta_i} \cdot w_i)$$
$$=\sum_{a \in \mathcal{A}} (\sum_{i=1}^m \pdv{\pi(s,a;\bm{\theta})}{\theta_i} \cdot w_i) = \sum_{i=1}^m (\sum_{a \in \mathcal{A}} \pdv{\pi(s,a;\bm{\theta})}{\theta_i})\cdot w_i$$
$$=\sum_{i=1}^m \frac {\partial} {\partial \theta_i} (\sum_{a \in \mathcal{A}} \pi(s,a; \bm{\theta}))\cdot w_i = \sum_{i=1}^m \pdv{1}{\theta_i} \cdot w_i = 0$$

Denoting $[\pdv{\log \pi(s,a;\bm{\theta})}{\theta_i}|i = 1, \ldots, m]$ as the score column vector $\bm{SC}(s,a;\bm{\theta})$ and assuming compatible linear-approximation critic:
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a;  \bm{\theta}) \cdot \bm{SC}(s, a; \bm{\theta}) \cdot (\bm{SC}(s,a;\bm{\theta})^T \cdot \bm{w})\\
 & = \sum_{s \in \mathcal{N}} \rho^{\pi}(s) \cdot \sum_{a \in \mathcal{A}} \pi(s,a;  \bm{\theta}) \cdot (\bm{SC}(s, a; \bm{\theta}) \cdot \bm{SC}(s,a;\bm{\theta})^T) \cdot \bm{w} \\
 & = \mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi}[\bm{SC}(s, a; \bm{\theta}) \cdot \bm{SC}(s,a;\bm{\theta})^T] \cdot \bm{w}
\end{align*}

Note that $\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi}[\bm{SC}(s, a; \bm{\theta}) \cdot \bm{SC}(s,a;\bm{\theta})^T]$ is the [Fisher Information Matrix](https://en.wikipedia.org/wiki/Fisher_information) $\bm{FIM}_{\rho^{\pi}, \pi}(\bm{\theta})$ with respect to $s \sim \rho^{\pi}, a \sim \pi$. Therefore, we can write $\nabla_{\bm{\theta}} J(\bm{\theta})$ more succinctly as:
\begin{equation}
\nabla_{\bm{\theta}} J(\bm{\theta}) = \bm{FIM}_{\rho^{\pi}, \pi}(\bm{\theta}) \cdot \bm{w}
\label{eq:pgt-fisher-information}
\end{equation}

Thus, we can update $\bm{\theta}$ after each atomic experience by calculating the gradient of $J(\bm{\theta})$ for the atomic experience as the outer product of $\bm{SC}(S_t,A_t;\bm{\theta})$ with itself (which gives a $m \times m$ matrix), then multiply this matrix with the vector $\bm{w}$, and then scale by $\gamma^t$, i.e.
$$\Delta \bm{\theta} = \alpha_{\bm{\theta}} \cdot \gamma^t \cdot \bm{SC}(S_t,A_t;\bm{w}) \cdot \bm{SC}(S_t,A_t;\bm{w})^T \cdot \bm{w}$$

The update for $\bm{w}$ after each atomic experience is the usual Q-Value Function Approximation update with Q-Value loss function gradient for the atomic experience calculated as:
$$\Delta \bm{w} = \alpha_{\bm{w}} \cdot (R_{t+1} + \gamma \cdot \bm{SC}(S_{t+1}, A_{t+1}; \bm{\theta})^T \cdot \bm{w} - \bm{SC}(S_t,A_t; \bm{\theta})^T \cdot \bm{w}) \cdot \bm{SC}(S_t,A_t;\bm{\theta})$$

This completes our coverage of the basic Policy Gradient Methods. Next, we cover a couple of special Policy Gradient Methods that have worked well in practice - Natural Policy Gradient and Deterministic Policy Gradient.

### Policy Gradient Methods in Practice

#### Natural Policy Gradient

Natural Policy Gradient (abbreviated NPG) is due to [a paper by Kakade](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf) [@conf/nips/Kakade01] that utilizes the idea of [Natural Gradient first introduced by Amari](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf) [@amari_natural_1998]. We won't cover the theory of Natural Gradient in detail here, and refer you to the above two papers instead. Here we give a high-level overview of the concepts, and describe the algorithm. 

The core motivation for Natural Gradient is that when the parameters space has a certain underlying structure (as is the case with the parameter space of $\bm{\theta}$ in the context of maximizing $J(\bm{\theta})$), the usual gradient does not represent it's steepest descent direction, but the Natural Gradient does. The steepest descent direction of an arbitrary function $f(\bm{\theta})$ to be minimized is defined as the vector $\Delta \bm{\theta}$ that minimizes $f(\bm{\theta} + \Delta \bm{\theta})$ under the constraint that the length $|\Delta \bm{\theta}|$ is a constant. In general, the length $|\Delta \bm{\theta}|$ is defined with respect to some positive-definite matrix $\bm{G}(\bm{\theta})$ governed by the underlying structure of the $\bm{\theta}$ parameters space, i.e.,
$$|\Delta \bm{\theta}|^2 = (\Delta \bm{\theta})^T \cdot \bm{G}(\bm{\theta}) \cdot \Delta \bm{\theta}$$
We can show that under the length metric defined by the matrix $\bm{G}$, the steepest descent direction is:

$$\nabla_{\bm{\theta}}^{nat} f(\bm{\theta}) = \bm{G}^{-1}(\bm{\theta}) \cdot \nabla_{\bm{\theta}} f({\bm{\theta}})$$

We refer to this steepest descent direction $\nabla_{\bm{\theta}}^{nat} f(\bm{\theta})$ as the Natural Gradient. We can update the parameters $\bm{\theta}$ in this Natural Gradient direction in order to achieve steepest descent (according to the matrix $\bm{G}$), as follows:

$$\Delta \bm{\theta} = \alpha \cdot \nabla_{\bm{\theta}}^{nat} f(\bm{\theta})$$

Amari showed that for a supervised learning problem of estimating the conditional probability distribution of $y|x$ with a function approximation (i.e., where the loss function is defined as the KL divergence between the data distribution and the model distribution), the matrix $\bm{G}$ is the Fisher Information Matrix for $y|x$.

Kakade specialized this idea of Natural Gradient to the case of Policy Gradient (naming it Natural Policy Gradient) with the objective function $f(\bm{\theta})$ equal to the negative of the Expected Returns $J(\bm{\theta})$. This gives the NPG $\nabla_{\bm{\theta}}^{nat} J(\bm{\theta})$ defined as:

$$\nabla_{\bm{\theta}}^{nat} J(\bm{\theta}) = \bm{FIM}_{\rho^{\pi}, \pi}^{-1}(\bm{\theta}) \cdot \nabla_{\bm{\theta}} J({\bm{\theta}})$$

where $\bm{FIM}_{\rho_{\pi}, \pi}$ denotes the Fisher Information Matrix with respect to $s \sim \rho^{\pi}, a \sim \pi$.

We've noted in the previous section that if we enable Compatible Function Approximation with a linear function approximation for $Q^{\pi}(s,a)$, then we have Equation \eqref{eq:pgt-fisher-information}, i.e.,

$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \bm{FIM}_{\rho^{\pi}, \pi}(\bm{\theta}) \cdot \bm{w}$$

This means:

$$\nabla_{\bm{\theta}}^{nat} J(\bm{\theta}) = \bm{w}$$

This compact result enables a simple algorithm for NPG:

* After each atomic experience, update Critic parameters $\bm{w}$ with the critic loss gradient as:
$$\Delta \bm{w} = \alpha_{\bm{w}} \cdot (R_{t+1} + \gamma \cdot \bm{SC}(S_{t+1}, A_{t+1}; \bm{\theta})^T \cdot \bm{w} - \bm{SC}(S_t,A_t; \bm{\theta})^T \cdot \bm{w}) \cdot \bm{SC}(S_t,A_t; \bm{\theta})$$
* After each atomic experience, update Actor parameters $\bm{\theta}$ in the direction of $\bm{w}$:
$$\Delta \bm{\theta} = \alpha_{\bm{\theta}} \cdot \bm{w}$$

#### Deterministic Policy Gradient

Deterministic Policy Gradient (abbreviated DPG) is a creative adaptation of Policy Gradient wherein instead of a parameterized function approximation for a stochastic policy, we have a parameterized function approximation for a deterministic policy for the case of continuous action spaces. DPG is due to [a paper by Silver, Lever, Heess, Degris, Wiestra, Riedmiller](http://proceedings.mlr.press/v32/silver14.pdf) [@conf/icml/SilverLHDWR14]. DPG is expressed in terms of the Expected Gradient of the Q-Value Function and can be estimated much more efficiently that the usual (stochastic) PG. (Stochastic) PG integrates over both the state and action spaces, whereas DPG integrates over only the state space. As a result, computing (stochastic) PG would require more samples if the action space has many dimensions. 

Since the policy approximated is Deterministic, we need to address the issue of exploration - this is typically done with Off-Policy Control wherein we employ an exploratory (stochastic) behavior policy, while the policy being approximated (and learnt with DPG) is the target (deterministic) policy. In Actor-Critic DPG, the Actor is the function approximation for the deterministic policy and the Critic is the function approximation for the Q-Value Function. The paper by Sutton et al. provides a Compatible Function Approximation Theorem for DPG to overcome Critic approximation bias. The paper also shows that DPG is the limiting case of (Stochastic) PG, as policy variance tends to 0. This means the usual machinery of PG (such as Actor-Critic, Compatible Function Approximation, Natural Policy Gradient etc.) is also applicable to DPG.

To avoid confusion with the notation $\pi(s,a;\bm{\theta})$ that we used for the stochastic policy function approximation in PG, here we use the notation $a = \mu(s; \bm{\theta})$ that represents (a potentially multi-dimensional) continuous-valued action $a$ equal to the value of policy function approximation $\mu$ (parameterized by $\bm{\theta}$), when evaluated for a state $s$. So we use the notation $\mu$ for the deterministic target policy approximation and we use the notation $\beta$ for the exploratory behavior policy.

The core idea of DPG is well-understood by orienting on the basics of GPI and specifically, on Policy Improvement in GPI. For continuous action spaces, greedy policy improvement (with an argmax over actions, for each state) is problematic. So a simple and attractive alternative is to move the policy in the direction of the gradient of the Q-Value Function (rather than globally maximizing the Q-Value Function, at each step). Specifically, for each state $s$ that is encountered, the policy approximation parameters $\bm{\theta}$ are updated in proportion to $\nabla_{\bm{\theta}} Q(s, \mu(s; \bm{\theta}))$. Note that the direction of policy improvement is different for each state, and so the average direction of policy improvements is given by:

$$\mathbb{E}_{s \sim \rho^{\mu}}[\nabla_{\bm{\theta}} Q(s,\mu(s; \bm{\theta}))]$$
where $\rho^{\mu}$ is the same Discounted-Aggregate State-Visitation Measure we had defined for PG (now for deterministic policy $\mu$).

Using chain-rule, the above expression can be written as:
$$\mathbb{E}_{s \sim \rho^{\mu}}[\nabla_{\bm{\theta}} \mu(s; \bm{\theta}) \cdot \nabla_a Q^{\mu}(s,a) \Bigr\rvert_{a=\mu(s;\bm{\theta})}]$$

Note that $\nabla_{\bm{\theta}} \mu(s;\bm{\theta})$ is a Jacobian matrix as it takes the partial derivatives of a potentially multi-dimensional action $a = \mu(s; \bm{\theta})$ with respect to each parameter in $\bm{\theta}$. As we've pointed out during the coverage of (stochastic) PG, when $\bm{\theta}$ changes, policy $\mu$ changes, which changes the state distribution $\rho^{\mu}$. So it's not clear that this calculation indeed guarantees improvement - it doesn't take into account the effect of changing $\bm{\theta}$ on $\rho^{\mu}$. However, as was the case in PGT, Deterministic Policy Gradient (abbreviated DPGT) ensures that there is no need to compute the gradient of $\rho^{\mu}$ with respect to $\bm{\theta}$, and that the update described above indeed follows the gradient of the Expected Return objective function. We formalize this now by stating the DPGT.

Analogous to the Expected Returns Objective defined for (stochastic) PG, we define the Expected Returns Objective $J(\bm{\theta})$ for DPG as:

\begin{align*}
J(\bm{\theta}) & = \mathbb{E}_{\mu}[\sum_{t=0}^\infty \gamma^t \cdot R_{t+1}]\\
& = \sum_{s \in \mathcal{N}} \rho^{\mu}(s) \cdot \mathcal{R}_s^{\mu(s;\bm{\theta})} \\
& = \mathbb{E}_{s \sim \rho^{\mu}}[\mathcal{R}_s^{\mu(s;\bm{\theta})}]
\end{align*}

where

$$\rho^{\mu}(s) = \sum_{S_0 \in \mathcal{N}}  \sum_{t=0}^\infty \gamma^t \cdot p_0(S_0) \cdot p(S_0 \rightarrow s, t, \mu)$$

is the Discounted-Aggregate State-Visitation Measure when following deterministic policy $\mu(s; \bm{\theta})$.

With a derivation similar to the proof of the PGT, we have the DPGT, as follows:

\begin{theorem}[Deterministic Policy Gradient Theorem]
Given an MDP with action space $\mathbb{R}^k$, 
\begin{align*}
\nabla_{\bm{\theta}} J(\bm{\theta}) & = \sum_{s \in \mathcal{N}} \rho^{\mu}(s) \cdot \nabla_{\bm{\theta}} \mu(s; \bm{\theta}) \cdot \nabla_a Q^{\mu}(s,a) \Bigr\rvert_{a=\mu(s;\bm{\theta})}\\
& = \mathbb{E}_{s \sim \rho^{\mu}}[\nabla_{\bm{\theta}} \mu(s; \bm{\theta}) \cdot \nabla_a Q^{\mu}(s,a) \Bigr\rvert_{a=\mu(s;\bm{\theta})}]
\end{align*}
\label{th:deterministic-policy-gradient-theorem}
\end{theorem}

In practice, we use an Actor-Critic algorithm with a function approximation $Q(s,a;\bm{w})$ for the Q-Value Function as the Critic. To ensure exploration, we employ an exploratory behavior policy so we can do an Off-Policy DPG algorithm. We avoid importance sampling in the Actor because DPG doesn't involve an integral over actions, and we avoid importance sampling in the Critic by employing Q-Learning. 

The Critic parameters $\bm{w}$ are updated after each atomic experience as:
$$\Delta \bm{w} = \alpha_{\bm{w}} \cdot (R_{t+1} + \gamma \cdot Q(S_{t+1}, \mu(S_{t+1}; \bm{\theta}); \bm{w}) - Q(S_t,A_t;\bm{w})) \cdot \nabla_{\bm{w}} Q(S_t,A_t; \bm{w})$$

The Actor parameters $\bm{\theta}$ are updated after each atomic experience as:
$$\Delta \bm{\theta} = \alpha_{\bm{\theta}} \cdot \gamma^t \cdot \nabla_{\bm{\theta}} \mu(S_t; \bm{\theta}) \cdot \nabla_a Q(S_t,A_t;\bm{w}) \Bigr\rvert_{a=\mu(S_t;\bm{\theta})}$$

Critic Bias can be resolved with a Compatible Function Approximation Theorem for DPG (see Silver et al. paper for details). Instabilities caused by Bootstrapped Off-Policy Learning with Function Approximation can be resolved with Gradient Temporal Difference (GTD).


### Evolutionary Strategies

We conclude this chapter with a section on Evolutionary Strategies - a class of algorithms to solve MDP Control problems. We want to highlight right upfront that Evolutionary Strategies are technically not RL algorithms (for reasons we shall illuminate once we explain the technique of Evolutionary Strategies). However, Evolutionary Strategies can sometimes be quite effective in solving MDP Control problems and so, we give them appropriate coverage as part of a wide range of approaches to solve MDP Control. We cover them in this chapter because of their superficial resemblance to Policy Gradient Algorithms (again, they are not RL algorithms and hence, not Policy Gradient algorithms).

Evolutionary Strategies (abbreviated as ES) actually refers to a technique/approach that is best understood as a type of Black-Box Optimization. It was popularized in the 1970s as *Heuristic Search Methods*. It is loosely inspired by natural evolution of living beings. We focus on a subclass of ES known as Natural Evolutionary Strategies (abbreviated as NES).

The original setting for this approach was quite generic and not at all specific to solving MDPs. Let us understand this generic setting first.  Given an objective function $F(\bm{\psi})$, where $\bm{\psi}$ refers to parameters, we consider a probability distribution $p_{\bm{\theta}}(\bm{\psi})$ over $\bm{\psi}$, where $\bm{\theta}$ refers to the parameters of the probability distribution. The goal in this generic setting is to maximize the average objective $\mathbb{E}_{\bm{\psi} \sim p_{\bm{\theta}}}[F(\bm{\psi})]$.

We search for optimal $\theta$ with stochastic gradient ascent as follows:

\begin{align}
\nabla_{\bm{\theta}} (\mathbb{E}_{\bm{\psi} \sim p_{\bm{\theta}}}[F(\bm{\psi})]) & = \nabla_{\bm{\theta}} (\int_{\bm{\psi}} p_{\bm{\theta}}(\bm{\psi}) \cdot F(\bm{\psi}) \cdot d\bm{\psi}) \nonumber \\
& = \int_{\bm{\psi}} \nabla_{\bm{\theta}}(p_{\bm{\theta}}(\bm{\psi})) \cdot F(\bm{\psi}) \cdot d\bm{\psi} \nonumber \\
& = \int_{\bm{\psi}} p_{\bm{\theta}}(\bm{\psi}) \cdot \nabla_{\bm{\theta}}(\log{p_{\bm{\theta}}(\bm{\psi})}) \cdot F(\bm{\psi}) \cdot d\bm{\psi} \nonumber \\
& = \mathbb{E}_{\bm{\psi} \sim p_{\bm{\theta}}}[\nabla_{\bm{\theta}}(\log{p_{\bm{\theta}}(\bm{\psi})}) \cdot F(\bm{\psi})] \label{eq:nes-gradient}
\end{align}

Now let's see how NES can be applied to solving MDP Control. We set $F(\cdot)$ to be the (stochastic) *Return* of an MDP. $\bm{\psi}$ corresponds to the parameters of a deterministic policy $\pi_{\bm{\psi}} : \mathcal{N} \rightarrow \mathcal{A}$. $\bm{\psi} \in \mathbb{R}^m$ is drawn from an isotropic $m$-variate Gaussian distribution, i.e., Gaussian with mean vector $\bm{\theta} \in \mathbb{R}^m$ and fixed diagonal covariance matrix $\sigma^2 \bm{I_m}$ where $\sigma \in \mathbb{R}$ is kept fixed and $\bm{I_m}$ is the $m \times m$ identity matrix. The average objective (*Expected Return*) can then be written as:
$$\mathbb{E}_{\bm{\psi} \sim p_{\bm{\theta}}}[F(\bm{\psi})] = \mathbb{E}_{\bm{\epsilon} \sim \mathcal{N}(0,\bm{I_m})}[F(\bm{\theta} + \sigma \cdot \bm{\epsilon})]$$
where $\bm{\epsilon} \in \mathbb{R}^m$ is the standard normal random variable generating $\bm{\psi}$.
Hence, from Equation \eqref{eq:nes-gradient}, the gradient ($\nabla_{\bm{\theta}}$) of *Expected Return* can be written as:
$$\mathbb{E}_{\bm{\psi} \sim p_{\bm{\theta}}}[\nabla_{\bm{\theta}}(\log{p_{\bm{\theta}}(\bm{\psi})}) \cdot F(\bm{\psi})]$$
$$= \mathbb{E}_{\bm{\psi} \sim \mathcal{N}(\bm{\theta}, \sigma^2 \bm{I_m})}[\nabla_{\bm{\theta}} ( \frac {-(\bm{\psi} - \bm{\theta})^T \cdot (\bm{\psi} - \bm{\theta})} {2\sigma^2}) \cdot F(\bm{\psi})] $$
$$ =\frac 1 {\sigma} \cdot \mathbb{E}_{\bm{\epsilon} \sim \mathcal{N}(0,\bm{I_m})}[\bm{\epsilon} \cdot F(\bm{\theta} + \sigma \cdot \bm{\epsilon})]$$

Now we come up with a sampling-based algorithm to solve the MDP. The above formula helps estimate the gradient of *Expected Return* by sampling several $\bm{\epsilon}$ (each $\bm{\epsilon}$ represents a *Policy* $\pi_{\bm{\theta} + \sigma \cdot \bm{\epsilon}}$), and averaging $\bm{\epsilon} \cdot F(\bm{\theta} + \sigma \cdot \bm{\epsilon})$ across a large set ($n$) of $\bm{\epsilon}$ samples.

Note that evaluating $F(\bm{\theta} + \sigma \cdot \bm{\epsilon})$ involves playing an episode for a given sampled $\bm{\epsilon}$, and obtaining that episode's *Return* $F(\bm{\theta} + \sigma \cdot \bm{\epsilon})$. Hence, we have $n$ values of $\bm{\epsilon}$, $n$ *Policies* $\pi_{\bm{\theta} + \sigma \cdot \bm{\epsilon}}$, and $n$ *Returns* $F(\bm{\theta} + \sigma \cdot \bm{\epsilon})$.

Given the gradient estimate, we update $\bm{\theta}$ in this gradient direction, which in turn leads to new samples of $\bm{\epsilon}$ (new set of *Policies* $\pi_{\bm{\theta} + \sigma \cdot \bm{\epsilon}}$), and the process repeats until $\mathbb{E}_{\bm{\epsilon} \sim \mathcal{N}(0,\bm{I_m})}[F(\bm{\theta} + \sigma \cdot \bm{\epsilon})]$ is maximized.

The key inputs to the algorithm will be:

* Learning rate (SGD Step Size) $\alpha$
* Standard Deviation $\sigma$
* Initial value of parameter vector $\bm{\theta_0}$

With these inputs, for each iteration $t = 0, 1, 2, \ldots$, the algorithm performs the following steps:

* Sample $\bm{\epsilon_1}, \bm{\epsilon_2}, \ldots \bm{\epsilon_n} \sim \mathcal{N}(0, \bm{I_m})$.
* Compute Returns $F_i \leftarrow F(\bm{\theta_t} + \sigma \cdot \bm{\epsilon_i})$ for $i = 1,2, \ldots, n$.
* $\bm{\theta_{t+1}} \leftarrow \bm{\theta_t} + \frac {\alpha} {n \sigma} \sum_{i=1}^n \bm{\epsilon_i} \cdot F_i$

On the surface, this NES algorithm looks like PG because it's not Value Function-based (it's Policy-based, like PG). Also, similar to PG, it uses a gradient to move the policy towards optimality. But, ES does not interact with the environment (like PG/RL does). ES operates at a high-level, ignoring the (state, action, reward) interplay. Specifically, it does not aim to assign credit to actions in specific states. Hence, ES doesn't have the core essence of RL: *Estimating the Q-Value Function of a Policy and using it to Improve the Policy*. Therefore, we don't classify ES as Reinforcement Learning. Rather, we consider ES to be an alternative approach to RL Algorithms.

What is the effectiveness of ES compared to RL? The traditional view has been that ES won't work on high-dimensional problems. Specifically, ES has been shown to be data-inefficient relative to RL. This is because ES resembles simple hill-climbing based only on finite differences along a few random directions at each step. However, ES is very simple to implement (no Value Function approximation or back-propagation needed), and is highly parallelizable. ES has the benefits of being indifferent to distribution of rewards and to action frequency, and is tolerant of long horizons.
[A paper from OpenAI Research](https://arxiv.org/pdf/1703.03864.pdf) [@salimans2017evolution] shows techniques to make NES more robust and more data-efficient, and they demonstrate that NES has more exploratory behavior than advanced PG algorithms.

### Key Takeaways from this Chapter

* Policy Gradient Algorithms are based on GPI with Policy Improvement as a Stochastic Gradient Ascent for "Expected Returns" Objective $J(\bm{\theta})$ where $\bm{\theta}$ are the parameters of the function approximation for the Policy,
* Policy Gradient Theorem gives us a simple formula for $\nabla_{\bm{\theta}} J(\bm{\theta})$ in terms of the gradient/score of the policy function approximation with respect to the parameters $\bm{\theta}$ of the function approximation.
* We can reduce variance in PG algorithms by using a critic and using an estimate of the advantage function for the Q-Value Function.
* Compatible Function Approximation Theorem enables us to overcome bias in PG Algorithms.
* Natural Policy Gradient and Deterministic Policy Gradient are specialized PG algorithms that have worked well in practice.
* Evolutionary Strategies are technically not RL, but they resemble PG Algorithms and can sometimes be quite effective in solving MDP Control problems.
