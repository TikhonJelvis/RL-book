## Multi-Armed Bandits: Exploration versus Exploitation {#sec:multi-armed-bandits-chapter}

We learnt in Chapter [-@sec:rl-control-chapter] that balancing exploration and exploitation is vital in RL Control algorithms. While we want to exploit actions that seem to be fetching good returns,  we also want to adequately explore all possible actions so we can obtain an accurate-enough estimate of their Q-Values. We had mentioned that this is essentially the Explore-Exploit dilemma of the famous [Multi-Armed Bandit Problem](https://en.wikipedia.org/wiki/Multi-armed_bandit). The Multi-Armed Bandit problem provides a simple setting to understand the explore-exploit tradeoff and to develop explore-exploit balancing algorithms. The approaches followed by the Multi-Armed Bandits algorithms are then well-transportable to the more complex setting of RL Control.

In this Chapter, we start by specifying the Multi-Armed Bandit problem, followed by coverage of a variety of techniques to solve the Multi-Armed Bandit problem (i.e., effectively balancing exploration with exploitation). We've actually seen one of these algorithms already for RL Control - following an $\epsilon$-greedy policy, which naturally is applicable to the simpler setting of Multi-Armed Bandits. We had mentioned in Chapter [-@sec:rl-control-chapter] that we can simply replace the $\epsilon$-greedy approach with any other algorithm for explore-exploit tradeoff. In this chapter, we consider a variety of such algorithms, many of which are far more sophisticated compared to the simple $\epsilon$-greedy approach. However, we cover these algorithms for the simple setting of Multi-Armed Bandits as it promotes understanding and development of intuition. After covering a range of algorithms for Multi-Armed Bandits, we consider an extended problem known as Contextual Bandits, that is a step between the Multi-Armed Bandits problem and the RL Control problem (in terms of problem complexity). We explain how the algorithms for Multi-Armed Bandits can be easily transported to the more nuanced/extended setting of Contextual Bandits. We finish this chapter by explaining how we can further extend Contextual Bandits to the RL Control problem, and that the algorithms developed for both Multi-Armed Bandits and Contextual Bandits are also applicable for RL Control. 

### Introduction to the Multi-Armed Bandit Problem

We've already got a fairly good understanding of the Explore-Exploit tradeoff in the context of RL Control - selecting actions for any given state that balances the notions of exploration and exploitation. If you think about it, you will realize that many situations in business (and in our lives!) present this explore-exploit dilemma on choices one has to make. *Exploitation* involves picking choices that *seem to be best* based on past outcomes, while *Exploration* involves picking choices one hasn't yet tried (or not tried sufficiently enough).

Exploitation has intuitive notions of "being greedy" and of being "short-sighted", and too much exploitation could lead to some regret of having missing out on unexplored "gems". Exploration has intuitive notions of "gaining information" and of being "long-sighted", and too much exploration could lead to some regret of having wasting time on "duds". This naturally leads to the idea of balancing exploration and exploitation so we can combine *information-gains* and *greedy-gains* in the most optimal manner. The natural question then is whether we can set up this problem of explore-exploit dilemma in a mathematically disciplined manner. Before we do that, let's look at a few common examples of the explore-exploit dilemma. 

#### Some Examples of Explore-Exploit Dilemma

* Restaurant Selection: We like to go to our favorite restaurant (Exploitation) but we also like to try out a new restaurant (Exploration).
* Online Banner Advertisements: We like to repeat the most successful advertisement (Exploitation) but we also like to show a new advertisement (Exploration).
* Oil Drilling: We like to drill at the best known location (Exploitation) but we also like to drill at a new location (Exploration).
* Learning to play a game: We like to play the move that has worked well for us so far (Exploitation) but we also like to play a new experimental move (Exploration).

The term *Multi-Armed Bandit* (abbreviated as MAB) is a spoof name that stands for "Many One-Armed Bandits" and the term *One-Armed Bandit* refers to playing a slot-machine in a casino (that has a single lever to be pulled, that presumably addicts us and eventually takes away all our money, hence the term "bandit"). Multi-Armed Bandit refers to the problem of playing several slot machines (each of which has an unknown fixed payout probability distribution) in a manner that we can make the maximum cumulative gains by playing over multiple rounds (by selecting a single slot machine in a single round). The core idea is that to achieve maximum cumulative gains, one would need to balance the notions of exploration and exploitation, no matter which selection strategy one would pursue.

#### Problem Definition

\begin{definition}
A {\em Multi-Armed Bandit} (MAB) comprises of:
\begin{itemize}
\item A finite set of actions $\mathcal{A}$ (known as the "arms")
\item Each action ("arm") $a \in \mathcal{A}$ is associated with a probability distribution over $\mathbb{R}$ (unknown to the AI Agent) denoted as $\mathcal{R}^a$, defined as:
$$\mathcal{R}^a(r) = \mathbb{P}[r|a] \text{ for all } r \in \mathbb{R}$$
\item A time-indexed sequence of AI agent-selected actions $A_t \in \mathcal{A}$ for time steps $t=1, 2, \ldots$, and a time-indexed sequence of Environment-generated {\em Reward} random variables $R_t \in \mathbb{R}$ for time steps $t=1, 2, \ldots$, with $R_t$ randomly drawn from the probability distribution $\mathcal{R}^{A_t}$.
\end{itemize}
\end{definition}

The AI agent's goal is to maximize the following *Cumulative Rewards* over a certain number of time steps $T$:
$$\sum_{t=1}^T R_t$$

So the AI agent has $T$ selections of actions to make (in sequence), basing each of those selections only on the rewards it has observed before that time step (specifically, the Agent does not have knowledge of the probability distributions $\mathcal{R}^a$). Any selection strategy to maximize the Cumulative Rewards risks wasting time on "duds" while exploring and also risks missing untapped "gems" while exploiting.

It is immediately observable that the environment doesn't have a notion of *State*. When the Agent selects an arm, the Environment simply samples from the probability distribution for that arm. However, the agent might maintain a statistic of history as it's *State*, which would help the agent in making the arm-selection (action) decision. The action is then based on a (*Policy*) function of the agent's *State*. So, the agent's arm-selection strategy is basically this *Policy*. Thus, even though a MAB is not posed as an MDP, the agent could model it as an MDP and solve it with an appropriate Planning or Learning algorithm. However, many MAB algorithms don't take this formal MDP approach. Instead, they rely on heuristic methods that don't aim to *optimize* - they simply strive for *good* Cumulative Rewards (in Expectation). Note that even in a simple heuristic algorithm, $A_t$ is a random variable simply because it is a function of past (random) rewards.

#### Regret

The idea of *Regret* is quite fundamental in designing algorithms for MAB. In this section, we illuminate this idea.

We define the *Action Value* $Q(a)$ as the (unknown) mean reward of action $a$, i.e., 
$$Q(a) = \mathbb{E}[r|a]$$
We define the *Optimal Value* $V^*$ and *Optimal Action* $a^*$ (noting that there could be multiple optimal actions) as:
$$V^* = \max_{a\in\mathcal{A}} Q(a) = Q(a^*)$$
We define *Regret* $l_t$ as the opportunity loss at a single time step $t$, as follows:
$$l_t = \mathbb{E}[V^* - Q(A_t)]$$
We define the *Total Regret* $L_T$ as the total opportunity loss, as follows:
$$L_T = \sum_{t=1}^T l_t = \sum_{t=1}^T \mathbb{E}[V^* - Q(A_t)]$$
Maximizing the *Cumulative Reward* is the same as Minimizing *Total Regret*.

#### Counts and Gaps

Let $N_t(a)$ be the (random) number of selections of an action $a$ across the first $t$ steps. Let us refer to $\mathbb{E}[N_t(a)]$ for a given action-selection strategy as the *Count* of an action $a$ over $t$ steps, denoted as $Count_t(a)$. Let us refer to the Value difference between an action $a$ and the optimal action $a^*$ as the $Gap$ for $a$, denoted as $\Delta_a$, i.e,
$$\Delta_a = V^* - Q(a) $$
We define Total Regret as the sum-product (over actions) of $Count$s and $Gap$s, as follows:
\begin{align*}
L_T & = \sum_{t=1}^T \mathbb{E}[V^* - Q(A_t)]
 & = \sum_{a\in\mathcal{A}} \mathbb{E}[N_T(a)] \cdot (V^* - Q(a))
 & = \sum_{a\in\mathcal{A}} Count_T(a) \cdot \Delta_a
\end{align*}
A good algorithm ensures small $Count$s for large $Gap$s. The core challenge though is that *we don't know the $Gap$s*.

Next, we cover some simple heuristic algorithms.


### Simple Algorithms

We consider algorithms that estimate a Q-Value $\hat{Q}_t(a)$ for each $a\in \mathcal{A}$, as an approximation to the true Q-Value $Q(a)$. The subscript $t$ in $\hat{Q}_t$ refers to the fact that this is an estimate after $t$ time steps that takes into account all of the information available up to $t$ time steps.

A natural way of estimating $\hat{Q}_t(a)$ is by *rewards-averaging*, i.e.,

$$\hat{Q}_t(a) = \frac 1 {N_t(a)} \sum_{s=1}^t R_s \cdot \mathbb{I}_{A_s=a}$$

where $\mathbb{I}$ refers to the indicator function.

#### Greedy and $\epsilon$-Greedy

First consider an algorithm that *never* explores (i.e., *always* exploits). This is known as the *Greedy Algorithm* which selects the action with highest estimated value, i.e.,
$$A_t = \argmax_{a\in \mathcal{A}} \hat{Q}_{t-1}(a)$$

We've noted in Chapter [-@sec:rl-control-chapter] that such an algorithm can lock into a suboptimal action forever (suboptimal $a$ is an action for which $\Delta_a > 0$). This results in $Count_T(a)$ being a linear function of $T$ for some suboptimal $a$, which means the Total Regret is a linear function of $T$ (we refer to this as *Linear Total Regret*). 

Now let's consider the $\epsilon$-greedy algorithm, which explores forever. At each time-step $t$:

* With probability $1-\epsilon$, select $A_t=\argmax_{a\in\mathcal{A}} \hat{Q}_{t-1}(a)$
* With probability $\epsilon$, select a random action (uniformly) from $\mathcal{A}$

A constant value of $\epsilon$ ensures a minimum regret proportional to the mean gap, i.e.,
$$ l_t \geq \frac {\epsilon} {|\mathcal{A}|} \sum_{a\in\mathcal{A}} \Delta_a$$

Hence, the $\epsilon$-Greedy algorithm also has Linear Total Regret. 

#### Optimistic Initialization

Next we consider a simple and practical idea: Initialize $\hat{Q}_0(a)$ to a high value for all $a\in \mathcal{A}$ and update action values by incremental-averaging.  Starting with $N_0(a) \geq 0$ for all $a\in \mathcal{A}$, the updates at each time step $t$ are as follows:

$$N_t(a) = N_{t-1}(a) + \mathbb{I}_{a = A_t} \mbox{ for all } a \in \mathcal{A}$$
$$\hat{Q}_t(A_t) = \hat{Q}_{t-1}(A_t) + \frac 1 {N_t(A_t)} (R_t - \hat{Q}_{t-1}(A_t))$$
$$\hat{Q}_t(a) = \hat{Q}_{t-1}(a) \mbox{ for all } a \neq A_t$$

The idea here is that by setting a high initial value for the estimate of Q-Values (which we refer to as *Optimistic Initialization*), we encourage systematic exploration early on. Another way of doing optimistic initialization is to set a high value for $N_0(a)$ for all $a \in \mathcal{A}$, which likewise encourages systematic exploration early on. However, these optimistic initialization ideas only serve to promote exploration early on and eventually, one can still lock into a suboptimal action. Specifically, the Greedy algorithm together with optimistic initialization has Linear Total Regret. Likewise, the $\epsilon$-Greedy algorithm together with optimistic initialization also has Linear Total Regret. But in practice, these simple ideas of doing optimistic initialization work quite well. 

#### Decaying $\epsilon_t$-Greedy Algorithm

The natural question that emerges is whether it is possible to construct an algorithm with Sublinear Total Regret. Along these lines, we consider an $\epsilon$-Greedy algorithm with the $\epsilon$ decaying as time progresses. We call such an algorithm Decaying $\epsilon_t$-Greedy.

For any fixed $c > 0$, consider a decay schedule for $\epsilon_1, \epsilon_2, \ldots$ as follows:

$$d = \min_{a|\Delta_a > 0} \Delta_a$$
$$\epsilon_t = \min(1, \frac {c|\mathcal{A}|} {d^2t}\}$$

It can be shown that this decay schedule achieves *Logarithmic* Total Regret. However, note that the above schedule requires advance knowledge of the gaps $\Delta_a$. Practically, implementing *some* decay schedule helps considerably. Here's some [Educational Code](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter14/epsilon_greedy.py) for decaying $\epsilon$-greedy with optimistic initialization.

![Total Regret Curves \label{fig:total_regret_curves}](./chapter14/total_regret_curves.png "Total Regret Curves")

### Optimism in the Face of Uncertainty

![Q-Value Distributions \label{fig:q_value_distribution1}](./chapter14/q_value_distribution1.png "Q-Value Distributions")
![Q-Value Distributions \label{fig:q_value_distribution2}](./chapter14/q_value_distribution2.png "Q-Value Distributions")

### Probability Matching

### Gradient Bandits

### Information State Space MDP

### Contextual Bandits

### Extending to RL Control
