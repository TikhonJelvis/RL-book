# Chapter 3

We've said before that this book is about "sequential decisioning" under "sequential uncertainty". In the previous chapter, we covered the "sequential uncertainty" aspect with the framework of Markov Processes, and we extended the framework to also incoporate the notion of uncertain "Reward" each time we make a state transition - we called this extended framework Markov Reward Processes. However, this framework had no notion of "sequential decisioning". In this chapter, we will extend the framework of Markov Reward Processes further to incorporate the notion of "sequential decisioning", formally known as Markov Decision Processes. Before we step into the formalism of Markov Decision Processes, let us develop some intuition and motivation for the need to have such a framework to handle sequential decisioning. Let's do this by re-visiting the simple inventory example we covered in the previous chapter.

## Simple Inventory Example: How much to Order?

When we covered the simple inventory example in the previous chapter in the context of Markov Decision Processes, the ordering policy was established according to the formula:

$$U = \max(C - (\alpha + \beta), 0)$$

where $U \in \mathbb{Z}_{\geq 0}$ is the order quantity, $C \in \mathbb{Z}_{\geq 0}$ is the capacity (in units) at the store, $\alpha$ is the On-Hand Inventory and $\beta$ is the On-Order Inventory ($(\alpha, \beta)$ comprising the *State*). We calculated the Value Function for the Markov Reward Process that results from following this policy. Now we ask the question if this Value Function is good enough. More importantly, we ask the question, can we improve this Value Function by following a different ordering policy?  Perhaps by ordering less than the above formula for $U$. This leads to the natural question - Can we identify the ordering policy that yields the *Optimal* Value Function (one with the highest expected returns, i.e., lowest expected accumulated costs, from each state). Let us get an intuitive sense for this optimization problem by considering a concrete example.

Assume that instead of bicycles, we are trying to control the inventory of a specific type of toothpaste in the store. Assume we have space for 20 units of toothpaste on the shelf assigned to the toothpaste (assume there is no space in the backroom of the store). Asssume that customer demand follows a Poisson distribution with Poisson parameter $\lambda = 3.0$. At 6pm store-closing each evening, when you observe the *State* as $(\alpha, \beta)$, we now have a choice of ordering a quantity of toothpastes from any of the following values of the order quantity $U: \{0, 1, \ldots, \max(20 - (\alpha - \beta), 0)\}$. Let's say on Monday 6pm store-closing $\alpha = 4$ and $\beta = 3$. So, you have a choice of order quantities from among the integers in the range of 0 to 20 - (4+3)=13 (i.e., 14 choices). Previously, in the Markov Reward Process situation, we would have ordered 13 units on Monday store-closing. This means on Wednesday morning at 6am, a truck would have arrived with 13 units of the toothpaste. If you sold say 2 units of the toothpaste on Tuesday, then on Wednesday 8am at store-opening, you'd have 4+3-2+13=18 units of toothpaste on your shelf. If you keep following this policy, you'd typically have almost a full shelf at store-opening each day, which covers almost a week worth of expected demand for the toothpaste. This means our risk of going out-of-stock on the toothpaste is extremely low, but we'd be incurring considerable holding cost since each night, we'd have close to a full shelf of toothpastes sitting around. So as a store manager, you'd be thinking - "I can lower my costs by ordering less than that prescribed by the formula of $20 - (\alpha + \beta)$". But how much less? If you order too little, you'd start the day with too little inventory and might risk going out-of-stock. That's a risk you are are highly uncomfortable with since the stock-out cost per unit of missed demand (we called it $p$) is typically much higher than the holding cost per unit (we called it $h$). So you'd rather "err" on the side of having more inventory. But how much more? We also need to factor in the fact that the 36-hour lead time means a large order incurs large holding costs *two days later*. Most importantly, to find this right balance in terms of a precise mathematical optimization of the Value Function, we'd have to factor in the uncertainty of demand (based on daily Poisson probabilities) in our calculations. Now this gives you a flavor of the problem of sequential decisioning (each day you have to decide how much to order) under sequential uncertainty.

To deal with the "decisioning" aspect, we will introduce the notion of *Action* to complement the previously introduced notions of *State* and *Reward*. In the inventory example, the order quantity is our *Action*. After observing the *State*, we choose from among a set of Actions (in this case, we choose from within the set $\{0, 1, \ldots, \max(C - (\alpha + \beta), 0)\}$). We note that the Action we take upon observing a state affects the next day's state - since the next day's state includes On-Order which is exactly equal to the Action (i.e., Order) quantity. This in turn might affect our next day's action since typically the Action (order quantity) is a function of the state (On-Hand and On-Order inventory). Moreover, the Action we take on a given day will influence our Rewards after a couple of days (i.e. after the order arrives). It may affect our holding cost adversely if we had ordered too much or it may affect our stockout cost adversely if we had too little inventory or too much unexpected demand and if we had chosen to order too little.

## The Difficulty of Sequential Decisioning under Uncertainty

This simple inventory example has giving us a peek into the world of Markov Decision Processes, which in general, have two distinct (and inter-dependent) high-level features:

* At each time step $t$, an *Action* $A_t$ in picked (from among a specified choice of actions) upon observing the *State* $S_t$
* Given an observed *State* $S_t$ and a performed *Action* $A_t$, the probabilities of the state and reward of the next time step ($S_{t+1}$ and $R_{t+1}$) are in general a function of not just the state $S_t$ but also of the action $A_t$.

We are tasked with maximizing the expected returns from each state (i.e., maximizing the Value Function). This seems like a pretty hard problem in the general case because there is a cyclic interplay between:

* actions dependening on state on one hand, and

* next state/reward probabilities depending on action (and state) on the other hand.

There is also the challenge that actions might have delayed consequences on rewards, and it's not clear how to disentangle the effects of actions from different time steps on a future reward. So without direct correspondences between actions and rewards, how can we control the actions so as to maximize expected accumulated rewards? To answer this question, we will need to set up some notation and theory. Before we formally define the Markov Decision Process framework and it's associated (elegant) theory, let us set up a bit of terminology.

Using the language of AI, we say that at each time step $t$, the *Agent* (the algorithm we design) observes the state $S_t$, after which the Agent performs action $A_t$, after which the *Environment* (upon seeing $S_t$ and $A_t$) produces a random pair: the next state state $S_{t+1}$ and the next reward $R_{t+1}$, after which the *Agent* oberves this next state $S_{t+1}$, and the cycle repeats. This cyclic interplay is depicted in Figure \ref{fig:mdp_cycle}. Note that time ticks over from $t$ to $t+1$ when the environment sees the state $S_t$ and action $A_t$.

![Markov Decision Process \label{fig:mdp_cycle}](./chapter3/mdp.png "Agent-Environment interaction in a Markov Decision Process")

## Formal Definition of a Markov Decision Process

Similar to the definitions of Markov Processes and Markov Reward Processes, for ease of exposition, the definitions and theory of Markov Decision  Processes below will be for discrete-time, for countable state spaces and countable set of pairs of next state and reward transitions (with the knowledge that the definitions and theory are analogously extensible to continuous-time and uncountable spaces, which we shall indeed encounter in this book). 

\begin{definition}

 A {\em Markov Decision Process} comprises of:

 \begin{itemize}

\item A countable set of states $\mathcal{S}$ and a countable set of actions $\mathcal{A}$

 \item A time-indexed sequence of environment-generated random states $S_t$ for each time $t=0, 1, 2, \ldots$, with each $S_t \in \mathcal{S}$

 \item A time-indexed sequence of environment-generated {\em Reward} random variables $R_t \in \mathbb{R}$ for each time $t=1, 2, \ldots$

\item A time-indexed sequence of agent-controllable actions $A_t$ for each time $t=0, 1, 2, \ldots$, with each $A_t \in \mathcal{A}$. Sometimes we restrict the set of actions allowable from specific states, in which case, we abuse the $\mathcal{A}$ notation to refer to a function whose domain is $\mathcal{S}$ and range is $\mathcal{A}$, and say that the set of actions allowable from a state $s\in \mathcal{S}$ is $\mathcal{A}(s)$.

 \item Markov Property: $\mathbb{P}[(R_{t+1}, S_{t+1}) | (S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0)] = \mathbb{P}[(R_{t+1}, S_{t+1}) | (S_t, A_t)]$ for all $t \geq 0$

 \item Specification of a discount factor $\gamma \in [0,1]$

 \end{itemize}

 \end{definition}

Like in the case of Markov Reward Processes, the role of $\gamma$ only comes in discounting future rewards when accumulating rewards from a given state - more on this later.

Like in the case of Markov Processes and Markov Reward Processes, we shall (by default) assume Stationarity for Markov Decision Processes, i.e., $\mathbb{P}[(R_{t+1}, S_{t+1}) | (S_t, A_t)]$ is independent of $t$. This means the transition probabilities of a Markov Decision Process can, in the most general case, be expressed as a state-reward transition probability function:

 $$\mathcal{P}_R: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \times \mathcal{S} \rightarrow [0,1]$$

 defined as:

 $$\mathcal{P}_R(s,a,r,s') = \mathbb{P}[(R_{t+1}=r, S_{t+1}=s') |(S_t=s, A_t=a)]$$ such that $$\sum_{s'\in \mathcal{S}} \sum_{r \in \mathbb{R}} \mathcal{P}_R(s,a,r,s') = 1 \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}$$

Henceforth, any time we say Markov Decision Process, assume we are refering to a Discrete-Time Stationary Markov Decision Process with the above-mentioned countable spaces/transitions assumptions (unless explicitly specified otherwise), which in turn will be characterized by the state-reward transition probability function $\mathcal{P}_R$. 

Let us now proceed to write some code that captures this formalism. 

TODO: Show the code for abstract class MarkovDecisionProcess

Now let us develop some more theory. Given a specification of $\mathcal{P}_R$, we can construct:

* The state transition probability function

$$\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$$

 defined as:

 $$\mathcal{P}(s, a, s') = \sum_{r\in \mathbb{R}} \mathcal{P}_R(s,a, r,s')$$

* The reward transition function:

 $$\mathcal{R}_T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$$

defined as:

$$\mathcal{R}_T(s,a,s') = \mathbb{E}[R_{t+1}|(S_{t+1}=s',S_t=s,A_t=a)]$$_

$$ = \sum_{r\in \mathcal{R}} \frac {\mathcal{P}_R(s,a,r,s')} {\mathcal{P}(s,a,s')} \cdot r = \sum_{r\in \mathcal{R}} \frac {\mathcal{P}_R(s,a,r,s')} {\sum_{r\in \mathbb{R}} \mathcal{P}_R(s,a,r,s')} \cdot r$$

The Rewards specification of most Markov Decision Processes we encounter in practice can be directly expressed as the reward transition function $\mathcal{R}_T$ (versus the more general specification of $\mathcal{P}_R$). Lastly, we want to highlight that we can transform either of $\mathcal{P}_R$ or $\mathcal{R}_T$ into a "more compact" reward function that is sufficient to perform key calculations involving Markov Decision Processes. This reward function

 $$\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

 is defined as:

 $$\mathcal{R}(s,a) = \mathbb{E}[R_{t+1}|(S_t=s, A_t=a)]$$_

$$= \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \mathcal{R}_T(s,a,s') = \sum_{s'\in \mathcal{S}} \sum_{r\in\mathbb{R}} \mathcal{P}_R(s,a,r,s') \cdot r$$

## Policy

Having understood the dynamics of a Markov Decision Process, we now move on to the specification of the *Agent*'s actions as a function of the observed state. In the general case, we assume that the Agent will perform a random action, according to a probability distribution that is a function of the state the Agent just observed. We refer to this function as a *Policy*. Formally, a *Policy* is a function

$$\pi: \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$$

defined as:

$$\pi(a, s) = \mathbb{P}[A_t = a|S_t = s] \text{ for all } t = 0, 1, 2, \ldots, \text{ for all } s\in \mathcal{S}, a \in \mathcal{A}$$

Note in the definition above that we've assumed that a Policy is stationary, i.e., $\pi$ is invariant in time $t$. If we do encounter a situation where the policy would need to depend on the time $t$, we'll simply include $t$ to be part of the state, which would make the Policy stationary.

When we have a policy such that the action probability distribution for each state is concentrated on a single action, we refer to it as a deterministic policy. Formally, a deterministic policy has the property that for all $s\in \mathcal{S}$,

$$\pi(\pi_D(s), s) = 1 \text{ and } \pi(a, s) = 0 \text{ for all } a\in \mathcal{A} \text{ with } a \neq \pi_D(s)$$

where $\pi_D: \mathcal{S} \rightarrow \mathcal{A}$.

So we shall specify deterministic policies simply with the function $\pi_D$. We shall refer to non-deterministic policies as stochastic policies (the word stochastic reflecting the fact that the agent will perform a random action according to the probability distribution specified by $\pi$). So when we use the notation $\pi$, assume that we are dealing with a stochastic (i.e., non-deterministic) policy and when we use the notation $\pi_D$, assume that we are dealing with a deterministic policy.

TODO: Show code for Policy class

Note that in the previous chapter, the Deterministic Policy we followed when constructing the simple inventory examples was:

$$\pi_D((\alpha, \beta)) = \max(C - (\alpha _ \beta), 0)$$ where $C$ is the capacity in the store for bicycles, $\alpha$ is the On-Hand Inventory at store-closing and $\beta$ is the On-Order Inventory at store-closing.

TODO: Show this above deterministic policy in code. Also show in code an example of a stochastic policy.

## Simple Inventory Example with Unlimited Capacity (Infinite State/Action Space)

TODO: Show the class deriving from MarkovDecisionProcess for simple inventory example with unlimited capacity. Do simulations, plot some graphs

## Finite Markov Decision Processes

TODO: Show code for class FiniteMarkovDecisionProcess

## Simple Inventory Example as a Finite Markov Decision Process

TODO: Set up the mathematical specification of simple Inventory example as an MDP

TODO: Show code for class SimpleInventoryMDP as a derived class of FiniteMarkovDecisionProcess


## [Markov Decision Process, Policy] := Markov Reward Process

This section has an important insight - that if we evaluate a Markov Decision Process (MDP) with a fixed policy $\pi$, we get the Markov Reward Process (MRP) that is implied by the evaluation of the MDP with the policy $\pi$ and the MDP. Let's clarify this with notational precision. But first we need to point out that we have some notation clashes between MDP and MRP. We used $\mathcal{P}_R$ to denote the transition probability function of the MRP as well as to denote the state-reward transition probability function of the MDP. We used $\mathcal{P}$ to denote the transition probability function of the Markov Process implicit in the MRP as well as to denote the state transition probability function of the MDP. We used $\mathcal{R}_T$ to denote the reward transition function of the MRP as well as to denote the reward transition function of the MDP. We used $\mathcal{R}$ to denote the reward function of the MRP as well as to denote the reward function of the MDP. We can resolve these notation clashes by noting the arguments being passed to $\mathcal{P}_R$, $\mathcal{P}, \mathcal{R}_T$ and $\mathcal{R}$, but to be extra-clear, we'll put a superscript of $\pi \rightarrow MRP$ to each of the functions $\mathcal{P}_R$, $\mathcal{P}, \mathcal{R}_T$ for the $\pi$-implied MRP so as to distinguish between these functions for the MDP versus the $\pi$-implied MRP.

Let's say we are given a fixed policy $\pi$ and an MDP specified by it's state-reward transition probability function $\mathcal{P}_R$. Then the transition probability function $\mathcal{P}^{\pi \rightarrow MRP}$ of the MRP implied by the evaluation of the MDP with the policy $\pi$ is defined as:

$$\mathcal{P}_R^{\pi \rightarrow MRP}(s,s',r) = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{P}(s,a,s',r)$$ 

Likewise,

$$\mathcal{P}^{\pi \rightarrow MRP}(s,s') = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{P}(s,a,s')$$

$$\mathcal{R}_T^{\pi \rightarrow MRP}(s,s') = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{R}_T(s,a,s')$$

$$\mathcal{R}^{\pi \rightarrow MRP}(s) = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{R}(s,a)$$

So each time we talk about an MDP evaluated with a fixed policy, you should know that we are effectively talking about the implied MRP.

TODO: Code to convert MDP + Policy to an MRP

## MDP Value Function for a Fixed Policy

Now we are ready to talk about the Value Function for an MDP evaluated with a fixed policy $\pi$. Just like in the case of MRP, we define the Return $G_t$ at time step $t$ for an MDP as:
$$G_t = \sum_{i=t+1}^{\infty} \gamma^{i-t-1} \cdot R_i = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots$$

The Value Function for an MDP evaluated with a fixed policy $\pi$
$$V^{\pi}: \mathcal{S} \rightarrow \mathbb{R}$$
is defined as:
$$V^{\pi}(s) = \mathbb{E}_{\pi, \mathcal{P}_R}[G_t|S_t=s] \text{ for all } s \in \mathcal{S}, \text{ for all } t = 0, 1, 2, \ldots$$

Now let's expand $\mathbb{E}_{\pi, \mathcal{P}_R}[G_t|S_t=s]$.

\begin{equation*}
\begin{split}
V^{\pi}(s) & = \mathbb{E}_{\pi, \mathcal{P}_R}[R_{t+1}|S_t=s] + \gamma \cdot \mathbb{E}_{\pi, \mathcal{P}_R}[R_{t+2}|S_t=s] + \gamma^2 \cdot \mathbb{E}_{\pi, \mathcal{P}_R}[R_{t+3}|S_t=s] + \ldots \\
& = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{R}(s,a) + \gamma \cdot \sum_{a\in \mathcal{A}} \pi(s,a) \sum_{s'\in \mathcal{S}} \mathcal{P}(s, a, s') \sum_{a'\in \mathcal{A}} \pi(s',a') \cdot \mathcal{R}(s', a') \\
& \hspace{4mm} + \gamma^2 \cdot \sum_{a\in \mathcal{A}} \pi(s,a) \sum_{s' \in \mathcal{S}} \mathcal{P}(s, a', s') \sum_{a'\in \mathcal{A}} \pi(s',a') \sum_{s'' \in \mathcal{S}} \mathcal{P}(s', a'', s'') \sum_{a''\in \mathcal{A}} \pi(s'',a'') \cdot \mathcal{R}(s'', a'')  \\
& \hspace{4mm} + \ldots \\
& = \mathcal{R}^{\pi \rightarrow MRP}(s) + \gamma \cdot \sum_{s'\in \mathcal{S}} \mathcal{P}^{\pi \rightarrow MRP}(s, s') \cdot \mathcal{R}^{\pi \rightarrow MRP}(s') \\
& \hspace{4mm} + \gamma^2 \cdot \sum_{s' \in \mathcal{S}} \mathcal{P}^{\pi \rightarrow MRP}(s, s') \sum_{s'' \in \mathcal{S}} \mathcal{P}^{\pi \rightarrow MRP}(s', s'') \cdot \mathcal{R}^{\pi \rightarrow MRP}(s'') + \ldots
\end{split}
\end{equation*}

But from Equation \eqref{eq:mrp_bellman_eqn} in the previous chapter, we know that the last expression above is equal to $V^{\pi \rightarrow MRP}(s)$(i.e, the Value Function for state $s$ of the $\pi$-implied MRP). So, the Value Function $V^{\pi}(\cdot)$ of an MDP evaluated with a fixed policy $\pi$ is the same function as the Value Function $V^{\pi \rightarrow MRP}(\cdot)$ of the $\pi$-implied MRP. So we can apply the MRP Bellman Equation on $V^{\pi}$, i.e.,

\begin{equation}
\begin{split}
V^{\pi}(s) & = \mathcal{R}^{\pi \rightarrow MRP}(s) + \gamma \cdot \sum_{s' \in \mathcal{S}} \mathcal{P}^{\pi \rightarrow MRP}(s,s') \cdot V^{\pi}(s')\\
& = \sum_{a\in\mathcal{A}} \pi(s,a) \cdot \mathcal{R}(s,a) + \gamma \cdot \sum_{a\in \mathcal{A}} \pi(s,a) \sum_{s'\in \mathcal{S}} \mathcal{P}(s,a,s') \cdot V^{\pi}(s') \\
& = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot (\mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{S}} \mathcal{P}(s,a,s') \cdot V^{\pi}(s')) \text{ for all  } s \in \mathcal{S}
\end{split}
\label{eq:mdp_bellman_policy_eqn_vv}
\end{equation}

As we saw in the previous chapter, for finite state spaces that are not too large, Equation \eqref{eq:mdp_bellman_policy_eqn_vv} can be solved with a linear algebra solution (Equation \eqref{eq:mrp_bellman_linalg_solve} from the previous chapter). More broadly, Equation \eqref{eq:mdp_bellman_policy_eqn_vv} will be a key equation for the rest of the book in developing various Dynamic Programming and Reinforcement Algorithms. However, there is another Value Function that's also going to be crucial in developing MDP algorithms - one which maps a (state, action) pair to the expected return originating from the (state, action) pair when evaluated with a fixed policy. This is known as the *Action-Value Function* of an MDP evaluated with a fixed policy $\pi$:
$$Q^{\pi}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$
defined as:
$$Q^{\pi}(s, a) = \mathbb{E}_{\pi, \mathcal{P}_R}[G_t|(S_t=s, A_t=a)] \text{ for all } s \in \mathcal{S}, a\in \mathcal{A}, \text{ for all } t = 0, 1, 2, \ldots$$

To avoid terminology confusion, we refer to $V^{\pi}$ as the *State-Value Function* (albeit often simply abbreviated to *Value Function*) for policy $\pi$, to distinguish from the *Action-Value Function* $Q^{\pi}$. From the definition of $Q^{\pi}$, we have the following equation connecting $Q^{\pi}$ and $V^{\pi}$:

\begin{equation}
V^{\pi}(s) = \sum_{a\in\mathcal{A}} \pi(s, a) \cdot Q^{\pi}(s, a) \text{ for all } s \in \mathcal{S} \label{eq:mdp_bellman_policy_eqn_vq}
\end{equation}

Combining Equation \eqref{eq:mdp_bellman_policy_eqn_vv} and Equation \eqref{eq:mdp_bellman_policy_eqn_vq} yields:
\begin{equation}
Q^{\pi}(s, a) = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{S}} \mathcal{P}(s,a,s') \cdot V^{\pi}(s') \text{ for all  } s \in \mathcal{S}, a \in \mathcal{A} \label{eq:mdp_bellman_policy_eqn_qv}
\end{equation}

Equation \eqref{eq:mdp_bellman_policy_eqn_vv} is known as the MDP State-Value Function Bellman Policy Equation. Figure \ref{fig:mdp_bellman_policy_tree_vv} serves as a visualization aid for this equation (note that Equations \eqref{eq:mdp_bellman_policy_eqn_vq} and Equation \eqref{eq:mdp_bellman_policy_eqn_qv} are also embedded in this Figure). For the rest of the book, in these MDP transition figures, we shall always depict states as elliptical-shaped nodes and actions as rectangular-shaped nodes. Notice that transition from a state node to an action node is associated with a probability represented by $\pi$ and transition from an action node to a state node is associated with a probability represented by $\mathcal{P}$.

<div style="text-align:center" markdown="1">
![Visualization of MDP State-Value Function Bellman Policy Equation \label{fig:mdp_bellman_policy_tree_vv}](./chapter3/mdp_bellman_policy_tree_vv.png "Visualization of MDP State-Value Function Bellman Policy Equation")
</div>

<div style="text-align:center" markdown="1">
![Visualization of MDP Action-Value Function Bellman Policy Equation \label{fig:mdp_bellman_policy_tree_qq}](./chapter3/mdp_bellman_policy_tree_qq.png "Visualization of MDP Action-Value Function Bellman Policy Equation")
</div>

Combining Equation \eqref{eq:mdp_bellman_policy_eqn_qv} and Equation \eqref{eq:mdp_bellman_policy_eqn_vq} yields:
\begin{equation}
Q^{\pi}(s, a) = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{S}} \mathcal{P}(s,a,s') \sum_{a'\in \mathcal{A}} \pi(s', a') \cdot Q^{\pi}(s', a') \text{ for all  } s \in \mathcal{S}, a \in \mathcal{A} \label{eq:mdp_bellman_policy_eqn_qq}
\end{equation}

Equations \eqref{eq:mdp_bellman_policy_eqn_vv}, \eqref{eq:mdp_bellman_policy_eqn_vq}, \eqref{eq:mdp_bellman_policy_eqn_qv} and \eqref{eq:mdp_bellman_policy_eqn_qq} are collectively known as the MDP Bellman Policy Equations.

TODO: For the case of finite MDPs, we can solve state-value function and action-value function (for fixed policies) using matrix inverion method by converting it to an MRP.
TODO: Write code to show to solve V and Q for fixed policy in the FiniteMarkovDecisionProcess class, then solve V and Q for a fixed policy in the simple inventory example with capacity.

## Optimal Value Function and Optimal Policies 

* Define Optimal Value Function (V and Q)

* Define Optimal Policy

* Theorem of Optimal Policy existence, and proof

* Bellman Optimality Equations (unlike Bellman Policy Equations, this are non-linear)

* Solving the Bellman Optimality Equation (topic of DP and RL algorithms) 

## Variants and extensions of MDPs

* continuous states, continuous actions, continuous time, POMDPs
* Some discussion of MDPs in the real-world, and then explain next chapter is Dynamic Programming to solve Value Function for fixed policy and Optimal Value Function

## Summary of Key Learnings from this Chapter

