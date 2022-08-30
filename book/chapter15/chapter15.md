## Blending Learning and Planning {#sec:blending-learning-planning-chapter}

\index{planning|(}
\index{learning|(}

After coverage of the issue of Exploration versus Exploitation in the last chapter, in this chapter, we cover the topic of Planning versus Learning (and how to blend the two approaches) in the context of solving MDP Prediction and Control problems. In this chapter, we also provide some coverage of the much-celebrated Monte-Carlo Tree-Search (abbreviated as MCTS) algorithm and it's spiritual origin—the Adaptive Multi-Stage Sampling (abbreviated as AMS) algorithm. MCTS and AMS are examples of Planning algorithms tackled with sampling/RL-based techniques.

### Planning versus Learning

\index{planning versus learning}

In the language of AI, we use the terms *Planning* and *Learning* to refer to two different approaches to solve an AI problem. Let us understand these terms from the perspective of solving MDP Prediction and Control. Let us zoom out and look at the big picture. The AI Agent has access to an MDP Environment $E$. In the process of *interacting* with the MDP Environment $E$, the AI Agent receives  experiences data. Each unit of experience data is in the form of a (next state, reward) pair for the current state and action. The AI Agent's goal is to estimate the requisite Value Function/Policy through this process of interaction with the MDP Environment $E$ (for Prediction, the AI Agent estimates the Value Function for a given policy and for Control, the AI Agent estimates the Optimal Value Function and the Optimal Policy). The AI Agent can go about this in one of two ways:

\index{Markov decision process!agent}
\index{Markov decision process!environment}
\index{Markov decision process!environment!real environment}
\index{model}
\index{reinforcement learning!model-based}

1. By interacting with the MDP Environment $E$, the AI Agent can build a *Model of the Environment* (call it $M$) and then use that model to estimate the requisite Value Function/Policy. We refer to this as the *Model-Based* approach. Solving Prediction/Control using a Model of the Environment (i.e., *Model-Based* approach) is known as *Planning* the solution. The term *Planning* comes from the fact that the AI Agent projects (with the help of the model $M$) probabilistic scenarios of future states/rewards for various choices of actions from specific states, and solves for the requisite Value Function/Policy based on the model-projected future outcomes. 
2. By interacting with the MDP Environment $E$, the AI Agent can directly estimate the requisite Value Function/Policy, without bothering to build a Model of the Environment. We refer to this as the *Model-Free* approach. Solving Prediction/Control without using a model (i.e., *Model-Free* approach) is known as *Learning* the solution. The term *Learning* comes from the fact that the AI Agent "learns" the requisite Value Function/Policy directly from experiences data obtained by interacting with the MDP Environment $E$ (without requiring any model).

\index{reinforcement learning!model-free}

Let us now dive a bit deeper into both these approaches to understand them better.

#### Planning the Solution of Prediction/Control {#sec:planning-subsection}

In the first approach (*Planning* the solution of Prediction/Control), we first need to "build a model". By "model", we refer to the State-Reward Transition Probability Function $\mathcal{P}_R$. By "building a model", we mean estimating $\mathcal{P}_R$ from experiences data obtained by interacting with the MDP Environment $E$. How does the AI Agent do this? Well, this is a matter of estimating the conditional probability density function of pairs of (next state, reward), conditioned on a particular pair of (state, action). This is an exercise in Supervised Learning, where the $y$-values are (next state, reward) pairs and the $x$-values are (state, action) pairs. We covered how to do Supervised Learning in Chapter [-@sec:funcapprox-chapter]. Also, note that Equation \eqref{eq:mrp-mle} in Chapter [-@sec:rl-prediction-chapter] provides a simple tabular calculation to estimate the $\mathcal{P}_R$ function for an MRP from a fixed, finite set of atomic experiences of (state, reward, next state) triples. Following this Equation, we had written the function `finite_mrp` to construct a `FiniteMarkovRewardProcess` (which includes a tabular $\mathcal{P}_R$ function of explicit probabilities of transitions), given as input a `Sequence[TransitionStep[S]]` (i.e., fixed, finite set of MRP atomic experiences). This approach can be easily extended to estimate the $\mathcal{P}_R$ function for an MDP. Ok—now we have a model $M$ in the form of an estimated $\mathcal{P}_R$. The next thing to do in this approach of *Planning* the solution of Prediction/Control is to use the model $M$ to estimate the requisite Value Function/Policy. There are two broad approaches to do this:

\index{model!probabilities model}

1. By constructing $\mathcal{P}_R$ as an explicit representation of probabilities of transitions, the AI Agent can utilize one of the Dynamic Programming Algorithms (e.g., Policy Evaluation, Policy Iteration, Value Iteration) or a Tree-search method (by growing out a tree of future states/rewards/actions from a given state/action, e.g., the MCTS/AMS algorithms we will cover later in this chapter). Note that in this approach, there is *no need to interact with an MDP Environment* since a model of transition probabilities are available that can be used to project any (probabilistic) future outcome (for any choice of action) that is desired to estimate the requisite Value Function/Policy.
2. By treating $\mathcal{P}_R$ as a *sampling model*, by which we mean that the AI agent uses $\mathcal{P}_R$ as simply an (on-demand) interface to sample an individual pair of (next state, reward) from a given (state, action) pair. This means the AI Agent treats this *sampling model* view of $\mathcal{P}_R$ as a *Simulated MDP Environment* (let us refer to this Simulated MDP Environment as $S$). Note that $S$ serves as a proxy interaction-interface to the real MDP Environment $E$. A significant advantage of interacting with $S$ instead of $E$ is that we can sample infinitely many times without any of the real-world interaction constraints that a real MDP Environment $E$ poses. Think about a robot learning to walk on an actual street versus learning to walk on a simulator of the street's activities. Furthermore, the user could augment his/her views on top of an experiences-data-learnt simulator. For example, the user might say that the experiences data obtained by interacting with $E$ doesn't include certain types of scenarios but the user might have knowledge of how those scenarios would play out, thus creating a "human-knowledge-augmented simulator" (more on this in Chapter [-@sec:concluding-chapter]). By interacting with the simulated MDP Environment $S$ (instead of the real MDP Environment $E$), the AI Agent can use any of the RL Algorithms we covered in Module III of this book to estimate the requisite Value Function/Policy. Since this approach uses a model $M$ (albeit a sampling model) and since this approach uses RL, we refer to this approach as *Model-Based RL*. To summarize this approach, the AI Agent first learns (supervised learning) a model $M$ as an approximation of the real MDP Environment $E$, and then the AI Agent plans the solution to Prediction/Control by using the model $M$ in the form of a simulated MDP Environment $S$ which an RL algorithm interacts with. Here the Planning/Learning terminology often gets confusing to new students of this topic since this approach is supervised learning followed by planning (the planning being done with a Reinforcement Learning algorithm interacting with the learnt simulator).

\index{model!sampling model}
\index{simulator}
\index{Markov decision process!environment!simulated environment}
\index{supervised learning}
\index{reinforcement learning!model-based}

![Planning with a Supervised-Learnt Model \label{fig:planning}](./chapter15/planning.png "Planning with a Supervised-Learnt Model"){height=6cm}

Figure \ref{fig:planning} depicts the above-described approach of *Planning* the solution of Prediction/Control. We start with an arbitrary Policy that is used to interact with the Environment $E$ (upward-pointing arrow in the figure). These interactions generate Experiences, which are used to perform Supervised Learning (rightward-pointing arrow in the figure) to learn a model $M$. This model $M$ is used to plan the requisite Value Function/Policy (leftward-pointing arrow in the figure). The Policy produced through this process of Planning is then used to further interact with the Environment $E$, which in turn generates a fresh set of Experiences, which in turn are used to update the Model $M$ (incremental supervised learning), which in turn is used to plan an updated Value Function/Policy, and so the cycle repeats.

#### Learning the Solution of Prediction/Control {#sec:learning-subsection}

\index{Markov decision process!environment!real environment}

In the second approach (*Learning* the solution of Prediction/Control), we don't bother to build a model. Rather, the AI Agent directly estimates the requisite Value Function/Policy from the experiences data obtained by interacting with the real MDP Environment $E$. The AI Agent does this by using any of the RL algorithms we covered in Module III of this book. Since this approach is "model-free", we refer to this approach as *Model-Free RL*.

\index{reinforcement learning!model-free}

#### Advantages and Disadvantages of Planning versus Learning

In the previous two subsections, we covered the two different approaches to solving Prediction/Control, either by *Planning* (subsection [-@sec:planning-subsection]) or by *Learning* (subsection [-@sec:learning-subsection]). Let us now talk about their advantages and disadvantages.

*Planning* involves constructing a Model, so its natural advantage is to be able to construct a model (from experiences data) with efficient and robust supervised learning methods. The other key advantage of *Planning* is that we can reason about Model Uncertainty. Specifically, when we learn the Model $M$ using supervised learning, we typically obtain the standard errors for estimation of model parameters, which can then be used to create confidence intervals for the Value Function and Policy planned using the model. Furthermore, since modeling real-world problems tends to be rather difficult, it is valuable to create a family of models with differing assumptions, with different functional forms, with differing parameterizations etc., and reason about how the Value Function/Policy would disperse as a function of this range of models. This is quite beneficial in typical real-world problems since it enables us to do Prediction/Control in a *robust* manner.

The disadvantage of *Planning* is that we have two sources of approximation error—the first from supervised learning in estimating the model $M$, and the second from constructing the Value Function/Policy (given the model). The *Learning* approach (without resorting to a model, i.e., Model-Free RL) is thus advantageous is not having the first source of approximation error (i.e., Model Error).
\index{reinforcement learning!model-free}

#### Blending Planning and Learning

![Blending Planning and Learning \label{fig:planning_learning}](./chapter15/planning_learning.png "Blending Planning and Learning"){height=6cm}

In this subsection, we show a rather creative and practically powerful approach to solve real-world Prediction and Control problems. We basically extend Figure \ref{fig:planning} to Figure \ref{fig:planning_learning}. As you can see in Figure \ref{fig:planning_learning}, the change is that there is a downward-pointing arrow from the *Experiences* node to the *Policy* node. This downward-pointing arrow refers to *Model-Free Reinforcement Learning*, i.e., learning the Value Function/Policy directly from experiences obtained by interacting with Environment $E$, i.e., Model-Free RL. This means we obtain the requisite Value Function/Policy through the collaborative approach of *Planning* (using the model $M$) and *Learning* (using Model-Free RL).

\index{reinforcement learning!model-free}
\index{reinforcement learning!model-based}
\index{Markov decision process!environment!real environment}
\index{Markov decision process!environment!simulated environment}
\index{supervised learning}

Note that when Planning is based on RL using experiences obtained by interacting with the Simulated Environment $S$ (based on Model $M$), then we obtain the requisite Value Function/Policy from two sources of experiences (from $E$ and $S$) that are combined and provided to an RL Algorithm. This means we simultaneously do Model-Based RL and Model-Free RL. This is creative and powerful because it blends the best of both worlds—Planning (with Model-Based RL) and Learning (with Model-Free RL). Apart from Model-Free RL and Model-Based RL being blended here to obtain a more accurate Value Function/Policy, the Model is simultaneously being updated with incremental supervised learning (rightward-pointing arrow in Figure \ref{fig:planning_learning}) as new experiences are being generated as a result of the Policy interacting with the Environment $E$ (upward-pointing arrow in Figure \ref{fig:planning_learning}).

This framework of blending Planning and Learning was created by Richard Sutton which he named as [Dyna](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.329.6065&rep=rep1&type=pdf) [@journals/sigart/Sutton91].

\index{reinforcement learning!dyna}

### Decision-Time Planning

\index{planning!decision-time planning|(}

In the next two sections of this chapter, we cover a couple of Planning methods that are sampling-based (experiences obtained by interacting with a sampling model) and use RL techniques to solve for the requisite Value Function/Policy from the model-sampled experiences. We cover the famous Monte-Carlo Tree-Search (MCTS) algorithm, followed by an algorithm which is MCTS' spiritual origin—the Adaptive Multi-Stage Sampling (AMS) algorithm.

Both these algorithms are examples of *Decision-Time Planning*. The term *Decision-Time Planning* requires some explanation. When it comes to Planning (with a model), there are two possibilities:

\index{planning!background planning}

* Background Planning: This refers to a planning method where the AI Agent pre-computes the requisite Value Function/Policy *for all states*, and when it is time for the AI Agent to perform the requisite action for a given state, it simply has to refer to the pre-calculated policy and apply that policy to the given state. Essentially, in the *background*, the AI Agent is constantly improving the requisite Value Function/Policy, irrespective of which state the AI Agent is currently required to act on. Hence, the term *Background Planning*.
* Decision-Time Planning: This approach contrasts with Background Planning. In this approach, when the AI Agent has to identify the best action to take for a specific state that the AI Agent currently encounters, the calculations for that best-action-identification happens only when the AI Agent *reaches that state*. This is appropriate in situations when there are such a large number of states in the state space that Background Planning is infeasible. However, for Decision-Time Planning to be effective, the AI Agent needs to have sufficient time to be able to perform the calculations to identify the action to take *upon reaching a given state*. This is feasible in games like Chess where there is indeed some time for the AI Agent to make its move upon encountering a specific state of the chessboard (the move response doesn't need to be immediate). However, this is not feasible for a self-driving car, where the decision to accelerate/brake or to steer must be immediate (this requires *Background Planning*).

Hence, with Decision-Time Planning, the AI Agent focuses all of the available computation and memory resources for the sole purpose of identifying the best action for *a particular state* (the state that has just been reached by the AI Agent). Decision-Time Planning is typically successful because of this focus on a single state and consequently, on the states that are most likely to be reached within the next few time steps (essentially, avoiding any wasteful computation on states that are unlikely to be reached from the given state).

Decision-Time Planning typically looks much deeper than just a single time step ahead (DP algorithms only look a single time step ahead) and evaluates action choices leading to many different state and reward possibilities over the next several time steps. Searching deeper than a single time step ahead is required because these Decision-Time Planning algorithms typically work with imperfect Q-Values. 

\index{planning!heuristic search}

Decision-Time Planning methods sometimes go by the name *Heuristic Search*. Heuristic Search refers to the method of growing out a tree of future states/actions/rewards from the given state (which serves as the root of the tree). In classical Heuristic Search, an approximate Value Function is calculated at the leaves of the tree and the Value Function is then backed up to the root of the tree. Knowing the backed-up Q-Values at the root of the tree enables the calculation of the best action for the root state. Modern methods of Heuristic Search are very efficient in how the Value Function is approximated and backed up. Monte-Carlo Tree-Search (MCTS) in one such efficient method that we cover in the next section.

\index{planning!decision-time planning|)}

### Monte-Carlo Tree-Search (MCTS)

\index{reinforcement learning!monte carlo tree search|(}

[Monte-Carlo Tree-Search (abbreviated as MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) is a Heuristic Search method that involves growing out a Search Tree from the state for which we seek the best action (hence, it is a Decision-Time Planning algorithm). MCTS was popularized in 2016 by [Deep Mind's AlphaGo algorithm](https://www.nature.com/articles/nature16961}) [@silver2016mastering]. MCTS was first introduced by [Remi Coulom for game trees](https://hal.inria.fr/inria-00116992/document) [@conf/cg/Coulom06].

For every state in the Search Tree, we maintain the Q-Values for all actions from that state. The basic idea is to form several sampling traces from the root of the tree (i.e., from the given state) to terminal states. Each such sampling trace threads through the Search Tree to a leaf node of the tree, and then extends beyond the tree from the leaf node to a terminal state. This separation of the two pieces of each sampling trace is important—the first piece within the tree, and the second piece outside the tree. Of particular importance is the fact that the first piece of various sampling traces will pass through states (within the Search Tree) that are quite likely to be reached from the given state (at the root node). MCTS benefits from states in the tree being revisited several times, as it enables more accurate Q-Values for those states (and consequently, a more accurate Q-Value at the root of the tree, from backing-up of Q-Values). Moreover, these sampling traces prioritize actions with good Q-Values. Prioritizing actions with good Q-Values has to be balanced against actions that haven't been tried sufficiently, and this is essentially the explore-exploit tradeoff that we covered in detail in Chapter [-@sec:multi-armed-bandits-chapter].

Each sampling trace round of MCTS consists of four steps:

* Selection: Starting from the root node $R$ (given state), we successively select children nodes all the way until a leaf node $L$. This involves selecting actions based on a *tree policy*, and selecting next states by sampling from the model of state transitions. The trees in Figure \ref{fig:mcts} show states colored as white and actions colored as gray. This figure shows the Q-Values for a 2-player game (e.g., Chess) where the reward is 1 at termination for a win, 0 at termination for a loss, and 0 throughout the time the game is in play. So the Q-Values in the figure are displayed at each node in the form of Wins as a fractions of Games Played that passed through the node (Games through a node means the number of sampling traces that have run through the node). So the label "1/6" for one of the State nodes (under "Selection", the first image in the figure) means that we've had 6 sampling traces from the root node that have passed through this State node labeled "1/6", and 1 of those games was won by us. For Actions nodes (gray nodes), the labels correspond to *Opponent Wins* as a fraction of Games through the Action node. So the label "2/3" for one of the Action leaf nodes means that we've had 3 sampling traces from the root node that have passed through this Action leaf node, and 2 of those resulted in wins *for the opponent* (i.e., 1 win for us).
* Expansion: On some rounds, the tree is expanded from $L$ by adding a child node $C$ to it. In the figure, we see that $L$ is the Action leaf node labeled as "3/3" and we add a child node $C$ (state) to it labeled "0/0" (because we don't yet have any sampling traces running through this added state $C$).
* Simulation: From $L$ (or from $C$ if this round involved adding $C$), we complete the sampling trace (that started from $R$ and ran through $L$) all the way to a terminal state $T$. This entire sampling trace from $R$ to $T$ is known as a single Monte-Carlo Simulation Trace, in which actions are selected according to the *tree policy* when within the tree, and according to a *rollout policy* beyond the tree (the term "rollout" refers to "rolling out" a simulation from the leaf node to termination). The tree policy is based on an Explore-Exploit tradeoff using estimated Q-Values, and the rollout policy is typically a simple policy such as a uniform policy.
* Backpropagation: The return generated by the sampling trace is backed up ("backpropagated") to update (or initialize) the Q-Values at the nodes in the tree that are part of the sampling trace. Note that in the figure, the rolled out simulation trace resulted in a win for the opponent (loss for us). So the backed up Q-Values reflect an extra win for the opponent (on the gray nodes, i.e., action nodes) and an extra loss for us (on the white nodes, i.e., state nodes).

![Monte-Carlo Tree-Search (This wikipedia image is being used under the creative commons license CC BY-SA 4.0) \label{fig:mcts}](./chapter15/mcts.png "Monte-Carlo Tree-Search (This wikipedia image is being used under the creative commons license CC BY-SA 4.0)")

The Selection Step in MCTS involves picking a child node (action) with "most promise", for each state in the sampling trace of the Selection Step. This means prioritizing actions with higher Q-Value estimates. However, this needs to be balanced against actions that haven't been tried sufficiently (i.e., those actions whose Q-Value estimates have considerable uncertainty). This is our usual *Explore v/s Exploit* tradeoff that we covered in detail in Chapter [-@sec:multi-armed-bandits-chapter]. The Explore v/s Exploit formula for games was [first provided by Kocsis and Szepesvari](http://ggp.stanford.edu/readings/uct.pdf) [@kocsis2006a]. This formula is known as *Upper Confidence Bound 1 for Trees* (abbreviated as UCT). Most current MCTS Algorithms are based on some variant of UCT. UCT is based on the [UCB1 formula of Auer, Cesa-Bianchi, Fischer](https://homes.di.unimi.it/cesa-bianchi/Pubblicazioni/ml-02.pdf) [@Auer2002].

\index{reinforcement learning!monte carlo tree search|)}

### Adaptive Multi-Stage Sampling

\index{reinforcement learning!adaptive multi-stage sampling|(}

Its not well known that MCTS and UCT concepts first appeared in the [Adaptive Multi-Stage Sampling algorithm by Chang, Fu, Hu, Marcus](https://pdfs.semanticscholar.org/a378/b2895a3e3f6a19cdff1a0ad404b301b5545f.pdf) [@journals/ior/ChangFHM05]. Adaptive Multi-Stage Sampling (abbreviated as AMS) is a generic sampling-based algorithm to solve finite-horizon Markov Decision Processes (although the paper describes how to extend this algorithm for infinite-horizon MDPs). We consider AMS to be the "spiritual origin" of MCTS/UCT, and hence we dedicate this section to coverage of AMS.

AMS is a planning algorithm—a sampling model is provided for the next state (conditional on given state and action), and a model of Expected Reward (conditional on given state and action) is also provided. AMS overcomes the curse of dimensionality by sampling the next state. The key idea in AMS is to adaptively select actions based on a suitable tradeoff between Exploration and Exploitation. AMS was the first algorithm to apply the theory of Multi-Armed Bandits to derive a provably convergent algorithm for solving finite-horizon MDPs. Moreover, it performs far better than the typical backward-induction approach to solving finite-horizon MDPs, in cases where the state space is very large and the action space is fairly small.

We use the same notation we used in Section [-@sec:finite-horizon-section] of Chapter [-@sec:dp-chapter] for Finite-Horizon MDPs (time steps $t = 0, 1, \ldots T$). We assume that the state space $\mathcal{S}_t$ for time step $t$ is very large for all $t = 0, 1, \ldots, T-1$ (the state space $\mathcal{S}_T$ for time step $T$ consists of all terminal states). We assume that the action space $\mathcal{A}_t$ for time step $t$ is fairly small for all $t = 0, 1, \ldots, T-1$. We denote the probability distribution for the next state, conditional on the current state and action (for time step $t$) as the function $\mathcal{P}_t: (\mathcal{S}_t \times \mathcal{A}_t) \rightarrow (\mathcal{S}_{t+1} \rightarrow [0, 1])$, defined as:
$$\mathcal{P}_t(s_t, a_t)(s_{t+1}) = \mathbb{P}[S_{t+1}=s_{t+1}|(S_t=s_t, A_t=a_t)]$$
As mentioned above, for all $t = 0, 1, \ldots, T-1$, AMS has access to only a sampling model of $\mathcal{P}_t$, that can be used to fetch a sample of the next state from $\mathcal{S}_{t+1}$. We also assume that we are given the Expected Reward function $\mathcal{R}_t: \mathcal{S}_t \times \mathcal{A}_t \rightarrow \mathbb{R}$ for each time step $t = 0, 1, \ldots, T-1$ defined as:
$$\mathcal{R}_t(s_t, a_t) = \mathbb{E}[R_{t+1}|(S_t=s_t, A_t=a_t)]$$
We denote the Discount Factor as $\gamma$.

The problem is to calculate an approximation to the Optimal Value Function $V_t^*(s_t)$ for all $s_t \in \mathcal{S}_t$ for all $t = 0, 1, \ldots, T-1$. Using only samples from the state-transition probability distribution functions $\mathcal{P}_t$ and the Expected Reward functions $\mathcal{R}_t$, AMS aims to do better than backward induction for the case where $\mathcal{S}_t$ is very large and $\mathcal{A}_t$ is small for all $t= 0, 1, \ldots T-1$.

The AMS algorithm is based on a fixed allocation of the number of action selections for each state at each time step. Denote the number of action selections for each state at time step $t$ as $N_t$. We ensure that each action $a_t \in \mathcal{A}_t$ is selected at least once, hence $N_t \geq |\mathcal{A}_t|$. While the algorithm is running, we denote $N_t^{s_t,a_t}$ to be the number of selections of a particular action $a_t$ (for a given state $s_t$) *until that point in the algorithm*.

Denote $\hat{V}_t^{N_t}(s_t)$ as the AMS Algorithm's approximation of $V_t^*(s_t)$, utilizing all of the $N_t$ action selections. For a given state $s_t$, for each selection of an action $a_t$, *one* next state is sampled from the probability distribution $\mathcal{P}_t(s_t, a_t)$ (over the state space $\mathcal{S}_{t+1}$). For a fixed $s_t$ and fixed $a_t$, let us denote the $j$-th sample of the next state (for $j = 1, \ldots, N_t^{s_t,a_t}$) as $s_{t+1}^{(s_t,a_t,j)}$. Each such next state sample $s_{t+1}^{(s_t,a_t,j)} \sim \mathcal{P}_t(s_t,a_t)$ leads to a recursive call to $\hat{V}_{t+1}^{N_{t+1}}(s_{t+1}^{(s_t,a_t,j)})$ in order to calculate the approximation $\hat{Q}_t(s_t,a_t)$ of the Optimal Action Value Function $Q_t^*(s_t, a_t)$ as:
$$\hat{Q}_t(s_t,a_t) = \mathcal{R}_t(s_t,a_t) + \gamma \cdot \frac {\sum_{j=1}^{N_t^{s_t,a_t}} \hat{V}_{t+1}^{N_{t+1}}(s_{t+1}^{(s_t,a_t,j)})} {N_t^{s_t,a_t}}$$

Now let us understand how the $N_t$ action selections are done for a given state $s_t$.  First, we select each of the actions in $\mathcal{A}_t$ exactly once. This is a total of $|\mathcal{A}_t|$ action selections. Each of the remaining $N_t - |\mathcal{A}_t|$ action selections (indexed as $i$ ranging from $|\mathcal{A}_t|$ to $N_t - 1$) is made based on the action that maximizes the following UCT formula (thus balancing exploration and exploitation):
\begin{equation}
\hat{Q}_t(s_t,a_t) + \sqrt{\frac {2 \log{i}} {N_t^{s_t,a_t}}} \label{eq:ams_uct_formula}
\end{equation}

When all $N_t$ action selections are made for a given state $s_t$, $V_t^*(s_t) = \max_{a_t \in \mathcal{A}_t} Q_t^*(s_t,a_t)$ is approximated as:
\begin{equation}
\hat{V}_t^{N_t}(s_t) = \sum_{a_t \in \mathcal{A}_t} \frac {N_t^{s_t,a_t}} {N_t} \cdot \hat{Q}_t(s_t,a_t) \label{eq:ams_optimal_vf_approx}
\end{equation}

Now let's write a Python class to implement AMS. We start by writing its constructor. For convenience, we assume each of the state spaces $\mathcal{S}_t$ (for $t = 0, 1, \ldots, T$) is the same (denoted as $\mathcal{S}$) and the allowable actions are the same across all time steps (denoted as $\mathcal{A})$. 

\index{AMS@\texttt{AMS}}

```python
from rl.distribution import Distribution

A = TypeVar('A')
S = TypeVar('S')

class AMS(Generic[S, A]):

    def __init__(
        self,
        actions_funcs: Sequence[Callable[[S], Set[A]]],
        state_distr_funcs: Sequence[Callable[[S, A], Distribution[S]]],
        expected_reward_funcs: Sequence[Callable[[S, A], float]],
        num_samples: Sequence[int],
        gamma: float
    ) -> None:
        self.num_steps: int = len(actions_funcs)
        self.actions_funcs: Sequence[Callable[[S], Set[A]]] = \
            actions_funcs
        self.state_distr_funcs: Sequence[Callable[[S, A], Distribution[S]]] = \
            state_distr_funcs
        self.expected_reward_funcs: Sequence[Callable[[S, A], float]] = \
            expected_reward_funcs
        self.num_samples: Sequence[int] = num_samples
        self.gamma: float = gamma
```

Let us understand the inputs to the constructor `__init__`.   

* `actions_funcs` consists of a `Sequence` (for all of $t = 0, 1, \ldots, T-1$) of functions, each mapping a state in $\mathcal{S}_t$ to a set of actions within $\mathcal{A}$, that we denote as $\mathcal{A}_t(s_t)$ (i.e., `Callable[[S], Set[A]]`).
* `state_distr_funcs` represents $\mathcal{P}_t$ for all $t = 0, 1, \ldots, T-1$ (accessible only as a sampling model since the return type of each `Callable` is `Distribution`).
* `expected_reward_funcs` represents $\mathcal{R}_t$ for all $t = 0, 1, \ldots, T-1$.
* `num_samples` represents $N_t$ (the number of actions selections for each state $s_t$) for all $t = 0, 1, \ldots, T-1$.
* `gamma` represents the discount factor $\gamma$.

`self.num_steps` represents the number of time steps $T$.

Next, we write the method `optimal_vf_and_policy` to compute $\hat{V}_t^{N_t}(s_t)$ and the associated recommended action for state $s_t$ (note the type of the output, representing this pair as `Tuple[float, A]`).

In the code below, `vals_sum` builds up the sum $\sum_{j=1}^{N_t^{s_t,a_t}} \hat{V}_{t+1}^{N_{t+1}}(s_{t+1}^{(s_t,a_t,j)})$, and `counts` represents $N_t^{s_t,a_t}$. Before the `for` loop, we initialize `vals_sum` by selecting each action $a_t \in \mathcal{A}_t(s_t)$ exactly once. Then, for each iteration $i$ of the `for` loop (for $i$ ranging from $|\mathcal{A}_t(s_t)|$ to $N_t - 1$), we calculate the Upper-Confidence Value (`ucb_vals` in the code below) for each of the actions $a_t \in \mathcal{A}_t(s_t)$ using the UCT formula of Equation \eqref{eq:ams_uct_formula}, and pick an action $a_t^*$ that maximizes `ucb_vals`. After the termination of the `for` loop, `optimal_vf_and_policy` returns the Optimal Value Function approximation for $s_t$ based on Equation \eqref{eq:ams_optimal_vf_approx} and the recommended action for $s_t$ as the action that maximizes $\hat{Q}_t(s_t, a_t)$


```python
import numpy as np
from operator import itemgetter

    def optimal_vf_and_policy(self, t: int, s: S) -> \
            Tuple[float, A]:
        actions: Set[A] = self.actions_funcs[t](s)
        state_distr_func: Callable[[S, A], Distribution[S]] = \
            self.state_distr_funcs[t]
        expected_reward_func: Callable[[S, A], float] = \
            self.expected_reward_funcs[t]
        rewards: Mapping[A, float] = {a: expected_reward_func(s, a)
                                      for a in actions}
        val_sums: Dict[A, float] = {a: (self.optimal_vf_and_policy(
            t + 1,
            state_distr_func(s, a).sample()
        )[0] if t < self.num_steps - 1 else 0.) for a in actions}
        counts: Dict[A, int] = {a: 1 for a in actions}
        for i in range(len(actions), self.num_samples[t]):
            ucb_vals: Mapping[A, float] = \
                {a: rewards[a] + self.gamma * val_sums[a] / counts[a] +
                 np.sqrt(2 * np.log(i) / counts[a]) for a in actions}
            max_actions: Sequence[A] = [a for a, u in ucb_vals.items()
                                        if u == max(ucb_vals.values())]
            a_star: A = np.random.default_rng().choice(max_actions)
            val_sums[a_star] += (self.optimal_vf_and_policy(
                t + 1,
                state_distr_func(s, a_star).sample()
            )[0] if t < self.num_steps - 1 else 0.)
            counts[a_star] += 1

        return (
            sum(counts[a] / self.num_samples[t] *
                (rewards[a] + self.gamma * val_sums[a] / counts[a])
                for a in actions),
            max(
                [(a, rewards[a] + self.gamma * val_sums[a] / counts[a])
                 for a in actions],
                key=itemgetter(1)
            )[0]
        )
```

The above code is in the file [rl/chapter15/ams.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter15/ams.py). The `__main__` in this file tests the AMS algorithm for the simple case of the Dynamic Pricing problem that we had covered in Section [-@sec:dynamic-pricing-section] of Chapter [-@sec:dp-chapter], although the Dynamic Pricing problem itself is not a problem where AMS would do better than backward induction (since its state space is not very large). We encourage you to play with our implementation of AMS by constructing a finite-horizon MDP with a large state space (and small-enough action space). An example of such a problem is Optimal Stopping (in particular, pricing of American Options) that we had covered in Chapter [-@sec:derivatives-pricing-chapter].

Now let's analyze the running-time complexity of AMS. Let $N = \max{(N_0, N_1, \ldots, N_{T-1})}$. At each time step $t$, the algorithm makes at most $N$ recursive calls, and so the running-time complexity is $O(N^T)$. Note that since we need to select every action at least once for every state at every time step, $N \geq |\mathcal{A}|$, meaning the running-time complexity is at least $|\mathcal{A}|^T$. Compare this against the running-time complexity of backward induction, which is $O(|\mathcal{S}|^2 \cdot |\mathcal{A}| \cdot T)$. So, AMS is more efficient when $\mathcal{S}$ is very large (which is typical in many real-world problems). In their paper, Chang, Fu, Hu, Marcus proved that the Value Function approximation $\hat{V}_0^{N_0}$ is asymptotically unbiased, i.e., 
$$\lim_{N_0\rightarrow \infty} \lim_{N_1\rightarrow \infty} \ldots \lim_{N_{T-1}\rightarrow \infty} \mathbb{E}[\hat{V}_0^{N_0}(s_0)] = V_0^*(s_0) \mbox{ for all } s_0 \in \mathcal{S}$$
They also proved that the worst-possible bias is bounded by a quantity that converges to zero at the rate of $O(\sum_{t=0}^{T-1} \frac {\ln N_t} {N_t})$. Specifically,
$$0 \leq V_0^*(s_0) - \mathbb{E}[\hat{V}_0^{N_0}(s_0)] \leq O(\sum_{t=0}^{T-1} \frac {\ln N_t} {N_t}) \mbox{ for all } s_0 \in \mathcal{S}$$

\index{reinforcement learning!adaptive multi-stage sampling|)}
\index{planning|)}
\index{learning|)}

### Summary of Key Learnings from This Chapter

* Planning versus Learning, and how to blend Planning and Learning.
* Monte-Carlo Tree-Search (MCTS): An example of a Planning algorithm based on Tree-Search and based on sampling/RL techniques.
* Adaptive Multi-Stage Sampling (AMS): The spiritual origin of MCTS—it is an efficient algorithm for finite-horizon MDPs with very large state space and fairly small action space.
