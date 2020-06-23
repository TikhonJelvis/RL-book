# Chapter 3: Markov Decision Processes

We've said before that this book is about "sequential decisioning" under "sequential uncertainty". In the previous chapter, we covered the "sequential uncertainty" aspect with the framework of Markov Processes, and we extended the framework to also incoporate the notion of uncertain "Reward" each time we make a state transition - we called this extended framework Markov Reward Processes. However, this framework had no notion of "sequential decisioning". In this chapter, we will further extend the framework of Markov Reward Processes to incorporate the notion of "sequential decisioning", formally known as Markov Decision Processes. Before we step into the formalism of Markov Decision Processes, let us develop some intuition and motivation for the need to have such a framework - to handle sequential decisioning. Let's do this by re-visiting the simple inventory example we covered in the previous chapter.

## Simple Inventory Example: How much to Order?

When we covered the simple inventory example in the previous chapter as a Markov Reward Process, the ordering policy was:

$$\theta = \max(C - (\alpha + \beta), 0)$$

where $\theta \in \mathbb{Z}_{\geq 0}$ is the order quantity, $C \in \mathbb{Z}_{\geq 0}$ is the space capacity (in bicycle units) at the store, $\alpha$ is the On-Hand Inventory and $\beta$ is the On-Order Inventory ($(\alpha, \beta)$ comprising the *State*). We calculated the Value Function for the Markov Reward Process that results from following this policy. Now we ask the question: Is this Value Function good enough? More importantly, we ask the question: Can we improve this Value Function by following a different ordering policy?  Perhaps by ordering less than that implied by the above formula for $\theta$? This leads to the natural question - Can we identify the ordering policy that yields the *Optimal* Value Function (one with the highest expected returns, i.e., lowest expected accumulated costs, from each state)? Let us get an intuitive sense for this optimization problem by considering a concrete example.

Assume that instead of bicycles, we want to control the inventory of a specific type of toothpaste in the store. Assume we have space for 20 units of toothpaste on the shelf assigned to the toothpaste (assume there is no space in the backroom of the store). Asssume that customer demand follows a Poisson distribution with Poisson parameter $\lambda = 3.0$. At 6pm store-closing each evening, when you observe the *State* as $(\alpha, \beta)$, we now have a choice of ordering a quantity of toothpastes from any of the following values for the order quantity $\theta: \{0, 1, \ldots, \max(20 - (\alpha + \beta), 0)\}$. Let's say at Monday 6pm store-closing, $\alpha = 4$ and $\beta = 3$. So, you have a choice of order quantities from among the integers in the range of 0 to (20 - (4 + 3) = 13) (i.e., 14 choices). Previously, in the Markov Reward Process model, we would have ordered 13 units on Monday store-closing. This means on Wednesday morning at 6am, a truck would have arrived with 13 units of the toothpaste. If you sold say 2 units of the toothpaste on Tuesday, then on Wednesday 8am at store-opening, you'd have 4 + 3 - 2 + 13 = 18 units of toothpaste on your shelf. If you keep following this policy, you'd typically have almost a full shelf at store-opening each day, which covers almost a week worth of expected demand for the toothpaste. This means our risk of going out-of-stock on the toothpaste is extremely low, but we'd be incurring considerable holding cost (we'd have close to a full shelf of toothpastes sitting around almost each night). So as a store manager, you'd be thinking - "I can lower my costs by ordering less than that prescribed by the formula of $20 - (\alpha + \beta)$". But how much less? If you order too little, you'd start the day with too little inventory and might risk going out-of-stock. That's a risk you are highly uncomfortable with since the stockout cost per unit of missed demand (we called it $p$) is typically much higher than the holding cost per unit (we called it $h$). So you'd rather "err" on the side of having more inventory. But how much more? We also need to factor in the fact that the 36-hour lead time means a large order incurs large holding costs *two days later*. Most importantly, to find this right balance in terms of a precise mathematical optimization of the Value Function, we'd have to factor in the uncertainty of demand (based on daily Poisson probabilities) in our calculations. Now this gives you a flavor of the problem of sequential decisioning (each day you have to decide how much to order) under sequential uncertainty.

To deal with the "decisioning" aspect, we will introduce the notion of *Action* to complement the previously introduced notions of *State* and *Reward*. In the inventory example, the order quantity is our *Action*. After observing the *State*, we choose from among a set of Actions (in this case, we choose from within the set $\{0, 1, \ldots, \max(C - (\alpha + \beta), 0)\}$). We note that the Action we take upon observing a state affects the next day's state. This is because the next day's On-Order is exactly equal to today's order quantity (i.e., today's action). This in turn might affect our next day's action since the action (order quantity) is typically a function of the state (On-Hand and On-Order inventory). Also note that the Action we take on a given day will influence the Rewards after a couple of days (i.e. after the order arrives). It may affect our holding cost adversely if we had ordered too much or it may affect our stockout cost adversely if we had ordered too little and then experienced high demand.

## The Difficulty of Sequential Decisioning under Uncertainty

This simple inventory example has given us a peek into the world of Markov Decision Processes, which in general, have two distinct (and inter-dependent) high-level features:

* At each time step $t$, an *Action* $A_t$ in picked (from among a specified choice of actions) upon observing the *State* $S_t$
* Given an observed *State* $S_t$ and a performed *Action* $A_t$, the probabilities of the state and reward of the next time step ($S_{t+1}$ and $R_{t+1}$) are in general a function of not just the state $S_t$ but also of the action $A_t$.

We are tasked with maximizing the *Expected Return* from each state (i.e., maximizing the Value Function). This seems like a pretty hard problem in the general case because there is a cyclic interplay between:

* actions depending on state, on one hand, and

* next state/reward probabilities depending on action (and state) on the other hand.

There is also the challenge that actions might have delayed consequences on rewards, and it's not clear how to disentangle the effects of actions from different time steps on a future reward. So without direct correspondence between actions and rewards, how can we control the actions so as to maximize expected accumulated rewards? To answer this question, we will need to set up some notation and theory. Before we formally define the Markov Decision Process framework and it's associated (elegant) theory, let us set up a bit of terminology.

Using the language of AI, we say that at each time step $t$, the *Agent* (the algorithm we design) observes the state $S_t$, after which the Agent performs action $A_t$, after which the *Environment* (upon seeing $S_t$ and $A_t$) produces a random pair: the next state state $S_{t+1}$ and the next reward $R_{t+1}$, after which the *Agent* oberves this next state $S_{t+1}$, and the cycle repeats. This cyclic interplay is depicted in Figure \ref{fig:mdp_cycle}. Note that time ticks over from $t$ to $t+1$ when the environment sees the state $S_t$ and action $A_t$.

![Markov Decision Process \label{fig:mdp_cycle}](./chapter3/mdp.png "Agent-Environment interaction in a Markov Decision Process")

## Formal Definition of a Markov Decision Process

Similar to the definitions of Markov Processes and Markov Reward Processes, for ease of exposition, the definitions and theory of Markov Decision  Processes below will be for discrete-time, for countable state spaces and countable set of pairs of next state and reward transitions (with the knowledge that the definitions and theory are analogously extensible to continuous-time and uncountable spaces, which we shall indeed encounter later in this book). 

\begin{definition}

 A {\em Markov Decision Process} comprises of:

 \begin{itemize}

\item A countable set of states $\mathcal{S}$ and a countable set of actions $\mathcal{A}$

 \item A time-indexed sequence of environment-generated random states $S_t$ for each time $t=0, 1, 2, \ldots$, with each $S_t \in \mathcal{S}$

 \item A time-indexed sequence of environment-generated {\em Reward} random variables $R_t \in \mathbb{R}$ for each time $t=1, 2, \ldots$

\item A time-indexed sequence of agent-controllable actions $A_t$ for each time $t=0, 1, 2, \ldots$, with each $A_t \in \mathcal{A}$. (Sometimes we restrict the set of actions allowable from specific states, in which case, we abuse the $\mathcal{A}$ notation to refer to a function whose domain is $\mathcal{S}$ and range is $\mathcal{A}$, and we say that the set of actions allowable from a state $s\in \mathcal{S}$ is $\mathcal{A}(s)$.)

 \item Markov Property: $\mathbb{P}[(R_{t+1}, S_{t+1}) | (S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0)] = \mathbb{P}[(R_{t+1}, S_{t+1}) | (S_t, A_t)]$ for all $t \geq 0$

 \item Specification of a discount factor $\gamma \in [0,1]$

 \end{itemize}

 \end{definition}

Like in the case of Markov Reward Processes, the role of $\gamma$ only comes in discounting future rewards when accumulating rewards from a given state - more on this later.

Like in the case of Markov Processes and Markov Reward Processes, we shall (by default) assume Stationarity for Markov Decision Processes, i.e., $\mathbb{P}[(R_{t+1}, S_{t+1}) | (S_t, A_t)]$ is independent of $t$. This means the transition probabilities of a Markov Decision Process can, in the most general case, be expressed as a state-reward transition probability function:

$$\mathcal{P}_R: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \times \mathcal{S} \rightarrow [0,1]$$

defined as:

$$\mathcal{P}_R(s,a,r,s') = \mathbb{P}[(R_{t+1}=r, S_{t+1}=s') |(S_t=s, A_t=a)]$$ such that $$\sum_{s'\in \mathcal{S}} \sum_{r \in \mathbb{R}} \mathcal{P}_R(s,a,r,s') = 1 \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}$$

Henceforth, any time we say Markov Decision Process, assume we are refering to a Discrete-Time Stationary Markov Decision Process with countable spaces and countable transitions (unless explicitly specified otherwise), which in turn can be characterized by the state-reward transition probability function $\mathcal{P}_R$. Given a specification of $\mathcal{P}_R$, we can construct:

* The state transition probability function

$$\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$$

defined as:

$$\mathcal{P}(s, a, s') = \sum_{r\in \mathbb{R}} \mathcal{P}_R(s,a, r,s')$$

* The reward transition function:

$$\mathcal{R}_T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$$

defined as:

$$\mathcal{R}_T(s,a,s') = \mathbb{E}[R_{t+1}|(S_{t+1}=s',S_t=s,A_t=a)]$$

$$ = \sum_{r\in \mathcal{R}} \frac {\mathcal{P}_R(s,a,r,s')} {\mathcal{P}(s,a,s')} \cdot r = \sum_{r\in \mathcal{R}} \frac {\mathcal{P}_R(s,a,r,s')} {\sum_{r\in \mathbb{R}} \mathcal{P}_R(s,a,r,s')} \cdot r$$

The Rewards specification of most Markov Decision Processes we encounter in practice can be directly expressed as the reward transition function $\mathcal{R}_T$ (versus the more general specification of $\mathcal{P}_R$). Lastly, we want to highlight that we can transform either of $\mathcal{P}_R$ or $\mathcal{R}_T$ into a "more compact" reward function that is sufficient to perform key calculations involving Markov Decision Processes. This reward function

$$\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

is defined as:

$$\mathcal{R}(s,a) = \mathbb{E}[R_{t+1}|(S_t=s, A_t=a)]$$

$$= \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \mathcal{R}_T(s,a,s') = \sum_{s'\in \mathcal{S}} \sum_{r\in\mathbb{R}} \mathcal{P}_R(s,a,r,s') \cdot r$$

## Policy

Having understood the dynamics of a Markov Decision Process, we now move on to the specification of the *Agent*'s actions as a function of the current state. In the general case, we assume that the Agent will perform a random action $A_t$, according to a probability distribution that is a function of the current state $S_t$. We refer to this function as a *Policy*. Formally, a *Policy* is a function

$$\pi: \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$$

defined as:

$$\pi(a, s) = \mathbb{P}[A_t = a|S_t = s] \text{ for all } t = 0, 1, 2, \ldots, \text{ for all } s\in \mathcal{S}, a \in \mathcal{A}$$

Note that in the definition above, we've assumed that a Policy is stationary, i.e., $\mathbb{P}[A_t = a|S_t = s]$ is invariant in time $t$. If we do encounter a situation where the policy would need to depend on the time $t$, we'll simply include $t$ to be part of the state, which would make the Policy stationary (albeit at the cost of state-space bloat and hence, computational cost).

When we have a policy such that the action probability distribution for each state is concentrated on a single action, we refer to it as a deterministic policy. Formally, a deterministic policy has the property that for all $s\in \mathcal{S}$,

$$\pi(\pi_D(s), s) = 1 \text{ and } \pi(a, s) = 0 \text{ for all } a\in \mathcal{A} \text{ with } a \neq \pi_D(s)$$

where $\pi_D: \mathcal{S} \rightarrow \mathcal{A}$.

So we shall denote deterministic policies simply with the function $\pi_D$. We shall refer to non-deterministic policies as stochastic policies (the word stochastic reflecting the fact that the agent will perform a random action according to the probability distribution specified by $\pi$). So when we use the notation $\pi$, assume that we are dealing with a stochastic (i.e., non-deterministic) policy and when we use the notation $\pi_D$, assume that we are dealing with a deterministic policy.

Let's write some code to get a grip on the concept of Policy  - we start with the design of an abstract class called `Policy` that represents a general Policy, as we have articulated above. The only method it contains is an `@abstractmethod act` that accepts as input a `state: S` (as seen before in the classes `MarkovProcess` and `MarkovRewardProcess`, `S` is a generic type to represent a generic state) and produces as output an object of type `Distribution[A]` that represents the probability distribution of the random action as a function of the input `state`.

```python
A = TypeVar('A')
S = TypeVar('S')

class Policy(ABC, Generic[S, A]):

    @abstractmethod
    def act(self, state: S) -> Distribution[A]:
        pass
```

Now let's write some code to create some concrete policies for an example we are familiar with - the simple inventory example. We first create a concrete class `SimpleInventoryDeterministicPolicy` for deterministic inventory replenishment policies that implements the interface of the abstract class `Policy` (specifically implements the `@abstractmethod act`). Note that the generic state `S` is replaced here with the class `InventoryState` that represents a state in the inventory example, comprising of the On-Hand and On-Order inventory quantities. Also note that the generic action `A` is replaced here with the `int` type since in this example, the action is the quantity of inventory to be ordered at store-closing (which is an integer quantity). Note that since our class is meant to produce a deterministic policy, the `act` method returns a `Constant[int]` which is a probability distribution with 100% of the probability concentrated at a single `int` value (`int` represents the integer quantity of inventory to be ordered). The code in `act` implements the following deterministic policy:

$$\pi_D((\alpha, \beta)) = \max(C - (\alpha + \beta), 0)$$ where $C$ is a parameter representing the "reorder point" (meaning, we order only when the inventory position falls below the "reorder point"), $\alpha$ is the On-Hand Inventory at store-closing, $\beta$ is the On-Order Inventory at store-closing, and inventory position is equal to $\alpha + \beta$. In the previous chapter, we set the reorder point to be equal to the store capacity $C$.

```python
@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order

class SimpleInventoryDeterministicPolicy(Policy[InventoryState, int]):
    def __init__(self, reorder_point: int):
        self.reorder_point: int = reorder_point

    def act(self, state: InventoryState) -> Constant[int]:
        return Constant(max(self.reorder_point - state.inventory_position(),
                            0))
```

We can instantiate a specific deterministic policy with a reorder point of say 10 as:   

```python
si_dp = SimpleInventoryDeterministicPolicy(reorder_point=10)
```

Now let's write some code to create stochastic policies for the inventory example. Similar to `SimpleInventoryDeterministicPolicy`, we create a concrete class `SimpleInventoryStochasticPolicy` that implements the interface of the abstract class `Policy` (specifically implements the `@abstractmethod act`). The code in `act` implements the following stochastic policy:

$$\pi((\alpha, \beta), \theta) = \frac {e^{-\lambda} \lambda^{r}} {r!}$$
$$\theta = \max(r - (\alpha + \beta), 0)$$
where $r \in \mathbb{Z}_{\geq 0}$ is the random re-order point with poisson probability distribution given by a specified poisson mean parameter $\lambda \in \mathbb{R}_{\geq 0}$, and $\theta \in \mathbb{Z}_{\geq 0}$ is the order quantity (action).

```python
class SimpleInventoryStochasticPolicy(Policy[InventoryState, int]):
    def __init__(self, reorder_point_poisson_mean: float):
        self.reorder_point_poisson_mean: float = reorder_point_poisson_mean

    def act(self, state: InventoryState) -> SampledDistribution[int]:
        def action_func(state=state) -> int:
            reorder_point_sample: int = \
                np.random.poisson(self.reorder_point_poisson_mean)
            return max(reorder_point_sample - state.inventory_position(), 0)

        return SampledDistribution(action_func)
```

We can instantiate a specific stochastic policy with a reorder point poisson distribution mean of say 5.2 as:   

```python
si_sp = SimpleInventoryStochasticPolicy(reorder_point_poisson_mean=5.2)
```

We will revisit the simple inventory example in a bit after we cover the code for Markov Decision Processes, when we'll show how to simulate the Markov Decision Process for this simple inventory example, with the agent running a deterministic policy. But before we move on to the code design for Markov Decision Processes (to accompany the above implementation of Policies), we require an important insight linking Markov Decision Processes, Policies and Markov Reward Processes.

## [Markov Decision Process, Policy] := Markov Reward Process

This section has an important insight - that if we evaluate a Markov Decision Process (MDP) with a fixed policy $\pi$ (in general, with a fixed stochastic policy $\pi$), we get the Markov Reward Process (MRP) that is *implied* by the combination of the MDP and the policy $\pi$. Let's clarify this with notational precision. But first we need to point out that we have some notation clashes between MDP and MRP. We used $\mathcal{P}_R$ to denote the transition probability function of the MRP as well as to denote the state-reward transition probability function of the MDP. We used $\mathcal{P}$ to denote the transition probability function of the Markov Process implicit in the MRP as well as to denote the state transition probability function of the MDP. We used $\mathcal{R}_T$ to denote the reward transition function of the MRP as well as to denote the reward transition function of the MDP. We used $\mathcal{R}$ to denote the reward function of the MRP as well as to denote the reward function of the MDP. We can resolve these notation clashes by noting the arguments to $\mathcal{P}_R$, $\mathcal{P}, \mathcal{R}_T$ and $\mathcal{R}$, but to be extra-clear, we'll put a superscript of $\pi \rightarrow MRP$ to each of the functions $\mathcal{P}_R$, $\mathcal{P}, \mathcal{R}_T$ and $\mathcal{R}$ for the $\pi$-implied MRP so as to distinguish between these functions for the MDP versus the $\pi$-implied MRP.

Let's say we are given a fixed policy $\pi$ and an MDP specified by it's state-reward transition probability function $\mathcal{P}_R$. Then the transition probability function $\mathcal{P}_R^{\pi \rightarrow MRP}$ of the MRP implied by the evaluation of the MDP with the policy $\pi$ is defined as:

$$\mathcal{P}_R^{\pi \rightarrow MRP}(s,s',r) = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{P}_R(s,a,s',r)$$ 

Likewise,

$$\mathcal{P}^{\pi \rightarrow MRP}(s,s') = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{P}(s,a,s')$$

$$\mathcal{R}_T^{\pi \rightarrow MRP}(s,s') = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{R}_T(s,a,s')$$

$$\mathcal{R}^{\pi \rightarrow MRP}(s) = \sum_{a\in \mathcal{A}} \pi(s,a) \cdot \mathcal{R}(s,a)$$

So any time we talk about an MDP evaluated with a fixed policy, you should know that we are effectively talking about the implied MRP. This insight is now going to be key in the design of our code to represent Markov Decision Processes. We create an abstract class called `MarkovDecisionProcess` with just a single method - an `@abstractmethod` called `apply_policy` that would take as input a `Policy` object and would produce as output a `MarkovRewardProcess` object. Thanks to the above insight, we've essentially modeled the fact that an MDP is fully specified by the MRP implied by a provided policy the agent would evaluate in order to "run the MDP" (which effectively collapses the MDP into the implied MRP). The entire body of this lean abstract class `MarkovDecisionProcess` is shown below:

```python
class MarkovDecisionProcess(ABC, Generic[S, A]):
    @abstractmethod
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        pass
```

The above code for `Policy` and `MarkovDecisionProcess` is in the file [rl/markov_decision_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_decision_process.py).   

## Simple Inventory Example with Unlimited Capacity (Infinite State/Action Space)

Now we come back to our simple inventory example. Unlike previous situations of this example, here we will assume that there is no space capacity constraint on toothpaste. This means we have a choice of ordering any (unlimited) non-negative integer quantity of toothpaste units. Therefore, the action space is infinite. Also, since the order quantity shows up as On-Order the next day and as delivered inventory the day after the next day, the On-Hand and On-Order quantities are also unbounded. Hence, the state space is infinite. Due to the infinite state and action spaces, we won't be able to take advantage of the so-called "Tabular Dynamic Programming Algorithms" we will cover in the next chapter (algorithms that are meant for finite state and action spaces). There is still significant value in modeling infinite MDPs of this type because we can perform simulations (by sampling from an infinite space). Simulations are valuable not just to explore various properties and metrics relevant in the business problem modeled with an MDP, but simulations also enable us to design approximate algorithms to calculate Value Functions for given policies as well as Optimal Value Functions (which is the ultimate purpose of modeling MDPs).

We will cover details on these approximate algorithms later in the book - for now, it's important for you to simply get familiar with how to model infinite MDPs of this type. This infinite-space inventory example serves as a great learning for an introduction to modeling an infinite (but countable) MDP.

We create a concrete class `SimpleInventoryMDPNoCap` that implements the abstract class `MarkovDecisionProcess` (specifically implements the `@abstractmethod apply_policy`). The attributes `poisson_lambda`, `holding_cost` and `stockout_cost` have the same semantics as in the previous chapter in the context of a Markov Reward Process (`SimpleInventoryMRP`). The `apply_policy` method takes as input an object `policy` of abstract type `Policy`, so the only thing we can do with the `policy` object is to invoke the only method it has - `act(state)` - which gives us an object of type `Distribution[int]` representing an abstract probability distribution of actions (order quantities). What can we do with this abstract `Distribution[int]` object? Well, the only thing we can do with it is to invoke the only method it has - `sample()` - which gives us a sample of the action (`order: int`). Next, we sample from the poisson probability distribution of customer demand. From the samples of `order: int` and `demand_sample: int`, we obtain a sample of the pair of `next_state: InventoryState` and `reward: float`. This sample pair is returned as a `SampledDistribution` object by the implementation of the `transition_reward` method in the implied MRP class `ImpliedMRP`. The above sampling dynamics fully describe the MDP in terms of how a given policy interface helps generate samples in the implied MRP.

```python
class SimpleInventoryMDPNoCap(MarkovDecisionProcess[InventoryState, int]):
    def __init__(self, poisson_lambda: float, holding_cost: float,
                 stockout_cost: float):
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def apply_policy(self, policy: Policy[InventoryState, int])\
            -> MarkovRewardProcess[InventoryState]:

        mdp = self

        class ImpliedMRP(MarkovRewardProcess[InventoryState]):

            def transition_reward(self, state: InventoryState)\
                    -> SampledDistribution[Tuple[InventoryState, float]]:
                order = policy.act(state).sample()

                def sample_next_state_reward(
                    mdp=mdp,
                    state=state,
                    order=order
                ) -> Tuple[InventoryState, float]:
                    demand_sample: int = np.random.poisson(mdp.poisson_lambda)
                    ip: int = state.inventory_position()
                    next_state: InventoryState = InventoryState(
                        max(ip - demand_sample, 0),
                        order
                    )
                    reward: float = - mdp.holding_cost * state.on_hand\
                        - mdp.stockout_cost * max(demand_sample - ip, 0)
                    return next_state, reward

                return SampledDistribution(sample_next_state_reward)

        return ImpliedMRP()
```

We leave it to you as an exercise to run various simulations of the MRP implied by the deterministic and stochastic policy instances we had created earlier (the above code is in the file [rl/chapter3/simple_inventory_mdp_nocap.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter3/simple_inventory_mdp_nocap.py)). See the method `fraction_of_days_oos` in this file as an example of a simulation to calculate the percentage of days when we'd be unable to satisfy some customer demand for toothpaste due to too little inventory at store-opening (naturally, the higher the re-order point in the policy, the lesser the percentage of days when we'd be Out-of-Stock). This kind of simulation exercise will help build intuition on the tradeoffs we have to make between having too little inventory versus having too much inventory (holding costs versus stockout costs) - essentially leading to our ultimate goal of determining the Optimal Policy (more on this later). 

## Finite Markov Decision Processes

Certain calculations for Markov Decision Processes can be performed easily if:

* The state space is finite ($\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$),
* The action space $\mathcal{A}(s)$ is finite for each $s \in \mathcal{S}$.
* The set of pairs of next state and reward transitions from each pair of current state and action, is finite

If we satisfy the above three characteristics, we refer to the Markov Decision Process as a Finite Markov Decision Process. Let us write some code for a Finite Markov Decision Process. We create a concrete class `FiniteMarkovDecisionProcess` that implements the interface of the abstract class `MarkovDecisionProcess` (specifically implements the `@abstractmethod apply_policy`). Our first task is to think about the data structure required to specify an instance of `FiniteMarkovDecisionProcess` (i.e., the data structure we'd pass to the `__init__` method of `FiniteMarkovDecisionProcess`). Analogous to how we curried $\mathcal{P}_R$ for a Markov Reward Process as $\mathcal{S} \rightarrow (\mathcal{S} \times \mathbb{R} \rightarrow [0,1])$ (where $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$), here we curry $\mathcal{P}_R$ for the MDP as:
$$\mathcal{S} \rightarrow (\mathcal{A} \rightarrow (\mathcal{S} \times \mathbb{R} \rightarrow [0, 1]))$$
Since $\mathcal{S}$ is finite, $\mathcal{A}$ is finite, and the set of next state and reward transitions for each pair of current state and action is also finite, we can represent $\mathcal{P}_R$ as a data structure of type `StateActionMapping[S, A]` as shown below:

```python
StateReward = FiniteDistribution[Tuple[S, float]]
ActionMapping = Mapping[A, StateReward[S]]
StateActionMapping = Mapping[S, ActionMapping[A, S]]
```

The constructor (``__init__`` method) of `FiniteMarkovDecisionProcess` takes as input `mapping: StateActionMapping[S, A]` that represents the complete structure of the Finite MDP - it maps each state to an action map, and it maps each action in each action map to a finite probability distribution of pairs of next state and reward (essentially the structure of the $\mathcal{P}_R$ function). Now let's consider the implementation of the abstract method `apply_policy` of `MarkovDecisionProcess`. It's interface says that it's input is a `policy: Policy[S, A]`. Since `Policy[S, A]` is an abstract class with only an `@abstractmethod act`, all we can do in `apply_policy` is to call the `act` method of `Policy[S, A]`. This gives us an object of abstract type `Distribution[A]` and all we can do with it is to call it's only (abstract) method `sample`, upon which we get an action sample `action: A`. Given the `state: A` and the sample `action: A`, we can access `self.mapping[state][action]: FiniteDistribution[Tuple[S, float]]` which represents a finite probability distribution of pairs of next state and reward. We sample from this distribution and return the sampled pair of next state and reward as a `SampledDistribution` object. This satisfies the responsibility of `FiniteMarkovDecisionProcess` in terms of implementing the `@abstractmethod apply_policy` of the abstract class `MarkovDecisionProcess`. The code below also includes the `actions` method which produces the set of allowed actions $\mathcal{A}(s)$ for a given state $s\in \mathcal{S}$, and the `__repr__` method that pretty-prints `self.mapping`.

```python
class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):

    mapping: StateActionMapping[S, A]

    def __init__(self, mapping: StateActionMapping[S, A]]):
        self.mapping = mapping

    def __repr__(self) -> str:
        display = ""
        for s, d in self.mapping.items():
            display += f"From State {s}:\n"
            for a, d1 in d.items():
                display += f"  With Action {a}:\n"
                for (s1, r), p in d1.table():
                    display += f"    To [State {s} and "\
                        + f"Reward {r:.3f}] with Probability {p:.3f}\n"
        return display

    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        class Process(MarkovRewardProcess[S]):
            def transition_reward(self,
                                  state: S) -> Distribution[Tuple[S, float]]:
                def next_state():
                    action: A = policy.act(state).sample()
                    return self.mapping[state][action].sample()

                return SampledDistribution(next_state)

        return Process()

    def actions(self, state: S) -> Iterable[A]:
        return self.mapping[state].keys()
```

Now that we've implemented a finite MDP, let's implement a finite policy, i.e., a policy function whose domain is a finite set of states $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ and maps each state to a probability distribution over a finite set of actions $\mathcal{A} = \{a_1, a_2, \ldots, a_m\}$. So we create a concrete class `FinitePolicy` that implements the interface of the abstract class `Policy` (specifically implements the `@abstractmethod act`). The input to the constructor (`__init__` method) is `policy_map: Mapping[S, FiniteDistribution[A]]` since this type captures the structure of the $\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$ function in the curried form:
$$\mathcal{S} \rightarrow (\mathcal{A} \rightarrow [0, 1])$$
for the case of finite $\mathcal{S}$ and finite $\mathcal{A}$. The `act` method is straightforward. We also implement a `__repr__` method for pretty-printing of `self.policy_map`.

```python
class FinitePolicy(Policy[S, A]):

    policy_map: Mapping[S, FiniteDistribution[A]]

    def __init__(self, policy_map: Mapping[S, FiniteDistribution[A]]):
        self.policy_map = policy_map

    def __repr__(self) -> str:
        display = ""
        for s, d in self.policy_map.items():
            display += f"For State {s}:\n"
            for a, p in d.table():
                display += f"  Do Action {a} with Probability {p:.3f}\n"
        return display

    def act(self, state: S) -> FiniteDistribution[A]:
        return self.policy_map[state]
```   

Armed with a `FinitePolicy` class, we can now write a method `apply_finite_policy` in `FiniteMarkovDecisionProcess` that takes as input a `policy: FinitePolicy[S, A]` and returns a `FiniteMarkovRewardProcess[S]` by processing the finite structures of both of the MDP and the Policy, and producing a finite structure of the implied MRP.      

```python
    def apply_finite_policy(
            self, policy: FinitePolicy[S, A]) -> FiniteMarkovRewardProcess[S]:
        transition_mapping: Dict[S, StateReward[S]] = {}

        for state in self.mapping:
            outcomes: DefaultDict[Tuple[S, float], float] = defaultdict(float)

            for action, p_action in policy.act(state).table():
                for outcome, p_state in self.mapping[state][action].table():
                    outcomes[outcome] += p_action * p_state

            transition_mapping[state] = Categorical(outcomes.items())

        return FiniteMarkovRewardProcess(transition_mapping)
```      

The above code for `FiniteMarkovRewardProcess` and `FinitePolicy` is in the file [rl/markov_decision_process.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/markov_decision_process.py).   

## Simple Inventory Example as a Finite Markov Decision Process

Now we'd like to model the simple inventory example as a Finite Markov Decision Process so we can take advantage of the algorithms specifically for Finite Markov Decision Processes. To enable finite states and finite actions, we will re-introduce the constraint of space capacity $C$ and we will apply the restriction that the order quantity (action) cannot exceed $C - (\alpha + \beta)$ where $\alpha$ is the On-Hand component of the State and $\beta$ is the On-Order component of the State. Thus, the action space for any given state $(\alpha, \beta) \in \mathcal{S}$ is finite. Next, note that this ordering policy ensures that in steady-state, the sum of On-Hand and On-Order will not exceed the capacity $C$. So we will constrain the set of states to be the steady-state set of finite states
$$\mathcal{S} = \{(\alpha, \beta): 0 \leq \alpha + \beta \leq C\}$$
Although the set of states is finite, there are an infinite number of pairs of next state and reward outcomes possible from any given pair of current state and action. This is because there are an infinite set of possibilities of customer demand on any given day (resulting in infinite possibilities of stockout cost, i.e., negative reward, on any day). To qualify as a Finite Markov Decision Process, we'll need to model in a manner such that we have a finite set of pairs of next state and reward outcomes from any given pair of current state and action. So what we'll do is that instead of considering $(S_{t+1}, R_{t+1})$ as the pair of next state and reward, we will model the pair of next state and reward to instead be $(S_{t+1}, \mathbb{E}[R_{t+1}|(S_t, S_{t+1}, A_t)])$ (we know $\mathcal{P}_R$ due to the Poisson probabilities of customer demand, so we can actually calculate this conditional expectation of reward). So given a state $s$ and action $a$, the pairs of next state and reward would be: $(s', \mathcal{R}_T(s, a, s'))$ for all the $s'$ we transition to from $(s, a)$. Since the set of possible next states $s'$ are finite, these newly-modeled rewards associated with the transitions ($\mathcal{R}_T(s,a,s')$) are also finite and hence, the set of pairs of next state and reward from any pair of current state and action are also finite. Note that this creative alteration of the reward definition is purely to reduce this Markov Decision Process into a Finite Markov Decision Process. Let's now work out the calculation of the reward transition function $\mathcal{R}_T$.

When the next state's ($S_{t+1}$) On-Hand is greater than zero, it means all of the day's demand was satisfied with inventory that was available at store-opening ($=\alpha + \beta$), and hence, each of these next states $S_{t+1}$ correspond to no stockout cost and only an overnight holding cost of $h \alpha$. Therefore, for all $\alpha, \beta$ (with $0 \leq \alpha + \beta \leq C$) and for all order quantity (action) $\theta$ (with $0 \leq \theta \leq C - (\alpha + \beta)$):
$$\mathcal{R}_T((\alpha, \beta), \theta, (\alpha + \beta - i, \theta)) = - h \alpha \text{ for } 0 \leq i \leq \alpha + \beta - 1$$

When next state's ($S_{t+1}$) On-Hand is equal to zero, there are two possibilities: 

1. The demand for the day was exactly $\alpha + \beta$, meaning all demand was satisifed with available store inventory (so no stockout cost and only overnight holding cost), or
2. The demand for the day was strictly greater than $\alpha + \beta$, meaning there's some stockout cost in addition to overnight holding cost. The exact stockout cost is an expectation calculation involving the number of units of missed demand under the corresponding poisson probabilities of demand exceeding $\alpha + \beta$.

This calculation is shown below:
$$\mathcal{R}_T((\alpha, \beta), \theta, (0, \theta)) = - h \alpha - p (\sum_{j=\alpha+\beta+1}^{\infty} f(j) \cdot (j - (\alpha + \beta)))$$
 $$= - h \alpha - p (\lambda (1 - F(\alpha + \beta - 1)) -  (\alpha + \beta)(1 - F(\alpha + \beta)))$$ 

So now we have a specification of $\mathcal{R}_T$, but when it comes to our coding interface, we are expected to specify $\mathcal{P}_R$ as that is the interface through which we create a `FiniteMarkovDecisionProcess`. Fear not - a specification of $\mathcal{P}_R$ is easy once we have a specification of $\mathcal{R}_T$. We simply create 5-tuples $(s,a,r,s',p)$ for all $s,s' \in \mathcal{S}, a \in \mathcal{A}$ such that $r=\mathcal{R}_T(s,a,s')$ and $p=\mathcal{P}(s,a,s')$ (we know $\mathcal{P}$ along with $\mathcal{R}_T$), and the set of all these 5-tuples (for all $s,s' \in \mathcal{S}, a \in \mathcal{A}$) constitute the specification of $\mathcal{P}_R$, i.e., $\mathcal{P}_R(s,a,r,s') = p$. This turns our reward-definition-altered mathematical model of a Finite Markov Decision Process into a programming model of the `FiniteMarkovDecisionProcess` class. This reward-definition-altered model enables us to gain from the fact that we can leverage the algorithms we'll be writing for Finite Markov Decision Processes (specifically, the classical Dynamic Programming algorithms - covered in the next chapter). The downside of this reward-definition-altered model is that it prevents us from performing simulations of the specific rewards encountered when transitioning from one state to another (because we no longer capture the probabilities of individual reward outcomes). Note that we can indeed perform simulations, but each transition step in the simulation will only show us the "mean reward" (specifically, the expected reward conditioned on current state, action and next state).

In fact, most Markov Processes you'd encounter in practice can be modeled as a combination of $\mathcal{R}_T$ and $\mathcal{P}$, and you'd simply follow the above $\mathcal{R}_T$ to $\mathcal{P}_R$ representation transformation drill to present this information in the form of $\mathcal{P}_R$ to instantiate a `FiniteMarkovDecisionProcess`. We designed the interface to accept $\mathcal{P}_R$ as input since that is the most general interface for specifying Markov Decision Processes.

So now let's write some code for the simple inventory example as a Finite Markov Decision Process as described above. All we have to do is to create a derived class inherited from `FiniteMarkovDecisionProcess` and write a method to construct the `mapping: StateActionMapping` (i.e., $\mathcal{P}_R$) that the `__init__` constuctor of `FiniteMarkovRewardProcess` requires as input. Note that the generic state `S` is replaced here with the `@dataclass InventoryState` to represent the inventory state, comprising of the On-Hand and On-Order inventory quantities, and the generic action `A` is replaced here with `int` to represent the order quantity.

```python
InvOrderMapping = StateActionMapping[InventoryState, int]

class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state: InventoryState = InventoryState(alpha, beta)
                ip: int = state.inventory_position()
                base_reward: float = - self.holding_cost * alpha
                d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}

                for order in range(self.capacity - ip + 1):
                    sr_probs_list: List[Tuple[Tuple[InventoryState, float],
                                              float]] =\
                        [((InventoryState(ip - i, order), base_reward),
                          self.poisson_distr.pmf(i)) for i in range(ip)]

                    probability: float = 1 - self.poisson_distr.cdf(ip - 1)
                    reward: float = base_reward - self.stockout_cost *\
                        (probability * (self.poisson_lambda - ip) +
                         ip * self.poisson_distr.pmf(ip))
                    sr_probs_list.append(
                        ((InventoryState(0, order), reward), probability)
                    )
                    d1[order] = Categorical(sr_probs_list)

                d[state] = d1
        return d
```

Now let's test this out with some example inputs (as shown below). We construct an instance of the `SimpleInventoryMDPCap` class with these inputs (named `si_mdp` below), then construct an instance of the `FinitePolicy[InventoryState, int]` class (a deterministic policy, named `fdp` below), and combine them to produce the implied MRP (an instance of the `FiniteMarkovRewardProcess[InventoryState]` class). 

```python
user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0

user_gamma = 0.9

si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
    SimpleInventoryMDPCap(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

fdp: FinitePolicy[InventoryState, int] = FinitePolicy(
    {InventoryState(alpha, beta):
     Constant(user_capacity - (alpha + beta)) for alpha in
     range(user_capacity + 1) for beta in range(user_capacity + 1 - alpha)}
)

implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
    si_mdp.apply_finite_policy(fdp)
```   

The above code is in the file [rl/chapter3/simple_inventory_mdp_cap.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter3/simple_inventory_mdp_cap.py). We encourage you to play with the inputs in `__main__`, produce the resultant implied MRP, and explore it's characteristics (such as it's Reward Function and it's Value Function).

## MDP Value Function for a Fixed Policy

Now we are ready to talk about the Value Function for an MDP evaluated with a fixed policy $\pi$ (also known as the MDP *Prediction* problem). The term *Prediction* refers to the fact that this problem is about forecasting the expected future return when the agent follows a specific policy. Just like in the case of MRP, we define the Return $G_t$ at time step $t$ for an MDP as:
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

As we saw in the previous chapter, for finite state spaces that are not too large, Equation \eqref{eq:mdp_bellman_policy_eqn_vv} can be solved for $V^{\pi}$ (i.e. solution to the MDP *Prediction* problem) with a linear algebra solution (Equation \eqref{eq:mrp_bellman_linalg_solve} from the previous chapter). More generally, Equation \eqref{eq:mdp_bellman_policy_eqn_vv} will be a key equation for the rest of the book in developing various Dynamic Programming and Reinforcement Algorithms for the MDP *Prediction* problem. However, there is another Value Function that's also going to be crucial in developing MDP algorithms - one which maps a (state, action) pair to the expected return originating from the (state, action) pair when evaluated with a fixed policy. This is known as the *Action-Value Function* of an MDP evaluated with a fixed policy $\pi$:
$$Q^{\pi}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$
defined as:
$$Q^{\pi}(s, a) = \mathbb{E}_{\pi, \mathcal{P}_R}[G_t|(S_t=s, A_t=a)] \text{ for all } s \in \mathcal{S}, a\in \mathcal{A}, \text{ for all } t = 0, 1, 2, \ldots$$

To avoid terminology confusion, we refer to $V^{\pi}$ as the *State-Value Function* (albeit often simply abbreviated to *Value Function*) for policy $\pi$, to distinguish from the *Action-Value Function* $Q^{\pi}$. The way to interpret $Q^{\pi}(s, a)$ is that it's the Expected Return from a given state $s$ by first taking the action $a$ and subsequently, following policy $\pi$. With this interpretation of $Q^{\pi}(s, a)$, we can perceive $V^{\pi}(s)$ as the "weighted average" of $Q^{\pi}(s,a)$ (over all possible actions $a$ from state $s$) with the weights equal to probabilities of action $a$, given state $s$ (i.e., $\pi(s, a)$). Precisely,

\begin{equation}
V^{\pi}(s) = \sum_{a\in\mathcal{A}} \pi(s, a) \cdot Q^{\pi}(s, a) \text{ for all } s \in \mathcal{S} \label{eq:mdp_bellman_policy_eqn_vq}
\end{equation}

Combining Equation \eqref{eq:mdp_bellman_policy_eqn_vv} and Equation \eqref{eq:mdp_bellman_policy_eqn_vq} yields:
\begin{equation}
Q^{\pi}(s, a) = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{S}} \mathcal{P}(s,a,s') \cdot V^{\pi}(s') \text{ for all  } s \in \mathcal{S}, a \in \mathcal{A} \label{eq:mdp_bellman_policy_eqn_qv}
\end{equation}

Combining Equation \eqref{eq:mdp_bellman_policy_eqn_qv} and Equation \eqref{eq:mdp_bellman_policy_eqn_vq} yields:
\begin{equation}
Q^{\pi}(s, a) = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{S}} \mathcal{P}(s,a,s') \sum_{a'\in \mathcal{A}} \pi(s', a') \cdot Q^{\pi}(s', a') \text{ for all  } s \in \mathcal{S}, a \in \mathcal{A} \label{eq:mdp_bellman_policy_eqn_qq}
\end{equation}

Equation \eqref{eq:mdp_bellman_policy_eqn_vv} is known as the MDP State-Value Function Bellman Policy Equation (Figure \ref{fig:mdp_bellman_policy_tree_vv} serves as a visualization aid for this Equation).  Equation \eqref{eq:mdp_bellman_policy_eqn_qq} is known as the MDP Action-Value Function Bellman Policy Equation (Figure \ref{fig:mdp_bellman_policy_tree_qq} serves as a visualization aid for this Equation).  Note that Equation \eqref{eq:mdp_bellman_policy_eqn_vq} and Equation \eqref{eq:mdp_bellman_policy_eqn_qv} are embedded in Figure \ref{fig:mdp_bellman_policy_tree_vv} as well as in Figure \ref{fig:mdp_bellman_policy_tree_qq}. Equations \eqref{eq:mdp_bellman_policy_eqn_vv}, \eqref{eq:mdp_bellman_policy_eqn_vq}, \eqref{eq:mdp_bellman_policy_eqn_qv} and \eqref{eq:mdp_bellman_policy_eqn_qq} are collectively known as the MDP Bellman Policy Equations.

<div style="text-align:center" markdown="1">
![Visualization of MDP State-Value Function Bellman Policy Equation \label{fig:mdp_bellman_policy_tree_vv}](./chapter3/mdp_bellman_policy_tree_vv.png "Visualization of MDP State-Value Function Bellman Policy Equation")
</div>

<div style="text-align:center" markdown="1">
![Visualization of MDP Action-Value Function Bellman Policy Equation \label{fig:mdp_bellman_policy_tree_qq}](./chapter3/mdp_bellman_policy_tree_qq.png "Visualization of MDP Action-Value Function Bellman Policy Equation")
</div>

For the rest of the book, in these MDP transition figures, we shall always depict states as elliptical-shaped nodes and actions as rectangular-shaped nodes. Notice that transition from a state node to an action node is associated with a probability represented by $\pi$ and transition from an action node to a state node is associated with a probability represented by $\mathcal{P}$.

Note that for finite MDPs of state space not too large, we can solve the MDP Prediction problem (solving for $V^{\pi}$ and equivalently, $Q^{\pi}$)in a straightforward manner: Given a policy $\pi$, we can create the finite MRP implied by $\pi$, using the method `apply_policy` in `FiniteMarkovDecisionProcess`, then use the matrix-inversion method you learnt in Chapter 2 to calculate the Value Function $V^{\pi \rightarrow MRP}$ of the $\pi$-implied MRP. We know that $V^{\pi \rightarrow MRP}$ is the same as the State-Value Function $V^{\pi}$ of the MDP which can then be used to arrive at the Action-Value Function $Q^{\pi}$ of the MDP (using Equation \eqref{eq:mdp_bellman_policy_eqn_qv}). For large state spaces, we will use iterative/numerical methods (Dynamic Programming and Reinforcement Learning algorithms) to solve this Prediction problem (covered later in this book).

## Optimal Value Function and Optimal Policies

Finally, we arrive at the main purpose of a Markov Decision Process - to identify a policy (or policies) that would yield the Optimal Value Function (i.e., the best possible *Expected Return* from each of the states). We say that a Markov Decision Process is "solved" when we identify its Optimal Value Function (together with its associated Optimal Policy) - i.e., a Policy that yields the Optimal Value Function). The problem of identifying the Optimal Value Function and its associated Optimal Policy/Policies is known as the MDP *Control* problem. The term *Control* refers to the fact that this problem involves steering the actions (by iterative modifications of the policy) to drive the Value Function towards Optimality. Formally, the Optimal Value Function

$$V^*: \mathcal{S} \rightarrow \mathbb{R}$$

is defined as:

$$V^*(s) = \max_{\pi \in \Pi} V^{\pi}(s) \text{ for all } s \in \mathcal{S}$$

where $\Pi$ is the set of stationary (stochastic) policies over the spaces of $\mathcal{S}$ and $\mathcal{A}$.

The way to read the above definition is that for each state $s$, we consider all possible stochastic stationary policies $\pi$, and maximize $V_{\pi}(s)$ across all these choices of $\pi$ (note: the maximization over choices of $\pi$ is done separately for each $s$). Note that we haven't yet talked about how to achieve the maximization through an algorithm - we have simply defined the Optimal Value Function.

Likewise, the Optimal Action-Value Function

$$Q^*: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

is defined as:

$$Q^*(s, a) = \max_{\pi \in \Pi} Q^{\pi}(s, a) \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}$$

$V^*$ is often refered to as the Optimal State-Value Function to distinguish it from the Optimal Action-Value Function $Q^*$ (although, for succinctness, $V^*$ is often also refered to as simply the Optimal Value Function). To be clear, if someone says, Optimal Value Function, by default, they'd be refering to the Optimal State-Value Function $V^*$ (not $Q^*$).

Much like how the Value Function(s) for a fixed policy have a recursive formulation, we can create a recursive formulation for the Optimal Value Function(s). Let us start by unraveling the Optimal State-Value Function $V^*(s)$ for a given state $s$ - we consider all possible actions $a\in \mathcal{A}$ we can take from state $s$, and pick the action $a$ that yields the best Action-Value from thereon, i.e., the action that yields the best $Q^*(s,a)$. Formally, this gives us the following equation:

\begin{equation}
V^*(s) = \max_{a\in \mathcal{A}} Q^*(s,a) \label{eq:mdp_bellman_opt_eqn_vq}
\end{equation}

Likewise, let's think about what it means to be optimal at a given state-action pair $(s,a)$, i.e, let's unravel $Q^*(s,a)$. First, we get the immediate expected reward $\mathcal{R}(s,a)$. Next, we consider all possible random states $s' \in \mathcal{S}$ we can transition to, and from each of those states, we recursively act optimally. Formally, this gives us the following equation:

\begin{equation}
Q^*(s,a) = \mathcal{R}(s,a) + \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot V^*(s') \label{eq:mdp_bellman_opt_eqn_qv}
\end{equation}

Substituting for $Q^*(s,a)$ from Equation \eqref{eq:mdp_bellman_opt_eqn_qv} in Equation \eqref{eq:mdp_bellman_opt_eqn_vq} gives:

\begin{equation}
V^*(s) = \max_{a\in \mathcal{A}} \{ \mathcal{R}(s,a) + \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot V^*(s') \} \label{eq:mdp_bellman_opt_eqn_vv}
\end{equation}

Equation \eqref{eq:mdp_bellman_opt_eqn_vv} is known as the MDP State-Value Function Bellman Optimality Equation and is depicted in Figure \ref{fig:mdp_bellman_opt_tree_vv} as a visualization aid.

Substituting for $V^*(s)$ from Equation \eqref{eq:mdp_bellman_opt_eqn_vq} in Equation \eqref{eq:mdp_bellman_opt_eqn_qv} gives:

\begin{equation}
Q^*(s,a) = \mathcal{R}(s,a) + \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \max_{a'\in \mathcal{A}} Q^*(s',a') \label{eq:mdp_bellman_opt_eqn_qq}
\end{equation}

Equation \eqref{eq:mdp_bellman_opt_eqn_qq} is known as the MDP Action-Value Function Bellman Optimality Equation and is depicted in Figure \ref{fig:mdp_bellman_opt_tree_qq} as a visualization aid.


<div style="text-align:center" markdown="1">
![Visualization of MDP State-Value Function Bellman Optimality Equation \label{fig:mdp_bellman_opt_tree_vv}](./chapter3/mdp_bellman_opt_tree_vv.png "Visualization of MDP State-Value Function Bellman Optimality Equation")
</div>


<div style="text-align:center" markdown="1">
![Visualization of MDP Action-Value Function Bellman Optimality Equation \label{fig:mdp_bellman_opt_tree_qq}](./chapter3/mdp_bellman_opt_tree_qq.png "Visualization of MDP Action-Value Function Bellman Optimality Equation")
</div>

Note that Equation \eqref{eq:mdp_bellman_opt_eqn_vq} and Equation \eqref{eq:mdp_bellman_opt_eqn_qv} are embedded in Figure \ref{fig:mdp_bellman_opt_tree_vv} as well as in Figure \ref{fig:mdp_bellman_opt_tree_qq}. Equations \eqref{eq:mdp_bellman_policy_eqn_vv}, \eqref{eq:mdp_bellman_policy_eqn_vq}, \eqref{eq:mdp_bellman_policy_eqn_qv} and \eqref{eq:mdp_bellman_policy_eqn_qq} are collectively known as the MDP Bellman Optimality Equations. We should highlight that when someone says MDP Bellman Equation or simply Bellman Equation, unless they explicit state otherwise, they'd be refering to the MDP Bellman Optimality Equations (and typically specifically the MDP State-Value Function Bellman Optimality Equation). This is because the MDP Bellman Optimality Equations address the ultimate purpose of Markov Decision Processes - to identify the Optimal Value Function and the associated policy/policies that achieve the Optimal Value Function (i.e., enabling us to solve the MDP *Control* problem).

Again, it pays to emphasize that the Bellman Optimality Equations don't directly give us a recipe to calculate the Optimal Value Function or the policy/policies that achieve the Optimal Value Function - they simply state a powerful mathematical property of the Optimal Value Function that (as we shall see later in this book) will help us come up with algorithms (Dynamic Programming and Reinforcement Learning) to calculate the Optimal Value Function and the associated policy/policies that achieve the Optimal Value Function.

We have been using the phrase "policy/policies that achieve the Optimal Value Function", but we have't provided a clear definition of such a policy (or policies) - one that achieves the Optimal Value Function. Now we are ready to dig into the concept of *Optimal Policy*.  Here's the formal definition of an Optimal Policy $\pi^*: \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$:


$$\pi^* \in \Pi \text{ is an Optimal Policy if } V^{\pi^*}(s) \geq V^{\pi}(s) \text{ {\em for all} } \pi \in \Pi \text{ and {\em for all states} } s \in \mathcal{S}$$

As explained earlier, the definition of $V^*$ states that the maximization of the Value Function is separate for each state $s \in \mathcal{S}$ and so, presumably we could end up with different policies $\pi$ that maximize $V^{\pi}(s)$ for different states. Separately, the definition of Optimal Policy $\pi^*$ says that it is a policy that is "better than or equal to" (on the $V^{\pi}$ metric) all other stationary policies *for all* states (note that there could be multiple Optimal Policies). So the natural question to ask is whether there exists an Optimal Policy $\pi^*$ that maximizes $V^{\pi}(s)$  *for all* states $s \in \mathcal{S}$, i.e., $V^*(s) = V^{\pi^*}(s)$ for all $s \in \mathcal{S}$. On the face of it, this seems like a strong statement. However, this answers in the affirmative. In fact,

\begin{theorem}
For any Markov Decision Process
\begin{itemize}
\item There exists an Optimal Policy $\pi^* \in \Pi$, i.e., there exists a Policy $\pi^* \in \Pi$ such that $V^{\pi^*}(s) \geq V^{\pi}(s) \mbox{ for all policies  } \pi \in \Pi \mbox{ and for all states } s \in \mathcal{S}$
\item All Optimal Policies achieve the Optimal Value Function, i.e. $V^{\pi^*}(s) = V^*(s)$ for all $s \in \mathcal{S}$, for all Optimal Policies $\pi^*$
\item All Optimal Policies achieve the Optimal Action-Value Function, i.e. $Q^{\pi^*}(s,a) = Q^*(s,a)$ for all $s \in \mathcal{S}$, for all $a \in \mathcal{A}$, for all Optimal Policies $\pi^*$
\end{itemize}
\label{th:mdp_opt_vf_policy}
\end{theorem}

Before proceeding with the proof of Theorem (\ref{th:mdp_opt_vf_policy}), we establish a simple Lemma.
\begin{lemma}
For any two Optimal Policies $\pi_1$ and $\pi_2$, $V^{\pi_1}(s) = V^{\pi_2}(s)$ for all $s \in \mathcal{S}$
\end{lemma}
\begin{proof}
Since $\pi_1$ is an Optimal Policy, from the Optimal Policy definition, we have: $V^{\pi_1}(s) \geq V^{\pi_2}(s)$ for all $s \in \mathcal{S}$.
Likewise, since $\pi_2$ is an Optimal Policy, from the Optimal Policy definition, we have: $V^{\pi_2}(s) \geq V^{\pi_1}(s)$ for all $s \in \mathcal{S}$.
This implies: $V^{\pi_1}(s) = V^{\pi_2}(s)$ for all $s \in \mathcal{S}$
\end{proof}

Now we are ready to prove Theorem (\ref{th:mdp_opt_vf_policy}).
\begin{proof}
As a consequence of the above Lemma, all we need to do to prove the theorem is to establish an Optimal Policy $\pi^*$ that achieves the Optimal Value Function and the Optimal Action-Value Function. Consider the following Deterministic Policy (as a candidate Optimal Policy) $\pi_D^* : \mathcal{S} \rightarrow \mathcal{A}$:

$$\pi_D^*(s) = \argmax_{a \in \mathcal{A}} Q^*(s,a) \mbox{ for all } s \in \mathcal{S}$$

First we show that $\pi_D^*$ achieves the Optimal Value Function. Since $\pi_D^*(s) = \argmax_{a \in \mathcal{A}} Q^*(s,a)$ and $V^*(s) = \max_{a \in \mathcal{A}} Q^*(s,a)$ for all $s \in \mathcal{S}$, $\pi_D^*$ prescribes the optimal action for each state (that produces the Optimal Value Function $V^*$). Hence, following policy $\pi_D^*$ in each state will generate the same Value Function as the Optimal Value Function. In other words, $V^{\pi_D^*}(s) = V^*(s)$ for all $s \in \mathcal{S}$. Likewise, we can argue that: $Q^{\pi_D^*}(s,a) = Q^*(s,a)$ for all $s \in \mathcal{S}$ and for all $a \in \mathcal{A}$.

Finally, we prove by contradiction that $\pi_D^*$ is an Optimal Policy. So assume $\pi_D^*$ is not an Optimal Policy. Then there exists a policy $\pi \in \Pi$ and a state $s \in \mathcal{S}$ such that $V^{\pi}(s) > V^{\pi_D^*}(s)$. Since $V^{\pi_D^*}(s) = V^*(s)$, we have: $V^{\pi}(s) > V^*(s)$ which contradicts the Optimal Policy Definition: $V^*(s) = \max_{\pi \in \Pi} V^{\pi}(s)$ for all $s\in \mathcal{S}$.


\end{proof}

It pays to emphasize again that this section hasn't provided a recipe to calculate the Optimal Value Function and Optimal Policy (i.e., to solve the MDP Control problem) - it has only provided definitions and results that will help us develop Dynamic Programming and Reinforcement Learning algorithms (later in the book) to solve the MDP Control problem. We should also note that unlike the Prediction problem which has a straightforward linear-algebra solution for small state spaces, the Control problem is non-linear and so, doesn't have an analogous straightforward linear-algebra solution. The simplest solutions for the Control problem (even for small state spaces) are the Dynamic Programming algorithms we will cover in the next chapter.

## Variants and extensions of MDPs

* continuous states, continuous actions, continuous time, POMDPs
* Some discussion of MDPs in the real-world, and then explain next chapter is Dynamic Programming to solve Value Function for fixed policy and Optimal Value Function

## Summary of Key Learnings from this Chapter

* MDP Bellman Policy Equations
* MDP Bellman Optimality Equations
* Theorem \ref{th:mdp_opt_vf_policy} about the existence of an Optimal Policy, and of each Optimal Policy achieving the Optimal Value Function

