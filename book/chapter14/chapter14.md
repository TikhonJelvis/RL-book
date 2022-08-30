# Finishing Touches

## Multi-Armed Bandits: Exploration versus Exploitation {#sec:multi-armed-bandits-chapter}

\index{exploration versus exploitation|(}
\index{multi-armed bandits|(}

We learnt in Chapter [-@sec:rl-control-chapter] that balancing exploration and exploitation is vital in RL Control algorithms. While we want to exploit actions that seem to be fetching good returns,  we also want to adequately explore all possible actions so we can obtain an accurate-enough estimate of their Q-Values. We had mentioned that this is essentially the Explore-Exploit dilemma of the famous [Multi-Armed Bandit Problem](https://en.wikipedia.org/wiki/Multi-armed_bandit). The Multi-Armed Bandit problem provides a simple setting to understand the explore-exploit tradeoff and to develop explore-exploit balancing algorithms. The approaches followed by the Multi-Armed Bandit algorithms are then well-transportable to the more complex setting of RL Control.

In this chapter, we start by specifying the Multi-Armed Bandit problem, followed by coverage of a variety of techniques to solve the Multi-Armed Bandit problem (i.e., effectively balancing exploration against exploitation). We've actually seen one of these algorithms already for RL Control—following an $\epsilon$-greedy policy, which naturally is applicable to the simpler setting of Multi-Armed Bandits. We had mentioned in Chapter [-@sec:rl-control-chapter] that we can simply replace the $\epsilon$-greedy approach with any other algorithm for explore-exploit tradeoff. In this chapter, we consider a variety of such algorithms, many of which are far more sophisticated compared to the simple $\epsilon$-greedy approach. However, we cover these algorithms for the simple setting of Multi-Armed Bandits as it promotes understanding and development of intuition. After covering a range of algorithms for Multi-Armed Bandits, we consider an extended problem known as Contextual Bandits, that is a step between the Multi-Armed Bandits problem and the RL Control problem (in terms of problem complexity). Finally, we explain how the algorithms for Multi-Armed Bandits can be easily transported to the more nuanced/extended setting of Contextual Bandits, and further extended to RL Control.

### Introduction to the Multi-Armed Bandit Problem

At various points in past chapters, we've emphasized the importance of the Explore-Exploit tradeoff in the context of RL Control—selecting actions for any given state that balances the notions of exploration and exploitation. If you think about it, you will realize that many situations in business (and in our lives!) present this explore-exploit dilemma on choices one has to make. *Exploitation* involves making choices that *seem to be best* based on past outcomes, while *Exploration* involves making choices one hasn't yet tried (or not tried sufficiently enough).

Exploitation has intuitive notions of "being greedy" and of being "short-sighted", and too much exploitation could lead to some regret of having missing out on unexplored "gems". Exploration has intuitive notions of "gaining information" and of being "long-sighted", and too much exploration could lead to some regret of having wasting time on "duds". This naturally leads to the idea of balancing exploration and exploitation so we can combine *information-gains* and *greedy-gains* in the most optimal manner. The natural question then is whether we can set up this problem of explore-exploit dilemma in a mathematically disciplined manner. Before we do that, let's look at a few common examples of the explore-exploit dilemma. 

#### Some Examples of Explore-Exploit Dilemma

* Restaurant Selection: We like to go to our favorite restaurant (Exploitation) but we also like to try out a new restaurant (Exploration).
* Online Banner Advertisements: We like to repeat the most successful advertisement (Exploitation) but we also like to show a new advertisement (Exploration).
* Oil Drilling: We like to drill at the best known location (Exploitation) but we also like to drill at a new location (Exploration).
* Learning to play a game: We like to play the move that has worked well for us so far (Exploitation) but we also like to play a new experimental move (Exploration).

\index{multi-armed bandits!slot machines}

The term *Multi-Armed Bandit* (abbreviated as MAB) is a spoof name that stands for "Many One-Armed Bandits" and the term *One-Armed Bandit* refers to playing a slot-machine in a casino (that has a single lever to be pulled, that presumably addicts us and eventually takes away all our money, hence the term "bandit"). Multi-Armed Bandit refers to the problem of playing several slot machines (each of which has an unknown fixed payout probability distribution) in a manner that we can make the maximum cumulative gains by playing over multiple rounds (by selecting a single slot machine in a single round). The core idea is that to achieve maximum cumulative gains, one would need to balance the notions of exploration and exploitation, no matter which selection strategy one would pursue.


\index{multi-armed bandits|textbf}

#### Problem Definition

\begin{definition}
A {\em Multi-Armed Bandit} (MAB) comprises of:
\begin{itemize}
\item A finite set of {\em Action}s $\mathcal{A}$ (known as the "arms").
\item Each action ("arm") $a \in \mathcal{A}$ is associated with a probability distribution over $\mathbb{R}$ (unknown to the AI Agent) denoted as $\mathcal{R}^a$, defined as:
$$\mathcal{R}^a(r) = \mathbb{P}[r|a] \text{ for all } r \in \mathbb{R}$$
\item A time-indexed sequence of AI Agent-selected actions $A_t \in \mathcal{A}$ for time steps $t=1, 2, \ldots$, and a time-indexed sequence of Environment-generated {\em Reward} random variables $R_t \in \mathbb{R}$ for time steps $t=1, 2, \ldots$, with $R_t$ randomly drawn from the probability distribution $\mathcal{R}^{A_t}$.
\end{itemize}
\end{definition}

The AI Agent's goal is to maximize the following *Expected Cumulative Rewards* over a certain number of time steps $T$:
$$\mathbb{E}[\sum_{t=1}^T R_t]$$

So the AI Agent has $T$ selections of actions to make (in sequence), basing each of those selections only on the rewards it has observed before that time step (specifically, the AI Agent does not have knowledge of the probability distributions $\mathcal{R}^a$). Any selection strategy to maximize the Expected Cumulative Rewards risks wasting time on "duds" while exploring and also risks missing untapped "gems" while exploiting. 

It is immediately observable that the Environment doesn't have a notion of *State*. When the AI Agent selects an arm, the Environment simply samples from the probability distribution for that arm. However, the AI Agent might maintain relevant features of the history (of actions taken and rewards obtained) as its *State*, which would help the AI Agent in making the arm-selection (action) decision. The arm-selection action is then based on a (*Policy*) function of the agent's *State*. So, the agent's arm-selection strategy is basically this *Policy*. Thus, even though a MAB is not posed as an MDP, the agent could model it as an MDP and solve it with an appropriate Planning or Learning algorithm. However, many MAB algorithms don't take this formal MDP approach. Instead, they rely on heuristic methods that don't aim to *optimize*—they simply strive for *good* Cumulative Rewards (in Expectation). Note that even in a simple heuristic algorithm, $A_t$ is a random variable simply because it is a function of past (random) rewards.

\index{Markov decision process}

#### Regret

\index{multi-armed bandits!regret|textbf}
\index{multi-armed bandits!action value|textbf}

The idea of *Regret* is quite fundamental in designing algorithms for MAB. In this section, we illuminate this idea.

We define the *Action Value* $Q(a)$ as the (unknown) mean reward of action $a$, i.e., 
$$Q(a) = \mathbb{E}[r|a]$$
We define the *Optimal Value* $V^*$ and *Optimal Action* $a^*$ (noting that there could be multiple optimal actions) as:
$$V^* = \max_{a\in\mathcal{A}} Q(a) = Q(a^*)$$
We define *Regret* $l_t$ as the opportunity loss at a single time step $t$, as follows:
$$l_t = \mathbb{E}[V^* - Q(A_t)]$$
We define the *Total Regret* $L_T$ as the total opportunity loss, as follows:
$$L_T = \sum_{t=1}^T l_t = \sum_{t=1}^T \mathbb{E}[V^* - Q(A_t)]$$
Maximizing the *Expected Cumulative Rewards* is the same as Minimizing *Total Regret*.

#### Counts and Gaps

Let $N_t(a)$ be the (random) number of selections of an action $a$ across the first $t$ time steps. Let us refer to $\mathbb{E}[N_t(a)]$ for a given action-selection strategy as the *Count* of an action $a$ over the first $t$ steps, denoted as $Count_t(a)$. Let us refer to the Value difference between an action $a$ and the optimal action $a^*$ as the $Gap$ for $a$, denoted as $\Delta_a$, i.e,
$$\Delta_a = V^* - Q(a) $$
We define Total Regret as the sum-product (over actions) of $Count$s and $Gap$s, as follows:
\begin{align*}
L_T & = \sum_{t=1}^T \mathbb{E}[V^* - Q(A_t)]
 & = \sum_{a\in\mathcal{A}} \mathbb{E}[N_T(a)] \cdot (V^* - Q(a))
 & = \sum_{a\in\mathcal{A}} Count_T(a) \cdot \Delta_a
\end{align*}
A good algorithm ensures small $Count$s for large $Gap$s. The core challenge though is that *we don't know the $Gap$s*.

In this chapter, we implement (in code) a few different algorithms for the MAB problem. So let's invest in an abstract base class whose interface can be implemented by each of the algorithms we develop. The code for this abstract base class `MABBase` is shown below. Its constructor takes 3 inputs:

* `arm_distributions` which is a `Sequence` of `Distribution[float]`s, one for each arm. 
* `time_steps` which represents the number of time steps $T$
* `num_episodes` which represents the number of episodes we can run the algorithm on (each episode having $T$ time steps), in order to produce metrics to evaluate how well the algorithm does in expectation (averaged across the episodes).

Each of the algorithms we'd like to write simply needs to implement the `@abstractmethod get_episode_rewards_actions` which is meant to return a 1-D `ndarray` of actions taken by the algorithm across the $T$ time steps (for a single episode), and a 1-D `ndarray` of rewards produced in response to those actions.

\index{MABBase@\texttt{MABBase}}

```python
from rl.distribution import Distribution
from numpy import ndarray

class MABBase(ABC):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int
    ) -> None:
        self.arm_distributions: Sequence[Distribution[float]] = \
            arm_distributions
        self.num_arms: int = len(arm_distributions)
        self.time_steps: int = time_steps
        self.num_episodes: int = num_episodes

    @abstractmethod
    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        pass
```

We write the following self-explanatory methods for the abstract base class `MABBase`:

```python
from numpy import mean, vstack, cumsum, full, bincount

    def get_all_rewards_actions(self) -> Sequence[Tuple[ndarray, ndarray]]:
        return [self.get_episode_rewards_actions()
                for _ in range(self.num_episodes)]

    def get_rewards_matrix(self) -> ndarray:
        return vstack([x for x, _ in self.get_all_rewards_actions()])

    def get_actions_matrix(self) -> ndarray:
        return vstack([y for _, y in self.get_all_rewards_actions()])

    def get_expected_rewards(self) -> ndarray:
        return mean(self.get_rewards_matrix(), axis=0)

    def get_expected_cum_rewards(self) -> ndarray:
        return cumsum(self.get_expected_rewards())

    def get_expected_regret(self, best_mean) -> ndarray:
        return full(self.time_steps, best_mean) - self.get_expected_rewards()

    def get_expected_cum_regret(self, best_mean) -> ndarray:
        return cumsum(self.get_expected_regret(best_mean))

    def get_action_counts(self) -> ndarray:
        return vstack([bincount(ep, minlength=self.num_arms)
                       for ep in self.get_actions_matrix()])

    def get_expected_action_counts(self) -> ndarray:
        return mean(self.get_action_counts(), axis=0)
```

The above code is in the file [rl/chapter14/mab_base.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter14/mab_base.py).

Next, we cover some simple heuristic algorithms.

### Simple Algorithms

We consider algorithms that estimate a Q-Value $\hat{Q}_t(a)$ for each $a\in \mathcal{A}$, as an approximation to the true Q-Value $Q(a)$. The subscript $t$ in $\hat{Q}_t$ refers to the fact that this is an estimate after $t$ time steps that takes into account all of the information available up to $t$ time steps.

A natural way of estimating $\hat{Q}_t(a)$ is by *rewards-averaging*, i.e.,

$$\hat{Q}_t(a) = \frac 1 {N_t(a)} \sum_{s=1}^t R_s \cdot \mathbb{I}_{A_s=a}$$
where $\mathbb{I}$ refers to the indicator function.

#### Greedy and $\epsilon$-Greedy

\index{multi-armed bandits!greedy}

First, consider an algorithm that *never* explores (i.e., *always* exploits). This is known as the *Greedy Algorithm* which selects the action with highest estimated value, i.e.,
$$A_t = \argmax_{a\in \mathcal{A}} \hat{Q}_{t-1}(a)$$

As ever, $\argmax$ ties are broken with an arbitrary rule in prioritizing actions. We've noted in Chapter [-@sec:rl-control-chapter] that such an algorithm can lock into a suboptimal action forever (suboptimal $a$ is an action for which $\Delta_a > 0$). This results in $Count_T(a)$ being a linear function of $T$ for some suboptimal $a$, which means the Total Regret is a linear function of $T$ (we refer to this as *Linear Total Regret*). 

\index{multi-armed bandits!linear total regret}
\index{multi-armed bandits!epsilon greedy@$\epsilon$-greedy}

Now let's consider the $\epsilon$-greedy algorithm, which explores forever. At each time-step $t$:

* With probability $1-\epsilon$, select action equal to $\argmax_{a\in\mathcal{A}} \hat{Q}_{t-1}(a)$
* With probability $\epsilon$, select a random action (uniformly) from $\mathcal{A}$

A constant value of $\epsilon$ ensures a minimum regret proportional to the mean gap, i.e.,
$$ l_t \geq \frac {\epsilon} {|\mathcal{A}|} \sum_{a\in\mathcal{A}} \Delta_a$$

Hence, the $\epsilon$-Greedy algorithm also has Linear Total Regret. 

#### Optimistic Initialization

\index{multi-armed bandits!optimistic initialization}

Next, we consider a simple and practical idea: Initialize $\hat{Q}_0(a)$ to a high value for all $a\in \mathcal{A}$ and update $\hat{Q}_t$ by incremental-averaging. Starting with $N_0(a) \geq 0$ for all $a\in \mathcal{A}$, the updates at each time step $t$ are as follows:

$$N_t(A_t) = N_{t-1}(A_t) + 1$$
$$\hat{Q}_t(A_t) = \hat{Q}_{t-1}(A_t) + \frac {R_t - \hat{Q}_{t-1}(A_t)} {N_t(A_t)}$$

The idea here is that by setting a high initial value for the estimate of Q-Values (which we refer to as *Optimistic Initialization*), we encourage systematic exploration early on. Another way of doing optimistic initialization is to set a high value for $N_0(a)$ for all $a \in \mathcal{A}$, which likewise encourages systematic exploration early on. However, these optimistic initialization ideas only serve to promote exploration early on and eventually, one can still lock into a suboptimal action. Specifically, the Greedy algorithm together with optimistic initialization cannot be prevented from having Linear Total Regret in the general case. Likewise, the $\epsilon$-Greedy algorithm together with optimistic initialization cannot be prevented from having Linear Total Regret in the general case. But in practice, these simple ideas of doing optimistic initialization work quite well. 

\index{multi-armed bandits!linear total regret}

#### Decaying $\epsilon_t$-Greedy Algorithm

\index{multi-armed bandits!decaying epsilon greedy@decaying $\epsilon_t$-greedy}

The natural question that emerges is whether it is possible to construct an algorithm with Sublinear Total Regret in the general case. Along these lines, we consider an $\epsilon$-Greedy algorithm with $\epsilon$ decaying as time progresses. We call such an algorithm Decaying $\epsilon_t$-Greedy.

For any fixed $c > 0$, consider a decay schedule for $\epsilon_1, \epsilon_2, \ldots$ as follows:

$$d = \min_{a|\Delta_a > 0} \Delta_a$$
$$\epsilon_t = \min(1, \frac {c|\mathcal{A}|} {d^2 (t+1)})$$

It can be shown that this decay schedule achieves *Logarithmic* Total Regret. However, note that the above schedule requires advance knowledge of the gaps $\Delta_a$ (which by definition, is not known to the AI Agent). In practice, implementing *some* decay schedule helps considerably. Let's now write some code to implement Decaying $\epsilon_t$-Greedy algorithm along with Optimistic Initialization.

The class `EpsilonGreedy` shown below implements the interface of the abstract base class `MABBase`.  Its constructor inputs `arm_distributions`, `time_steps` and `num_episodes` are the inputs we have seen before (used to pass to the constructor of the abstract base class `MABBase`). `epsilon` and `epsilon_half_life` are the inputs used to specify the declining trajectory of $\epsilon_t$. `epsilon` refers to $\epsilon_0$ (initial value of $\epsilon$) and `epsilon_half_life` refers to the half life of an exponentially-decaying $\epsilon_t$ (used in the `@staticmethod get_epsilon_decay_func`). `count_init` and `mean_init` refer to values of $N_0$ and $\hat{Q}_0$, respectively. `get_episode_rewards_actions` implements `MABBase`'s `@abstracmethod` interface, and its code below should be self-explanatory.

\index{EpsilonGreedy@\texttt{EpsilonGreedy}}

```python
from operator import itemgetter
from rl.distribution import Distribution, Range, Bernoulli
from numpy import ndarray, empty

class EpsilonGreedy(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        epsilon: float,
        epsilon_half_life: float = 1e8,
        count_init: int = 0,
        mean_init: float = 0.,
    ) -> None:
        if epsilon < 0 or epsilon > 1 or \
                epsilon_half_life <= 1 or count_init < 0:
            raise ValueError

        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.epsilon_func: Callable[[int], float] = \
            EpsilonGreedy.get_epsilon_decay_func(epsilon, epsilon_half_life)
        self.count_init: int = count_init
        self.mean_init: float = mean_init

    @staticmethod
    def get_epsilon_decay_func(
        epsilon,
        epsilon_half_life
    ) -> Callable[[int], float]:

        def epsilon_decay(
            t: int,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life
        ) -> float:
            return epsilon * 2 ** -(t / epsilon_half_life)

        return epsilon_decay

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        counts: List[int] = [self.count_init] * self.num_arms
        means: List[float] = [self.mean_init] * self.num_arms
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.time_steps):
            max_action: int = max(enumerate(means), key=itemgetter(1))[0]
            epsl: float = self.epsilon_func(i)
            action: int = max_action if Bernoulli(1 - epsl).sample() else \
                Range(self.num_arms).sample()
            reward: float = self.arm_distributions[action].sample()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions
```

The above code is in the file [rl/chapter14/epsilon_greedy.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter14/epsilon_greedy.py).

![Total Regret Curves \label{fig:exp_cum_regret}](./chapter14/exp_cum_regret.png "Total Regret Curves"){height=7cm}

Figure \ref{fig:exp_cum_regret} shows the results of running the above code for 1000 time steps over 500 episodes, with $N_0$ and $\hat{Q}_0$ both set to 0. This graph was generated (see `__main__` in [rl/chapter14/epsilon_greedy.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter14/epsilon_greedy.py)) by creating 3 instances of `EpsilonGreedy`—the first with `epsilon` set to 0 (i.e., Greedy), the second with `epsilon` set to 0.12 and `epsilon_half_life` set to a very high value (i.e, $\epsilon$-Greedy, with no decay for $\epsilon$), and the third with `epsilon` set to 0.12 and `epsilon_half_life` set to 150 (i.e., Decaying $\epsilon_t$-Greedy). We can see that Greedy produces Linear Total Regret since it locks to a suboptimal value. We can also see that $\epsilon$-Greedy has higher total regret than Greedy initially because of exploration, and then settles in with Linear Total Regret, commensurate with the constant amount of exploration ($\epsilon = 0.12$ in this case). Lastly, we can see that Decaying $\epsilon_t$-Greedy produces Sublinear Total Regret as the initial effort spent in exploration helps identify the best action and as time elapses, the exploration keeps reducing so as to keep reducing the single-step regret.

\index{multi-armed bandits!linear total regret}
\index{multi-armed bandits!sublinear total regret}

In the `__main__` code in [rl/chapter14/epsilon_greedy.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter14/epsilon_greedy.py), we encourage you to experiment with different `arm_distributions`, `epsilon`, `epsilon_half_life`, `count_init` ($N_0$) and `mean_init` ($\hat{Q}_0$), observe how the graphs change, and develop better intuition for these simple algorithms.

### Lower Bound

\index{multi-armed bandits!Lai-Robbins lower bound|textbf}
\index{probability!Kullback-Leibler divergence}

It should be clear by now that we strive for algorithms with Sublinear Total Regret for any MAB problem (i.e., without any prior knowledge of the arm-reward distributions $\mathcal{R}^a$). Intuitively, the performance of any algorithm is determined by the similarity between the optimal arm's reward-distribution and the other arms's reward-distributions. Hard MAB problems are those with similar-distribution arms with different means $Q(a)$. This can be formally described in terms of the [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) $KL(\mathcal{R}^a||\mathcal{R}^{a^*})$ and gaps $\Delta_a$. Indeed, [Lai and Robbins](https://www.sciencedirect.com/science/article/pii/0196885885900028) [@lai-allocation] established a logarithmic lower bound for the Asymptotic Total Regret, with a factor expressed in terms of the KL Divergence $KL(\mathcal{R}^a||\mathcal{R}^{a^*})$ and gaps $\Delta_a$. Specifically,

\begin{theorem}[Lai and Robbins Lower-Bound]
Asymptotic Total Regret is at least logarithmic in the number of time steps, i.e., as $T \rightarrow \infty$, 
$$L_T \geq log T \sum_{a|\Delta_a > 0} \frac 1 {\Delta_a}  \geq \log T \sum_{a|\Delta_a > 0} \frac {\Delta_a} {KL(\mathcal{R}^a||\mathcal{R}^{a^*})}$$
\end{theorem}

\index{multi-armed bandits!logarithmic total regret}

This makes intuitive sense because it would be hard for an algorithm to have low total regret if the KL Divergence of arm reward-distributions (relative to the optimal arm's reward-distribution) are low (i.e., arms that look distributionally-similar to the optimal arm) but the Gaps (Expected Rewards of Arms relative to Optimal Arm) are not small—these are the MAB problem instances where the algorithm will have a hard time isolating the optimal arm simply from reward samples (we'd get similar sampling reward-distributions of arms), and suboptimal arm selections would inflate the Total Regret.

### Upper Confidence Bound Algorithms

\index{multi-armed bandits!optimism in the face of uncertainty}

Now we come to an important idea that is central to many algorithms for MAB. This idea goes by the catchy name of *Optimism in the Face of Uncertainty*. As ever, this idea is best understood with intuition first, followed by mathematical rigor. To develop intuition, imagine you are given 3 arms. You'd like to develop an estimate of $Q(a) = \mathbb{E}[r|a]$ for each of the 3 arms $a$. After playing the arms a few times, you start forming beliefs in your mind of what the $Q(a)$ might be for each arm. Unlike the simple algorithms we've seen so far where one averaged the sample rewards for each arm to maintain a $\hat{Q}(a)$ estimate for each $a$, here we maintain the sampling distribution of the mean rewards (for each $a$) that represents our (probabilistic) beliefs of what $Q(a)$ might be for each arm $a$.

![Q-Value Distributions \label{fig:q_value_distribution1}](./chapter14/q_value_distribution1.png "Q-Value Distributions"){height=7cm}

To keep things simple, let's assume the sampling distribution of the mean reward is a Gaussian distribution (for each $a$), and so we maintain an estimate of $\mu_a$ and $\sigma_a$ for each arm $a$ to represent the mean and standard deviation of the sampling distribution of mean reward for $a$. $\mu_a$ would be calculated as the average of the sample rewards seen so far for an arm $a$. $\sigma_a$ would be calculated as the standard error of the mean reward estimate, i.e., the sample standard deviation of the rewards seen so far, divided by the square root of the number of samples (for a given arm $a$). Let us say that after playing the arms a few times, we arrive at the Gaussian sampling distribution of mean reward for each of the 3 arms, as illustrated in Figure \ref{fig:q_value_distribution1}. Let's refer to the three arms as red, blue and green. The normal distributions in Figure \ref{fig:q_value_distribution1} show the red arm as the solid curve, the blue arm as the dashed curve and the green arm as the dotted-and-dashed curve. The blue arm has the highest $\sigma_a$. This could be either because the sample standard deviation is high or it could be because we have played the blue arm just a few times (remember the square root of number of samples appears in the denominator of the standard error calculation). Now looking at this figure, we have to decide which arm to select next. The intuition behind *Optimism in the Face of Uncertainty* is that the more uncertain we are about the $Q(a)$ for an arm $a$, the more important it is to play that arm. This is because more uncertainty on $Q(a)$ makes it more likely to be the best arm (all else being equal on the arms). The rough heuristic then would be to select the arm with the highest value of $\mu_a + c \cdot \sigma_a$ across the arms (for some fixed $c \in \mathbb{R}^+$). Thus, we are comparing (across actions) $c$ standard errors higher than the mean reward estimate (i.e., the upper-end of an appropriate confidence interval for the mean reward). In this figure, let's say $\mu_a + c \cdot \sigma_a$ is highest for the blue arm. So we play the blue arm, and let's say we get a somewhat low reward for the blue arm. This might do two things to the blue arm's sampling distribution—it can move blue's $\mu_a$ lower and it can also also lower blue's $\sigma_a$ (simply due to the fact that the number of blue arm samples has grown). With the new $\mu_a$ and $\sigma_a$ for the blue arm, let's say the updated sampling distributions are as shown in Figure \ref{fig:q_value_distribution2}. With the blue arm's sampling distribution of the mean reward narrower, let's say the red arm now has the highest $\mu_a + c \cdot \sigma_a$, and so we play the red arm. This process goes on until the sampling distributions get narrow enough to give us adequate confidence in the mean rewards for the actions (i.e., obtain confident estimates of $Q(a)$) so we can home in on the action with highest $Q(a)$.

It pays to emphasize that *Optimism in the Face of Uncertainty* is a great approach to resolve the Explore-Exploit dilemma because you gain regardless of whether the exploration due to Optimism produces large rewards or not. If it does produce large rewards, you gain immediately by collecting the large rewards. If it does not produce large rewards, you still gain by acquiring the knowledge that certain actions (that you have explored) might not be the best actions, which helps you in the long-run by focusing your attention on other actions.

![Q-Value Distributions \label{fig:q_value_distribution2}](./chapter14/q_value_distribution2.png "Q-Value Distributions"){height=7cm}


A formalization of the above intuition on *Optimism in the Face of Uncertainty* is the idea of *Upper Confidence Bounds* (abbreviated as UCB). The idea of UCB is that along with an estimate $\hat{Q}_t(a)$ (for each $a$ after $t$ time steps), we also maintain an estimate $\hat{U}_t(a)$ representing the upper confidence interval width for the mean reward of $a$ (after $t$ time steps) such that $Q(a) < \hat{Q}_t(a) + \hat{U}_t(a)$ with high probability. This naturally depends on the number of times that $a$ has been selected so far (call it $N_t(a)$). A small value of $N_t(a)$ would imply a large value of $\hat{U}_t(a)$ since the estimate of the mean reward would be fairly uncertain. On the other hand, a large value of $N_t(a)$ would imply a small value of $\hat{U}_t(a)$ since the estimate of the mean reward would be fairly certain. We refer to $\hat{Q}_t(a) + \hat{U}_t(a)$ as the *Upper Confidence Bound* (or simply UCB). The idea is to select the action that maximizes the UCB. Formally, the action $A_{t+1}$ selected for the next ($t+1$) time step is as follows:

\index{multi-armed bandits!upper confidence bound|textbf}

$$A_{t+1} = \argmax_{a\in\mathcal{A}} \{ \hat{Q}_t(a) + \hat{U}_t(a) \}$$

Next, we develop the famous UCB1 Algorithm. In order to do that, we tap into an important result from Statistics known as [Hoeffding's Inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality).

#### Hoeffding's Inequality

We state Hoeffding's Inequality without proof.

\index{probability!Hoeffding's inequality}

\begin{theorem}[Hoeffding's Inequality]
Let $X_1, \ldots, X_n$ be independent and identically distributed random variables in the range $[0,1]$, and let $$\bar{X}_n = \frac 1 n \sum_{i=1}^n X_i$$ be the sample mean. Then for any $u \geq 0$,
$$\mathbb{P}[\mathbb{E}[\bar{X}_n] > \bar{X}_n + u] \leq e^{-2nu^2}$$
\end{theorem}

We can apply Hoeffding's Inequality to MAB problem instances whose rewards have probability distributions with $[0,1]$-support. Conditioned on selecting action $a$ at time step $t$, sample mean $\bar{X}_n$ specializes to $\hat{Q}_t(a)$, and we set $n = N_t(a)$ and $u = \hat{U}_t(a)$. Therefore,
$$\mathbb{P}[Q(a) > \hat{Q}_t(a) + \hat{U}_t(a)] \leq e^{-2N_t(a) \cdot \hat{U}_t(a)^2}$$

Next, we pick a small probability $p$ for $Q(a)$ exceeding UCB $\hat{Q}_t(a) + \hat{U}_t(a)$. Now solve for $\hat{U}_t(a)$, as follows:
$$e^{-2N_t(a) \cdot \hat{U}_t(a)^2} = p \Rightarrow \hat{U}_t(a) = \sqrt{\frac {-\log p} {2 N_t(a)}}$$
We reduce $p$ as we observe more rewards, e.g., $p = t^{-\alpha}$ (for some fixed $\alpha > 0$). This ensures we select the optimal action as $t\rightarrow \infty$. Thus,
$$\hat{U}_t(a) = \sqrt{\frac {\alpha \log t} {2N_t(a)}}$$

#### UCB1 Algorithm

\index{multi-armed bandits!ucb1|(}
\index{multi-armed bandits!logarithmic total regret}

This yields the [UCB1 algorithm by Auer, Cesa-Bianchi, Fischer](https://homes.di.unimi.it/cesa-bianchi/Pubblicazioni/ml-02.pdf) [@Auer2002] for arbitrary-distribution arms bounded in $[0,1]$:
$$A_{t+1} = \argmax_{a\in \mathcal{A}} \{ \hat{Q}_t(a) + \sqrt{\frac {\alpha \log t} {2N_t(a)}} \}$$
It has been shown that the UCB1 Algorithm achieves logarithmic total regret asymptotically. Specifically,

\begin{theorem}[UCB1 Logarithmic Total Regret] As $T \rightarrow \infty$,
$$L_T \leq \sum_{a|\Delta_a > 0} \frac {4\alpha \cdot \log T} {\Delta_a} + \frac {2\alpha \cdot \Delta_a}{\alpha - 1}$$
\end{theorem}

Now let's implement the UCB1 Algorithm in code. The class `UCB1` below implements the interface of the abstract base class `MABBase`. We've implemented the below code for rewards range $[0,B]$ (adjusting the above UCB1 formula apropriately from $[0,1]$ range to $[0,B]$ range). $B$ is specified as the constructor input `bounds_range`. The constructor input `alpha` corresponds to the parameter $\alpha$ specified above. `get_episode_rewards_actions` implements `MABBase`'s `@abstracmethod` interface, and its code below should be self-explanatory.

\index{UCB1@\texttt{UCB1}}

```python
from numpy import ndarray, empty, sqrt, log
from operator import itemgetter

class UCB1(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        bounds_range: float,
        alpha: float
    ) -> None:
        if bounds_range < 0 or alpha <= 0:
            raise ValueError
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.bounds_range: float = bounds_range
        self.alpha: float = alpha

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.num_arms):
            ep_rewards[i] = self.arm_distributions[i].sample()
            ep_actions[i] = i
        counts: List[int] = [1] * self.num_arms
        means: List[float] = [ep_rewards[j] for j in range(self.num_arms)]
        for i in range(self.num_arms, self.time_steps):
            ucbs: Sequence[float] = [means[j] + self.bounds_range *
                                     sqrt(0.5 * self.alpha * log(i) /
                                          counts[j])
                                     for j in range(self.num_arms)]
            action: int = max(enumerate(ucbs), key=itemgetter(1))[0]
            reward: float = self.arm_distributions[action].sample()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions
```

The above code is in the file [rl/chapter14/ucb1.py](https://github.com/TikhonJelvis/RL-book/tree/master/rl/chapter14/ucb1.py). The code in `__main__` sets up a `UCB1` instance with 6 arms, each having a binomial distribution with $n=10$ and $p = \{0.4, 0.8, 0.1, 0.5, 0.9, 0.2\}$ for the 6 arms. When run with 1000 time steps, 500 episodes and $\alpha = 4$, we get the Total Regret Curve as shown in Figure \ref{fig:ucb1_total_regret_curve}.

![UCB1 Total Regret Curve \label{fig:ucb1_total_regret_curve}](./chapter14/ucb1_total_regret_curve.png "UCB1 Total Regret Curve"){height=7cm}

We encourage you to modify the code in `__main__` to model other distributions for the arms, examine the results obtained, and develop more intuition for the UCB1 Algorithm.

\index{multi-armed bandits!ucb1|)}

#### Bayesian UCB


The algorithms we have covered so far have not made any assumptions about the rewards distributions $\mathcal{R}^a$ (except for the range of the rewards). Now we assume that the rewards distributions are restricted to a family of analytically-tractable probability distributions, which enables us to make analytically-favorable inferences about the rewards distributions. Let us refer to the sequence of distributions $[\mathcal{R}^a|a \in \mathcal{A}]$ as $\mathcal{R}$. To be clear, the AI Agent (algorithm) does not have knowledge of $\mathcal{R}$ and aims to estimate $\mathcal{R}$ from the rewards data obtained upon performing actions. Bayesian Bandit Algorithms (abbreviated as *Bayesian Bandits*) achieve this by maintaining an estimate of the probability distribution over $\mathcal{R}$ based on rewards data seen for each of the selected arms. The idea is to compute the posterior distribution $\mathbb{P}[\mathcal{R}|H_t]$ by exploiting prior knowledge of $\mathbb{P}[\mathcal{R}]$, where $H_t = A_1,R_1, A_1, R_1, \ldots, A_t, R_t$ is the history. Note that the prior distribution $\mathbb{P}[\mathcal{R}]$ and the posterior distribution $\mathbb{P}[\mathcal{R}|H_t]$ are probability distributions over probability distributions (since each $\mathcal{R}^a$ in $\mathcal{R}$ is a probability distribution). This posterior distribution is then used to guide exploration. This leads to two types of algorithms:

* Upper Confidence Bounds (Bayesian UCB), which we give an example of below.
* Probability Matching, which we cover in the next section in the form of Thompson Sampling.

\index{probability!prior distribution}
\index{probability!posterior distribution}
\index{multi-armed bandits!bayesian ucb}

We get a better performance if our prior knowledge of $\mathbb{P}[\mathcal{R}]$ is accurate. A simple example of Bayesian UCB is to model independent Gaussian distributions. Assume the reward distribution is Gaussian: $\mathcal{R}^a(r) =\mathcal{N}(r;\mu_a, \sigma_a^2)$ for all $a \in \mathcal{A}$, where $\mu_a$ and $\sigma_a^2$ denote the mean and variance respectively of the Gaussian reward distribution of $a$. The idea is to compute a Gaussian posterior over $\mu_a,\sigma_a^2$, as follows:
$$\mathbb{P}[\mu_a, \sigma_a^2|H_t] \propto \mathbb{P}[\mu_a, \sigma_a^2] \cdot \prod_{t|A_t=a} \mathcal{N}(R_t;\mu_a, \sigma_a^2)$$
This posterior calculation can be performed in an incremental manner by updating $\mathbb{P}[\mu_{A_t}, \sigma_{A_t}^2|H_t]$ after each time step $t$ (observing $R_t$ after selecting action $A_t$). This incremental calculation with Bayesian updates to hyperparameters (parameters controlling the probability distributions of $\mu_a$ and $\sigma_a^2$) is described in detail in Section [-@sec:conjugate-prior-gaussian] in Appendix [-@sec:conjugate-priors-appendix]. 

\index{probability!prior distribution}
\index{probability!posterior distribution}

Given this posterior distribution for $\mu_a$ and $\sigma_a^2$ for all $a \in \mathcal{A}$ after each time step $t$, we select the action that maximizes the Expectation of "$c$ standard-errors above mean", i.e., 
$$A_{t+1} = \argmax_{a\in\mathcal{A}} \mathbb{E}_{\mathbb{P}[\mu_a,\sigma_a^2|H_t]}[\mu_a + \frac {c \cdot \sigma_a} {\sqrt{N_t(a)}}]$$

### Probability Matching

\index{multi-armed bandits!probability matching}

As mentioned in the previous section, calculating the posterior distribution $\mathbb{P}[\mathcal{R}|H_t]$ after each time step $t$ also enables a different approach known as *Probability Matching*.  The idea behind Probability Matching is to select an action $a$ probabilistically in proportion to the probability that $a$ might be the optimal action (based on the rewards data seen so far). Before describing Probability Matching formally, we illustrate the idea with a simple example to develop intuition.

Let us say we have only two actions $a_1$ and $a_2$. For simplicity, let us assume that the posterior distribution $\mathbb{P}[\mathcal{R}^{a_1}|H_t]$ has only two distribution outcomes (call them $\mathcal{R}^{a_1}_1$ and $\mathcal{R}^{a_1}_2$) and that the posterior distribution $\mathbb{P}[\mathcal{R}^{a_2}|H_t]$ also has only two distribution outcomes (call them $\mathcal{R}^{a_2}_1$ and $\mathcal{R}^{a_2}_2$). Typically, there will be an infinite (continuum) of distribution outcomes for $\mathbb{P}[\mathcal{R}|H_t]$—here we assume only two distribution outcomes for each of the actions' estimated conditional probability of rewards purely for simplicity so as to convey the intuition behind Probability Matching. Assume that $\mathbb{P}[\mathcal{R}^{a_1} = \mathcal{R}^{a_1}_1|H_t] = 0.7$ and $\mathbb{P}[\mathcal{R}^{a_1} = \mathcal{R}^{a_1}_2|H_t] = 0.3$, and that $\mathcal{R}^{a_1}_1$ has mean 5.0 and $\mathcal{R}^{a_1}_2$ has mean 10.0. Assume that $\mathbb{P}[\mathcal{R}^{a_2} = \mathcal{R}^{a_2}_1|H_t] = 0.2$ and $\mathbb{P}[\mathcal{R}^{a_2} = \mathcal{R}^{a_2}_2|H_t] = 0.8$, and that $\mathcal{R}^{a_2}_1$ has mean 2.0 and $\mathcal{R}^{a_2}_2$ has mean 7.0.

\index{probability!prior distribution}
\index{probability!posterior distribution}

Probability Matching calculates at each time step $t$ how often does each action $a$ have the maximum $\mathbb{E}[r|a]$ among all actions, across all the probabilistic outcomes for the posterior distribution $\mathbb{P}[\mathcal{R}|H_t]$, and then selects that action $a$ probabilistically in proportion to this calculation. Let's do this probability calculation for our simple case of two actions and two probabilistic outcomes each for the posterior distribution for each action. So here, we have 4 probabilistic outcomes when considering the two actions jointly, as follows:

* Outcome 1: $\mathcal{R}^{a_1}_1$ (with probability 0.7) and $\mathcal{R}^{a_2}_1$ (with probability 0.2). Thus, Outcome 1 has probability 0.7 * 0.2 = 0.14. In Outcome 1, $a_1$ has the maximum $\mathbb{E}[r|a]$ among all actions since $\mathcal{R}^{a_1}_1$ has mean 5.0 and $\mathcal{R}^{a_2}_1$ has mean 2.0.
* Outcome 2: $\mathcal{R}^{a_1}_1$ (with probability 0.7) and $\mathcal{R}^{a_2}_2$ (with probability 0.8). Thus, Outcome 2 has probability 0.7 * 0.8 = 0.56. In Outcome 2, $a_2$ has the maximum $\mathbb{E}[r|a]$ among all actions since $\mathcal{R}^{a_1}_1$ has mean 5.0 and $\mathcal{R}^{a_2}_2$ has mean 7.0.
* Outcome 3: $\mathcal{R}^{a_1}_2$ (with probability 0.3) and $\mathcal{R}^{a_2}_1$ (with probability 0.2). Thus, Outcome 3 has probability 0.3 * 0.2 = 0.06. In Outcome 3, $a_1$ has the maximum $\mathbb{E}[r|a]$ among all actions since $\mathcal{R}^{a_1}_2$ has mean 10.0 and $\mathcal{R}^{a_2}_1$ has mean 2.0.
* Outcome 4: $\mathcal{R}^{a_1}_2$ (with probability 0.3) and $\mathcal{R}^{a_2}_2$ (with probability 0.8). Thus, Outcome 4 has probability 0.3 * 0.8 = 0.24. In Outcome 4, $a_1$ has the maximum $\mathbb{E}[r|a]$ among all actions since $\mathcal{R}^{a_1}_2$ has mean 10.0 and $\mathcal{R}^{a_2}_2$ has mean 7.0.

Thus, $a_1$ has the maximum $\mathbb{E}[r|a]$ among the two actions in Outcomes 1, 3 and 4, amounting to a total outcomes probability of 0.14 + 0.06 + 0.24 = 0.44, and $a_2$ has the maximum $\mathbb{E}[r|a]$ among the two actions only in Outcome 2, which has an outcome probability of 0.56. Therefore, in the next time step ($t+1$), the Probability Matching method will select action $a_1$ with probability 0.44 and $a_2$ with probability 0.56.

Generalizing this Probability Matching method to an arbitrary number of actions and to an arbitrary number of probabilistic outcomes for the conditional reward distributions for each action, we can write the probabilistic selection of actions at time step $t+1$ as:

\begin{equation}
\mathbb{P}[A_{t+1}|H_t] = \mathbb{P}_{\mathcal{D}_t \sim \mathbb{P}[\mathcal{R}|H_t]}[\mathbb{E}_{\mathcal{D}_t}[r|A_{t+1}] > \mathbb{E}_{\mathcal{D}_t}[r|a] \text{ for all } a \neq A_{t+1}]
\label{eq:probability-matching}
\end{equation}
where $\mathcal{D}_t$ refers to a particular random outcome of a distribution of rewards for each action, drawn from the posterior distribution $\mathbb{P}[\mathcal{R}|H_t]$. As ever, ties between actions are broken with an arbitrary rule prioritizing actions.

Note that the Probability Matching method is also based on the principle of *Optimism in the Face of Uncertainty* because an action with more uncertainty in its mean reward is more likely to have the highest mean reward among all actions (all else being equal), and hence deserves to be selected more frequently.

We see that the Probability Matching approach is mathematically disciplined in driving towards cumulative reward maximization while balancing exploration and exploitation. However, the right-hand-side of Equation \ref{eq:probability-matching} can be difficult to compute analytically from the posterior distributions. We resolve this difficulty with a sampling approach to Probability Matching known as *Thompson Sampling*.

#### Thompson Sampling

\index{multi-armed bandits!Thompson sampling}

We can reformulate the right-hand-side of Equation \ref{eq:probability-matching} as follows:
\begin{align*}
\mathbb{P}[A_{t+1}|H_t] & = \mathbb{P}_{\mathcal{D}_t \sim \mathbb{P}[\mathcal{R}|H_t]}[\mathbb{E}_{\mathcal{D}_t}[r|A_{t+1}] > \mathbb{E}_{\mathcal{D}_t}[r|a] \text{for all } a \neq A_{t+1}] \\
& =\mathbb{E}_{\mathcal{D}_t \sim \mathbb{P}[\mathcal{R}|H_t]}[\mathbb{I}_{A_{t+1}=\argmax_{a\in\mathcal{A}} \mathbb{E}_{\mathcal{D}_t}[r|a]}]
\end{align*}
where $\mathbb{I}$ refers to the indicator function. This reformulation in terms of an *Expectation* is convenient because we can estimate the Expectation by sampling various $\mathcal{D}_t$ probability distributions and for each sample of $\mathcal{D}_t$, we simply check if an action has the best mean reward (compared to other actions) under the distribution $\mathcal{D}_t$. This sampling-based approach to Probability Matching is known as *Thompson Sampling*. Specifically, Thompson Sampling performs the following calculations for time step $t+1$:

* Compute the posterior distribution $\mathbb{P}[\mathcal{R}|H_t]$ by performing Bayesian updates of the hyperparameters that govern the estimated probability distributions of the parameters of the reward distributions for each action.
* *Sample* a joint (across actions) rewards distribution $\mathcal{D}_t$ from the posterior distribution $\mathbb{P}[\mathcal{R}|H_t]$.
* Calculate a sample Action-Value function with sample $\mathcal{D}_t$ as:
$$\hat{Q}_t(a) = \mathbb{E}_{\mathcal{D}_t}[r|a]$$
* Select the action (for time step $t+1$) that maximizes this sample Action-Value function:
$$A_{t+1} = \argmax_{a\in\mathcal{A}} \hat{Q}_t(a)$$

\index{probability!prior distribution}
\index{probability!posterior distribution}
\index{multi-armed bandits!logarithmic total regret}

It turns out that Thompson Sampling achieves the Lai-Robbins lower bound for Logarithmic Total Regret. To learn more about Thompson Sampling, we refer you to [the excellent tutorial on Thompson Sampling by Russo, Roy, Kazerouni, Osband, Wen](https://arxiv.org/abs/1707.02038) [@Russo_2018].

Now we implement Thompson Sampling by assuming a Gaussian distribution of rewards for each action. The posterior distributions for each action are produced by performing Bayesian updates of the hyperparameters that govern the estimated [Gaussian-Inverse-Gamma Probability Distributions](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution) of the parameters of the Gaussian reward distributions for each action. Section [-@sec:conjugate-prior-gaussian] of Appendix [-@sec:conjugate-priors-appendix] describes the Bayesian updates of the hyperparameters $\theta, \alpha, \beta$, and the code below implements this update in the variable `bayes` in method `get_episode_rewards_actions` (this method implements the `@abstractmethod` interface of abstract base class `MABBase`). The sample mean rewards are obtained by invoking the `sample` method of `Gaussian` and `Gamma` classes, and assigned to the variable `mean_draws`. The variable `theta` refers to the hyperparameter $\theta$, the variable `alpha` refers to the hyperparameter $\alpha$, and the variable `beta` refers to the hyperparameter $\beta$. The rest of the code in the method `get_episode_rewards_actions` should be self-explanatory.

\index{ThompsonSamplingGaussian@\texttt{ThompsonSamplingGaussian}}
\index{probability!normal distribution}

```python
from rl.distribution import Gaussian, Gamma
from operator import itemgetter
from numpy import ndarray, empty, sqrt

class ThompsonSamplingGaussian(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Gaussian],
        time_steps: int,
        num_episodes: int,
        init_mean: float,
        init_stdev: float
    ) -> None:
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.theta0: float = init_mean
        self.n0: int = 1
        self.alpha0: float = 1
        self.beta0: float = init_stdev * init_stdev

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        bayes: List[Tuple[float, int, float, float]] =\
            [(self.theta0, self.n0, self.alpha0, self.beta0)] * self.num_arms

        for i in range(self.time_steps):
            mean_draws: Sequence[float] = [Gaussian(
                mu=theta,
                sigma=1 / sqrt(n * Gamma(alpha=alpha, beta=beta).sample())
            ).sample() for theta, n, alpha, beta in bayes]
            action: int = max(enumerate(mean_draws), key=itemgetter(1))[0]
            reward: float = self.arm_distributions[action].sample()
            theta, n, alpha, beta = bayes[action]
            bayes[action] = (
                (reward + n * theta) / (n + 1),
                n + 1,
                alpha + 0.5,
                beta + 0.5 * n / (n + 1) * (reward - theta) * (reward - theta)
            )
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions
```

The above code is in the file [rl/chapter14/ts_gaussian.py](https://github.com/TikhonJelvis/RL-book/tree/master/rl/chapter14/ts_gaussian.py). The code in `__main__` sets up a `ThompsonSamplingGaussian` instance with 6 arms, each having a Gaussian rewards distribution. When run with 1000 time steps and 500 episodes, we get the Total Regret Curve as shown in Figure \ref{fig:ts_gaussian_total_regret_curve}.

![Thompson Sampling (Gaussian) Total Regret Curve \label{fig:ts_gaussian_total_regret_curve}](./chapter14/ts_gaussian_total_regret_curve.png "Thompson Sampling (Gaussian) Total Regret Curve"){height=7cm}

We encourage you to modify the code in `__main__` to try other mean and variance settings for the Gaussian reward distributions of the arms, examine the results obtained, and develop more intuition for Thompson Sampling for Gaussians.

\index{probability!prior distribution}
\index{probability!posterior distribution}

Now we implement Thompson Sampling by assuming a Bernoulli distribution of rewards for each action. The posterior distributions for each action are produced by performing Bayesian updates of the hyperparameters that govern the estimated [Beta Probability Distributions](https://en.wikipedia.org/wiki/Beta_distribution) of the parameters of the Bernoulli reward distributions for each action. Section [-@sec:conjugate-prior-bernoulli] of Appendix [-@sec:conjugate-priors-appendix] describes the Bayesian updates of the hyperparameters $\alpha$ and $\beta$, and the code below implements this update in the variable `bayes` in method `get_episode_rewards_actions` (this method implements the `@abstractmethod` interface of abstract base class `MABBase`). The sample mean rewards are obtained by invoking the `sample` method of the `Beta` class, and assigned to the variable `mean_draws`. The variable `alpha` refers to the hyperparameter $\alpha$ and the variable `beta` refers to the hyperparameter $\beta$. The rest of the code in the method `get_episode_rewards_actions` should be self-explanatory.

\index{ThompsonSamplingBernoulli@\texttt{ThompsonSamplingBernoulli}}
\index{probability!Bernoulli distribution}

```python
from rl.distribution import Bernoulli, Beta
from operator import itemgetter
from numpy import ndarray, empty

class ThompsonSamplingBernoulli(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Bernoulli],
        time_steps: int,
        num_episodes: int
    ) -> None:
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        bayes: List[Tuple[int, int]] = [(1, 1)] * self.num_arms

        for i in range(self.time_steps):
            mean_draws: Sequence[float] = \
                [Beta(alpha=alpha, beta=beta).sample() for alpha, beta in bayes]
            action: int = max(enumerate(mean_draws), key=itemgetter(1))[0]
            reward: float = float(self.arm_distributions[action].sample())
            alpha, beta = bayes[action]
            bayes[action] = (alpha + int(reward), beta + int(1 - reward))
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions
```

The above code is in the file [rl/chapter14/ts_bernoulli.py](https://github.com/TikhonJelvis/RL-book/tree/master/rl/chapter14/ts_bernoulli.py). The code in `__main__` sets up a `ThompsonSamplingBernoulli` instance with 6 arms, each having a Bernoulli rewards distribution. When run with 1000 time steps and 500 episodes, we get the Total Regret Curve as shown in Figure \ref{fig:ts_bernoulli_total_regret_curve}.

![Thompson Sampling (Bernoulli) Total Regret Curve \label{fig:ts_bernoulli_total_regret_curve}](./chapter14/ts_bernoulli_total_regret_curve.png "Thompson Sampling (Bernoulli) Total Regret Curve"){height=7cm}

We encourage you to modify the code in `__main__` to try other mean settings for the Bernoulli reward distributions of the arms, examine the results obtained, and develop more intuition for Thompson Sampling for Bernoullis.

### Gradient Bandits

\index{multi-armed bandits!gradient bandits|(}

Now we cover a MAB algorithm that is similar to Policy Gradient for MDPs. This MAB algorithm's action selection is randomized and the action selection probabilities are constructed through Gradient Ascent (much like Stochastic Policy Gradient for MDPs). This MAB Algorithm and its variants are cheekily referred to as *Gradient Bandits*. Our coverage below follows the coverage of [Gradient Bandit algorithm in the RL book by Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html) [@Sutton1998].

The basic idea is that we have $m$ *Score* parameters (to be optimized), one for each action, denoted as $\{s_a|a \in \mathcal{A}\}$ that define the action-selection probabilities, which in turn defines an *Expected Reward* Objective function to be maximized, as follows:

$$J(s_{a_1}, \ldots, s_{a_m}) = \sum_{a\in\mathcal{A}} \pi(a) \cdot \mathbb{E}[r|a]$$
where $\pi: \mathcal{A} \rightarrow [0, 1]$ refers to the function for action-selection probabilities, that is defined as follows:
$$\pi(a) = \frac {e^{s_a}} {\sum_{b\in \mathcal{A}} e^{s_b}} \text{ for all } a \in \mathcal{A}$$

\index{probability!score}
\index{probability!softmax distribution}

The *Score* parameters are meant to represent the relative value of actions based on the rewards seen until a certain time step, and are adjusted appropriately after each time step (using Gradient Ascent). Note that $\pi(\cdot)$ is a [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) of the *Score* parameters.

Gradient Ascent moves the *Score* parameters $s_a$ (and hence, action probabilities $\pi(a)$) in the direction of the gradient of the objective function $J(s_{a_1}, \ldots, s_{a_m})$ with respect to $(s_{a_1}, \ldots, s_{a_m})$. To construct this gradient of $J(\cdot)$, we calculate $\pdv{J}{s_a}$ for each $a\in \mathcal{A}$, as follows:
\begin{align*}
\pdv{J}{s_a} & = \pdv{(\sum_{a'\in\mathcal{A}} \pi(a') \cdot \mathbb{E}[r|a'])}{s_a} \\
 & = \sum_{a'\in\mathcal{A}} \mathbb{E}[r|a'] \cdot \pdv{\pi(a')} {s_a} \\
 & = \sum_{a'\in\mathcal{A}} \pi(a') \cdot \mathbb{E}[r|a'] \cdot \pdv{\log \pi(a')} {s_a} \\
 & = \mathbb{E}_{a'\sim \pi, r\sim \mathcal{R}^{a'}}[r \cdot \pdv{\log \pi(a')} {s_a}]
\end{align*}
We know from standard softmax-function calculus that:
$$\pdv{\log \pi(a')} {s_a} = \pdv{(\log\frac {e^{s_{a'}}} {\sum_{b\in \mathcal{A}} e^{s_b}})}{s_a} = \mathbb{I}_{a=a'} - \pi(a)$$
Therefore, $\pdv{J}{s_a}$ can be re-written as:
$$\pdv{J}{s_a} =\mathbb{E}_{a'\sim \pi, r\sim \mathcal{R}^{a'}}[r \cdot  (\mathbb{I}_{a=a'} - \pi(a))]$$
At each time step $t$, we approximate the gradient with the $(A_t, R_t)$ sample as:
$$R_t \cdot (\mathbb{I}_{a=A_t} - \pi_t(a)) \text{ for all } a \in \mathcal{A}$$
$\pi_t(a)$ is the probability of selecting action $a$ at time step $t$, derived from the *Score* $s_t(a)$ at time step $t$.

We can reduce the variance of this estimate with a baseline $B$ that is independent of $a$, as follows:
$$(R_t -B) \cdot (\mathbb{I}_{a=A_t} - \pi_t(a)) \mbox{ for all } a \in \mathcal{A}$$
This doesn't introduce any bias in the estimate of the gradient of $J(\cdot)$ because:
\begin{align*}
\mathbb{E}_{a'\sim \pi}[B \cdot (\mathbb{I}_{a=a'} - \pi(a))] & = \mathbb{E}_{a'\sim \pi}[B \cdot \pdv{\log \pi(a')} {s_a}] \\
& = B \cdot \sum_{a'\in\mathcal{A}} \pi(a') \cdot \pdv{\log \pi(a')} {s_a} \\
& = B \cdot \sum_{a'\in\mathcal{A}} \pdv{\pi(a')} {s_a} \\
& = B \cdot \pdv{(\sum_{a'\in\mathcal{A}} \pi(a'))}{s_a} \\
& = B \cdot \pdv{1}{s_a} \\
& = 0
\end{align*}

We can use $B = \bar{R}_t = \frac 1 t \sum_{s=1}^t R_s$ (average of all rewards obtained until time step $t$). So, the update to scores $s_t(a)$ for all $a\in\mathcal{A}$ is:
$$s_{t+1}(a) = s_t(a) + \alpha \cdot (R_t - \bar{R}_t) \cdot (\mathbb{I}_{a=A_t} - \pi_t(a))$$

It should be noted that this Gradient Bandit algorithm and its variant Gradient Bandit algorithms are simply a special case of policy gradient-based RL algorithms.

Now let's write some code to implement this Gradient Algorithm. Apart from the usual constructor inputs `arm_distributions`, `time_steps` and `num_episodes` that are passed along to the constructor of the abstract base class `MABBase`, `GradientBandits`' constructor also takes as input `learning_rate` (specifying the initial learning rate) and `learning_rate_decay` (specifying the speed at which the learning rate decays), which influence how the variable `step_size` is set at every time step. The variable `scores` represents $s_t(a)$ for all $a \in \mathcal{A}$ and the variable `probs` represents $\pi_t(a)$ for all $a \in \mathcal{A}$. The rest of the code below should be self-explanatory, based on the above description of the calculations.

\index{GradientBandits@\texttt{GradientBandits}}

```python
from rl.distribution import Distribution, Categorical
from operator import itemgetter
from numpy import ndarray, empty, exp

class GradientBandits(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        learning_rate: float,
        learning_rate_decay: float
    ) -> None:
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.learning_rate: float = learning_rate
        self.learning_rate_decay: float = learning_rate_decay

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        scores: List[float] = [0.] * self.num_arms
        avg_reward: float = 0.

        for i in range(self.time_steps):
            max_score: float = max(scores)
            exp_scores: Sequence[float] = [exp(s - max_score) for s in scores]
            sum_exp_scores = sum(exp_scores)
            probs: Sequence[float] = [s / sum_exp_scores for s in exp_scores]
            action: int = Categorical(
                {i: p for i, p in enumerate(probs)}
            ).sample()
            reward: float = self.arm_distributions[action].sample()
            avg_reward += (reward - avg_reward) / (i + 1)
            step_size: float = self.learning_rate *\
                (i / self.learning_rate_decay + 1) ** -0.5
            for j in range(self.num_arms):
                scores[j] += step_size * (reward - avg_reward) *\
                             ((1 if j == action else 0) - probs[j])

            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions
```

The above code is in the file [rl/chapter14/gradient_bandits.py](https://github.com/TikhonJelvis/RL-book/tree/master/rl/chapter14/gradient_bandits.py). The code in `__main__` sets up a `GradientBandits` instance with 6 arms, each having a Gaussian reward distribution. When run with 1000 time steps and 500 episodes, we get the Total Regret Curve as shown in Figure \ref{fig:gradient_bandits_total_regret_curve}.

![Gradient Algorithm Total Regret Curve \label{fig:gradient_bandits_total_regret_curve}](./chapter14/gradient_bandits_total_regret_curve.png "Gradient Algorithm Total Regret Curve"){height=7cm}

We encourage you to modify the code in `__main__` to try other mean and standard deviation settings for the Gaussian reward distributions of the arms, examine the results obtained, and develop more intuition for this Gradient Algorithm.

\index{multi-armed bandits!gradient bandits|)}

### Horse Race

We've implemented several algorithms for the MAB problem. Now it's time for a competition between them, that we will call a *Horse Race*. In this Horse Race, we will compare the *Total Regret* across the algorithms, and we will also examine the number of times the different arms get pulled by the various algorithms. We expect a good algorithm to have small total regret and we expect a good algorithm to pull the arms with high *Gap*s few number of times and pull the arms with low (and zero) gaps large number of times.

The code in the file [rl/chapter14/plot_mab_graphs.py](https://github.com/TikhonJelvis/RL-book/tree/master/rl/chapter14/plot_mab_graphs.py) has a function to run a horse race for Gaussian arms with the following algorithms:

* Greedy with Optimistic Initialization
* $\epsilon$-Greedy
* Decaying $\epsilon_t$-Greedy
* Thompson Sampling
* Gradient Bandit

![Gaussian Horse Race—Total Regret Curves \label{fig:gaussian_horse_race_total_regret}](./chapter14/gaussian_horse_race_total_regret.png "Gaussian Horse Race—Total Regret Curves"){height=7cm}

Running this horse race for 7 Gaussian arms with 500 time steps, 500 episodes and the settings as specified in the file rl/chapter14/plot_mab_graphs.py, we obtain Figure \ref{fig:gaussian_horse_race_total_regret} for the Total Regret Curves for each of these algorithms.

![Gaussian Horse Race—Arms Count \label{fig:gaussian_horse_race_arms_count}](./chapter14/gaussian_horse_race_arms_count.png "Gaussian Horse Race—Arms Count"){height=7cm}

Figure \ref{fig:gaussian_horse_race_arms_count} shows the number of times each arm is pulled (for each of these algorithms). The X-axis is sorted by the mean of the reward distributions of the arms. For each arm, the left-to-right order of the arm-pulls count is the order in which the 5 MAB algorithms are listed above. As we can see, the arms with low means are pulled only a few times and the arms with high means are pulled often.

The file rl/chapter14/plot_mab_graphs.py also has a function to run a horse race for Bernoulli arms with the following algorithms:

* Greedy with Optimistic Initialization
* $\epsilon$-Greedy
* Decaying $\epsilon_t$-Greedy
* UCB1
* Thompson Sampling
* Gradient Bandit

![Bernoulli Horse Race—Total Regret Curves \label{fig:bernoulli_horse_race_total_regret}](./chapter14/bernoulli_horse_race_total_regret.png "Bernoulli Horse Race—Total Regret Curves"){height=7cm}

Running this horse race for 9 Bernoulli arms with 500 time steps, 500 episodes and the settings as specified in the file rl/chapter14/plot_mab_graphs.py, we obtain Figure \ref{fig:bernoulli_horse_race_total_regret} for the Total Regret Curves for each of these algorithms.

![Bernoulli Horse Race—Arms Count \label{fig:bernoulli_horse_race_arms_count}](./chapter14/bernoulli_horse_race_arms_count.png "Bernoulli Horse Race—Arms Count"){height=7cm}

Figure \ref{fig:bernoulli_horse_race_arms_count} shows the number of times each arm is pulled (for each of the algorithms). The X-axis is sorted by the mean of the reward distributions of the arms. For each arm, the left-to-right order of the arm-pulls count is the order in which the 6 MAB algorithms are listed above. As we can see, the arms with low means are pulled only a few times and the arms with high means are pulled often.

We encourage you to experiment with the code in [rl/chapter14/plot_mab_graphs.py](https://github.com/TikhonJelvis/RL-book/tree/master/rl/chapter14/plot_mab_graphs.py): try different arm distributions, try different input parameters for each of the algorithms, plot the graphs, and try to explain the relative performance of the algorithms (perhaps by writing some more diagnostics code). This will help build tremendous intuition on the pros and cons of these algorithms.

### Information State Space MDP

\index{multi-armed bandits!information state space MDP|(}

We had mentioned earlier in this chapter that although a MAB problem is not posed as an MDP, the AI Agent could maintain relevant features of the history (of actions taken and rewards obtained) as its *State*, which would help the AI Agent in making the arm-selection (action) decision. So the AI Agent treats the MAB problem as an MDP and the arm-selection action is essentially a (*Policy*) function of the agent's *State*. One can then arrive at the Optimal arm-selection strategy by solving the Control problem of this MDP with an appropriate Planning or Learning algorithm. The representation of *State* as relevant features of history is known as *Information State* (to indicate that the agent captures all of the relevant information known so far in the *State* of the modeled MDP). Before we explain this *Information State Space MDP* approach in more detail, it pays to develop an intuitive understanding of the *Value of Information*.

The key idea is that *Exploration* enables the agent to acquire information, which in turn enables the agent to make more informed decisions as far as its future arm-selection strategy is concerned. The natural question to ask then is whether we can quantify the value of this information that can be acquired by *Exploration*. In other words, how much would a decision-maker be willing to pay to acquire information (through exploration), prior to making a decision? Vaguely speaking, the decision-maker should be paying an amount equal to the gains in long-term (accumulated) reward that can be obtained upon getting the information, less the sacrifice of excess immediate reward one would have obtained had one exploited rather than explored. We can see that this approach aims to settle the explore-exploit trade-off in a mathematically rigorous manner by establishing the *Value of Information*. Note that information gain is higher in a more uncertain situation (all else being equal). Therefore, it makes sense to explore uncertain situations more. By formalizing the value of information, we can trade-off exploration and exploitation *optimally*.

Now let us formalize the approach of treating a MAB as an Information State Space MDP. After each time step of a MAB, we construct an *Information State* $\tilde{s}$, which comprises of relevant features of the history until that time step. Essentially, $\tilde{s}$ summarizes all of the information accumulated so far that is pertinent to be able to predict the reward distribution for each action. Each action $a$ causes a transition to a new information state $\tilde{s}'$ (by adding information about the reward obtained after performing action $a$), with probability $\tilde{\mathcal{P}}(\tilde{s}, a, \tilde{s}')$. Note that this probability depends on the reward probability function $\mathcal{R}^a$ of the MAB. Moreover, the MAB reward $r$ obtained upon performing action $a$ constitutes the Reward of the Information State Space MDP for that time step. Putting all this together, we have an MDP $\tilde{M}$ in information state space as follows:

\index{Markov decision process!state space}
\index{Markov decision process!action space}
\index{Markov decision process!state transition probability function}
\index{Markov decision process!reward function}
\index{discount factor}

* Denote the Information State Space of $\tilde{M}$ as $\tilde{\mathcal{S}}$.
* The Action Space of $\tilde{M}$ is the action space of the given MAB: $\mathcal{A}$.
* The State Transition Probability function of $\tilde{M}$ is $\tilde{\mathcal{P}}$.
* The Reward function of $\tilde{M}$ is given by the Reward probability function $\mathcal{R}^a$ of the MAB.
* Discount Factor $\gamma = 1$.

The key point to note is that since $\mathcal{R}^a$ is unknown to the AI Agent in the MAB problem, the State Transition Probability function and the Reward function of the Information State Space MDP $\tilde{M}$ are unknown to the AI Agent. However, at any given time step, the AI Agent can utilize the information within $\tilde{s}$ to form an estimate of $\mathcal{R}^a$, which in turn gives estimates of the State Transition Probability function and the Reward function of the Information State Space MDP $\tilde{M}$.

Note that $\tilde{M}$ will typically be a fairly complex MDP over an infinite number of information states, and hence is not easy to solve. However, since it is after all an MDP, we can use Dynamic Programming or Reinforcement Learning algorithms to arrive at the Optimal Policy, which prescribes the optimal MAB action to take at that time step. If a Dynamic Programming approach is taken, then after each time step, as new information arrives (in the form of the MAB reward in response to the action taken), the estimates of the State Transition probability function and the Reward function change, meaning the Information State Space MDP to be solved changes, and consequently the Action-Selection strategy for the MAB problem (prescribed by the Optimal Policy of the Information State Space MDP) changes. A common approach is to treat the Information State Space MDP as a *Bayes-Adaptive MDP*. Specifically, if we have $m$ arms $a_1, \ldots, a_m$, the state $\tilde{s}$ is modeled as $(\tilde{s_{a_1}}, \ldots, \tilde{s_{a_m}})$ such that $\tilde{s_{a}}$ for any $a \in \mathcal{A}$ represents a posterior probability distribution over $\mathcal{R}^a$, which is Bayes-updated after observing the reward upon each pull of the arm $a$. This Bayes-Adaptive MDP can be tackled with the highly-celebrated Dynamic Programming method known as [Gittins Index](https://en.wikipedia.org/wiki/Gittins_index), which was introduced in a [1979 paper by Gittins](http://apdalab.org/mnk/RL/Gittins_1979.pdf) [@gittins1979bandit]. The Gittins Index approach finds the Bayes-optimal explore-exploit trade-off with respect to the prior distribution.

\index{probability!prior distribution}
\index{probability!posterior distribution}

\index{multi-armed bandits!bayes-adaptive MDP}
\index{multi-armed bandits!Gittins index}

\index{probability!Bernoulli distribution}

To grasp the concept of Information State Space MDP, let us consider a Bernoulli Bandit problem with $m$ arms with arm $a$'s reward probability distribution $\mathcal{R}^a$ given by the Bernoulli distribution $\mathcal{B}(\mu_a)$, where $\mu_a \in [0, 1]$ (i.e., reward = 1 with probability $\mu_a$, and reward = 0 with probability $1 - \mu_a$). If we denote the $m$ arms by $a_1, a_2, \ldots, a_m$, then the information state is $\tilde{s} = (\alpha_{a_1}, \beta_{a_1}, \alpha_{a_2}, \beta_{a_2}\ldots, \alpha_{a_m}, \beta_{a_m})$, where $\alpha_a$ is the number of pulls of arm $a$ (so far) for which the reward was 1 and $\beta_a$ is the number of pulls of arm $a$ (so far) for which the reward was 0. Note that by the Law of Large Numbers, in the long-run, $\frac {\alpha_a} {\alpha_a + \beta_a} \rightarrow \mu_a$.

\index{probability!beta distribution}

We can treat this as a Bayes-adaptive MDP as follows: We model the prior distribution over $\mathcal{R}^a$ as the Beta Distribution $Beta(\alpha_a, \beta_a)$ over the unknown parameter $\mu_a$. Each time arm $a$ is pulled, we update the posterior for $\mathcal{R}^a$ as:
\index{probability!prior distribution}
\index{probability!posterior distribution}

* $Beta(\alpha_a+1, \beta_a)$ if $r=1$
* $Beta(\alpha_a, \beta_a+1)$ if $r=0$

Note that the component $(\alpha_a, \beta_a)$ within the information state provides the model $Beta(\alpha_a, \beta_a)$ as the probability distribution over $\mu_a$. Moreover, note that each state transition (updating either $\alpha_a$ or $\beta_a$ by 1) is essentially a Bayesian model update (Section [-@sec:conjugate-prior-bernoulli] in Appendix [-@sec:conjugate-priors-appendix] provides details of Bayesian updates to a Beta distribution over a Bernoulli parameter).

Note that in general, an exact solution to a Bayes-adaptive MDP is typically intractable. In 2014, [Guez, Heess, Silver, Dayan](https://proceedings.neurips.cc/paper/2014/file/839ab46820b524afda05122893c2fe8e-Paper.pdf) [@conf/nips/GuezHSD14] came up with a Simulation-based Search method, which involves a forward search in information state space using simulations from current information state, to solve a Bayes-adaptive MDP.

\index{multi-armed bandits!information state space MDP|)}
\index{multi-armed bandits|)}

### Extending to Contextual Bandits and RL Control

\index{contextual bandits|(}

A Contextual Bandit problem is a natural extension of the MAB problem, by introducing the concept of *Context* that has an influence on the rewards probability distribution for each arm. Before we provide a formal definition of a Contextual Bandit problem, we will provide an intuitive explanation with a canonical example. Consider the problem of showing a banner advertisement on a web site where there is a choice of displaying one among $m$ different advertisements at a time. If the user clicks on the advertisement, there is a reward of 1 (if the user doesn't click, the reward is 0). The selection of the advertisement to display is the arm-selection (out of $m$ arms, i.e., advertisements). This seems like a standard MAB problem, except that on a web site, we don't have a single user. In each round, a random user (among typically millions of users) appears. Each user will have their own characteristics of how they would respond to advertisements, meaning the rewards probability distribution for each arm would depend on the user. We refer to the user characteristics (as relevant to their likelihood to respond to specific advertisements) as the *Context*. This means, the *Context* influences the rewards probability distribution for each arm. This is known as the *Contextual Bandit* problem, which we formalize below:

\begin{definition}
A {\em Contextual Bandit} comprises of:
\begin{itemize}
\item A finite set of {\em Action}s $\mathcal{A}$ (known as the "arms").
\item A probability distribution $\mathcal{C}$ over {\em Context}s,  defined as:
$$\mathcal{C}(c) = \mathbb{P}[c] \text{ for all Contexts } c$$
\item Each pair of a context $c$ and an action ("arm") $a \in \mathcal{A}$ is associated with a probability distribution over $\mathbb{R}$ (unknown to the AI Agent) denoted as $\mathcal{R}^a_c$, defined as:
$$\mathcal{R}^a_c(r) = \mathbb{P}[r|c,a] \text{ for all } r \in \mathbb{R}$$
\item A time-indexed sequence of Environment-generated random Contexts $C_t$ for time steps $t=1, 2, \ldots$, a time-indexed sequence of AI Agent-selected actions $A_t \in \mathcal{A}$ for time steps $t=1, 2, \ldots$, and a time-indexed sequence of Environment-generated {\em Reward} random variables $R_t \in \mathbb{R}$ for time steps $t=1, 2, \ldots$, such that for each time step $t$, $C_t$ is first randomly drawn from the probability distribution $\mathcal{C}$, after which the AI Agent selects the action $A_t$, after which $R_t$ is randomly drawn from the probability distribution $\mathcal{R}^{A_t}_{C_t}$.
\end{itemize}
\end{definition}

The AI Agent's goal is to maximize the following *Expected Cumulative Rewards* over a certain number of time steps $T$:
$$\mathbb{E}[\sum_{t=1}^T R_t]$$

Each of the algorithms we've covered for the MAB problem can be easily extended to the Contextual Bandit problem. The key idea in the extension of the MAB algorithms is that we have to take into account the Context, when dealing with the rewards probability distribution. In the MAB problem, the algorithms deal with a finite set of reward distributions, one for each of the actions. Here in the Contextual Bandit problem, the algorithms work with function approximations for the rewards probability distributions where each function approximation takes as input a pair of (Context, Action). 

We won't cover the details of the extensions of all MAB Algorithms to Contextual Bandit algorithms. Rather, we simply sketch a simple Upper-Confidence-Bound algorithm for the Contextual Bandit problem to convey a sense of how to extend the MAB algorithms to the Contextual Bandit problem. Assume that the sampling distribution of the mean reward for each (Context, Action) pair is a Gaussian distribution, and so we maintain two function approximations $\mu(c,a; \bm{w})$ and $\sigma(c, a; \bm{v})$ to represent the mean and standard deviation of the sampling distribution of mean reward for any context $c$ and any action $a$. It's important to note that for MAB, we simply maintained a finite set of estimates $\mu_a$ and $\sigma_a$, i.e., two parameters for each action $a$. Here we replace $\mu_a$ with function approximation $\mu(c,a; \bm{w})$ and we replace $\sigma_a$ with function approximation $\sigma(c, a; \bm{v})$. After the receipt of a reward from the Environment, the parameters $\bm{w}$ and $\bm{v}$ are appropriately updated. We essentially perform supervised learning in an incremental manner when updating these parameters of the function approximations. Note that $\sigma(c,a;\bm{v})$ represents a function approximation for the standard error of the mean reward for a given context $c$ and given action $a$. A simple Upper-Confidence-Bound algorithm would then select the action for a given context $C_t$ at time step $t$ that maximizes $\mu(C_t,a; \bm{w}) + \alpha \cdot \sigma(C_t, a; \bm{v})$ over all choices of $a \in \mathcal{A}$, for some fixed $\alpha$. Thus, we are comparing (across actions) $\alpha$ standard errors higher than the mean reward estimate (i.e., the upper-end of an appropriate confidence interval for the mean reward) for Context $C_t$. 

We want to highlight that many authors refer to the *Context* in Contextual Bandits as *State*. We desist from using the term *State* in Contextual Bandits since we want to reserve the term *State* to refer to the concept of "transitions" (as is the case in MDPs). Note that the Context does not "transition" to the next Context in the next time step in Contextual Bandits problems. Rather, the Context is drawn at random independently at each time step from the Context probability distribution $\mathcal{C}$. This is in contrast to the *State* in MDPs which transitions to the next state at the next time step based on the State Transition probability function of the MDP.

\index{contextual bandits|)}

We finish this chapter by simply pointing out that the approaches of the MAB algorithms can be further extended to resolve the Explore-Exploit dilemma in RL Control. From the perspective of this extension, it pays to emphasize that MAB algorithms that fall under the category of *Optimism in the Face of Uncertainty* can be roughly split into:

* Those that estimate the $Q$-Values (i.e., estimate $\mathbb{E}[r|a]$ from observed data) and the uncertainty of the $Q$-Values estimate. When extending to RL Control, we estimate the $Q$-Value Function for the (unknown) MDP and the uncertainty of the $Q$-Value Function estimate. Note that when moving from MAB to RL Control, the $Q$-Values are no longer simply the Expected Reward for a given action—rather, they are the Expected Return (i.e., accumulated rewards) from a given state and a given action. This extension from Expected Reward to Expected Return introduces significant complexity in the calculation of the uncertainty of the $Q$-Value Function estimate.
* Those that estimate the Model of the MDP, i.e., estimate of the State-Reward Transition Probability function $\mathcal{P}_R$ of the MDP, and the uncertainty of the $\mathcal{P}_R$ estimate. This includes extension of Bayesian Bandits, Thompson Sampling and Bayes-Adaptive MDP (for Information State Space MDP) where we replace $\mathbb{P}[\mathcal{R}|H_t]$ in the case of Bandits with $\mathbb{P}[\mathcal{P}_R|H_t]$ in the case of RL Control. Some of these algorithms sample from the estimated $\mathcal{P}_R$, and learn the Optimal Value Function/Optimal Policy from the samples. Some other algorithms are Planning-oriented. Specifically, the Planning-oriented approach is to run a Planning method (e.g., Policy Iteration, Value Iteration) using the estimated $\mathcal{P}_R$, then generate more data using the Optimal Policy (produced by the Planning method), use the generated data to improve the $\mathcal{P}_R$ estimate, then run the Planning method again to come up with the Optimal Policy (for the MDP based on the improved $\mathcal{P}_R$ estimate), and loop on in this manner until convergence. As an example of this Planning-oriented approach, we refer you to the [paper on RMax Algorithm](https://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf) [@conf/ijcai/BrafmanT01] to learn more.

\index{exploration versus exploitation|)}

### Key Takeaways from This Chapter

* The Multi-Armed Bandit problem provides a simple setting to understand and appreciate the nuances of the Explore-Exploit dilemma that we typically need to resolve within RL Control algorithms.
* In this chapter, we covered the following broad approaches to resolve the Explore-Exploit dilemma:
    - Naive Exploration, e.g., $\epsilon$-greedy
    - Optimistic Initialization
    - Optimism in the Face of Uncertainty, e.g., UCB, Bayesian UCB
    - Probability Matching, e.g., Thompson Sampling
    - Gradient Bandit Algorithms
    - Information State Space MDPs (incorporating value of Information), typically solved by treating as Bayes-Adaptive MDPs
* The above MAB algorithms are well-extensible to Contextual Bandits and RL Control.
