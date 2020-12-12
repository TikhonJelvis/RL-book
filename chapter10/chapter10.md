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

With this high-level perspective, we are now ready to dive into the actual RL algorithms. As has been our practice, we will start with the Prediction problem (this chapter) and then move to the Control problem (next chapter). 

### RL for Prediction

We shall re-use a lot of the notation we had developed in Module I. As a reminder, Prediction is the problem of estimating the Value Function of an MDP for a given policy $\pi$. We know from Chapter [-@sec:mdp-chapter] that this is equivalent to estimating the Value Function of the $\pi$-implied MRP. So in this chapter, we assume that we are working with an MRP (rather than an MDP) and we assume that the MRP is available in the form of an interface that serves up a sample of (next state, reward) pair, given a current state. Running this sampling interface in succession gives us a trace of experiences (which we call an episode):

$$S_0, R_1, S_1, R_2, S_2, \ldots$$

for some starting state $S_0$ for the episode.

Given a sufficient set of such episodes, the RL Prediction problem is to estimate the *Value Function* $V: \mathcal{N} \rightarrow \mathbb{R}$ of the MRP defined as:

$$V(s) = \mathbb{E}[G_t|S_t = s] \text{ for all } s in \mathcal{N}, \text{ for all } t = 0, 1, 2, \ldots$$

where the *Return* $G_t$ for each $t = 0, 1, 2, \ldots$ is defined as:

$$G_t = \sum_{i=t=1}^{\infty} \gamma^{i-t-1} \cdot R_i = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots$$

We use the above definition of *Return* even for a terminating sequence (say terminating at $t=T$, i.e., $S_T \in \mathcal{T}$), by treating $R_i = 0$ for all $i > T$.

We take you back to the code in Chapter [-@sec:mrp-chapter] where we had set up a `@dataclass TransitionStep` that serves as a building block in the method `simulate_reward` in the `@abstractclass MarkovRewardProcess`. Let's add a method called `add_return` to `TransitionStep` so we can augment the triple (state, reward, next state) with a return attribute that is comprised of the reward plus gamma times the return from the next state. The `returnStep` class (Derived from `TransitionStep`) includes the additional attribute named `return_` 

```python
@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: S
    next_state: S
    reward: float

    def add_return(self, γ: float, return_: float) -> ReturnStep[S]:
        return ReturnStep(
            self.state,
            self.next_state,
            self.reward,
            return_=self.reward + γ * return_
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

### Prediction with Monte-Carlo

Monte-Carlo is a very simple RL algorithm, and it is best understood in the context of a Tabular Monte-Carlo algorithm (which we can later generalize to Monte-Carlo with Function Approximation). In the Tabular setting, we have a not-so-large finite set of states $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ and so, we can represent the Value Function in the form of a "table" (specifically, in the form of a Python dictionary where the states are the keys and the associated estimated expected returns are the values). We should bear in mind that if the state space is not too large, we can employ the linear algebra-based solution (Equation \eqref{eq:mrp_bellman_linalg_solve} in Chapter [-@sec:mrp-chapter]) or the Policy Evaluation algorithm that we had covered in Chapter [-@sec:mdp-chapter]. So you should bear in mind that we are covering Tabular Prediction algorithms mainly for pedagogical purposes (since the core concepts of RL are best understood in the simple setting of Tabular Prediction). 


