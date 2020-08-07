# Approximate Dynamic Programming

In the previous chapter, we covered Dynamic Programming algorithms where the MDP is specified in the form of a finite data structure and the Value Function is represented as a finite "table" of states and values. These Dynamic Programming algorithms made a sweep through all states in each iteration to update the value function. But when the state space is large (as is the case in real-world applications), these Dynamic Programming algorithm won't work because:

1. Large state spaces could disallow a "tabular" representation of the MDP or Value Function, due to storage limits
2. Large state spaces would be time-prohibitive in terms of sweeping through all states (or simple impossible, in the case of infinite state spaces)

When the state space is very large, we need to resort to function approximation of the Value Function and the Dynamic Programming algorithms would be suitably modified to their Approximate Dynamic Programming (abbreviated as ADP) form. It's not hard to modify each of the (tabular) Dynamic Programming algorithms such that instead of sweeping through all the states at each step, we simply sample an appropriate subset of states, update the Value Function for those states (with the same Bellman Operator calculations as for the case of tabular), and then create a function approximation for the Value Function using just the updated values for the sample of states. The fundamental structure of the algorithms and the fundamental principles (Fixed-Point and Bellman Operators) would still be the same.

So, in this chapter, we do a quick review of function approximation, write some code for a couple for a couple of standard function approximation methods, and then utilize these function approximation methods to developed Approximate Dynamic Programming (in particular, Approximate Value Iteration and Approximate Backward Induction). If you are reading this book, it's highly likely that you are already familiar with the simple and standard function approximation methods such as linear function approximation and function approximation using neural networks supervised learning. So we shall go through the background on linear function approximation and neural networks supervised learning in a quick and terse manner, with the goal of developing some code for these methods that we can use not just for the ADP algorithms for this chapter, but also for RL algorithms later in the book. Note also that apart from function approximation of Value Functions $\mathcal{N} \rightarrow \mathbb{R}$, these function approximation methods can also be used for approximation of Stochastic Policies $\mathcal{N} \times \mathcal{A} \rightarrow [0, 1]$ in Policy-based RL algorithms.

## Function Approximation
In this section, we describe function approximation in a fairly generic setting (not specific to approximation of Value Functions or Policies). We denote the predictor variable as $x$, belonging to an arbitrary domain denoted $\mathcal{X}$ and the response variable as $y \in \mathbb{R}$. We treat $x$ and $y$ as unknown random variables and our goal is to estimate the probability distribution function $f$ of the conditional random variable $y|x$ from data provided in the form of a sequence of $(x,y)$ pairs. We shall consider parameterized function $f$ with the parameters denoted as $w$, The exact data type of $w$ will depend on the specific form of function approximation. We denote the estimated probability of $y$ conditional on $x$ as $f(x; w)(y)$. Assume we are given the following data in the form of a sequence of $n$ $(x,y)$ pairs:
$$[(x_i, y_i)|1 \leq i \leq n]$$
The notion of estimating the conditional probability $\mathbb{P}[y|x]$ is formalized by solving for $w=w^*$ such that:
$$w^* = \argmax\{ \prod_{i=1}^n f(x_i; w)(y_i)\} = \argmax\{ \sum_{i=1}^n \log f(x_i; w)(y_i)\}$$
In other words, we shall be operating in the framework of [Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). We say that the data $[(x_i, y_i)|1 \leq i \leq n]$ gives us the *empirical probability distribution* $D$ of $y|x$ and the function $f$ (parameterized by $w$) gives us the *model probability distribution* $M$ of $y|x$. With maximum likelihood estimation, we are essentially trying to reconcile the model probability distribution $M$ with the empirical probability distribution $D$. Maximum likelihood estimation is essentially minimization of a loss function defined as the [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) $\mathcal{H}(D, M) = -\mathbb{E}_D[\log M]$ between the probability distribution $D$ and $M$.

Our framework will allow for incremental estimation wherein at each time step $t$ (for $t=1, 2, \ldots$), data of the form

$$[(x_{t,i}, y_{t,i})|1 \leq i \leq n_t]$$

is used to update the parameters from $w_{t-1}$ to $w_t$ (parameters initialized at time $t=0$ to $w_0$). This framework can be used to update the parameters incrementally with a gradient descent algorithm, either stochastic gradient descent (where a single $(x,y)$ pair is used for each time step's gradient calculation) or mini-batch gradient descent (where an appropriate subset of the entire data is used for each time step's gradient calculation) or simply re-using the entire data available for each time step's gradient calculation (and consequent, parameter update). The flexibility of our framework to allow for incremental estimation is particularly important for Reinforcement Learning algorithms wherein we update the parameters of the function approximation from the new data that is generated from each state transition as a result of interaction with either the real environment or a simulated environment.
 
Among other things, the estimate $f$ (parameterized by $w^*$) gives us the model expected value of $y$ conditional on $x$, i.e.

$$\mathbb{E}_M[y|x] = \int_{-\infty}^{+\infty} y \cdot f(x;w^*)(y) \cdot dy$$

For the purposes of Approximate Dynamic Programming and Reinforcement Learning, the above expectation will provide an estimate of the Value Function for any state ($x$ takes the role of the state, and $y$ takes the role of the Value Function for that state). In the case of function approximation for policies, $x$ takes the role of the state, and $y$ takes the role of the action for that policy, and $f(x;w)$ will provide the probability distribution of the actions for state $x$ (for a stochastic policy). It's also worthwhile pointing out that the broader theory of function approximations covers the case of multi-dimensional $y$ (where $y$ is a real-valued vector, rather than scalar) - this allows us to solve classification problems, along with regression problems. However, for ease of exposition and for sufficient coverage of function approximation applications in this book, we will only cover the case of scalar $y$.

Now let us write code for this framework - for incremental estimation of $f$ with updates to $w$ at each time step $t$ and for evaluation of $\mathbb{E}_M[y|x]$ using the function $f(x;w)$.


## Linear Function Approximation

We define a sequence of feature functions
$$\phi_j: \mathcal{X} \rightarrow \mathbb{R} \text{ for each } j = 1, 2, \ldots, m$$
We define the weights $w$ as a vector $\bm{w} = (w_1, w_2, \ldots, w_m) \in \mathbb{R}^m$. Linear function approximation is based on the assumption of a Gaussian distribution for $y|x$ with mean
$$\sum_{j=1}^m \phi_j(x) \cdot w_j$$
and constant variance $\sigma^2$, i.e.,
$$\mathbb{P}[y|x] = f(x;\bm{w})(y) = \frac {1} {\sqrt{2\pi \sigma^2}} \cdot e^{-\frac {(y - \sum_{j=1}^m \phi_j(x) \cdot w_j)^2} {2\sigma^2}}$$

So, the cross-entropy loss function (ignoring constant terms associated with $\sigma^2$) is defined as:
$$\mathcal{L}(\bm{w}) = \frac 1 2 \cdot \sum_{i=1}^n (\sum_{j=1}^m \phi_j(x_i) \cdot w_j - y_i)^2$$
Note that this loss function is identical to the mean-squared error of the linear (in $\bm{w}$) predictions $\sum_{j=1}^m \phi_j(x_i) \cdot w_j$ relative to the response values $y_i$ associated with the predictor values $x_i$, over all $1\leq i \leq n$.

The gradient of $\mathcal{L}(\bm{w})$ with respect to $\bm{w}$ works out to:

$$\nabla_{\bm{w}} \mathcal{L}(\bm{w}) = \sum_{i=1}^n \bm{\phi}(x_i) \cdot (\sum_{j=1}^m \phi_j(x_i) \cdot w_j - y_i)$$

where $$\bm{\phi}: \mathcal{X} \rightarrow \mathbb{R}^m$$ is defined as:
$$\bm{\phi}(x) = (\phi_1(x), \phi_2(x), \ldots, \phi_m(x)) \text{ for all } x \in \mathcal{X}$$

We can solve for $\bm{w^*}$ by incremental estimation using gradient descent (change in $\bm{w}$ proportional to the gradient estimate of $\mathcal{L}(\bm{w})$ with respect to $\bm{w}$), where the gradient estimate at time step $t$ is:
$$\sum_{i=1}^{n_t} \bm{\phi}(x_{t,i}) \cdot (\sum_{j=1}^m \phi_j(x_{t,i}) \cdot w_j - y_{t,i})$$
which can be interpreted as the sum (over the data at time $t$) of the feature vectors $\bm{\phi}(x_{t,i})$ scaled by the (scalar) linear prediction errors $\sum_{j=1}^m \phi_j(x_{t,i}) \cdot w_j - y_{t,i}$.

Note that for linear function approximation, we can directly solve for $w^*$ if the number of feature function $m$ is not too large. If the entire provided data is $[(x_i, y_i)|1\leq i \leq n]$, then the gradient estimate based on this data can be set to 0 to solve for $\bm{w^*}$, i.e.,
$$\sum_{i=1}^n \bm{\phi}(x_i) \cdot (\sum_{j=1}^m \phi_j(x_i) \cdot w_j^* - y_i) = 0$$
We denote $\bm{\Phi}$ as the $n$ rows $\times$ $m$ columns matrix defined as $\bm{\Phi}_{i,j} = \phi_j(x_i)$ and the column vector $\bm{Y} \in \mathbb{R}^n$ as $\bm{Y}_i = y_i$. Then we can write the above equation as:
$$\bm{\Phi}^T \cdot \bm{\Phi} \cdot \bm{w^*} = \bm{\Phi}^T \cdot \bm{Y}$$
$$\Rightarrow \bm{w^*} = (\bm{\Phi}^T \cdot \bm{\Phi})^{-1} \cdot \bm{\Phi}^T \cdot \bm{Y}$$
This requires inversion of the $m \times m$ matrix $\bm{\Phi}^T \cdot \bm{\Phi}$ and so, this direct solution for $\bm{w^*}$ requires that $m$ not be too large.

## Neural Network Function Approximation

## Tabular as an Exact Approximation

## Approximate Value Iteration

## Approximate Backward Induction

## Key Takeaways from this Chapter
