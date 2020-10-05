# Function Approximation and Approximate Dynamic Programming {#sec:funcapprox-chapter}

In Chapter [-@sec:dp-chapter], we covered Dynamic Programming algorithms where the MDP is specified in the form of a finite data structure and the Value Function is represented as a finite "table" of states and values. These Dynamic Programming algorithms swept through all states in each iteration to update the value function. But when the state space is large (as is the case in real-world applications), these Dynamic Programming algorithm won't work because:

1. A "tabular" representation of the MDP or of the Value Function, won't fit within storage limits
2. Sweeping through all states and their transition probabilities would be time-prohibitive (or simply impossible, in the case of infinite state spaces)

Hence, when the state space is very large, we need to resort to approximation of the Value Function.  The Dynamic Programming algorithms would need to be suitably modified to their Approximate Dynamic Programming (abbreviated as ADP) versions. The good news is that it's not hard to modify each of the (tabular) Dynamic Programming algorithms such that instead of sweeping through all the states in each iteration, we simply sample an appropriate subset of states, update the Value Function for those states (with the same Bellman Operator calculations as for the case of tabular), and then construct an approximation for the Value Function using just the updated values for the sample of states. Furthermore, if the set of transitions from a given state is large (or infinite), instead of using the explicit probabilities of those transitions, we can sample from the transitions probability distribution. The fundamental structure of the algorithms and the fundamental principles (Fixed-Point and Bellman Operators) would still be the same.

So, in this chapter, we do a quick review of function approximation, write some code for a couple for a couple of standard function approximation methods, and then utilize these function approximation methods to develop Approximate Dynamic Programming algorithms (in particular, Approximate Policy Evaluation, Approximate Value Iteration and Approximate Backward Induction). Since you are reading this book, it's highly likely that you are already familiar with the simple and standard function approximation methods such as linear function approximation and function approximation using neural networks supervised learning. So we shall go through the background on linear function approximation and neural networks supervised learning in a quick and terse manner, with the goal of developing some code for these methods that we can use not just for the ADP algorithms for this chapter, but also for RL algorithms later in the book. Note also that apart from approximation of Value Functions $\mathcal{N} \rightarrow \mathbb{R}$, these function approximation methods can also be used for approximation of Stochastic Policies $\mathcal{N} \times \mathcal{A} \rightarrow [0, 1]$ in Policy-based RL algorithms.

## Function Approximation

In this section, we describe function approximation in a fairly generic setting (not specific to approximation of Value Functions or Policies). We denote the predictor variable as $x$, belonging to an arbitrary domain denoted $\mathcal{X}$ and the response variable as $y \in \mathbb{R}$. We treat $x$ and $y$ as unknown random variables and our goal is to estimate the probability distribution function $f$ of the conditional random variable $y|x$ from data provided in the form of a sequence of $(x,y)$ pairs. We shall consider parameterized functions $f$ with the parameters denoted as $w$. The exact data type of $w$ will depend on the specific form of function approximation. We denote the estimated probability of $y$ conditional on $x$ as $f(x; w)(y)$. Assume we are given the following data in the form of a sequence of $n$ $(x,y)$ pairs:
$$[(x_i, y_i)|1 \leq i \leq n]$$
The notion of estimating the conditional probability $\mathbb{P}[y|x]$ is formalized by solving for $w=w^*$ such that:
$$w^* = \argmax\{ \prod_{i=1}^n f(x_i; w)(y_i)\} = \argmax\{ \sum_{i=1}^n \log f(x_i; w)(y_i)\}$$
In other words, we shall be operating in the framework of [Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). We say that the data $[(x_i, y_i)|1 \leq i \leq n]$ gives us the *empirical probability distribution* $D$ of $y|x$ and the function $f$ (parameterized by $w$) gives us the *model probability distribution* $M$ of $y|x$. With maximum likelihood estimation, we are essentially trying to reconcile the model probability distribution $M$ with the empirical probability distribution $D$. Maximum likelihood estimation is essentially minimization of a loss function defined as the [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) $\mathcal{H}(D, M) = -\mathbb{E}_D[\log M]$ between the probability distribution $D$ and $M$.

Our framework will allow for incremental estimation wherein at each iteration $t$ of the incremental estimation (for $t=1, 2, \ldots$), data of the form

$$[(x_{t,i}, y_{t,i})|1 \leq i \leq n_t]$$

is used to update the parameters from $w_{t-1}$ to $w_t$ (parameters initialized at iteration $t=0$ to $w_0$). This framework can be used to update the parameters incrementally with a gradient descent algorithm, either stochastic gradient descent (where a single $(x,y)$ pair is used for each iteration's gradient calculation) or mini-batch gradient descent (where an appropriate subset of the available data is used for each iteration's gradient calculation) or simply re-using the entire data available for each iteration's gradient calculation (and consequent, parameter update). Moreover, the flexibility of our framework, allowing for incremental estimation, is particularly important for Reinforcement Learning algorithms wherein we update the parameters of the function approximation from the new data that is generated from each state transition as a result of interaction with either the real environment or a simulated environment.
 
Among other things, the estimate $f$ (parameterized by $w^*$) gives us the model expected value of $y$ conditional on $x$, i.e.

$$\mathbb{E}_M[y|x] = \int_{-\infty}^{+\infty} y \cdot f(x;w^*)(y) \cdot dy$$

For the purposes of Approximate Dynamic Programming and Reinforcement Learning, the above expectation will provide an estimate of the Value Function for any state ($x$ takes the role of the state, and $y$ takes the role of the Value Function for that state). In the case of function approximation for policies, $x$ takes the role of the state, and $y$ takes the role of the action for that policy, and $f(x;w)$ will provide the probability distribution of the actions for state $x$ (for a stochastic policy). It's also worthwhile pointing out that the broader theory of function approximations covers the case of multi-dimensional $y$ (where $y$ is a real-valued vector, rather than scalar) - this allows us to solve classification problems, along with regression problems. However, for ease of exposition and for sufficient coverage of function approximation applications in this book, we will only cover the case of scalar $y$.

Now let us write code for this framework - for incremental estimation of $f$ with updates to $w$ at each iteration $t$ and for evaluation of $\mathbb{E}_M[y|x]$ using the function $f(x;w)$. We write an abstract base class `FunctionApprox` parameterized by `X` (to permit arbitrary data types $\mathcal{X}$). The first `@abstractmethod` is `update` that takes as input a sequence of $(x,y)$ pairs and is meant to update the weights $w$ that define $f(x;w)$. The second `@abstractmethod` is `evaluate` that takes as input a sequence of $x$ values and is meant to calculate $f(x;w)$ for each of those $x$ values and produce the output sequence of $f(x;w)$ values in the form of an `np.ndarray`. The concrete classes that implement this abstract class `FunctionApprox` will implement these two `abstractmethod`s according to the functional form assumptions for $f$.

```python
class FunctionApprox(ABC, Generic[X]):

from abc import ABC, abstractmethod
import numpy as np

X = TypeVar('X')

    @abstractmethod
    def update(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> FunctionApprox[X]:
        pass

    @abstractmethod
    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        pass
```

We also need a couple of helper methods in `FunctionApprox`. One is a `@staticmethod` method `within` that runs through an `Iterator` of `FunctionApprox` objects (obtained from the sequence of updates to the weights $w$) and terminates the updates when the weights $w$ in two successive `FunctionApprox` objects in the `Iterator` are "close enough". The judgement of "close enough" is made by an `@abstractmethod within` that is meant to examine the "closeness" of weights $w$ within two `FunctionApprox` objects. Specific classes that implement `FunctionApprox` will need to implement the method `within` along with the methods `update` and `evaluate`. The complete code for `FunctionApprox` is in the file [rl/function_approx.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/function_approx.py).

```python
    @abstractmethod
    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        pass

    @staticmethod
    def converged(iterator: Iterator[FunctionApprox[X]],
                  tolerance: float = 0.0001) -> FunctionApprox[X]:
        def done(a, b):
            return a.within(b, tolerance)

        return iterate.converged(iterator, done=done)
```

Now we are ready to cover a concrete but simple function approximation - the case of linear function approximation. 

## Linear Function Approximation

We define a sequence of feature functions
$$\phi_j: \mathcal{X} \rightarrow \mathbb{R} \text{ for each } j = 1, 2, \ldots, m$$
We define the weights $w$ as a vector $\bm{w} = (w_1, w_2, \ldots, w_m) \in \mathbb{R}^m$. Linear function approximation is based on the assumption of a Gaussian distribution for $y|x$ with mean
$$\sum_{j=1}^m \phi_j(x) \cdot w_j$$
and constant variance $\sigma^2$, i.e.,
$$\mathbb{P}[y|x] = f(x;\bm{w})(y) = \frac {1} {\sqrt{2\pi \sigma^2}} \cdot e^{-\frac {(y - \sum_{j=1}^m \phi_j(x) \cdot w_j)^2} {2\sigma^2}}$$

So, the cross-entropy loss function (ignoring constant terms associated with $\sigma^2$) for a given set of data points $[x_i, y_i|1 \leq i \leq n]$ is defined as:
$$\mathcal{L}(\bm{w}) = \frac 1 {2n} \cdot \sum_{i=1}^n (\sum_{j=1}^m \phi_j(x_i) \cdot w_j - y_i)^2$$
Note that this loss function is identical to the mean-squared-error of the linear (in $\bm{w}$) predictions $\sum_{j=1}^m \phi_j(x_i) \cdot w_j$ relative to the response values $y_i$ associated with the predictor values $x_i$, over all $1\leq i \leq n$.

If we include $L^2$ regularization (with $\lambda$ as the regularization coefficient), then the regularized loss function is:

$$\mathcal{L}(\bm{w}) = \frac 1 {2n} (\sum_{i=1}^n (\sum_{j=1}^m \phi_j(x_i) \cdot w_j - y_i)^2) + \frac 1 2 \cdot\lambda \cdot \sum_{j=1}^m w_j^2$$

The gradient of $\mathcal{L}(\bm{w})$ with respect to $\bm{w}$ works out to:

$$\nabla_{\bm{w}} \mathcal{L}(\bm{w}) = \frac 1 n \cdot (\sum_{i=1}^n \bm{\phi}(x_i) \cdot (\bm{\phi}(x_i) \cdot \bm{w} - y_i)) + \lambda \cdot \bm{w}$$

where $$\bm{\phi}: \mathcal{X} \rightarrow \mathbb{R}^m$$ is defined as:
$$\bm{\phi}(x) = (\phi_1(x), \phi_2(x), \ldots, \phi_m(x)) \text{ for all } x \in \mathcal{X}$$

We can solve for $\bm{w^*}$ by incremental estimation using gradient descent (change in $\bm{w}$ proportional to the gradient estimate of $\mathcal{L}(\bm{w})$ with respect to $\bm{w}$). If the $(x_t, y_t)$ data at time $t$ is:

$$[(x_{t,i}, y_{t,i})|1 \leq i \leq n_t]$$,

then the gradient estimate $\mathcal{G}_{(x_t,y_t)}(\bm{w}_t)$ at time $t$ is given by:

$$\mathcal{G}_{(x_t, y_t)}(\bm{w}_t) = \frac 1 n \cdot (\sum_{i=1}^{n_t} \bm{\phi}(x_{t,i}) \cdot (\bm{\phi}(x_{t,i}) \cdot \bm{w}_t - y_{t,i})) + \lambda \cdot \bm{w}_t$$
which can be interpreted as the mean (over the data in iteration $t$) of the feature vectors $\bm{\phi}(x_{t,i})$ weghted by the (scalar) linear prediction errors $\bm{\phi}(x_{t,i}) \cdot \bm{w} - y_{t,i}$ (plus regularization term $\lambda \cdot \bm{w}$).

Then, the update to the weights is given by:

$$\bm{w}_{t+1} = \bm{w}_t - \alpha_t \cdot \mathcal{G}_{(x_t, y_t)}(\bm{w}_t)$$

where $\alpha_t$ is the learning rate for the gradient descent at time $t$. To facilitate numerical convergence, we require $\alpha_t$ to be an appropriate function of time $t$. There are a number of numerical algorithms to achieve the appropriate time-trajectory of $\alpha_t$. We shall go with one such numerical algorithm - [ADAM](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam), which we shall use not just for linear function approximation but also for the deep neural network function approximation. Before we write code for linear function approximation, we need to write some helper code to implement the ADAM gradient descent algorithm.

We create an `@dataclass Weights` to represent and update the weights of a function approximation. The `Weights` dataclass has 5 attributes: `adam_gradient` that captures the ADAM parameters, including the base learning rate and the decay parameter, `time` that represents how many times the weights have been updated, `weights` that represents the weight parameters of the function approximation as a numpy array (1-D array for linear function approximation and 2-D array for each layer of deep neural network function approximation), and the two ADAM cache parameters. The `update` method of this `Weights` dataclass produces a new instance of the `Weights` dataclass that represents the updated weight parameters together with the incremented `time` and the updated ADAM cache parameters. We will follow a programming design pattern wherein we don't update anything in-place - rather, we create a new object with updated values. This ensures we don't get unexpected/undesirable updates in-place, which are typically the cause of bugs in numerical code. Finally, we write the `within` method which will be required to implement the `within` method in the linear function approximation class as well as in the deep neural network function approximation class.

```python
@dataclass(frozen=True)
class AdamGradient:
    learning_rate: float
    decay1: float
    decay2: float


@dataclass(frozen=True)
class Weights:
    adam_gradient: AdamGradient
    time: int
    weights: np.ndarray
    adam_cache1: np.ndarray
    adam_cache2: np.ndarray

    @staticmethod
    def create(
        adam_gradient: AdamGradient,
        weights: np.ndarray,
        adam_cache1: Optional[np.ndarray] = None,
        adam_cache2: Optional[np.ndarray] = None
    ) -> Weights:
        return Weights(
            adam_gradient=adam_gradient,
            time=0,
            weights=weights,
            adam_cache1=np.zeros_like(
                weights
            ) if adam_cache1 is None else adam_cache1,
            adam_cache2=np.zeros_like(
                weights
            ) if adam_cache2 is None else adam_cache2
        )

    def update(self, gradient: np.ndarray) -> Weights:
        time: int = self.time + 1
        new_adam_cache1: np.ndarray = self.adam_gradient.decay1 * \
            self.adam_cache1 + (1 - self.adam_gradient.decay1) * gradient
        new_adam_cache2: np.ndarray = self.adam_gradient.decay2 * \
            self.adam_cache2 + (1 - self.adam_gradient.decay2) * gradient ** 2
        new_weights: np.ndarray = self.weights - \
            self.adam_gradient.learning_rate * self.adam_cache1 / \
            (np.sqrt(self.adam_cache2) + SMALL_NUM) * \
            np.sqrt(1 - self.adam_gradient.decay2 ** time) / \
            (1 - self.adam_gradient.decay1 ** time)
        return replace(
            self,
            time=time,
            weights=new_weights,
            adam_cache1=new_adam_cache1,
            adam_cache2=new_adam_cache2,
        )

    def within(self, other: Weights[X], tolerance: float) -> bool:
        return np.all(np.abs(self.weights - other.weights) <= tolerance).item()
```

Given this `Weights` dataclass, we are now ready to write the `@dataclass LinearFunctionApprox` for linear function approximation that implements the abstract base class `FunctionApprox`. It has attributes `feature_functions` that represents $\phi_j: \mathcal{X} \rightarrow \mathbb{R}$ for all $j = 1, 2, \ldots, m$, `regularization_coeff` that represents the regularization coefficient $\lambda$, and `weights` which is an instance of the `Weights` class we wrote above. Note that the `get_feature_values` method adds a feature function $\phi(x) = 1$ for all $x \in \mathcal{X}$, so as to include the bias term. The method `evaluate` (an `@abstractmethod` in `FunctionApprox`) calculates the prediction $\mathbb{E}_M[y|x]$ for each $x$ in the input sequence as: $\bm{\phi}(x) \cdot \bm{w} = \sum_{j=1}^m \phi_j(x) \cdot w_i$. The method `regularized_loss_gradient` performs the calculation $\mathcal{G}_{(x_t, y_t)}(\bm{w}_t)$ shown above. Finally, the method `update` (`@abstractmethod` in `FunctionApprox`) invokes `regularized_loss_gradient` and returns a new instance of `LinearFunctionApprox` that contains the updated weights, along with the ADAM cache updates (invoking the `update` method of the `Weights` class to ensure there are no in-place updates).


```python
@dataclass(frozen=True)
class LinearFunctionApprox(FunctionApprox[X]):

    feature_functions: Sequence[Callable[[X], float]]
    regularization_coeff: float
    weights: Weights

    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        adam_gradient: AdamGradient,
        regularization_coeff: float = 0.,
        weights: Optional[Weights] = None
    ) -> LinearFunctionApprox[X]:
        return LinearFunctionApprox(
            feature_functions=feature_functions,
            regularization_coeff=regularization_coeff,
            weights=Weights.create(
                adam_gradient=adam_gradient,
                weights=np.zeros(len(feature_functions) + 1)
            ) if weights is None else weights
        )

    def get_feature_values(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.array([[1.] + [f(x) for f in self.feature_functions]
                         for x in x_values_seq])

    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.dot(
            self.get_feature_values(x_values_seq),
            self.weights.weights
        )

    def regularized_loss_gradient(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> np.ndarray:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: np.ndarray = self.get_feature_values(x_vals)
        diff: np.ndarray = np.dot(feature_vals, self.weights.weights) \
            - np.array(y_vals)
        return np.dot(feature_vals.T, diff) / len(diff) \
            + self.regularization_coeff * self.weights.weights

    def update(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> LinearFunctionApprox[X]:
        gradient: np.ndarray = self.regularized_loss_gradient(xy_vals_seq)
        new_weights: np.ndarray = self.weights.update(gradient)
        return replace(self, weights=new_weights)
```

We also require the `within` method, that simply delegates to the `within` method of the `Weights` class.

```python
    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, LinearFunctionApprox):
            return self.weights.within(other.weights, tolerance)
        else:
            return False
```

Note that for linear function approximation, we can directly solve for $\bm{w^*}$ if the number of feature functions $m$ is not too large. If the entire provided data is $[(x_i, y_i)|1\leq i \leq n]$, then the gradient estimate based on this data can be set to 0 to solve for $\bm{w^*}$, i.e.,
$$\frac 1 n \cdot (\sum_{i=1}^n \bm{\phi}(x_i) \cdot (\bm{\phi}(x_i) \cdot \bm{w^*} - y_i)) + \lambda \cdot \bm{w^*} = 0$$
We denote $\bm{\Phi}$ as the $n$ rows $\times$ $m$ columns matrix defined as $\bm{\Phi}_{i,j} = \phi_j(x_i)$ and the column vector $\bm{Y} \in \mathbb{R}^n$ defined as $\bm{Y}_i = y_i$. Then we can write the above equation as:
$$\frac 1 n \cdot \bm{\Phi}^T \cdot (\bm{\Phi} \cdot \bm{w^*} - \bm{Y}) + \lambda \cdot \bm{w^*} = 0$$
$$\Rightarrow (\bm{\Phi}^T \cdot \bm{\Phi} + n \lambda \cdot \bm{I_m}) \cdot \bm{w^*} = \bm{\Phi}^T \cdot \bm{Y}$$
$$\Rightarrow \bm{w^*} = (\bm{\Phi}^T \cdot \bm{\Phi} + n \lambda \cdot \bm{I_m})^{-1} \cdot \bm{\Phi}^T \cdot \bm{Y}$$
where $\bm{I_m}$ is the $m \times m$ identity matrix. Note that this requires inversion of the $m \times m$ matrix $\bm{\Phi}^T \cdot \bm{\Phi} + n \lambda \cdot \bm{I_m}$ and so, this direct solution for $\bm{w^*}$ requires that $m$ not be too large.

So let's also implement this direct-solve for $\bm{w^*}$ as a method in `LinearFunctionApprox`.

```python
    def direct_solve(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> LinearFunctionApprox[X]:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: np.ndarray = self.get_feature_values(x_vals)
        feature_vals_T: np.ndarray = feature_vals.T
        left: np.ndarray = np.dot(feature_vals_T, feature_vals) \
            + feature_vals.shape[0] * self.regularization_coeff * \
            np.eye(len(self.weights.weights))
        right: np.ndarray = np.dot(feature_vals_T, y_vals)
        return replace(
            self,
            weights=Weights.create(
                adam_gradient=self.weights.adam_gradient,
                weights=np.dot(np.linalg.inv(left), right)
            )
        )
```

The above code is in the file [rl/function_approx.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/function_approx.py).

## Neural Network Function Approximation

Now we generalize the linear function approximation to accommodate non-linear functions with a simple deep neural network, specifically a feed-forward fully-connected neural network. We work with the same notation $\bm{\phi}(\cdot) = (\phi_1(\cdot), \phi_2(\cdot), \ldots, \phi_m(\cdot))$ for feature functions that we covered for the case of linear function approximation. Assume we have $L$ hidden layers in the neural network. Layers numbered $l = 0, 1, \ldots, L - 1$ carry the hidden layer neurons and layer $l = L$ carries the output layer neurons.

A couple of things to note about our notation for vectors and matrices when performing linear algebra operations: Vectors will be treated as column vectors (including gradient of a scalar with respect to a vector). When our notation expresses gradient of a vector of dimension $m$ with respect to a vector of dimension $n$, we treat it as a Jacobian matrix with $m$ rows and $n$ columns. We use the notation $dim(\bm{V})$ to refer to the dimension of a vector $\bm{V}$.

We denote the input to layer $l$ as vector $\bm{I_l}$ and the output to layer $l$ as vector $\bm{O_l}$, for all $l = 0, 1, \ldots, L$. Denoting the predictor variable as $x \in \mathcal{X}$, we have:

\begin{equation}
\bm{I_0} = \bm{\phi}(x) \in \mathbb{R}^m \text{ and } \bm{I_{l+1}} = \bm{O_l} \text{ for all } l = 0, 1, \ldots, L - 1
\label{eq:layers_input_output_connect}
\end{equation}

We denote the parameters for layer $l$ as the matrix $\bm{w_l}$ with $dim(\bm{O_l})$ rows and $dim(\bm{I_l})$ columns.  Note that the number of neurons in layer $l$ is equal to $dim(\bm{O_l})$. Since we are restricting ourselves to scalar $y$, $dim(\bm{O_L}) = 1$ and so, the number of neurons in the output layer is 1. 

The neurons in layer $l$ define a linear transformation from layer input $\bm{I_l}$ to a variable we denote as $\bm{S_l}$. Therefore,

\begin{equation}
\bm{S_l} = \bm{w_l} \cdot \bm{I_l} \text{ for all } l = 0, 1, \ldots, L
\label{eq:layer_linearity}
\end{equation}

We denote the activation function of layer $l$ as $g_l: \mathbb{R} \rightarrow \mathbb{R}$ for all $l = 0, 1, \ldots, L$. The activation function $g_l$ applies point-wise on each dimension of vector $\bm{S_l}$, so we take notational liberty with $g_l$ by writing:
\begin{equation}
\bm{O_l} = g_l(\bm{S_l}) \text{ for all } l = 0, 1, \ldots, L
\label{eq:layer_non_linearity}
\end{equation}

Equations \eqref{eq:layers_input_output_connect}, \eqref{eq:layer_linearity} and \eqref{eq:layer_non_linearity} together define the calculation of the neural network prediction $\bm{O_L}$ (associated with the response variable $y$), given the predictor variable $x$. This calculation is known as *forward-propagation* and will define the `evaluate` method of the deep neural network function approximation class we shall soon write.

Our goal is to derive an expression for the loss gradient $\nabla_{\bm{w_l}} \mathcal{L}$ for all $l = 0, 1, \ldots, L$. We can reduce this problem of calculating the loss gradient to the problem of calculating $\bm{P_l} = \nabla_{\bm{S_l}} \mathcal{L}$ for all $l = 0, 1, \ldots, L$, as revealed by the following chain-rule calculation:
$$\nabla_{\bm{w_l}} \mathcal{L} = (\nabla_{\bm{S_l}} \mathcal{L})^T \cdot \nabla_{\bm{w_l}} \bm{S_l} = \bm{P_l}^T \cdot \nabla_{\bm{w_l}} \bm{S_l} = \bm{P_l} \cdot \bm{I_l}^T = \bm{P_l} \otimes \bm{I_l} \text{ for all } l = 0, 1, \ldots L$$
where the symbol $\otimes$ refers to the [outer-product](https://en.wikipedia.org/wiki/Outer_product) of two vectors resulting in a matrix. Note that the outer-product of the $dim(\bm{O_l})$ size vector $\bm{P_l}$ and the $dim(\bm{I_l})$ size vector $\bm{I_l}$ gives a matrix of size $dim(\bm{O_l}) \times dim(\bm{I_l})$.

If we include $L^2$ regularization (with $\lambda_l$ as the regularization coefficient for layer $l$), then:

\begin{equation}
\nabla_{\bm{w_l}} \mathcal{L} = \bm{P_l} \otimes \bm{I_l} + \lambda_l \cdot \bm{w_l} \text{ for all } l = 0, 1, \ldots, L
\label{eq:loss_gradient_formula}
\end{equation}

Here's the summary of our notation:

\begin{center} 
\begin{tabular}{|c|c|}
\hline
\textbf{Notation} & \textbf{Description} \\
\hline
$\bm{I_l}$ & Vector Input to layer $l$ for all $l = 0, 1, \ldots, L$  \\
\hline
$\bm{O_l}$ & Vector Output of layer $l$ for all $l = 0, 1, \ldots, L$ \\
\hline
$\bm{\phi}(x)$ & Feature Vector for predictor variable $x$ \\
\hline
 $y$ & Response variable associated with predictor variable $x$ \\
\hline
 $\bm{w_l}$ & Matrix of Parameters for layer $l$ for all $l = 0, 1, \ldots, L$ \\
 \hline
 $g_l(\cdot)$ & Activation function for layer $l$ for $l = 0, 1, \ldots, L$ \\
 \hline
 $\bm{S_l}$ & $\bm{S_l} = \bm{w_l} \cdot \bm{I_l}, \bm{O_l} = g_l(\bm{S_l})$ for all $l = 0, 1, \ldots L$ \\
 \hline
 $\bm{P_l}$ & $\bm{P_l} = \nabla_{\bm{S_l}} \mathcal{L}$ for all $l = 0, 1, \ldots, L$\\
 \hline
 $\lambda_l$ & Regularization coefficient for layer $l$ for all $l = 0, 1, \ldots, L$ \\
 \hline
\end{tabular}
\end{center}

Now that we have reduced the loss gradient calculation to calculation of $\bm{P_l}$, we spend the rest of this section deriving the analytical calculation of $\bm{P_l}$. The following theorem tells us that $\bm{P_l}$ has a recursive formulation that forms the core of the *back-propagation* algorithm for a feed-forward fully-connected deep neural network.

\begin{theorem}
For all $l = 0, 1, \ldots, L-1$,
$$\bm{P_l} = (\bm{w_{l+1}}^T \cdot \bm{P_{l+1}}) \circ g_l'(\bm{S_l})$$
where the symbol $\cdot$ represents vector-matrix multiplication and the symbol $\circ$ represents the \href{https://en.wikipedia.org/wiki/Hadamard_product_(matrices)}{Hadamard Product}, i.e., point-wise multiplication of two vectors of the same dimension.
\label{th:recursive_gradient_formulation}
\end{theorem}

\begin{proof}
We start by applying the chain rule on $\bm{P_l}$.
\begin{equation}
\bm{P_l} = \nabla_{\bm{S_l}} \mathcal{L} = (\nabla_{\bm{S_l}} \bm{S_{l+1}})^T \cdot \nabla_{\bm{S_{l+1}}} \mathcal{L} = (\nabla_{\bm{S_l}} \bm{S_{l+1}})^T \cdot \bm{P_{l+1}}
\label{eq:recursive_chain_rule}
\end{equation}
Next, note that:
$$\bm{S_{l+1}} = \bm{w_{l+1}} \cdot g_l(\bm{S_l})$$
Therefore,
$$\nabla_{\bm{S_l}} \bm{S_{l+1}} = \bm{w_{l+1}} \cdot \bm{Diagonal}(g_l'(\bm{S_l}))$$
Substituting this in Equation \eqref{eq:recursive_chain_rule} yields:
$$\bm{P_l} = (\bm{w_{l+1}} \cdot \bm{Diagonal}(g_l'(\bm{S_l})))^T \cdot \bm{P_{l+1}} = \bm{Diagonal}(g_l'(\bm{S_l})) \cdot \bm{w_{l+1}}^T \cdot \bm{P_{l+1}}$$
$$= g_l'(\bm{S_l}) \circ (\bm{w_{l+1}}^T \cdot \bm{P_{l+1}}) = (\bm{w_{l+1}}^T \cdot \bm{P_{l+1}}) \circ g_l'(\bm{S_l})$$
\end{proof}

Now all we need to do is to calculate $\bm{P_L} = \nabla_{\bm{S_L}} \mathcal{L}$ so that we can run this recursive formulation for $\bm{P_l}$, estimate the loss gradient $\nabla_{\bm{w_l}} \mathcal{L}$ for any given data (using Equation \eqref{eq:loss_gradient_formula}), and perform gradient descent to arrive at $\bm{w_l^*}$ for all $l = 0, 1, \ldots L$.

Firstly, note that $\bm{S_L}, \bm{O_L}, \bm{P_L}$ are all scalars, so let's just write them as $S_L, O_L, P_L$ respectively (without the bold-facing) to make it explicit in the derivation that they are scalars. Specifically, the gradient
$$\nabla_{\bm{S_L}} \mathcal{L} = \frac {\partial \mathcal{L}}{\partial S_L}$$

To calculate $\frac {\partial \mathcal{L}} {\partial S_L}$, we need to assume a functional form for $\mathbb{P}[y|S_L]$. We work with a fairly generic exponential functional form for the probability distribution function:

$$p(y|\theta, \tau) = h(y, \tau) \cdot e^{\frac {\theta \cdot y - A(\theta)} {d(\tau)}}$$

where $\theta$ should be thought of as the "center" parameter (related to the mean) of the probability distribution and $\tau$ should be thought of as the "dispersion" parameter (related to the variance) of the distribution. $h(\cdot, \cdot), A(\cdot), d(\cdot)$ are general functions whose specializations define the family of distributions that can be modeled with this fairly generic exponential functional form (note that this structure is adopted from the framework of [Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model)). 

For our neural network function approximation, we assume that $\tau$ is a constant, and we set $\theta$ to be $S_L$. So,

$$\mathbb{P}[y|S_L] = p(y|S_L, \tau) = h(y, \tau) \cdot e^{\frac {S_L \cdot y - A(S_L)} {d(\tau)}}$$

We require the scalar prediction of the neural network $O_L = g_L(S_L)$ to be equal to $\mathbb{E}_p[y|S_L]$. So the question is: What function $g_L: \mathbb{R} \rightarrow \mathbb{R}$ (in terms of the functional form of $p(y|S_L, \tau)$) would satisfy the requirement of $O_L = g_L(S_L) = \mathbb{E}_p[y|S_L]$? To answer this question, we first establish the following Lemma:

\begin{lemma}
$$\mathbb{E}_p[y|S_L] = A'(S_L)$$
\end{lemma}
\begin{proof}
Since

$$\int_{-\infty}^{\infty} p(y | S_L, \tau) \cdot dy = 1,$$

the partial derivative of the left-hand-side of the above equation with respect to $S_L$ is zero. In other words,

$$\frac {\partial \{\int_{-\infty}^{\infty} p(y | S_L, \tau) \cdot dy\}}{\partial S_L} = 0$$

Hence,

$$\frac {\partial \{\int_{-\infty}^{\infty}  h(y, \tau) \cdot e^{\frac {S_L \cdot y - A(S_L)} {d(\tau)}} \cdot dy\}}{\partial S_L} = 0$$

Taking the partial derivative inside the integral, we get:

$$\int_{-\infty}^{\infty}  h(y, \tau) \cdot e^{\frac {S_L \cdot y - A(S_L)} {d(\tau)}} \cdot \frac {y - A'(S_L)} {d(\tau)} \cdot dy = 0$$

$$\Rightarrow \int_{-\infty}^{\infty}  p(y | S_L, \tau) \cdot (y - A'(S_L)) \cdot dy = 0$$

$$\Rightarrow \mathbb{E}_p[y|S_L] = A'(S_L)$$
\end{proof}

So to satisfy $O_L = g_L(S_L) = \mathbb{E}_p[y|S_L]$, we require that
\begin{equation}
O_L = g_L(S_L) = A'(S_L)
\label{eq:glm_eqn}
\end{equation}
The above equation is important since it tells us that the output layer activation function $g_L(\cdot)$ must be set to be the derivative of the $A(\cdot)$ function. In the theory of generalized linear models, the derivative of the $A(\cdot)$ function serves as the *canonical link function* for a given probability distribution of the response variable conditional on the predictor variable.

Now we are equipped to derive a simple expression for $P_L$.

\begin{theorem}
$$P_L = \frac {\partial \mathcal{L}}{\partial S_L} = \frac {O_L - y} {d(\tau)}$$
\end{theorem}

\begin{proof}
The Cross-Entropy Loss (Negative Log-Likelihood) for a single training data point $(x, y)$ is given by:

$$\mathcal{L} = - \log{(h(y, \tau))} + \frac {A(S_L) - S_L \cdot y} {d(\tau)}$$

Therefore,

$$P_L = \frac {\partial \mathcal{L}}{\partial S_L} = \frac {A'(S_L) - y} {d(\tau)}$$
But from Equation \eqref{eq:glm_eqn}, we know that $A'(S_L) = O_L$. Therefore,

$$P_L = \frac {\partial \mathcal{L}}{\partial S_L} = \frac {O_L - y}{d(\tau)}$$

\end{proof}

At each iteration of gradient descent, we require an estimate of the loss gradient up to a constant factor. So we can ignore the constant $d(\tau)$ and simply say that $P_L = O_L - y$ (up to a constant factor). This is a rather convenient estimate of $P_L$ for a given data point $(x,y)$ since it represents the neural network prediction error for that data point. When presented with a sequence of data points $[(x_{t,i}, y_{t,i})|1\leq i \leq n_t]$ in iteration $t$, we simply average the prediction errors across these presented data points. Then, beginning with this estimate of $P_L$, we can use the recursive formulation of $\bm{P_l}$ (Theorem \ref{th:recursive_gradient_formulation}) to calculate the gradient of the loss function (Equation \eqref{eq:loss_gradient_formula}) with respect to all the parameters of the neural network (this is known as the back-propagation algorithm for a fully-connected feed-forward deep neural network).

Here are some common specializations of the functional form for the conditional probability distribution $\mathbb{P}[y|S_L]$, along with the corresponding activation function $g_L$ of the output layer:

* Normal distribution $y \sim \mathcal{N}(\mu, \sigma^2)$: $S_L = \mu, \tau = \sigma, h(y, \tau) = \frac {e^{\frac {-y^2} {2 \tau^2}}} {\sqrt{2 \pi} \tau}, A(S_L) = \frac {S_L^2} {2}, d(\tau) = \tau^2$. $g_L(S_L) = \mathbb{E}[y|S_L] = S_L$, hence the output layer activation function $g_L$ is the identity function. This means that the linear function approximation of the previous section is exactly the same as a neural network with 0 hidden layers (just the output layer) and with the output layer activation function equal to the identity function.
* Bernoulli distribution for binary-valued $y$, parameterized by $p$: $S_L = \log{(\frac p {1-p})}, \tau = 1, h(y, \tau) = 1, d(\tau) = 1, A(S_L) = \log{(1+e^{S_L})}$. $g_L(S_L) = \mathbb{E}[y|S_L] = \frac 1 {1+e^{-S_L}}$, hence the output layer activation function $g_L$ is the logistic function. This generalizes to [softmax](https://en.wikipedia.org/wiki/Softmax_function) $g_L$ when we generalize this framework to multivariate $y$, which in turn enables us to classify inputs $x$ into a finite set of categories represented by $y$ as [one-hot-encodings](https://en.wikipedia.org/wiki/One-hot).
* Poisson distribution for $y$ parameterized by $\lambda$: $S_L = \log{\lambda}, \tau = 1, d(\tau) = 1, h(y, \tau) = \frac 1 {y!}, A(S_L) = e^{S_L}$. $g_L(S_L) = \mathbb{E}[y|S_L] = e^{S_L}$, hence the output layer activation function $g_L$ is the exponential function.

Now we are ready to write a class for function approximation with the deep neural network framework described above. We shall assume that the activation functions $g_l(\cdot)$ are identical for all $l = 0, 1, \ldots, L-1$ (known as the hidden layers activation function) and the activation function $g_L(\cdot)$ will be known as the output layer activation function. First, we write a `@dataclass` to hold the configuration of a deep neural network (number of neurons in the layers, hidden layers activation function and output layer activation function).

```python
@dataclass(frozen=True)
class DNNSpec:
    neurons: Sequence[int]
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    output_activation: Callable[[np.ndarray], np.ndarray]
```

Note that along with specifying the hidden and output layers activation functions $g_l(\cdot)$ defined as $g_l(\bm{S_l}) = \bm{O_l}$, we also specify the hidden layers activation function derivative (`hidden_activation_deriv`) in the form of a function $h_l(\cdot)$ defined as $h_l(g(\bm{S_l})) = h_l(\bm{O_l}) = g_l'(\bm{S_l})$ (as we know, this derivative is required in the back-propagation calculation). We shall soon see that in the code, $h_l(\cdot)$ is a more convenient specification than the direct specification of $g_l'(\cdot)$. Now we write the `@dataclass DNNApprox` that implements the abstract base class `FunctionApprox`. It has attributes:

* `feature_functions` that represents $\phi_j: \mathcal{X} \rightarrow \mathbb{R}$ for all $j = 1, 2, \ldots, m$
* `dnn_spec` that specifies the neural network configuration (instance of `DNNSpec`)
* `regularization_coeff` that represents the common regularization coefficient $\lambda$ for the weights across all layers
* `weights` which is a sequence of `Weights` objects (to represent and update the weights of all layers).

Note that the `get_feature_values` method adds a feature function $\phi(x) = 1$ for all $x \in \mathcal{X}$, so as to include the bias term. The method `forward_propagation` implements the forward-propagation calculation that was covered earlier (combining Equations \eqref{eq:layers_input_output_connect}, \eqref{eq:layer_linearity} and \eqref{eq:layer_non_linearity}), computing a sequence of $y$ values as an `np.ndarray` corresponding to each of the sequence of $x$ values provided as input in `x_values_seq`. `forward_propagation` returns a list whose last element represents the final output of the neural network $\bm{O_L} = \mathbb{E}_M[y|x]$ and the remaining elements represent $\bm{I_l}$ for all $l = 0, 1, \ldots L$ (for each of the $x$ values provided as input in `x_values_seq`). The method `evaluate` (an `@abstractmethod` in `FunctionApprox`) returns the last element ($\bm{O_L} = \mathbb{E}_M[y|x]$) from the output of `forward_propagation`. The method `backward_propagation` is the most important method of `DNNApprox` and deserves a detailed explanation.

`backward_propagation` takes as input `xy_vals_seq` which is a sequence of $(x,y)$ pairs, and the output of `backward_propagation` is an estimate of $\nabla_{\bm{w_l}} \mathcal{L} = \bm{P_l} \otimes \bm{I_l}$ (i.e., without the regularization term) for all $l = 0, 1, \ldots L$, using the input data of $(x,y)$ pairs. The first step in this method is to invoke `forward_propagation` and store the results in the variable `fwd_prop`, whose last element represents $\bm{O_L}$ and whose remaining elements represent $\bm{I_l}$ for all $l = 0, 1, \ldots L$ (for each of the $x$ values provided as input in `xy_vals_seq`). This sequence of $\bm{I_l}$ is stored in the variable `layer_inputs`. The variable `deriv` represents $\bm{P_l} = \nabla_{\bm{S_l}} \mathcal{L}$, computed for each of the $x$ values provided as input in `xy_vals_seq` (note that `deriv` is updated in each iteration of the loop). `deriv` is initialized to the value of $P_L = O_L - y$. Within the loop, we perform the calculations of Theorem \ref{th:recursive_gradient_formulation}: $\bm{P_l} = (\bm{w_{l+1}}^T \cdot \bm{P_{l+1}}) \circ g_l'(\bm{S_l})$ (updating the `deriv` variable) and Equation \eqref{eq:loss_gradient_formula}: $\nabla_{\bm{w_l}} \mathcal{L} = \bm{P_l} \otimes \bm{I_l}$ (storing the results in the variable `back_prop`).

The method `regularized_loss_gradient` simply adds on the regularization term $\lambda \cdot \bm{w_l}$ to the output of `backward_propagation`. Finally, the method `update` (`@abstractmethod` in `FunctionApprox`) invokes `regularized_loss_gradient` and returns a new instance of `DNNApprox` that contains the updated weights, along with the ADAM cache updates (invoking the `update` method of the `Weights` class to ensure there are no in-place updates).

```python
@dataclass(frozen=True)
class DNNApprox(FunctionApprox[X]):

    feature_functions: Sequence[Callable[[X], float]]
    dnn_spec: DNNSpec
    regularization_coeff: float
    weights: Sequence[Weights]

    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: DNNSpec,
        adam_gradient: AdamGradient,
        regularization_coeff: float = 0.,
        weights: Optional[Sequence[Weights]] = None
    ) -> DNNApprox[X]:
        if weights is None:
            augmented_layers = [len(feature_functions)] + \
                dnn_spec.neurons + [1]
            wts = [Weights.create(
                adam_gradient,
                np.random.randn(output, inp + 1) / np.sqrt(inp + 1)
            ) for inp, output in pairwise(augmented_layers)]
        else:
            wts = weights

        return DNNApprox(
            feature_functions=feature_functions,
            dnn_spec=dnn_spec,
            regularization_coeff=regularization_coeff,
            weights=wts
        )

    def get_feature_values(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.array([[1.] + [f(x) for f in self.feature_functions]
                         for x in x_values_seq])

    def forward_propagation(
        self,
        x_values_seq: Sequence[X]
    ) -> Sequence[np.ndarray]:
        inp: np.ndarray = self.get_feature_values(x_values_seq)
        outputs: List[np.ndarray] = [inp]
        for w in self.weights[:-1]:
            out: np.ndarray = self.dnn_spec.hidden_activation(
                np.dot(inp, w.weights.T)
            )
            inp: np.ndarray = np.insert(out, 0, 1., axis=1)
            outputs.append(inp)
        outputs.append(
            self.dnn_spec.output_activation(
                np.dot(inp, self.weights[-1].weights.T)
            )
        )
        return outputs

    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return self.forward_propagation(x_values_seq)[-1][:, 0]

    def backward_propagation(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> Sequence[np.ndarray]:
        x_vals, y_vals = zip(*xy_vals_seq)
        fwd_prop: Sequence[np.ndarray] = self.forward_propagation(x_vals)
        layer_inputs: Sequence[np.ndarray] = fwd_prop[:-1]
        deriv: np.ndarray = (
            fwd_prop[-1][:, 0] - np.array(y_vals)
        ).reshape(1, -1)
        back_prop: List[np.ndarray] = []
        for i in reversed(range(len(self.weights))):
            back_prop.append(np.dot(deriv, layer_inputs[i]) / deriv.shape[1])
            deriv = (np.dot(self.weights[i].weights.T, deriv) *
                     self.dnn_spec.hidden_activation_deriv(
                         layer_inputs[i].T
                     ))[1:]
        return back_prop[::-1]

    def regularized_loss_gradient(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> Sequence[np.ndarray]:
        return [x + self.regularization_coeff * self.weights[i].weights
                for i, x in enumerate(self.backward_propagation(xy_vals_seq))]

    def update(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> DNNApprox[X]:
        return replace(
            self,
            weights=[w.update(g) for w, g in zip(
                self.weights,
                self.regularized_loss_gradient(xy_vals_seq)
            )]
        )
```
We also require the `within` method, that simply delegates to the `within` method of the `Weights` class.

```python
    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, DNNApprox):
            return all(w1.within(w2, tolerance)
                       for w1, w2 in zip(self.weights, other.weights))
        else:
            return False
```

We also need a couple of utilities to be able to use `LinearFunctionApprox` and `DNNApprox` objects:

* `sgd` which performs a stochastic gradient descent for an instance of `FunctionApprox` using an Iterator on a sequence of $(x,y)$ pairs (note how `sgd` calls `update` repeatedly).
* `rmse` which calculates the Root-Mean-Squared-Error of the predictions from an instance of `FunctionApprox`, based on a sequence of $(x,y)$ pairs of test data.

```python
def sgd(
    func_approx: FunctionApprox[X],
    xy_seq_stream: Iterator[Sequence[Tuple[X, float]]]
) -> Iterator[FunctionApprox[X]]:
    for xy_seq in xy_seq_stream:
        yield func_approx
        func_approx = func_approx.update(xy_seq)

def rmse(
    func_approx: FunctionApprox[X],
    xy_seq: Sequence[Tuple[X, float]]
) -> float:
    x_seq, y_seq = zip(*xy_seq)
    errors: np.ndarray = func_approx.evaluate(x_seq) - np.array(y_seq)
    return np.sqrt(np.mean(errors * errors))
```

All of the above code is in the file [rl/function_approx.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/function_approx.py).   

Let us now write some code to create function approximations with `LinearFunctionApprox` and `DNNApprox`, given a stream of data from a simple data model - one that has some noise around a linear function. Here's some code to create an Iterator of $(x, y)$ pairs (where $x = (x_1, x_2, x_3)$) for the data model:

$$y = 2 + 10x_1 +4x_2 - 6x_3 + \mathcal{N}(0, 0.3)$$

```python
def example_model_data_generator() -> Iterator[Tuple[Triple, float]]:

    coeffs: Aug_Triple = (2., 10., 4., -6.)
    d = norm(loc=0., scale=0.3)

    while True:
        pt: np.ndarray = np.random.randn(3)
        x_val: Triple = (pt[0], pt[1], pt[2])
        y_val: float = coeffs[0] + np.dot(coeffs[1:], pt) + \
            d.rvs(size=1)[0]
        yield (x_val, y_val)
```

Next we wrap this in an Iterator that returns a certain number of $(x,y)$ pairs upon each request for data points.

```python
def data_seq_generator(
    data_generator: Iterator[Tuple[Triple, float]],
    num_pts: int
) -> Iterator[DataSeq]:
    while True:
        pts: DataSeq = islice(data_generator, num_pts)
        yield pts
```

Now let's write a function to create a `LinearFunctionApprox`.

```python
def feature_functions():
    return [lambda x: x[0], lambda x: x[1], lambda x: x[2]]

def adam_gradient():
    return AdamGradient(
        learning_rate=0.1,
        decay1=0.9,
        decay2=0.999
    )

def get_linear_model() -> LinearFunctionApprox[Triple]:
    ffs = feature_functions()
    ag = adam_gradient()
    return LinearFunctionApprox.create(
         feature_functions=ffs,
         adam_gradient=ag,
         regularization_coeff=0.
    )
```   

Likewise, let's write a function to create a `DNNApprox` with 1 hidden layer with 2 neurons and a little bit of regularization since this deep neural network is somewhat over-parameterized to fit the data generated from the linear data model with noise.

```python
def get_dnn_model() -> DNNApprox[Triple]:
    ffs = feature_functions()
    ag = adam_gradient()

    def relu(arg: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: x if x > 0. else 0.)(arg)

    def relu_deriv(res: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: 1. if x > 0. else 0.)(res)

    def identity(arg: np.ndarray) -> np.ndarray:
        return arg

    ds = DNNSpec(
        neurons=[2],
        hidden_activation=relu,
        hidden_activation_deriv=relu_deriv,
        output_activation=identity
    )

    return DNNApprox.create(
        feature_functions=ffs,
        dnn_spec=ds,
        adam_gradient=ag,
        regularization_coeff=0.05
    )
```

Now let's write some code to do a `direct_solve` with the `LinearFunctionApprox` based on the data from the data model we have set up. 

```python
training_num_pts: int = 1000
test_num_pts: int = 10000
training_iterations: int = 300
data_gen: Iterator[Tuple[Triple, float]] = example_model_data_generator()
training_data_gen: Iterator[DataSeq] = data_seq_generator(
    data_gen,
    training_num_pts
)
test_data: DataSeq = list(islice(data_gen, test_num_pts))

direct_solve: LinearFunctionApprox[Triple] = \
    get_linear_model().direct_solve(next(training_data_gen))
direct_solve_rmse: float = rmse(direct_solve, test_data)
```

Running the above code, we see that the Root-Mean-Squared-Error (`direct_solve_rmse`) is indeed 0.3, matching the standard deviation of the noise in the linear data model (which is used above to generate the training data as well as the test data).

Now let us perform stochastic gradient descent with instances of `LinearFunctionApprox` and `DNNApprox` and examine the Root-Mean-Squared-Errors on the two function approximations as a function of number of iterations in the gradient descent.

```python
linear_model_rmse_seq: Sequence[float] = \
    [rmse(lfa, test_data) for lfa in islice(
        sgd(get_linear_model(), training_data_gen),
        training_iterations
    )]

dnn_model_rmse_seq: Sequence[float] = \
    [rmse(dfa, test_data) for dfa in islice(
        sgd(get_dnn_model(), training_data_gen),
        training_iterations
    )]
```

The plot of `linear_model_rmse_seq` and `dnn_model_rmse_seq` is shown in Figure \ref{fig:sgd_convergence}.

<div style="text-align:center" markdown="1">
![SGD Convergence \label{fig:sgd_convergence}](./chapter5/rmse.png "SGD Convergence")
</div>

## Tabular as a form of `FunctionApprox`

Now we consider the simple case where we have a fixed set of $x$-values $\mathcal{X} = \{x_1, x_2, \ldots, x_n\}$, i.e., the $n$ $x$-values are stored in a table, and the corresponding prediction for each $x \in \mathcal{X}$ (to be stored in the table next to $x$) needs to be calculated only from the $y$-values associated with $x$ within the data set of $(x,y)$ pairs. In other words, the $y$-values in the data associated with other $x$ should not influence the prediction for $x$. Since we'd like the prediction for $x$ to be $\mathbb{E}[y|x]$, it would make sense for the prediction for a given $x$ to be the average of all the $y$ associated with $x$ within the data set of $(x,y)$ pairs seen so far. So the calculations for Tabular prediction are particularly straightforward. What is interesting though is the fact that Tabular prediction actually fits the interface of `FunctionApprox` in terms of implementing an `update` function (updating the average of $y$-values from each new $y$ in the data, associated with a given $x$) and an `evaluate` function (simply reporting the currently calculated average of $y$-values, associated with a given $x$). This view of Tabular prediction as a special case of `FunctionApprox` also permits us to cast the tabular algorithms of Dynamic Programming and Reinforcement Learning as special cases of the function approximation versions of the algorithms (using the `Tabular` class we develop below).

So now let us write the code for `@dataclass Tabular` as an implementation of the abstract base class `FunctionApprox`. The attributes of `@dataclass Tabular` are:

* `values_map` which is a dictionary mapping each $x$ value to the average of the $y$-values seen so far in the data, associated with $x$.
* `counts_map` which is a dictionary mapping each $x$ value to the count of $y$-values seen so far in the data, associated with $x$. We need to track the count of $y$-values associated with each $x$ because this enables us to update `values_map` appropriately upon seeing a new $y$-value associated a given $x$.
* `count_to_weight_func` which defines a function from number of $y$-values seen so far (associated with a given $x$) to the weight assigned to the most recent $y$. This enables us to do a weighted average of the $y$-values seen so far, controlling the emphasis to be placed on more recent $y$-values relative to previously seen $y$-values (associated with a given $x$).

The `evaluate`, `update` and `within` methods are now self-explanatory.

```python
@dataclass(frozen=True)
class Tabular(FunctionApprox[X]):

    values_map: Mapping[X, float] =\
        field(default_factory=lambda: defaultdict(float))
    counts_map: Mapping[X, int] =\
        field(default_factory=lambda: defaultdict(int))
    count_to_weight_func: Callable[[int], float] =\
        field(default_factory=lambda: lambda n: 1. / n)

    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.array([self.values_map[x] for x in x_values_seq])

    def update(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> Tabular[X]:
        values_map: Dict[X, float] = self.values_map.copy()
        counts_map: Dict[X, int] = self.counts_map.copy()
        for x, y in xy_vals_seq:
            counts_map[x] += 1
            weight: float = self.count_to_weight_func(counts_map[x])
            values_map[x] += weight * (y - values_map[x])
        return replace(
            self,
            values_map=values_map,
            counts_map=counts_map
        )

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, Tabular):
            return\
                all(abs(self.values_map[s] - other.values_map[s]) <= tolerance
                    for s in self.values_map)
        else:
            return False
```
   
Note that in the tabular Dynamic Programming algorithms, the set of finite states take the role of $\mathcal{X}$ and the Value Function for a given state $x=s$ takes the role of the "predicted" $y$-value associated with $x$. We also note that in the Dynamic Programming algorithms, in each iteration of sweeping through all the states, the Value Function for a state $x=s$ is set to the current $y$ value (not the average of all $y$-values seen so far). The current $y$-value is simply the right-hand-side of the Bellmen Equation corresponding to the Dynamic Programmming algorithm. Consequently, for tabular Dynamic Programming, we'd need to set `count_to_weight_func` to be the function `lambda _: 1` (this is because a weight of 1 for the current $y$-value sets `values_map[x]` equal to the current $y$-value). Later, when we get to Reinforcement Learning algorithms, we will be averaging all the Returns observed for a given state. If we choose to do a plain average (equal importance for all $y$-values see so far, associated with a given $x$), then we'd need to set `count_to_weights_func` to be the function `lambda n: 1. / n`. Note that this also means tabular RL is a special case of RL with linear function approximation by setting a feature function $\phi_i(\cdot)$ for each $x_i$ as: $\phi_i(x) = 1$ for $x=x_i$ and $\phi_(x) = 0$ for each $x \neq x_i$ (i.e., $\phi_i(x)$ is the indicator function for $x_i$, and the $\bm{\Phi}$ matrix is the identity matrix). This also means that the `count_to_weights_func` plays the role of the learning rate function (as a function of the number of iterations in stochastic gradient descent). Please do bear this in mind when we get to tabular RL.

Again, we want to emphasize that tabular algorithms are just a special case of algorithms with function approximation. However, we give special coverage in this book to tabular algorithms because they help us conceptualize the core concepts in a simple (tabular) setting without the distraction of some of the details and complications in the apparatus of function approximation.

Now we are ready to write algorithms for Approximate Dynamic Programming (ADP).

## Approximate Policy Evaluation

The first ADP algorithm we cover is Approximate Policy Evaluation, i.e., evaluating the Value Function for a Markov Reward Process (MRP). Approximate Policy Evaluation is fundamentally the same as Tabular Policy Evaluation in terms of repeatedly applying the Bellman Policy Operator $\bm{B^}\pi$ on the Value Function $V: \mathcal{N} \rightarrow \mathbb{R}$. However, unlike Tabular Policy Evaluation algorithm, the Value Function $V(\cdot)$ is set up and updated as an instance of `FunctionApprox` rather than as a table of values for the states. This is because unlike Tabular Policy Evaluation which operates on an instance of a `FiniteMarkovRewardProcess`, Approximate Policy Evaluation algorithm operates on an instance of `MarkovRewardProcess`. So we do not have an enumeration of states of the MRP and we do not have the transition probabilities of the MRP. This is typical in many real-world problems where the state space is either very large or is continuous-valued, and the transitions could be too many or could be continuous-valued transitions. So, here's what we do to overcome these challenges:

* We specify a sampling probability distribution of non-terminal states (argument `non_terminal_states_distribution` in the code below) from which we shall sample a specified number (`num_state_samples` in the code below) of state samples and construct a list of non-terminal states (`nt_states` in the code below) in each iteration.
* We sample pairs of (next state $s'$, reward $r$) from a given state $s$, and calculate the expectation $\mathbb{E}[r + \gamma \cdot V(s')]$ by averaging $r+\gamma \cdot V(s')$ across the sampled pairs. Note that the method `expectation` of a `Distribution` object performs a sampled expectation. $V(s')$ is obtained from the function approximation instance of `FunctionApprox` that is being updated in each iteration.
* The sampled list of non-terminal states $s$ comprise our $x$-values and the associated sampled expectations described above comprise our $y$-values. This list of $(x,y)$ pairs are used to update the approximation of the Value Function in each iteration (producing a new instance of `FunctionApprox` using it's `update` method).

The entire code is shown below. `evaluate_mrp` produces an `Iterator` on `FunctionApprox` instances, and the code that calls `evaluate_mrp` can decide when/how to terminate the iterations of Approximate Policy Evaluation.

```python
from rl.iterate import iterate

def evaluate_mrp(
    mrp: MarkovRewardProcess[S],
    gamma: float,
    approx_0: FunctionApprox[S],
    non_terminal_states_distribution: Distribution[S],
    num_state_samples: int
) -> Iterator[FunctionApprox[S]]:

    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        nt_states: Sequence[S] = non_terminal_states_distribution.sample_n(
            num_state_samples
        )

        def return_(s_r: Tuple[S, float]) -> float:
            s, r = s_r
            return r + gamma * v.evaluate([s]).item()

        return v.update(
            [(s, mrp.transition_reward(s).expectation(return_))
             for s in nt_states]
        )

    return iterate(update, approx_0)
```

## Approximate Value Iteration

Now that we've understood and coded Approximate Policy Evaluation (to solve the Prediction problem), we can extend the same concepts to Approximate Value Iteration (to solve the Control problem). The code below in `value_iteration` is almost the same as the code above in `evaluate_mrp`, except that instead of a `MarkovRewardProcess` at each time step, here we have a `MarkovDecisionProcess` at each time step, and instead of the Bellman Policy Operator update, here we have the Bellman Optimality Operator update. Therefore, in the Value Function update, we maximize the $Q$-value function (over all actions $a$) for each state $s$. Also, similar to `evaluate_mrp`, `value_iteration` produces an `Iterator` on `FunctionApprox` instances, and the code that calls `value_iteration` can decide when/how to terminate the iterations of Approximate Value Iteration.

```python
from rl.iterate import iterate

def value_iteration(
    mdp: MarkovDecisionProcess[S, A],
    gamma: float,
    approx_0: FunctionApprox[S],
    non_terminal_states_distribution: Distribution[S],
    num_state_samples: int
) -> Iterator[FunctionApprox[S]]:

    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        nt_states: Sequence[S] = non_terminal_states_distribution.sample_n(
            num_state_samples
        )

        def return_(s_r: Tuple[S, float]) -> float:
            s, r = s_r
            return r + gamma * v.evaluate([s]).item()

        return v.update(
            [(s, max(mdp.step(s, a).expectation(return_,)
                     for a in mdp.actions(s)))
             for s in nt_states]
        )

    return iterate(update, approx_0)

```

## Finite-Horizon Approximate Policy Evaluation

Next, we move on to Approximate Policy Evaluation in a finite-horizon setting, meaning we will perform Approximate Policy Evaluation with a backward induction algorithm, much like how we did backward induction for finite-horizon Tabular Policy Evaluation. We will of course make the same types of adaptations from Tabular to Approximate as we did in the functions `evaluate_mrp` and `value_iteration` above.

In the `backward_evaluate` code below, the input argument `mrp_f0_mu_triples` is a list of triples, with each triple corresponding to each non-terminal time step in the finite horizon. Each triple consists of:

* An instance of `MarkovRewardProceess` (note that each time step has it's own instance of `MarkovRewardProcess` representation of transitions from the states $s$ in one time step $t$ to the (state $s'$, reward $r$) pairs in the next time step $t+1$ (variable `mrp` in the code below).
* An instance of `FunctionApprox` to capture the approximate Value Function for the time step (variable `approx0` in the code below, representing the initial `FunctionApprox` instance).
* A sampling probability distribution of states in the time step (variable `mu` in the code below).

The backward induction code below should be pretty self-explanatory. Note that in backward induction, we don't invoke the `update` method of `FunctionApprox` like we did in the non-finite-horizon cases - here we invoke the `sgd` method (that we wrote earlier in this chapter) which internally performs a sequence of `update`s on the `FunctionApprox` for a given time step (until we converge to within a specified level of `error_tolerance`). In the non-finite-horizon cases, it was okay to simply do a single `update` in each iteration because we revisit the same set of states in further iterations. Here, once we converge to an acceptable `FunctionApprox` (using `sgd`) for a specific time step, we won't be performing any more updates to the Value Function for that time step (since we move on to the next time step, in reverse). `backward_evaluate` returns an Iterator over `FunctionApprox` objects, from time step 0 to the horizon time step.

```python
from itertools import repeat
from rl.function_approx import sgd

MRP_FuncApprox_Distribution = \
    Tuple[MarkovRewardProcess[S], FunctionApprox[S], Distribution[S]]

def backward_evaluate(
    mrp_f0_mu_triples: Sequence[MRP_FuncApprox_Distribution[S]],
    gamma: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[FunctionApprox[S]]:
    v: List[FunctionApprox[S]] = []

    for i, (mrp, approx0, mu) in enumerate(reversed(mrp_f0_mu_triples)):

        def return_(s_r: Tuple[S, float], i=i) -> float:
            s, r = s_r
            return r + gamma * (v[i-1].evaluate([s]).item() if i > 0 else 0.)

        v.append(
            FunctionApprox.converged(
                sgd(
                    approx0,
                    repeat([(s, mrp.transition_reward(s).expectation(return_))
                            for s in mu.sample_n(num_state_samples)])
                ),
                error_tolerance
            )
        )

    return reversed(v)
```

## Finite-Horizon Approximate Value Iteration

Now that we've understood and coded finite-horizon Approximate Policy Evaluation (to solve the finite-horizon Prediction problem), we can extend the same concepts to finite-horizon Approximate Value Iteration (to solve the finite-horizon Control problem). The code below in `back_opt_vf_and_policy` is almost the same as the code above in `backward_evaluate`, except that instead of a `MarkovRewardProcess`, here we have a `MarkovDecisionProcess`. For each non-terminal time step, we maximize the $Q$-value function (over all actions $a$) for each state $s$. `back_opt_vf_and_policy` returns an Iterator over pairs of `FunctionApprox` and `Policy` objects (representing the Optimal Value Function and the Optimal Policy respectively), from time step 0 to the horizon time step.

```python
from itertools import repeat
from rl.function_approx import sgd
from rl.distribution import Constant
from operator import itemgetter

MDP_FuncApprox_Distribution = \
    Tuple[MarkovDecisionProcess[S, A], FunctionApprox[S], Distribution[S]]

def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApprox_Distribution[S, A]],
    gamma: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[Tuple[FunctionApprox[S], Policy[S, A]]]:
    vp: List[Tuple[FunctionApprox[S], Policy[S, A]]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[S, float], i=i) -> float:
            s, r = s_r
            return r + gamma * (vp[i-1][0].evaluate([s]).item() if i > 0 else 0.)

        this_v = FunctionApprox.converged(
            sgd(
                approx0,
                repeat([(
                    s,
                    max(mdp.step(s, a).expectation(return_)
                        for a in mdp.actions(s))
                ) for s in mu.sample_n(num_state_samples)])
            ),
            error_tolerance
        )

        class ThisPolicy(Policy[S, A]):
            def act(self, state: S) -> Constant[A]:
                return Constant(max(
                    ((mdp.step(state, a).expectation(return_), a)
                     for a in mdp.actions(state)),
                    key=itemgetter(0)
                )[1])

        vp.append((this_v, ThisPolicy()))

    return reversed(vp)
```

All of the above code for Approximate Dynamic Programming algorithms is in the file [rl/approximate_dynamic_programming.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/approximate_dynamic_programming.py). We encourage you to create instances of `MarkovRewardProcess` and `MarkovDecisionProcess` (including finite-horizon instances) and play with the above ADP code with different choices of function approximations, state sampling distributions, and number of samples. A simple but valuable exercise is to reproduce the tabular versions of these algorithms by using the `Tabular` implementation of `FunctionApprox` (note: the `count_to_weights_func` would need to be lambda _: 1.) in the above ADP functions.

## Key Takeaways from this Chapter

* The Function Approximation interface involves two key methods - A) updating the parameters of the Function Approximation based on training data available from each iteration of a data stream, and B) evaluating the expectation of the response variable whose conditional probability distribution is modeled by the Function Approximation. Linear Function Approximation and Deep Neural Network Function Approximation are the two main Function Approximations we've implemented and will be using in the rest of the book.
* Tabular is a special type of Function Approximation, and Tabular RL is a special case of linear function approximation with feature functions as indicator functions for each of the states.
* All the Tabular DP algorithms can be generalized to ADP algorithms replacing tabular Value Function updates with updates to Function Approximation parameters (where the Function Approximation represents the Value Function). Sweep over all states in the tabular case is replaced by sampling states in the ADP case. Expectation calculations in Bellman Operators are handled in ADP as averages of the corresponding calculations over transition samples (versus calculations using explicit transition probabilities in the tabular algorithms).
