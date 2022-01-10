## Introduction to and Overview of Stochastic Calculus Basics {#sec:stochasticcalculus-appendix}

In this Appendix, we provide a quick introduction to the *Basics of Stochastic Calculus*. To be clear, Stochastic Calculus is a vast topic requiring an entire graduate-level course to develop a good understanding. We shall only be scratching the surface of Stochastic Calculus and even with the very basics of this subject, we will focus more on intuition than rigor, and familiarize you with just the most important results relevant to this book. For an adequate treatment of Stochastic Calculus relevant to Finance, we recommend Steven Shreve's two-volume discourse [Stochastic Calculus for Finance I](https://www.amazon.com/Stochastic-Calculus-Finance-Binomial-Springer/dp/0387249680) [@shreve03] and [Stochastic Calculus for Finance II](https://www.amazon.com/Stochastic-Calculus-Finance-II-Continuous-Time/dp/144192311X) [@shreve04]. For a broader treatment of Stochastic Calculus, we recommend [Bernt Oksendal's book on Stochastic Differential Equations](https://www.amazon.com/Stochastic-Differential-Equations-Introduction-Applications/dp/3540047581) [@UBMA_106106503]. 

### Simple Random Walk

The best way to get started with Stochastic Calculus is to first get familiar with key properties of a *simple random walk* viewed as a discrete-time, countable state-space, time-homogeneous Markov Process. The state space is the set of integers $\mathbb{Z}$. Denoting the random state at time $t$ as $Z_t$, the state transitions are defined in terms of the independent and identically distributed (i.i.d.) random variables $Y_t$ for all $t = 0, 1, \ldots$

$$Z_{t+1} = Z_t + Y_t \mbox{ and } \mathbb{P}[Y_t = 1] = \mathbb{P}[Y_t = -1] = 0.5 \mbox{ for all } t = 0, 1, \ldots$$

A quick point on notation: We refer to the random state at time $t$ as $Z_t$ (i.e., as a random variable at time $t$), whereas we refer to the Markov Process for this simple random walk as $Z$ (i.e., without any subscript).

Since the random variables $\{Y_t|t = 0, 1, \ldots\}$ are i.i.d, the *increments* $Z_{t_{i+1}} - Z_{t_i}, i = 0, 1, \ldots n-1$ in the random walk states for any set of time steps $t_0 < t_1 < \ldots < t_n$ have the following properties:

* **Independent Increments**: Increments $Z_{t_1} - Z_{t_0}, Z_{t_2} - Z_{t_1}, \ldots, Z_{t_n} - Z_{t_{n-1}}$ are independent of each other
* **Martingale (i.e., Zero-Drift) Property**: Expected Value of Increment $\mathbb{E}[(Z_{t_{i+1}} - Z_{t_i})] = 0$ for all $i = 0, 1, \ldots, n-1$
* **Variance of Increment equals Time Steps**: Variance of Increment
$$\mathbb{E}[(Z_{t_{i+1}} - Z_{t_i})^2] = \sum_{j=t_i}^{t_{i+1} - 1} \mathbb{E}[(Z_{j+1} - Z_j)^2] = t_{i+1} - t_i \mbox{ for all } i = 0, 1, \ldots, n-1$$

Moreover, we have an important property that **Quadratic Variation equals Time Steps**. Quadratic Variation over the time interval $[t_i, t_{i+1}]$ for all $i = 0, 1, \ldots, n-1$ is defined as:

$$\sum_{j=t_i}^{t_{i+1} - 1} (Z_{j+1} - Z_j)^2$$

Since $(Z_{j+1} - Z_j)^2 = Y_j^2 = 1$ for all $j = t_i, t_i + 1, \ldots, t_{i+1} - 1$, Quadratic Variation

$$\sum_{j=t_i}^{t_{i+1} - 1} (Z_{j+1} - Z_j)^2 = t_{i+1} - t_i \mbox{ for all } i = 0, 1, \ldots n-1$$

It pays to emphasize the important conceptual difference between the Variance of Increment property and Quadratic Variation property: Variance of Increment property is a statement about *expectation* of the square of the $Z_{t_{i+1}} - Z_{t_i}$ increment whereas Quadratic Variation property is a statement of certainty (note: there is no $\mathbb{E}[\cdots]$ in this statement) about the sum of squares of *atomic* increments $Y_j$ over the discrete-steps time-interval $[t_i, t_{i+1}]$. The Quadratic Variation property owes to the fact that $\mathbb{P}[Y_t^2 = 1] = 1$ for all $t = 0, 1, \ldots$.

We can view the Quadratic Variations of a Process $X$ over all discrete-step time intervals $[0, t]$ as a Process denoted $[X]$, defined as:

$$[X]_t = \sum_{j=0}^t (X_{j+1} - X_j)^2$$

Thus, for the simple random walk Markov Process $Z$, we have the succinct formula: $[Z]_t = t$ for all $t$ (i.e., this Quadratic Variation process is a deterministic process).

### Brownian Motion as Scaled Random Walk

Now let us take our simple random walk process $Z$, and simultaneously A) speed up time and B) scale down the size of the atomic increments $Y_t$. Specifically, define for any fixed positive integer $n$:

$$z^{(n)}_t = \frac 1 {\sqrt{n}} \cdot Z_{nt} \mbox{ for all } t \in \frac {\mathbb{Z}_{\geq 0}} n$$

It's easy to show that the above properties of the simple random walk process holds for the $z^{(n)}$ process as well. Now consider the continuous-time process $z$ defined as:

$$z_t = \lim_{n\rightarrow \infty} z^{(n)}_t \mbox{ for all } t \in \mathbb{R}_{\geq 0}$$

This continuous-time process $z$ with $z_0 = 0$ is known as standard Brownian Motion. $z$ retains the same properties as those of the simple random walk process that we have listed above (independent increments, martingale, increment variance equal to time interval, and quadratic variation equal to the time interval). Also, by Central Limit Theorem,

$$z_t | z_s \sim \mathcal{N}(z_s, t-s) \mbox{ for any } 0 \leq s < t$$

We denote $dz_t$ as the increment in $z$ over the infinitesimal time interval $[t, t + dt]$.

$$dz_t \sim \mathcal{N}(0, dt)$$

### Continuous-Time Stochastic Processes

Brownian motion $z$ was our first example of a continuous-time stochastic process. Now let us define a general continuous-time stochastic process, although for the sake of simplicity, we shall restrict ourselves to one-dimensional real-valued continuous-time stochastic processes.

\begin{definition}
A {\em One-dimensional Real-Valued Continuous-Time Stochastic Process} denoted $X$ is defined as a collection of real-valued random variables $\{X_t|t \in [0, T]\}$ (for some fixed $T \in \mathbb{R}$, with index $t$ interpreted as continuous-time) defined on a common probability space $(\Omega, \mathcal{F}, \mathbb{P})$, where $\Omega$ is a sample space, $\mathcal{F}$ is a $\sigma$-algebra and $\mathbb{P}$ is a probability measure (so, $X_t: \Omega \rightarrow \mathbb{R}$ for each $t \in [0, T])$.
\end{definition}

We can view a stochastic process $X$ as an $\mathbb{R}$-valued function of two variables:

* $t \in [0, T]$
* $\omega \in \Omega$

As a two-variable function, if we fix $t$, then we get the random variable $X_t: \Omega \rightarrow \mathbb{R}$ for time $t$ and if we fix $\omega$, then we get a single $\mathbb{R}$-valued outcome for each random variable across time (giving us a *sample path* in time, denoted $X(\omega)$).

Now let us come back to Brownian motion, viewed as a Stochastic Process.

### Properties of Brownian Motion sample paths

* Sample paths $z(\omega)$ of Brownian motion $z$ are continuous
* Sample paths $z(\omega)$ are almost always non-differentiable, meaning:
$$\mbox{Random variable } \lim_{h \rightarrow 0} \frac {z_{t+h} - z_t} {h} \mbox{ is almost always infinite}$$
The intuition is that $\frac {dz_t} {dt}$ has standard deviation of $\frac 1 {\sqrt{dt}}$, which goes to $\infty$ as $dt$ goes to 0
* Sample paths $z(\omega)$ have infinite total variation, meaning:
$$\mbox{Random variable } \int_S^T |dz_t| = \infty \mbox{ (almost always)}$$

The quadratic variation property can be expressed as:

$$\int_S^T (dz_t)^2 = T-S$$

This means each sample random path of brownian motion has quadratic variation equal to the time interval of the path. The quadratic variation of $z$ expressed as a process $[z]$ has the deterministic value of $t$ at time $t$. Expressed in infinitesimal terms, we say that:

$$(dz_t)^2 = dt$$

This formula generalizes to:

$$(dz^{(1)}_t) \cdot (dz^{(2)}_t) = \rho \cdot dt$$

where $z^{(1)}$ and $z^{(2)}$ are two different brownian motions with correlation between the random variables $z^{(1)}_t$ and $z^{(2)}_t$ equal to $\rho$ for all $t > 0$.

You should intuitively interpret the formula $(dz_t)^2 = dt$ (and it's generalization) as a deterministic statement, and in fact this statement is used as an algebraic convenience in Brownian motion-based stochastic calculus, forming the core of *Ito Isometry* and *Ito's Lemma* (which we cover shortly, but first we need to define the Ito Integral).

### Ito Integral

We want to define a stochastic process $Y$ from a stochastic process $X$ as follows:

$$Y_t = \int_0^t X_s \cdot dz_s$$

In the interest of focusing on intuition rather than rigor, we skip the technical details of filtrations and adaptive processes that make the above integral sensible. Instead, we simply say that this integral makes sense only if random variable $X_s$ for any time $s$ is disallowed from depending on $z_{s'}$ for any $s' > s$ (i.e., the stochastic process $X$ cannot peek into the future) and that the time-integral
$\int_0^t X_s^2 \cdot ds$ is finite for all $t \geq 0$. So we shall roll forward with the assumption that the stochastic process $Y$ is defined as the above-specified integral (known as the *Ito Integral*) of a stochastic process $X$ with respect to Brownian motion. The equivalent notation is:

$$dY_t = X_t \cdot dz_t$$

We state without proof the following properties of the Ito Integral stochastic process $Y$:

* $Y$ is a martingale, i.e., $\mathbb{E}[Y_t|Y_s] = 0$ for all $0 \leq s < t$
* **Ito Isometry**: $\mathbb{E}[Y_t^2] = \int_0^t \mathbb{E}[X_s^2] \cdot ds$. 
* Quadratic Variance formula: $[Y]_t = \int_0^t X_s^2 \cdot ds$

Ito Isometry generalizes to:

$$\mathbb{E}[(\int_S^T X^{(1)}_t \cdot dz^{(1)}_t)(\int_S^T X^{(2)}_t \cdot dz^{(2)}_t)] = \int_S^T \mathbb{E}[X^{(1)}_t\cdot X^{(2)}_t \cdot \rho \cdot dt]$$

where $X^{(1)}$ and $X^{(2)}$ are two different stochastic processes, and $z^{(1)}$ and $z^{(2)}$ are two different brownian motions with correlation between the random variables $z^{(1)}_t$ and $z^{(2)}_t$ equal to $\rho$ for all $t > 0$.

Likewise, the Quadratic Variance formula generalizes to:

$$\int_S^T (X^{(1)}_t \cdot dz^{(1)}_t)(X^{(2)}_t \cdot dz^{(2)}_t) = \int_S^T X^{(1)}_t\cdot X^{(2)}_t \cdot \rho \cdot dt$$

### Ito's Lemma {#sec:itos-lemma-section}

We can extend the above Ito Integral to an Ito process $Y$ as defined below:

$$dY_t = \mu_t \cdot dt + \sigma_t \cdot dz_t$$

We require the same conditions for the stochastic process $\sigma$ as we required above for $X$ in the definition of the Ito Integral. Moreover, we require that: $\int_0^t |\mu_s| \cdot ds$ is finite for all $t\geq 0$.

In the context of this Ito process $Y$ described above, we refer to $\mu$ as the *drift* process and we refer to $\sigma$ as the *dispersion* process.

Now, consider a twice-differentiable function $f: [0, T] \times \mathbb{R} \rightarrow \mathbb{R}$. We define a stochastic process whose (random) value at time $t$ is $f(t, Y_t)$. Let's write it's Taylor series with respect to the variables $t$ and $Y_t$.

$$df(t,Y_t) = \pdv{f(t,Y_t)}{t} \cdot dt + \pdv{f(t,Y_t)}{Y_t} \cdot dY_t + \frac 1 2 \cdot \pdv[2]{f(t,Y_t)}{Y_t} \cdot (dY_t)^2 + \ldots$$

Substituting for $dY_t$ and lightening notation, we get:

$$df(t, Y_t) = \pdv{f}{t} \cdot dt + \pdv{f}{Y_t} \cdot (\mu_t \cdot dt + \sigma_t \cdot dz_t) + \frac 1 2 \cdot \pdv[2]{f}{Y_t} \cdot (\mu_t \cdot dt + \sigma_t \cdot dz_t )^2 + \ldots$$

Next, we use the rules: $(dt)^2 = 0, dt \cdot dz_t = 0, (dz_t)^2 = dt$ to get **Ito's Lemma**:

\begin{equation}
df(t, Y_t) = (\pdv{f}{t} + \mu_t \cdot \pdv{f}{Y_t} + \frac {\sigma_t^2} 2 \cdot \pdv[2]{f}{Y_t}) \cdot dt + \sigma_t \cdot \pdv{f}{Y_t} \cdot dz_t
\label{eq:itos-lemma}
\end{equation}

Ito's Lemma describes the stochastic process of a function ($f$) of an Ito Process ($Y$) in terms of the partial derivatives of $f$, and in terms of the drift ($\mu$) and dispersion ($\sigma$) processes that define $Y$.

If we generalize $\bm{Y}$ to be an $n$-dimensional stochastic process (as a column vector) with $\bm{\mu}_t$ as an $n$-dimensional (stochastic) column vector, $\bm{\sigma}_t$ as an $n \times m$ (stochastic) matrix, and $\bm{z}_t$ as an $m$-dimensional vector of $m$ independent standard brownian motions (as follows)

$$d\bm{Y}_t = \bm{\mu}_t \cdot dt + \bm{\sigma}_t \cdot d\bm{z}_t$$

then we get the multi-variate version of Ito's Lemma, as follows:

\begin{equation}
df(t, \bm{Y}_t) = (\pdv{f}{t} + (\nabla_{\bm{Y}} f)^T \cdot \bm{\mu}_t + \frac 1 2 Tr[\bm{\sigma}_t^T \cdot (\Delta_{\bm{Y}} f) \cdot \bm{\sigma}_t]) \cdot dt + (\nabla_{\bm{Y}} f)^T \cdot \bm{\sigma}_t \cdot d\bm{z}_t
\label{eq:itos-lemma-multi}
\end{equation}

where the symbol $\nabla$ represents the gradient of a function, the symbol $\Delta$ represents the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) of a function, and the symbol $Tr$ represents the [Trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) of a matrix.

Next, we cover two common Ito processes, and use Ito's Lemma to solve the Stochastic Differential Equation represented by these Ito Processes:

### A Lognormal Process {#sec:lognormal-process-section}

Consider a stochastic process $x$ described in the form of the following Ito process:

$$dx_t = \mu(t) \cdot x_t \cdot dt + \sigma(t) \cdot x_t \cdot dz_t$$

Note that here $z$ is standard (one-dimensional) Brownian motion, and $\mu$, $\sigma$ are deterministic functions of time $t$. This is solved easily by defining an appropriate function of $x_t$ and applying Ito's Lemma, as follows:

$$y_t = \log(x_t)$$

Applying Ito's Lemma on $y_t$ with respect to $x_t$, we get:

\begin{align*}
dy_t & = (\mu(t) \cdot x_t \cdot \frac 1 {x_t} - \frac {\sigma^2(t) \cdot x_t^2} 2 \cdot \frac 1 {x_t^2}) \cdot dt + \sigma(t) \cdot x_t \cdot \frac 1 {x_t} \cdot dz_t \\
& = (\mu(t) - \frac {\sigma^2(t)} 2) \cdot dt + \sigma(t) \cdot dz_t
\end{align*}

So,
$$y_T = y_S + \int_S^T (\mu(t) - \frac {\sigma^2(t)} 2) \cdot dt + \int_S^T  \sigma(t) \cdot dz_t$$
$$x_T = x_S \cdot e^{\int_S^T (\mu(t) - \frac {\sigma^2(t)} 2) \cdot dt + \int_S^T  \sigma(t) \cdot dz_t}$$

$x_T | x_S$ follows a lognormal distribution, i.e., 

$$y_T = \log(x_T) \sim \mathcal{N}(\log(x_S) + \int_S^T (\mu(t) - \frac {\sigma^2(t)} 2) \cdot dt, \int_S^T  \sigma^2(t) \cdot dt)$$

$$E[x_T|x_S] = x_S \cdot e^{\int_S^T \mu(t) \cdot dt}$$
$$E[x_T^2|x_S] = x_S^2 \cdot e^{\int_S^T (2 \mu(t) + \sigma^2(t)) \cdot dt}$$
$$Variance[x_T|x_S] = E[x_T^2|x_S] - (E[x_T|x_S])^2 = x_S^2 \cdot e^{\int_S^T 2 \mu(t) \cdot dt} \cdot (e^{\int_S^T \sigma^2(t) \cdot dt} - 1)$$

The special case of $\mu(t) = \mu$ (constant) and $\sigma(t) = \sigma$ (constant) is a very common Ito process used all over Finance/Economics (for its simplicity, tractability as well as practicality), and is known as [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion),  to reflect the fact that the stochastic increment of the process ($\sigma \cdot x_t \cdot dz_t$) is multiplicative to the level of the process $x_t$. If we consider this special case, we get:

$$y_T = \log(x_T) \sim \mathcal{N}(\log(x_S) + (\mu - \frac {\sigma^2} 2)(T-S), \sigma^2 (T-S))$$
$$E[x_T|x_S] = x_S \cdot e^{\mu (T-S)}$$
$$Variance[x_T|x_S] = x_S^2 \cdot e^{2 \mu (T-S)} \cdot (e^{\sigma^2 (T-S)} - 1)$$

### A Mean-Reverting Process {#sec:mean-reverting-process-section}

Now we consider a stochastic process $x$ described in the form of the following Ito process:

$$dx_t = \mu(t) \cdot x_t \cdot dt + \sigma(t) \cdot dz_t$$

As in the process of the previous section, $z$ is standard (one-dimensional) Brownian motion, and $\mu$, $\sigma$ are deterministic functions of time $t$. This is solved easily by defining an appropriate function of $x_t$ and applying Ito's Lemma, as follows:

$$y_t = x_t \cdot e^{-\int_0^t \mu(u) \cdot du}$$
Applying Ito's Lemma on $y_t$ with respect to $x_t$, we get:

\begin{align*}
dy_t & = (- x_t \cdot \mu(t) \cdot e^{-\int_0^t \mu(u) \cdot du} + \mu(t) \cdot x_t \cdot e^{-\int_0^t \mu(u) \cdot du}) \cdot dt + \sigma(t) \cdot e^{-\int_0^t \mu(u) \cdot du} \cdot dz_t \\
& = \sigma(t) \cdot e^{-\int_0^t \mu(u) \cdot du} \cdot dz_t
\end{align*}

So the process $y$ is a martingale. Using Ito Isometry, we get:

$$y_T \sim \mathcal{N}(y_S, \int_S^T \sigma^2(t) \cdot e^{-\int_0^t 2 \mu(u) \cdot du} \cdot dt)$$

Therefore,

$$x_T \sim \mathcal{N}(x_S \cdot e^{\int_S^T \mu(t) \cdot dt}, e^{\int_0^T 2\mu(t) \cdot dt} \cdot \int_S^T \sigma^2(t) \cdot e^{-\int_0^t 2 \mu(u) \cdot du} \cdot dt)$$

We call this process "mean-reverting" because with negative $\mu(t)$, the process is "pulled" to a baseline level of 0, at a speed whose expectation is proportional to $-\mu(t)$ and proportional to the distance from the baseline (so we say the process reverts to a baseline of 0 and the strength of mean-reversion is greater if the distance from the baseline is greater). If $\mu(t)$ is positive, then we say that the process is "mean-diverting" to signify that it gets pulled away from the baseline level of 0.

The special case of $\mu(t) = \mu$ (constant) and $\sigma(t) = \sigma$ (constant) is a fairly common Ito process (again for it's simplicity, tractability as well as practicality), and is known as the [Ornstein-Uhlenbeck Process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) with the mean (baseline) level set to 0. If we consider this special case, we get:

$$x_T \sim \mathcal{N}(x_S \cdot e^{\mu (T-S)}, \frac {\sigma^2} {2 \mu} \cdot (e^{2 \mu (T-S)} - 1))$$
