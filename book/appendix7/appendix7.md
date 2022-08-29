## Conjugate Priors for Gaussian and Bernoulli Distributions {#sec:conjugate-priors-appendix}
\index{conjugate prior|(}

The setting for this Appendix is that we receive data incrementally as $x_1, x_2, \ldots$ and we assume a certain probability distribution (e.g., Gaussian, Bernoulli) for each $x_i, i = 1, 2, \ldots$. We utilize an appropriate conjugate prior for the assumed data distribution so that we can derive the posterior distribution for the parameters of the assumed data distribution. We can then say that for any $n \in \mathbb{Z}^+$, the conjugate prior is the probability distribution for the parameters of the assumed data distribution, conditional on the first $n$ data points $(x_1, x_2, \ldots x_n)$ and the posterior is the probability distribution for the parameters of the assumed distribution, conditional on the first $n+1$ data points $(x_1, x_2, \ldots, x_{n+1})$. This amounts to performing Bayesian updates on the hyperparameters upon receipt of each incremental data $x_i$ (hyperparameters refer to the parameters of the prior and posterior distributions). In this appendix, we shall not cover the derivations of the posterior distribution from the prior distribution and the data distribution. We shall simply state the results (references for derivations can be found on the [Conjugate Prior Wikipedia Page](https://en.wikipedia.org/wiki/Conjugate_prior)).

### Conjugate Prior for Gaussian Distribution {#sec:conjugate-prior-gaussian}

\index{probability!normal distribution}

Here we assume that each data point is Gaussian-distributed in $\mathbb{R}$. So when we receive the $n$-th data point $x_n$, we assume:

$$x_n \sim \mathcal{N}(\mu, \sigma^2)$$
and we assume both $\mu$ and $\sigma^2$ are unknown random variables with [Gaussian-Inverse-Gamma Probability Distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution) Conjugate Prior for $\mu$ and $\sigma^2$, i.e.,

\index{probability!inverse-gamma distribution}

$$\mu | x_1, \ldots, x_n \sim \mathcal{N}(\theta_n, \frac {\sigma^2} n)$$
$$\sigma^2 | x_1, \ldots, x_n \sim IG(\alpha_n, \beta_n)$$
where $IG(\alpha_n, \beta_n)$ refers to the Inverse Gamma distribution with parameters $\alpha_n$ and $\beta_n$. This means $\frac 1 {\sigma^2} | x_1, \ldots, x_n$ follows a Gamma distribution with parameters $\alpha_n$ and $\beta_n$, i.e., the probability of $\frac 1 {\sigma^2}$ having a value $y \in \mathbb{R}^+$ is:

$$\frac {\beta^{\alpha} \cdot y^{\alpha - 1} \cdot e^{- \beta y}} {\Gamma(\alpha)}$$
where $\Gamma(\cdot)$ is the [Gamma Function](https://en.wikipedia.org/wiki/Gamma_function). 

\index{functions!gamma function}
\index{hyperparameter}

$\theta_n, \alpha_n, \beta_n$ are hyperparameters determining the probability distributions of $\mu$ and $\sigma^2$, conditional on data $x_1, \ldots, x_n$.

Then, the posterior distribution is given by:

$$\mu | x_1, \ldots, x_{n+1} \sim \mathcal{N}(\frac {n \theta_n + x_{n+1}} {n + 1}, \frac {\sigma^2} {n+1})$$
$$\sigma^2 | x_1, \ldots, x_{n+1} \sim IG(\alpha_n + \frac 1 2, \beta_n + \frac {n (x_{n+1} - \theta_n)^2} {2(n +1)})$$

This means upon receipt of the data point $x_{n+1}$, the hyperparameters can be updated as:

$$\theta_{n+1} = \frac {n \theta_n + x_{n+1}} {n + 1}$$
$$\alpha_{n+1} = \alpha_n + \frac 1 2$$
$$\beta_{n+1} = \beta_n + \frac {n (x_{n+1} - \theta_n)^2} {2(n+1)}$$

### Conjugate Prior for Bernoulli Distribution {#sec:conjugate-prior-bernoulli}

\index{probability!Bernoulli distribution}
\index{probability!beta distribution}

Here we assume that each data point is Bernoulli-distributed. So when we receive the $n$-th data point $x_n$, we assume $x_n = 1$ with probability $p$ and $x_n = 0$ with probability $1-p$. We assume $p$ is an unknown random variable with [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution) Conjugate Prior for $p$, i.e, 
\index{functions!gamma function}

$$p | x_1, \ldots, x_n \sim Beta(\alpha_n, \beta_n)$$
where $Beta(\alpha_n, \beta_n)$ refers to the Beta distribution with parameters $\alpha_n$ and $\beta_n$, i.e., the probability of $p$ having a value $y \in [0, 1]$ is:
$$\frac {\Gamma(\alpha + \beta)} {\Gamma(\alpha) \cdot \Gamma(\beta)} \cdot y^{\alpha - 1} \cdot (1 - y)^{\beta - 1}$$
where $\Gamma(\cdot)$ is the [Gamma Function](https://en.wikipedia.org/wiki/Gamma_function). 

\index{functions!gamma function}
\index{hyperparameter}

$\alpha_n, \beta_n$ are hyperparameters determining the probability distribution of $p$, conditional on data $x_1, \ldots, x_n$.

Then, the posterior distribution is given by:

$$p | x_1, \ldots, x_{n+1} \sim Beta(\alpha_n + \mathbb{I}_{x_{n+1} = 1}, \beta_n + \mathbb{I}_{x_{n+1} = 0})$$
where $\mathbb{I}$ refers to the indicator function.

This means upon receipt of the data point $x_{n+1}$, the hyperparameters can be updated as:

$$\alpha_{n+1} = \alpha_n + \mathbb{I}_{x_{n+1} = 1}$$
$$\beta_{n+1} = \beta_n + \mathbb{I}_{x_{n+1} = 0}$$
\index{conjugate prior|)}
