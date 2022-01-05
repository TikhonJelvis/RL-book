\appendix

# Appendix {.unnumbered}

## Moment Generating Function and its Applications {#sec:mgf-appendix}

The purpose of this Appendix is to introduce the *Moment Generating Function (MGF)* and demonstrate it's utility in several applications in Applied Mathematics. 

### The Moment Generating Function (MGF)

The Moment Generating Function (MGF) of a random variable $x$ (discrete or continuous) is defined as a function $f_x : \mathbb{R} \rightarrow \mathbb{R}^+$ such that:

\begin{equation}
f_x(t) = \mathbb{E}_x[e^{tx}] \mbox{ for all } t \in \mathbb{R} \label{eq:mgfdef}
\end{equation}

Let us denote the $n^{th}$-derivative of $f_x$ as $f_x^{(n)} : \mathbb{R} \rightarrow \mathbb{R}$ for all $n\in \mathbb{Z}_{\geq 0}$ ($f_x^{(0)}$ is defined to be simply the MGF $f_x$).

\begin{equation}
f_x^{(n)}(t) = \mathbb{E}_x[x^n \cdot e^{tx}] \mbox{ for all } n\in \mathbb{Z}_{\geq 0} \mbox{ for all } t\in \mathbb{R} \label{eq:mgfderiv}
\end{equation}

\begin{equation}
f_x^{(n)}(0) = \mathbb{E}_x[x^n]  \label{eq:derivat0}
\end{equation}

\begin{equation}
f_x^{(n)}(1) = \mathbb{E}_x[x^n \cdot e^x]  \label{eq:derivat1}
\end{equation}

Equation \eqref{eq:derivat0} tells us that $f_x^{(n)}(0)$ gives us the $n^{th}$ moment of $x$. In particular, $f_x^{(1)}(0) = f_x'(0)$ gives us the mean and $f_x^{(2)}(0) - (f_x^{(1)}(0))^2 = f_x''(0) - (f_x'(0))^2$ gives us the variance. Note that this holds true for any distribution for $x$. This is rather convenient since all we need is the functional form for the distribution of $x$. This would lead us to the expression for the MGF (in terms of $t$). Then, we take derivatives of this MGF and evaluate those derivatives at 0 to obtain the moments of $x$.

Equation \eqref{eq:derivat1} helps us calculate the often-appearing expectation $\mathbb{E}_x[x^n \cdot e^x]$. In fact, $\mathbb{E}_x[e^x]$ and $\mathbb{E}_x[x \cdot e^x]$ are very common in several areas of Applied Mathematics. Again, note that this holds true for any distribution for $x$.

MGF should be thought of as an alternative specification of a random variable (alternative to specifying it's Probability Distribution). This alternative specification is very valuable because it can sometimes provide better analytical tractability than working with the Probability Density Function or Cumulative Distribution Function (as an example, see the below section on MGF for linear functions of independent random variables).

### MGF for Linear Functions of Random Variables

Consider $m$ independent random variables $x_1, x_2, \ldots, x_m$. Let $\alpha_0, \alpha_1, \ldots, \alpha_m \in \mathbb{R}$. Now consider the random variable
$$x = x_0 + \sum_{i=1}^m \alpha_i x_i$$

The Probability Density Function of $x$ is complicated to calculate as it involves convolutions. However, observe that the MGF $f_x$ of $x$ is given by:
\begin{equation*}
f_x(t) = \mathbb{E}[e^{t(\alpha_0 + \sum_{i=1}^m \alpha_i x_i)}] = e^{\alpha_0 t} \cdot \prod_{i=1}^m \mathbb{E}[e^{t\alpha_i x_i}] = e^{\alpha_0 t}  \cdot \prod_{i=1}^m f_{\alpha_i x_i}(t) = e^{\alpha_0 t}  \cdot \prod_{i=1}^m f_{x_i}(\alpha_i t)
\end{equation*}
This means the MGF of $x$ can be calculated as $e^{\alpha_0 t}$ times the product of the MGFs of $\alpha_i x_i$ (or of $\alpha_i$-scaled MGFs of $x_i$) for all $i = 1, 2, \ldots, m$. This gives us a much better way to analytically tract the probability distribution of $x$ (compared to the convolution approach).


### MGF for the Normal Distribution

Here we assume that the random variables $x$ follows a normal distribution. Let $x \sim \mathcal{N}(\mu, \sigma^2)$.

\begin{align}
f_{x\sim \mathcal{N}(\mu, \sigma^2)}(t) & = \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[e^{tx}] \nonumber \\
& = \int_{-\infty}^{+\infty} \frac {1} {\sqrt{2\pi} \sigma} \cdot e^{-\frac {(x - \mu)^2} {2\sigma^2}} \cdot e^{tx} \cdot dx \nonumber \\
& = \int_{-\infty}^{+\infty} \frac {1} {\sqrt{2\pi} \sigma} \cdot e^{-\frac {(x-(\mu +t\sigma^2))^2} {2\sigma^2}} \cdot e^{\mu t + \frac {\sigma^2 t^2} {2}} \cdot dx \nonumber \\
& = e^{\mu t + \frac {\sigma^2 t^2} 2} \cdot \mathbb{E}_{x\sim \mathcal{N}(\mu + t\sigma^2, \sigma^2)}[1] \nonumber \\
& = e^{\mu t + \frac {\sigma^2 t^2} 2} \label{eq:normmgf}
\end{align}

\begin{equation}
f_{x\sim \mathcal{N}(\mu, \sigma^2)}'(t) = \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[x \cdot e^{tx}] = (\mu + \sigma^2t)\cdot e^{\mu t + \frac {\sigma^2 t^2} 2} \label{eq:normmgfderiv}
\end{equation}
\begin{equation}
f_{x\sim \mathcal{N}(\mu, \sigma^2)}''(t) = \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[x^2 \cdot e^{tx}] = ((\mu + \sigma^2t)^2 + \sigma^2)\cdot e^{\mu t + \frac {\sigma^2 t^2} 2} \label{eq:normmgfdoublederiv}
\end{equation}
\begin{equation*}
f_{x\sim \mathcal{N}(\mu, \sigma^2)}'(0) = \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[x] = \mu
\end{equation*}
\begin{equation*}
f_{x\sim \mathcal{N}(\mu, \sigma^2)}''(0) = \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[x^2] = \mu^2 + \sigma^2
\end{equation*}
\begin{equation*}
f_{x\sim \mathcal{N}(\mu, \sigma^2)}'(1) = \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[x\cdot e^x] = (\mu + \sigma^2)e^{\mu+ \frac {\sigma^2} 2}
\end{equation*}
\begin{equation*}
f_{x\sim \mathcal{N}(\mu, \sigma^2)}''(1) = \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[x^2\cdot e^x] = ((\mu + \sigma^2)^2 + \sigma^2)e^{\mu+ \frac {\sigma^2} 2}
\end{equation*}

### Minimizing the MGF

Now let us consider the problem of minimizing the MGF. The problem is to:
$$\min_{t\in \mathbb{R}} f_x(t) = \min_{t\in \mathbb{R}} \mathbb{E}_x[e^{tx}]$$

This problem of minimizing $\mathbb{E}_x[e^{tx}]$ shows up a lot in various places in Applied Mathematics when dealing with exponential functions (eg: when optimizing the Expectation of a Constant Absolute Risk-Aversion (CARA) Utility function $U(y) = \frac {1 - e^{-\gamma y}} {\gamma}$ where $\gamma$ is the coefficient of risk-aversion and where $y$ is a parameterized function of a random variable $x$).

Let us denote $t^*$ as the value of $t$ that minimizes the MGF. Specifically,
$$t^* = \argmin_{t\in \mathbb{R}} f_x(t) = \argmin_{t \in \mathbb{R}} \mathbb{E}_x[e^{tx}]$$

#### Minimizing the MGF when $x$ follows a normal distribution {#sec:norm-distrib-mgf-min}

Here we consider the fairly typical case where $x$ follows a normal distribution. Let $x\sim \mathcal{N}(\mu, \sigma^2)$. Then we have to solve the problem:
$$\min_{t\in \mathbb{R}} f_{x\sim \mathcal{N}(\mu, \sigma^2)}(t) = \min_{t\in \mathbb{R}} \mathbb{E}_{x\sim \mathcal{N}(\mu, \sigma^2)}[e^{tx}] = \min_{t\in \mathbb{R}} e^{\mu t + \frac {\sigma^2 t^2} 2}$$
From Equation \eqref{eq:normmgfderiv} above, we have:
$$f_{x\sim \mathcal{N}(\mu, \sigma^2)}'(t) = (\mu + \sigma^2t)\cdot e^{\mu t + \frac {\sigma^2 t^2} 2}$$
Setting this to 0 yields:
$$(\mu + \sigma^2t^*)\cdot e^{\mu t^* + \frac {\sigma^2 {t^*}^2} 2} = 0$$
which leads to:
\begin{equation}
t^* = \frac {-\mu} {\sigma^2}
\end{equation}

From Equation \eqref{eq:normmgfdoublederiv} above, we have:
$$f_{x\sim \mathcal{N}(\mu, \sigma^2)}''(t) = ((\mu + \sigma^2t)^2 + \sigma^2)\cdot e^{\mu t + \frac {\sigma^2 t^2} 2} > 0 \mbox{ for all } t \in \mathbb{R}$$
which confirms that $t^*$ is a minima.

Substituting $t=t^*$ in $f_{x\sim \mathcal{N}(\mu, \sigma^2)}(t) = e^{\mu t + \frac {\sigma^2 t^2} 2}$ yields:
\begin{equation}
\min_{t\in \mathbb{R}} f_{x\sim \mathcal{N}(\mu, \sigma^2)}(t) = e^{\mu t^* + \frac {\sigma^2 {t^*}^2} 2} = e^{\frac {-\mu^2} {2\sigma^2}}
\label{eq:normmgfminvalue}
\end{equation}

#### Minimizing the MGF when $x$ is a symmetric binary distribution

Here we consider the case where $x$ follows a binary distribution: $x$ takes values $\mu + \sigma$ and $\mu - \sigma$ with probability 0.5 each. Let us refer to this distribution as $x \sim \mathcal{B}(\mu + \sigma, \mu - \sigma)$. Note that the mean and variance of $x$ under $\mathcal{B}(\mu + \sigma, \mu - \sigma)$ are $\mu$ and $\sigma^2$ respectively. So we have to solve the problem:
$$\min_{t\in \mathbb{R}} f_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}(t) = \min_{t\in \mathbb{R}} \mathbb{E}_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}[e^{tx}] = \min_{t\in \mathbb{R}} 0.5(e^{(\mu + \sigma)t} + e^{(\mu - \sigma)t})$$
$$f_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}'(t) = 0.5((\mu + \sigma) \cdot e^{(\mu + \sigma)t} + (\mu - \sigma) \cdot e^{(\mu - \sigma)t})$$
Note that unless $\mu \in$ open interval $(-\sigma, \sigma)$ (i.e., absolute value of mean is less than standard deviation), $f_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}'(t)$ will not be 0 for any value of $t$. Therefore, for this minimization to be non-trivial, we will henceforth assume $\mu \in (-\sigma, \sigma)$.
With this assumption in place, setting $f_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}'(t)$ to 0 yields:
$$(\mu + \sigma) \cdot e^{(\mu + \sigma)t^*} + (\mu - \sigma) \cdot e^{(\mu - \sigma)t^*} = 0$$
which leads to:
\begin{equation*}
t^* = \frac 1 {2\sigma} \ln{(\frac {\sigma - \mu} {\mu + \sigma})}
\end{equation*}
Note that
$$f_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}''(t) =  0.5((\mu + \sigma)^2 \cdot e^{(\mu + \sigma)t} + (\mu - \sigma)^2 \cdot e^{(\mu - \sigma)t}) > 0 \mbox{ for all } t \in \mathbb{R}$$
which confirms that $t^*$ is a minima.

Substituting $t=t^*$ in $f_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}(t) = 0.5(e^{(\mu + \sigma)t} + e^{(\mu - \sigma)t})$ yields:
\begin{equation*}
\min_{t\in \mathbb{R}} f_{x\sim \mathcal{B}(\mu + \sigma, \mu - \sigma)}(t) = 0.5(e^{(\mu + \sigma)t^*} + e^{(\mu - \sigma)t^*}) = 0.5((\frac {\sigma - \mu} {\mu+ \sigma})^{\frac {\mu + \sigma} {2\sigma}} + (\frac {\sigma - \mu} {\mu+ \sigma})^{\frac {\mu - \sigma} {2\sigma}})
\end{equation*}

