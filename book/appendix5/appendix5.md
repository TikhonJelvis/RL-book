## Black-Scholes Equation and it's Solution for Call/Put Options {#sec:black-scholes-appendix}

In this Appendix, we sketch the derivation of the [much-celebrated Black-Scholes equation and it's solution for Call and Put Options](https://www.cs.princeton.edu/courses/archive/fall09/cos323/papers/black_scholes73.pdf) [@BlackScholes1973]. As is the norm in the Appendices in this book, we will compromise on some of the rigor and emphasize the intuition to develop basic familiarity with concepts in continuous-time derivatives pricing and hedging.

### Assumptions
The Black-Scholes Model is about pricing and hedging of a derivative on a single underlying asset (henceforth, simply known as "underlying"). The model makes several simplifying assumptions for analytical convenience. Here are the assumptions:

* The underlying (whose price we denote as $S_t$ as time $t$) follows a special case of the lognormal process we covered in Section [-@sec:lognormal-process-section] of Appendix [-@sec:stochasticcalculus-appendix], where the drift $\mu(t)$ is a constant (call it $\mu \in \mathbb{R}$) and the dispersion $\sigma(t)$ is also a constant (call it $\sigma \in \mathbb{R}^+$):
\begin{equation}
dS_t = \mu \cdot S_t \cdot dt + \sigma \cdot S_t \cdot dz_t
\label{eq:black-scholes-underlying-process}
\end{equation}
This process is often refered to as *Geometric Brownian Motion* to reflect the fact that the stochastic increment of the process ($\sigma \cdot S_t \cdot dz_t$) is multiplicative to the level of the process $S_t$.
* The derivative has a known payoff at time $t=T$, as a function $f: \mathbb{R}^+ \rightarrow \mathbb{R}$ of the underlying price $S_T$ at time $T$.
* Apart from the underlying, the market also includes a riskless asset (which should be thought of as lending/borrowing money at a constant infinitesimal rate of annual return equal to $r$). The riskless asset (denote it's price as $R_t$ at time $t$) movements can thus be described as:
$$dR_t = r \cdot R_t \cdot dt$$
* Assume that we can trade in any real-number quantity in the underlying as well as in the riskless asset, in continuous-time, without any transaction costs (i.e., the typical "frictionless" market assumption).

### Derivation of the Black-Scholes Equation

We denote the price of the derivative at any time $t$ for any price $S_t$ of the underlying as $V(t, S_t)$. Thus, $V(T, S_T)$ is equal to the payoff $f(S_T)$. Applying Ito's Lemma on $V(t, S_t)$ (see Equation \eqref{eq:itos-lemma} in Appendix [-@sec:stochasticcalculus-appendix]), we get:

\begin{equation}
dV(t, S_t) = (\pdv{V}{t} + \mu \cdot S_t \cdot \pdv{V}{S_t} + \frac {\sigma^2} 2 \cdot S_t^2 \cdot \pdv[2]{V}{S_t}) \cdot dt + \sigma \cdot S_t \cdot \pdv{V}{S_t} \cdot dz_t
\label{eq:black-scholes-derivative-process}
\end{equation}

Now here comes the key idea: create a portfolio comprising of the derivative and the underlying so as to eliminate the incremental uncertainty arising from the brownian motion increment $dz_t$. It's clear from the coefficients of $dz_t$ in Equation \eqref{eq:black-scholes-underlying-process} and \eqref{eq:black-scholes-derivative-process} that this can be accomplished with a portfolio comprising of $\pdv{V}{S_t}$ units of the underlying and -1 units of the derivative (i.e., by selling a derivative contract written on a single unit of the underlying). Let us refer to the value of this portfolio as $\Pi_t$ at time $t$. Thus,

\begin{equation}
\Pi_t = - V(t, S_t) + \pdv{V}{S_t} \cdot S_t
\label{eq:black-scholes-portfolio-value}
\end{equation}

Over an infinitesimal time-period $[t, t+dt]$, the change in the portfolio value $\Pi_t$ is given by:

$$d\Pi_t = - dV(t, S_t) + \pdv{V}{S_t} \cdot dS_t$$

Substituting for $dS_t$ and $dV(t, S_t)$ from Equations \eqref{eq:black-scholes-underlying-process} and \eqref{eq:black-scholes-derivative-process}, we get:
\begin{equation}
d\Pi_t = (- \pdv{V}{t} - \frac {\sigma^2} 2 \cdot S_t^2 \cdot \pdv[2]{V}{S_t}) \cdot dt
\label{eq:black-scholes-portfolio-process}
\end{equation}

Thus, we have eliminated the incremental uncertainty arising from $dz_t$ and hence, this is a riskless portfolio. To ensure the market remains free of arbitrage, the infinitesimal rate of annual return for this riskless portfolio must be the same as that for the riskless asset, i.e., must be equal to $r$. Therefore,

\begin{equation}
d\Pi_t = r \cdot \Pi_t \cdot dt
\label{eq:black-scholes-portfolio-riskless}
\end{equation}

From Equations \eqref{eq:black-scholes-portfolio-process} and \eqref{eq:black-scholes-portfolio-riskless}, we infer that:

$$- \pdv{V}{t} - \frac {\sigma^2} 2 \cdot S_t^2 \cdot \pdv[2]{V}{S_t} = r \cdot \Pi_t$$

Substituting for $\Pi_t$ from Equation \eqref{eq:black-scholes-portfolio-value}, we get:

$$- \pdv{V}{t} - \frac {\sigma^2} 2 \cdot S_t^2 \cdot \pdv[2]{V}{S_t} = r \cdot (- V(t, S_t) + \pdv{V}{S_t} \cdot S_t)$$

Re-arranging, we arrive at the famous Black-Scholes equation:

\begin{equation}
\pdv{V}{t} + \frac {\sigma^2} 2 \cdot S_t^2 \cdot \pdv[2]{V}{S_t} + r \cdot S_t \cdot \pdv{V}{S_t} + r \cdot V(t, S_t) = 0
\label{eq:black-scholes-equation}
\end{equation}

A few key points to note here:

1. The Black-Scholes equation is a partial differential equation (PDE) in $t$ and $S_t$, and it is valid for any derivative with arbitary payoff $f(S_T)$ at a fixed time $t=T$, and the derivative price function $V(t, S_t)$ needs to be twice differentiable with respect to $S_t$ and once differentiable with respect to $t$.
2. The infinitesimal change in the portfolio value ($=d\Pi_t$) incorporates only the infinitesimal changes in the prices of the underlying and the derivative, and not the changes in the units held in the underlying and the derivative (meaning the portfolio is assumed to be self-financing). The portfolio composition does change continuously though since the units held in the underlying at time $t$ needs to be $\pdv{V}{S_t}$, which in general would change as time evolves and as the price $S_t$ of the underlying changes. Note that $-\pdv{V}{S_t}$ represents the hedge units in the underlying at any time $t$ for any underlying price $S_t$, which nullifies the risk of changes to the derivative price $V(t, S_t)$.
3. The drift $\mu$ of the underlying price movement (interpreted as expected annual rate of return of the underlying) does not appear in the Black-Scholes Equation and hence, the price of any derivative will be independent of the expected rate of return of the underlying. Note though the prominent appearance of $\sigma$ (refered to as the underlying volatility) and the riskless rate of return $r$ in the Black-Scholes equation.

### Solution of the Black-Scholes Equation for Call/Put Options

The Black–Scholes PDE can be solved numerically using standard methods such as finite-differences. It turns out we can solve this PDE as an exact formula (closed-form solution) for the case of European call and put options, whose payoff functions are $\max(S_T-K, 0)$ and $\max(K-S_T, 0)$ respectively, where $K$ is the option strike. We shall denote the call and put option prices at time $t$ for underlying price of $S_t$ as $C(t, S_t)$  and $P(t, S_t)$ respectively (as specializations of $V(t, S_t)$). We derive the solution below for call option pricing, with put option pricing derived similarly. Note that we could simply use the put-call parity: $C(t, S_t) - P(t, S_t) = S_t - K$ to obtain the put option price from the call option price. The put-call parity holds because buying a call option and selling a put option is a combined payoff of $S_T - K$ - this means owning the underlying and borrowing $K\cdot e^{-rT}$, which at any time $t$ would be valued at $S_t - K \cdot e^{-r\cdot (T-t)}$.

To derive the formula for $C(t, S_t)$, we perform the following change-of-variables transformation:

$$\tau = T - t$$
$$x = \log(\frac {S_t} K) + (r - \frac {\sigma^2} 2) \cdot \tau$$
$$u(\tau, x) = C(t, S_t) \cdot e^{r \tau}$$

This reduces the Black-Scholes PDE into the *Heat Equation*:

$$\pdv{u}{\tau} = \frac {\sigma^2} 2 \cdot \pdv[2]{u}{x}$$

The terminal condition $C(T, S_T) = \max(S_T-K, 0)$ transforms into the Heat Equation's initial condition:

$$u(0, x) = K \cdot (e^{\max(x, 0)} - 1)$$

Using the standard convolution method for solving this Heat Equation with initial condition $u(0,x)$, we obtain the [Green's Function Solution](https://en.wikipedia.org/wiki/Heat_equation#Some_Green's_function_solutions_in_1D):

$$u(\tau, x) = \frac 1 {\sigma \sqrt{2 \pi \tau}} \cdot \int_{-\infty}^{+\infty} u(0, y) \cdot e^{-\frac {(x-y)^2} {2\sigma^2 \tau}} \cdot dy$$

With some manipulations, this yields:

$$u(\tau, x) = K \cdot e^{x + \frac {\sigma^2 \tau} 2} \cdot N(d_1) - K \cdot N(d_2)$$
where $N(\cdot)$ is the standard normal cumulative distribution function:
$$N(z) = \frac 1 {\sigma \sqrt{2\pi}} \int_{-\infty}^z e^{-\frac {y^2} 2} \cdot dy$$
and $d_1, d_2$ are the quantities:

$$d_1 = \frac {x + \sigma^2 \tau} {\sigma \sqrt{\tau}}$$
$$d_2 = d_1 - \sigma \sqrt{\tau}$$

Substituting for $\tau, x, u(\tau, x)$ with $t, S_t, C(t,S_t)$, we get:

\begin{equation}
C(t, S_t) = S_t \cdot N(d_1) - K \cdot e^{-r\cdot (T-t)} \cdot N(d_2)
\label{eq:black-scholes-call-option-pricing}
\end{equation}
where
$$d_1 = \frac {\log(\frac {S_t} K) + (r + \frac {\sigma^2} 2) \cdot (T-t)} {\sigma \cdot \sqrt{T-t}}$$
$$d_2 = d_1 - \sigma \sqrt{T-t}$$

The put option price is:

\begin{equation}
P(t, S_t) = K \cdot e^{-r\cdot (T-t)} \cdot N(-d_2) - S_t \cdot N(-d_1)
\label{eq:black-scholes-put-option-pricing}
\end{equation}
