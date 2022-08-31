## Portfolio Theory {#sec:portfoliotheory-appendix}
\index{portfolio theory|(}

In this Appendix, we provide a quick and terse introduction to *Portfolio Theory*. While this topic is not a direct pre-requisite for the topics we cover in the chapters, we believe one should have some familiarity with the risk versus reward considerations when constructing portfolios of financial assets, and know of the important results. To keep this Appendix brief, we will provide the minimal content required to understand the *essence* of the key concepts. We won't be doing rigorous proofs. We will also ignore details pertaining to edge-case/irregular-case conditions so as to focus on the core concepts.

### Setting and Notation

In this section, we go over the core setting of Portfolio Theory, along with the requisite notation.

Assume there are $n$ assets in the economy and that their mean returns are represented in a column vector $R \in \mathbb{R}^n$. We denote the covariance of returns of the $n$ assets by an $n \times n$ non-singular matrix $V$.

We consider arbitrary portfolios $p$ comprised of investment quantities in these $n$ assets that are normalized to sum up to 1. Denoting column vector $X_p \in \mathbb{R}^n$ as the investment quantities in the $n$ assets for portfolio $p$, we can write the normality of the investment quantities in vector notation as:

$$X_p^T \cdot 1_n = 1$$
where $1_n \in \mathbb{R}^n$ is a column vector comprising of all 1's.

We shall drop the subscript $p$ in $X_p$ whenever the reference to portfolio $p$ is clear.

### Portfolio Returns

* A single portfolio's mean return is $X^T \cdot R \in \mathbb{R}$.
* A single portfolio's variance of return is the quadratic form $X^T \cdot V \cdot X \in \mathbb{R}$.
* Covariance between portfolios $p$ and $q$ is the bilinear form $X_p^T \cdot V \cdot X_q \in \mathbb{R}$.
* Covariance of the $n$ assets with a single portfolio is the vector $V \cdot X \in\mathbb{R}^n$.

### Derivation of Efficient Frontier Curve
\index{portfolio theory!efficient frontier}
\index{portfolio theory!efficient portfolio}

An asset which has no variance in terms of how its value evolves in time is known as a riskless asset.  The Efficient Frontier is defined for a world with no riskless assets. The Efficient Frontier is the set of portfolios with minimum variance of return for each level of portfolio mean return (we refer to a portfolio in the Efficient Frontier as an *Efficient Portfolio*). Hence, to determine the Efficient Frontier, we solve for $X$ so as to minimize portfolio variance $X^T \cdot V \cdot X$ subject to constraints:
$$X^T \cdot 1_n = 1$$
$$X^T \cdot R = r_p$$
where $r_p$ is the mean return for Efficient Portfolio $p$.
We set up the Lagrangian and solve to express $X$ in terms of $R, V, r_p$. Substituting for $X$ gives us the efficient frontier parabola of Efficient Portfolio Variance $\sigma_p^2$ as a function of its mean $r_p$:
$$\sigma_p^2 = \frac {a - 2 b r_p + c r_p^2} {ac - b^2}$$
where

* $a = R^T \cdot V^{-1} \cdot R$
* $b = R^T \cdot V^{-1} \cdot 1_n$
* $c = 1_n^T \cdot V^{-1} \cdot 1_n$


### Global Minimum Variance Portfolio (GMVP)

The global minimum variance portfolio (GMVP) is the portfolio at the tip of the efficient frontier parabola, i.e., the portfolio with the lowest possible variance among all portfolios on the Efficient Frontier. Here are the relevant characteristics for the GMVP:

* It has mean $r_0 = \frac b c$.
* It has variance $\sigma_0^2 = \frac 1 c$.
* It has investment proportions $X_0 = \frac {V^{-1} \cdot 1_n} c$.

GMVP is positively correlated with all portfolios and with all assets. GMVP's covariance with all portfolios and with all assets is a constant value equal to $\sigma_0^2 = \frac 1 c$ (which is also equal to its own variance).

### Orthogonal Efficient Portfolios
\index{portfolio theory!efficient portfolio}

For every efficient portfolio $p$ (other than GMVP), there exists a unique orthogonal efficient portfolio $z$ (i.e. $Covariance(p,z) = 0$) with finite mean
$$r_z = \frac {a - b r_p} {b - c r_p}$$

$z$ always lies on the opposite side of $p$ on the (efficient frontier) parabola. If we treat the Efficient Frontier as a curve of mean (y-axis) versus variance (x-axis), the straight line from $p$ to GMVP intersects the mean axis (y-axis) at $r_z$. If we treat the Efficient Frontier as a curve of mean (y-axis) versus standard deviation (x-axis), the tangent to the efficient frontier at $p$ intersects the mean axis (y-axis) at $r_z$. Moreover, all portfolios on one side of the efficient frontier are positively correlated with each other.

### Two-Fund Theorem
\index{portfolio theory!two-fund theorem}

The $X$ vector (normalized investment quantities in assets) of any efficient portfolio is a linear combination of the $X$ vectors of two other efficient portfolios. Notationally,
$$X_p = \alpha X_{p_1} + (1-\alpha) X_{p_2} \mbox{ for some scalar } \alpha$$
Varying $\alpha$ from $-\infty$ to $+\infty$ basically traces the entire efficient frontier. So to construct all efficient portfolios, we just need to identify two canonical efficient portfolios. One of them is GMVP. The other is a portfolio we call Special Efficient Portfolio (SEP) with:

* Mean $r_1  = \frac a b$.
* Variance $\sigma_1^2 = \frac a {b^2}$.
* Investment proportions $X_1 = \frac {V^{-1} \cdot R} {b}$.

The orthogonal portfolio to SEP has mean $r_z = \frac {a - b \frac a b} {b - c \frac a b} = 0$

### An Example of the Efficient Frontier for 16 Assets
\index{portfolio theory!efficient frontier}

Figure \ref{fig:efficient_frontier} shows a plot of the mean daily returns versus the standard deviation of daily returns collected over a 3-year period for 16 assets. The curve is the Efficient Frontier for these 16 assets. Note the special portfolios GMVP and SEP on the Efficient Frontier. This curve was generated from the code at [rl/appendix2/efficient_frontier.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/appendix2/efficient_frontier.py). We encourage you to play with different choices (and count) of assets, and to also experiment with different time ranges as well as to try weekly and monthly returns.

<div style="text-align:center" markdown="1">
![Efficient Frontier for 16 Assets \label{fig:efficient_frontier}](./appendix2/EffFront.png "Efficient Frontier for 16 Assets"){height=7cm}
</div>

### CAPM: Linearity of Covariance Vector w.r.t. Mean Returns
\index{portfolio theory!capital asset pricing model}

**Important Theorem**: The covariance vector of individual assets with a portfolio (note: covariance vector $= V \cdot X \in \mathbb{R}^n$) can be expressed as an exact linear function of the individual assets' mean returns vector if and only if the portfolio is efficient. If the efficient portfolio is $p$ (and its orthogonal portfolio $z$), then:
$$R = r_z 1_n + \frac {r_p - r_z} {\sigma_p^2} (V \cdot X_p) = r_z 1_n +  (r_p - r_z) \beta_p$$
where $\beta_p = \frac {V \cdot X_p} {\sigma_p^2} \in \mathbb{R}^n$ is the vector of slope coefficients of regressions where the explanatory variable is the portfolio mean return $r_p \in \mathbb{R}$ and the $n$ dependent variables are the asset mean returns $R \in \mathbb{R}^n$.

The linearity of $\beta_p$ w.r.t. mean returns $R$ is famously known as the Capital Asset Pricing Model (CAPM).

### Useful Corollaries of CAPM

* If $p$ is SEP, $r_z = 0$ which would mean:
$$R = r_p \beta_p = \frac {r_p} {\sigma_p^2} \cdot V \cdot X_p$$
* So, in this case, covariance vector $V \cdot X_p$ and $\beta_p$ are just scalar multiples of asset mean vector.
* The investment proportion $X$ in a given individual asset changes monotonically along the efficient frontier.
* Covariance $V \cdot X$ is also monotonic along the efficient frontier.
* But $\beta$ is not monotonic, which means that for every individual asset, there is a unique pair of efficient portfolios that result in maximum and minimum $\beta$s for that asset.

### Cross-Sectional Variance

* The cross-sectional variance in $\beta$s (variance in $\beta$s across assets for a fixed efficient portfolio) is zero when the efficient portfolio is GMVP and is also zero when the efficient portfolio has infinite mean.
* The cross-sectional variance in $\beta$s is maximum for the two efficient portfolios with means: $r_0 + \sigma_0^2 \sqrt{|A|}$ and $r_0 - \sigma_0^2 \sqrt{|A|}$ where $A$ is the 2 $\times$ 2 symmetric matrix consisting of $a,b,b,c$.
* These two portfolios lie symmetrically on opposite sides of the efficient frontier (their $\beta$s are equal and of opposite signs), and are the only two orthogonal efficient portfolios with the same variance ( $= 2 \sigma_0^2$).

\index{portfolio theory!efficient portfolio}

### Efficient Set with a Risk-Free Asset
\index{portfolio theory!efficient set}
\index{finance!riskless asset}

If we have a riskless asset with return $r_F$, then $V$ is singular. So we first form the Efficient Frontier without the riskless asset. The Efficient Set (including the riskless asset) is defined as the tangent to this Efficient Frontier (without the riskless asset) from the point $(0, r_F)$ when the Efficient Frontier is considered to be a curve of mean returns (y-axis) against standard deviation of returns (x-axis).

Let's say the tangent touches the Efficient Frontier at the point (Portfolio) $T$ and let its return be $r_T$. Then:

* If $r_F < r_0, r_T > r_F$.
* If $r_F > r_0, r_T < r_F$.
* All portfolios on this efficient set are perfectly correlated.

\index{portfolio theory|)}
