# Derivatives Pricing {#sec:derivatives-pricing-chapter}

In this chapter, we cover two applications of MDP Control regarding financial derivatives pricing and hedging (the word *hedging* refers to reducing or eliminating market risks associated with a derivative). The first application is to identify the optimal time/state to exercise an American Option (a type of financial derivative) in an idealized market setting (akin to the idealized market setting of Merton's Portfolio problem from the previous chapter). Optimal exercise of an American Option is the key to determining it's fair price. The second application is to identify the optimal hedging strategy for derivatives in real-world situations (technically refered to as *incomplete markets*, a term we will define shortly). The optimal hedging strategy of a derivative is the key to determining it's fair price in the real-world (incomplete market) setting. Both of these applications can be cast as Markov Decision Processes where the Optimal Policy gives the Optimal Exercise/Optimal Hedging in the respective applications, leading to the fair price of the derivatives under consideration. Casting these derivatives applications as MDPs means that we can tackle them with Dynamic Programming or Reinforcement Learning algorithms, providing an interesting and valuable alternative to the traditional methods of pricing derivatives.

In order to understand and appreciate the modeling of these derivatives applications as MDPs, one requires some background in the classical theory of derivatives pricing. Unfortunately, thorough coverage of this theory is beyond the scope of this book and we refer you to [Tomas Bjork's book on Arbitrage Theory in Continuous Time](https://www.amazon.com/Arbitrage-Theory-Continuous-Oxford-Finance-ebook/dp/B082HRGDJV) for a thorough understanding of this theory. We shall spend much of this chapter covering the very basics of this theory, and in particular explaining the key technical concepts (such as arbitrage, replication, risk-neutral measure, market-completeness etc.) in a simple and intuitive manner. In fact, we shall cover the theory for the very simple case of discrete-time with a single-period. While that is nowhere near enough to do justice to the rich continuous-time theory of derivatives pricing and hedging, this is the best we can do in a single chapter. The good news is that MDP-modeling of the two problems we want to solve - optimal exercise of American Options and optimal hedging of derivatives in a real-world (incomplete market) setting - doesn't require one to have a thorough understanding of the classical theory. Rather, an intuitive understanding of the key technical and economic concepts should suffice, which we bring to life in the simple setting of discrete-time with a single-period. We start this chapter with a quick introduction to derivatives, next we describe the simple setting of a single-period with formal mathematical notation, covering the key concepts (arbitrage, replication, risk-neutral measure, market-completeness etc.), state and prove the all-important fundamental theorems of asset pricing (only for the single-period setting), and finally show how these two derivatives applications can be cast as MDPs, along with the appropriate algorithms to solve the MDPs. 

## A Brief Introduction to Derivatives

If you are reading this book, you likely already have some familiarity with Financial Derivatives (or at least have heard of them, given that derivatives were at the center of the 2008 financial crisis). In this section, we sketch an overview of financial derivatives and refer you to [the book by John Hull](https://www.amazon.com/Options-Futures-Other-Derivatives-10th/dp/013447208X) for a thorough coverage of Derivatives. The term "Derivative" is based on the word "derived" - it refers to the fact that a derivative is a financial instrument whose structure and hence, value is derived from the *performance* of an underlying entity or entities (which we shall simply refer to as "underlying"). The underlying can be pretty much any financial entity - it could be a stock, currency, bond, basket of stocks, or something more exotic like another derivative. The term *performance* also refers to something fairly generic - it could be the price of a stock or commodity, it could be the interest rate a bond yields, it could be average price of a stock over a time interval, it could be a market-index, or it could be something more exotic like the implied volatility of an option (which itself is a type of derivative). Technically, a derivative is a legal contract between the derivative buyer and seller that either:

* Entitles the derivative buyer to cashflow (which we'll refer to as derivative *payoff*) at future point(s) in time, with the payoff being contingent on the underlying's performance (i.e., the payoff is a precise mathematical function of the underlying's performance, eg: a function of the underlying's price at a future point in time). This type of derivative is known as a "lock-type" derivative.
* Provides the derivative buyer with choices at future points in time, upon making which, the derivative buyer can avail of cashflow (i.e., *payoff*) that is contingent on the underlying's performance. This type of derivative is known as an "option-type" derivative (the word "option" refering to the choice or choices the buyer can make to trigger the contingent payoff).

Although both "lock-type" and "option-type" derivatives can both get very complex (with contracts running over several pages of legal descriptions), we now illustrate both these types of derivatives by going over the most basic derivative structures. In the following descriptions, current time (when the derivative is bought/sold) is denoted as time $t=0$.

### Forwards

The most basic form of Forward Contract involves specification of:

* A future point in time $t=T$ (we refer to $T$ as expiry of the forward contract).
* The fixed payment $K$ to be made by the forward contract buyer to the seller at time $t=T$.

In addition, the contract establishes that at time $t=T$, the forward contract seller needs to deliver the underlying (say a stock with price $S_t$ at time $t$) to the forward contract buyer. This means at time $t=T$, effectively the payoff for the buyer is $S_T - K$ (likewise, the payoff for the seller is $K-S_T$). This is because the buyer, upon receiving the underlying from the seller, can immediately sell the underlying in the market for the price of $S_T$ and so, would have made a gain of $S_T-K$ (note $S_T-K$ can be negative, in which case the payoff for the buyer is negative). 

The problem of forward contract "pricing" is to determine the fair value of $K$ so that the price of this forward contract derivative at the time of contract creation is 0. As time $t$ progresses, the underlying price might fluctuate, which would cause a movement away from the initial price of 0. If the underlying price increases, the price of the forward would naturally increase (and if the underlying price decreases, the price of the forward would naturally decrease). This is an example of a "lock-type" derivative since neither the buyer nor the seller of the forward contract need to make any choices. Rather, the payoff for the buyer is determined directly by the formula $S_T - K$ and the payoff for the seller is determined by the formula $K - S_T$.

### European Options

The most basic forms of European Options are European Call and Put Options. The most basic European Call Option contract involves specification of:

* A future point in time $t=T$ (we refer to $T$ as the expiry of the Call Option).
* Underlying Price $K$ known as strike.

The contract gives the buyer (owner) of the European Call Option the right, but not the obligation, to buy the underlying at time $t=T$ for the price of $K$. Since the option owner doesn't have the obligation to buy, if the price $S_T$ of the underlying at time $t=T$ ends up being equal to or below $K$, the rational decision for the option owner would be to not buy (at price $K$), which would result in a payoff of 0 (in this outcome, we say that the call option is *out-of-the-money*). However, if $S_T > K$, the option owner would make an instant profit of $S_T - K$ by *exercising* her right to buy the underlying at the price of $K$. Hence, the payoff in this case is $S_T-K$ (in this outcome, we say that the call option is *in-the-money*). We can combine the two cases and say that the payoff is $f(S_T) = \max(S_T - K, 0)$. Since the payoff is always non-negative, the call option owner would need to pay for this privilege. The amount the option owner would need to pay to own this call option is known as the fair price of the call option. Identifying the value of this fair price is the highly celebrated problem of *Option Pricing* (which you will learn more about as this chapter progresses).

A European Put Option is very similar to a European Call Option with the only difference being that the owner of the European Put Option has the right (but not the obligation) to *sell* the underlying at time $t=T$ for the price of $K$. This means that the payoff is $f(S_T) = \max(K - S_T, 0)$. Payoffs for these Call and Put Options are known as "hockey-stick" payoffs because if you plot the $f(\cdot)$ function, it is a flat line on the *out-of-the-money* side and a sloped line on the *in-the-money* side. Such European Call and Put Options are "Option-Type" (and not "Lock-Type") derivatives since they involve a choice to be made by the option owner (the choice of exercising the right to buy/sell at the strike price $K$). However, it is possible to construct derivatives with the same payoff as these European Call/Put Options by simply writing in the contract that the option owner will get paid $\max(S_T - K, 0)$ (in case of Call Option) or will get paid $\max(K - S_T, 0)$ (in case of Put Option) at time $t=T$. Such derivatives contracts do away with the option owner's exercise choice and hence, they are "Lock-Type" contracts. There is a subtle difference - setting these derivatives up as "Option-Type" means the option owner might act "irrationally" - the call option owner might mistakenly buy even if $S_T < K$, or the call option owner might for some reason forget/neglect to exercise her option even when $S_T > K$. Setting up such contracts as "Lock-Type" takes away the possibilities of these types of irrationalities from the option owner. However, note that the typical European Call and Put Options are set up as "Option-Type" contracts.

A more general European Derivative involves an arbitrary function $f(\cdot)$ (generalizing from the hockey-stick payoffs) and could be set up as "Option-Type" or "Lock-Type". 

### American Options

The term "European" above refers to the fact that the option to exercise is available only at a fixed point in time $t=T$. Even if it is set up as "Lock-Type", the term "European" typically means that the payoff can happen only at a fixed point in time $t=T$. This is in contrast to American Options. The most basic forms of American Options are American Call and Put Options. American Call and Put Options are essentially extensions of the corresponding European Call and Put Options by allowing the buyer (owner) of the American Option to exercise the option to buy (in the case of Call) or sell (in the case of Put) at any time $t \leq T$. The allowance of exercise at any time at or before the expiry time $T$ can often be a tricky financial decision for the option owner. At each point in time when the American Option is *in-the-money* (i.e., positive payoff upon exercise), the option owner might be tempted to exercise and collect the payoff but might as well be thinking that if she waits, the option might become more *in-the-money* (i.e., prospect of a bigger payoff if she waits for a while). Hence, it's clear that an American Option is always of the "Option-Type" (and not "Lock-Type") since the timing of the decision (option) to exercise is very important in the case of an American Option. This also means that the problem of pricing an American Option (the fair price the buyer would need to pay to own an American Option) is much harder than the problem of pricing a European Option.

So what purpose do derivatives serve? There are actually many motivations for different market participants, but we'll just list two key motivations. The first reason is to protect against adverse market movements that might damage the value of one's portfolio (this is known as *hedging*). As an example, buying a put option can reduce or eliminate the risk associated with ownership of the underlying. The second reason is operational or financial convenience in trading to express a speculative view of market movements. For instance, if one thinks a stock will increase in value by 50\% over the next two years, instead of paying say \$100,000 to buy the stock (hoping to make \$50,000 after two years), one can simply buy a call option on \$100,000 of the stock (paying the option price of say \$5,000). If the stock price indeed appreciates by 50\% after 2 years, one makes \$50,000 - \$5,000 = \$45,000. Although one made \$5000 less than the alternative of simply buying the stock, the fact that one needs to pay \$5000 (versus \$50,000) to enter into the trade means the  potential *return on investment* is much higher.

Next, we embark on the journey of learning how to value derivatives, i.e., how to figure out the fair price that one would be willing to buy or sell the derivative for at any point in time. As mentioned earlier, the general theory of derivatives pricing is quite rich and elaborate (based on continuous-time stochastic processes) but beyond the scope of this book. Instead, we will provide intuition for the core concepts underlying derivatives pricing theory in the context of a simple, special case - that of discrete-time with a single-period. We formalize this simple setting in the next section.

## Notation for the Single-Period Simple Setting

Our simple setting involves discrete time with a single-period from $t=0$ to $t=1$. Time $t=0$ has a single state which we shall refer to as the "Spot" state.  Time $t=1$ has $n$ random outcomes formalized by the sample space $\Omega = \{\omega_1, \ldots, \omega_n\}$. The probability distribution of this finite sample space is given by the probability mass function
$$\mu: \Omega \rightarrow [0,1]$$
such that 
$$\sum_{i=1}^n \mu(\omega_i) = 1$$

This simple single-period setting involves $m + 1$ fundamental assets $A_0, A_1, \ldots, A_m$ where $A_0$ is a riskless asset (i.e., it's price will evolve deterministically from $t=0$ to $t=1$) and $A_1, \ldots, A_m$ are risky assets. We denote the Spot Price (at $t=0$) of $A_j$ as $S_j^{(0)}$ for all $j = 0, 1, \ldots, m$. We denote the Price of $A_j$ in  $\omega_i$ as $S_j^{(i)}$ for all $j = 0, \ldots, m, i = 1, \ldots, n$. Assume that all asset prices are real numbers, i.e., in $\mathbb{R}$ (negative prices are typically unrealistic, but we still assume it for simplicity of exposition). For convenience, we normalize the Spot Price (at $t=0$) of the riskless asset $A_O$ to be 1. Therefore,

$$S_0^{(0)} = 1 \text{ and } S_0^{(i)}= 1 + r \text{ for all } i = 1, \ldots, n$$
where $r$ represents the constant risk-free rate of growth. We should interpret this risk-free rate of growth as the 
["time value of money"](https://en.wikipedia.org/wiki/Time_value_of_money) and $\frac 1 {1+r}$ as the risk-free discount factor corresponding to the "time value of money".

## Portfolios, Arbitrage and Risk-Neutral Probability Measure

We define a portfolio as a vector $\theta = (\theta_0, \theta_1, \ldots, \theta_m) \in \mathbb{R}^{m+1}$. We interpret $\theta_j$ as the number of units held in asset $A_j$ for all $j = 0, 1, \ldots, m$. The Spot Value (at $t=0$) of portfolio $\theta$ denoted $V_{\theta}^{(0)}$ is:
\begin{equation}
V_{\theta}^{(0)} = \sum_{j=0}^m \theta_j \cdot S_j^{(0)}
\label{eq:portfolio-spot-value}
\end{equation}

The Value of portfolio $\theta$ in random outcome $\omega_i$ (at $t=1$) denoted $V_{\theta}^{(i)}$ is:
\begin{equation}
V_{\theta}^{(i)} = \sum_{j=0}^m \theta_j \cdot S_j^{(i)} \mbox{ for all } i = 1, \ldots, n
\label{eq:portfolio-random-value}
\end{equation}

Next, we cover an extremely important concept in Mathematical Economics/Finance, the concept of *Arbitrage*. An Arbitrage Portfolio $\theta$ is one that "makes money from nothing". Formally, an arbitrage portfolio is a portfolio $\theta$ such that:

* $V_{\theta}^{(0)} \leq 0$
* $V_{\theta}^{(i)} \geq 0 \mbox{ for all } i = 1, \ldots,n$
* There exists an $i \in \{1, \ldots, n\}$ such that $\mu(\omega_i) > 0$ and $V_{\theta}^{(i)} > 0$

Thus, with an Arbitrage Portfolio, we never end up (at $t=0$) with less value than what we start with (at $t=1$) and we end up with expected value strictly greater than what we start with. This is the formalism of the notion of [*arbitrage*](https://en.wikipedia.org/wiki/Arbitrage), i.e., "making money from nothing". Arbitrage allows market participants to make infinite returns. In an [efficient market](https://en.wikipedia.org/wiki/Efficient-market_hypothesis), arbitrage would disappear as soon as it appears since market participants would immediately exploit it and through the process of exploiting the arbitrage, immediately eliminate the arbitrage. Hence, Finance Theory typically assumes "arbitrage-free" markets (i.e., financial markets with no arbitrage opportunities).

Next, we describe another very important concept in Mathematical Economics/Finance, the concept of a *Risk-Neutral Probability Measure*. Consider a Probability Distribution $\pi : \Omega \rightarrow [0,1]$ such that 
$$\pi(\omega_i) = 0 \mbox{ if and only if } \mu(\omega_i) = 0 \mbox{ for all } i = 1, \ldots, n$$
Then, $\pi$ is said to be a Risk-Neutral Probability Measure if:
\begin{equation}
S_j^{(0)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot S_j^{(i)} \mbox{ for all } j = 0, 1, \ldots, m
\label{eq:disc-exp-asset-value}
\end{equation}
So for each of the $m+1$ assets, the asset spot price (at $t=0$) is the risk-free rate-discounted expectation (under $\pi$) of the asset price at $t=1$. The term "risk-neutral" here is the same as the term "risk-neutral" we used in Chapter [-@sec:utility-theory-chapter], meaning it's a situation where one doesn't need to be compensated for taking risk (the situation of a linear utility function). However, we are not saying that the market is risk-neutral - if that were the case, the market probability measure $\mu$ would be a risk-neutral probability measure. We are simply defining $\pi$ as a *hypothetical construct* under which each asset's spot price is equal to the risk-free rate-discounted expectation (under $\pi$) of the asset's price at $t=1$. This means that under the hypothetical $\pi$, there's no return in excess of $r$ for taking on the risk of variables outcomes at $t=1$ (note: outcome probabilities are governed by the hypothetical $\pi$). Hence, we refer to $\pi$ as a risk-neutral probability measure.

Before we cover the two fundamental theorems of asset pricing, we need to cover an important lemma that we will utilize in the proofs of the two fundamental theorems of asset pricing.

\begin{lemma}
For any portfolio $\theta = (\theta_0, \theta_1, \ldots, \theta_m) \in \mathbb{R}^{m+1}$ and any risk-neutral probability measure $\pi: \Omega \rightarrow [0, 1]$,
$$V_{\theta}^{(0)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_{\theta}^{(i)}$$
\label{th:disc-exp-portfolio-value}
\end{lemma}
\begin{proof}
Using Equations \eqref{eq:portfolio-spot-value}, \eqref{eq:disc-exp-asset-value} and \eqref{eq:portfolio-random-value}, the proof is straightforward:
\begin{align*}
V_{\theta}^{(0)} & = \sum_{j=0}^m \theta_j \cdot S_j^{(0)} = \sum_{j=0}^m \theta_j \cdot \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot S_j^{(i)} \\
& = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot \sum_{j=0}^m \theta_j \cdot S_j^{(i)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_{\theta}^{(i)}
\end{align*}
\end{proof}

Now we are ready to cover the two fundamental theorems of asset pricing (sometimes, also refered to as the fundamental theorems of arbitrage and the fundamental theorems of finance!). We start with the first fundamental theorem of asset pricing, which associates absence of arbitrage with existence of a risk-neutral probability measure.

## First Fundamental Theorem of Asset Pricing (1st FTAP)

\begin{theorem}[First Fundamental Theorem of Asset Pricing (1st FTAP)]
Our simple setting of discrete time with single-period will not admit arbitrage portfolios if and only if there exists a Risk-Neutral Probability Measure.
\end{theorem}

\begin{proof}
First we prove the easy implication - if there exists a Risk-Neutral Probability Measure $\pi$ , then we cannot have any arbitrage portfolios. Let's review what it takes to have an arbitrage portfolio $\theta = (\theta_0, \theta_1, \ldots, \theta_m)$. The following are two of the three conditions to be satisfied to qualify as an arbitrage portfolio $\theta$ (according to the definition of arbitrage portfolio we gave above):

\begin{itemize}
\item $V_{\theta}^{(i)} \geq 0$ for all $i = 1, \ldots,n$
\item There exists an $i \in \{1, \ldots, n\}$ such that $\mu(\omega_i) > 0$ ($\Rightarrow \pi(\omega_i) > 0$) and $V_{\theta}^{(i)} > 0$
\end{itemize}

But if these two conditions are satisfied, the third condition $V_{\theta}^{(0)} \leq 0$ cannot be satisfied because from Lemma \eqref{th:disc-exp-portfolio-value}, we know that:
$$V_{\theta}^{(0)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_{\theta}^{(i)}$$
which is strictly greater than 0, given the two conditions stated above. Hence, all three conditions cannot be simultaneously satisfied which eliminates the possibility of arbitrage for any portfolio $\theta$.

Next, we prove the reverse (harder to prove) implication - if a risk-neutral probability measure doesn't exist, there exists an arbitrage portfolio $\theta$. We define $\mathbb{V} \subset \mathbb{R}^m$ as the set of vectors $v = (v_1, \ldots, v_m)$ such that
$$v_j = \frac 1 {1+r} \cdot \sum_{i=1}^n \mu(\omega_i) \cdot S_j^{(i)} \mbox{ for all } j = 1, \ldots, m$$
with $\mathbb{V}$ defined as spanning over all possible probability distributions $\mu: \Omega \rightarrow [0,1]$.
$\mathbb{V}$ is a \href{https://en.wikipedia.org/wiki/Convex_polytope}{bounded, closed, convex polytope} in $\mathbb{R}^m$. If a risk-neutral probability measure doesn't exist, the vector $(S_1^{(0)}, \ldots, S_m^{(0)}) \not\in \mathbb{V}$. The \href{https://en.wikipedia.org/wiki/Hyperplane_separation_theorem}{Hyperplane Separation Theorem} implies that there exists a non-zero vector $(\theta_1, \ldots, \theta_m)$ such that
for any $v = (v_1, \ldots, v_m) \in \mathbb{V}$,
$$\sum_{j=1}^m \theta_j \cdot v_j > \sum_{j=1}^m \theta_j \cdot S_j^{(0)}$$
In particular, consider vectors $v$ corresponding to the corners of $\mathbb{V}$, those for which the full probability
 mass is on a particular $\omega_i \in \Omega$, i.e.,
 $$\sum_{j=1}^m \theta_j \cdot (\frac 1 {1+r} \cdot S_j^{(i)}) > \sum_{j=1}^m \theta_j \cdot S_j^{(0)} \mbox{ for all } i = 1, \ldots, n$$
 Since this is a strict inequality, we will be able to choose a $\theta_0 \in \mathbb{R}$ such that:
 $$\sum_{j=1}^m \theta_j \cdot (\frac 1 {1+r} \cdot S_j^{(i)}) > -\theta_0 > \sum_{j=1}^m \theta_j \cdot S_j^{(0)} \mbox{ for all } i = 1, \ldots, n$$
 Therefore,
 $$\frac 1 {1+r} \cdot \sum_{j=0}^m \theta_j \cdot S_j^{(i)} > 0 > \sum_{j=0}^m \theta_j \cdot S_j^{(0)} \mbox{ for all } i = 1, \ldots, n$$
 This can be rewritten in terms of the Values of portfolio $\theta = (\theta_0, \theta_1, \ldots, \theta)$ at $t=0$ and $t=1$, as follows:
 $$\frac 1 {1+r} \cdot V_{\theta}^{(i)} > 0 > V_{\theta}^{(0)} \mbox{ for all } i = 1, \ldots, n$$

 Thus, we can see that all three conditions in the definition of arbitrage portfolio are satisfied and hence, $\theta = (\theta_0, \theta_1, \ldots, \theta_m)$ is an arbitrage portfolio.

\end{proof}

Now we are ready to move on to the second fundamental theorem of asset pricing, which associates replication of derivatives with a unique risk-neutral probability measure.

## Second Fundamental Theorem of Asset Pricing (2nd FTAP)

Before we state and prove the 2nd FTAP, we need some definitions.

\begin{definition} A Derivative $D$ (in our simple setting of discrete-time with a single-period) is specified as a vector payoff at time $t=1$, denoted as:
$$(V_D^{(1)}, V_D^{(2)}, \ldots, V_D^{(n)})$$
where $V_D^{(i)}$ is the payoff of the derivative in random outcome $\omega_i$ for all $i = 1, \ldots, n$
\end{definition}

\begin{definition}
A Portfolio $\theta = (\theta_0, \theta_1, \ldots, \theta_m) \in \mathbb{R}^{m+1}$ is a {\em Replicating Portfolio} for derivative $D$ if:
\begin{equation}
V_D^{(i)} = V_{\theta}^{(i)} = \sum_{j=0}^m \theta_j \cdot S_j^{(i)} \mbox{ for all } i = 1, \ldots, n \label{eq:endreplport}
\end{equation}
\end{definition}

The negatives of the components $(\theta_0, \theta_1, \ldots, \theta_m)$ are known as the *hedges* for $D$ since they can be used to offset the risk in the payoff of $D$ at $t=1$.

\begin{definition}
An arbitrage-free market (i.e., a market devoid of arbitrage) is said to be {\em Complete} if every derivative in the market has a replicating portfolio.
\end{definition}

\begin{theorem}[Second Fundamental Theorem of Asset Pricing (2nd FTAP)]
A market (in our simple setting of discrete-time with a single-period) is Complete if and only if there is a unique risk-neutral probability measure.
\end{theorem}

\begin{proof}
We will first prove that in an arbitrage-free market, if every derivative has a replicating portfolio (i.e., the market is complete), there is a unique risk-neutral probability measure.  We define $n$ special derivatives (known as \href{https://en.wikipedia.org/wiki/State_prices}{{\em Arrow-Debreu securities}}), one for each random outcome in $\Omega$ at $t=1$. We define the time $t=1$ payoff of {\em Arrow-Debreu security} $D_k$ (for each of $k = 1, \ldots, n$) as follows:
$$V_{D_k}^{(i)} = \mathbb{I}_{i=k} \text{ for all } i = 1, \ldots, n$$
where $\mathbb{I}$ represents the indicator function. This means the payoff of derivative $D_k$ is 1 for random outcome $\omega_k$ and 0 for all other random outcomes.

Since each derivative has a replicating portfolio, denote $\theta^{(k)} = (\theta_0^{(k)}, \theta_1^{(k)}, \ldots, \theta_m^{(k)})$ as the replicating portfolio for $D_k$ for each $k = 1, \ldots, m$. Therefore, for each $k= 1, \ldots, m$:

$$V_{\theta^{(k)}}^{(i)} = \sum_{j=0}^m \theta_j^{(k)} \cdot S_j^{(i)} = V_{D_k}^{(i)} = \mathbb{I}_{i=k} \text{ for all } i = 1, \ldots, n$$

Using Lemma \eqref{th:disc-exp-portfolio-value}, we can write the following equation for any risk-neutral probability measure $\pi$, for each $k = 1, \ldots, m$:

$$\sum_{j=0}^m \theta_j^{(k)} \cdot S_j^{(0)} = V_{\theta^{(k)}}^{(0)} = \frac 1 {1 + r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_{\theta^{(k)}}^{(i)} = \frac 1 {1 + r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot \mathbb{I}_{i=k} = \frac 1 {1+r} \cdot \pi(\omega_k)$$

We note that the above equation is satisfied for a unique $\pi: \Omega \rightarrow [0, 1]$, defined as:

$$\pi(\omega_k) = (1 + r) \cdot \sum_{j=0}^m \theta_j^{(k)} \cdot S_j^{(0)} \text{ for all } k = 1, \ldots, n$$

which implies that we have a unique risk-neutral probability measure.

Next, we prove the other direction of the 2nd FTAP. We need to prove that if there exists a risk-neutral probability measure $\pi$ and if there exists a derivative $D$ with no replicating portfolio, we can construct a risk-neutral probability measure different than $\pi$.

Consider the following vectors in the vector space $\mathbb{R}^n$
$$v = (V_D^{(1)}, \ldots, V_D^{(n)}) \mbox{ and } v_j = (S_j^{(1)}, \ldots, S_j^{(n)}) \mbox{ for all } j = 0, 1, \ldots, m$$
Since $D$ does not have a replicating portfolio, $v$ is not in the span of $v_0, v_1, \ldots, v_m$, which means $v_0, v_1, \ldots, v_m$ do not span $\mathbb{R}^n$.  Hence, there exists a non-zero vector $u = (u_1, \ldots, u_n) \in \mathbb{R}^n$ orthogonal to each of $v_0, v_1, \ldots, v_m$, i.e.,
\begin{equation}
\sum_{i=1}^n u_i \cdot S_j^{(i)} = 0 \mbox{ for all } j = 0,1, \ldots, n \label{eq:orthogonal}
\end{equation}
Note that $S_0^{(i)} = 1 + r$ for all $i = 1, \ldots, n$ and so,
\begin{equation}
\sum_{i=1}^n u_i = 0 \label{eq:partitionunity}
\end{equation}
Define $\pi' : \Omega \rightarrow \mathbb{R}$ as follows (for some $\epsilon > 0 \in \mathbb{R})$:
\begin{equation}
\pi'(\omega_i) = \pi(\omega_i) + \epsilon \cdot u_i \mbox{ for all } i = 1, \ldots, n \label{eq:newmeasure}
\end{equation}
To establish $\pi'$ as a risk-neutral probability measure different than $\pi$, note:
\begin{itemize}
\item Since $\sum_{i=1}^n \pi(\omega_i) = 1$ and since $\sum_{i=1}^n u_i = 0$, $\sum_{i=1}^n \pi'(\omega_i) = 1$
\item Construct $\pi'(\omega_i) > 0$ for each $i$ where $\pi(\omega_i) > 0$ by making $\epsilon > 0$ sufficiently small, and set $\pi'(\omega_i) = 0$ for each $i$ where $\pi(\omega_i) = 0$
\item From Equations \eqref{eq:newmeasure}, \eqref{eq:disc-exp-asset-value} and \eqref{eq:orthogonal}, we have for each $j = 0, 1, \ldots, m$:
$$\frac 1 {1 + r} \cdot \sum_{i=1}^n \pi'(\omega_i) \cdot S_j^{(i)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot S_j^{(i)} + \frac {\epsilon} {1 + r} \cdot \sum_{i=1}^n u_i \cdot S_j^{(i)} = S_j^{(0)}$$
\end{itemize}
\end{proof}

Together, the two FTAPs classify markets into:

* Market with arbitrage $\Leftrightarrow$ No risk-neutral probability measure
* Complete (arbitrage-free) market $\Leftrightarrow$ Unique risk-neutral probability measure
* Incomplete (arbitrage-free) market $\Leftrightarrow$ Multiple risk-neutral probability measures

The next topic is derivatives pricing that is based on the concepts of *replication of derivatives* and *risk-neutral probability measures*, and so is tied to the concepts of *arbitrage* and *completeness*.

## Derivatives Pricing

In this section, we cover the theory of derivatives pricing for our simple setting of discrete-time with a single-period. To develop the theory of how to price a derivative, first we need to define the notion of a *Position*.
\begin{definition}
A {\em Position} involving a derivative $D$ is the combination of holding some units in $D$ and some units in the fundamental assets $A_0, A_1, \ldots, A_m$, which can be formally represented as a vector $\gamma_D = (\alpha, \theta_0, \theta_1, \ldots, \theta_m) \in \mathbb{R}^{m+2}$ where $\alpha$ denotes the units held in derivative $D$ and $\alpha_j$ denotes the units held in $A_j$ for all $j = 0, 1 \ldots, m$.
\end{definition}
Therefore, a *Position* is an extension of the Portfolio concept that includes a derivative. Hence, we can naturally extend the definition of *Portfolio Value* to *Position Value* and we can also extend the  definition of *Arbitrage Portfolio* to *Arbitrage Position*.

We need to consider derivatives pricing in three market situations:

* When the market is complete
* When the market is incomplete
* When the market has arbitrage

### Derivatives Pricing when Market is Complete {#sec:pricing-complete-market-subsection}

\begin{theorem}
For our simple setting of discrete-time with a single-period, if the market is complete, then any derivative $D$ with replicating portfolio $\theta = (\theta_0, \theta_1, \ldots, \theta_m)$ has price at time $t=0$ (denoted as value $V_D^{(0)}$):
\begin{equation}
V_D^{(0)} = V_{\theta}^{(0)} = \sum_{j=0}^n \theta_j \cdot S_j^{(i)}
\label{eq:derivatives-pricing-replication}
\end{equation}
Furthermore, if the unique risk-neutral probability measure is $\pi: \Omega \rightarrow [0,1]$, then:
\begin{equation}
V_D^{(0)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_D^{(i)}
\label{eq:derivatives-pricing-risk-neutral}
\end{equation}
\label{th:derivatives-pricing-complete}
\end{theorem}

\begin{proof}
It seems quite reasonable that since $\theta$ is the replicating portfolio for $D$, the value of the replicating portfolio at time $t=0$ (equal to $V_{\theta}^{(0)} = \sum_{j=0}^n \theta_j \cdot S_j^{(i)}$) should be the price (at $t=0$) of derivative $D$. However, we will formalize the proof by first arguing that any candidate derivative price for $D$ other than $V_{\theta}^{(0)}$ leads to arbitrage, thus dismissing those other candidate derivative prices, and then argue that with $V_{\theta}^{(0)}$ as the price of derivative $D$, we eliminate the possibility of an arbitrage position involving $D$.

Consider candidate derivative prices $V_{\theta}^{(0)} - x$ for any positive real number $x$. Position $(1, -\theta_0 + x, -\theta_1, \ldots, -\theta_m)$ has value $x \cdot (1 + r) > 0$ in each of the random outcomes at $t=1$. But this position has spot ($t=0$) value of 0, which means this is an Arbitrage Position, rendering these candidate derivative prices invalid.  Next consider candidate derivative prices $V_{\theta}^{(0)} + x$  for any positive real number $x$. Position $(-1, \theta_0 + x, \theta_1, \ldots, \theta_m)$ has value $x \cdot (1 + r) > 0$ in each of the random outcomes at $t=1$. But this position has spot ($t=0$) value of 0, which means this is an Arbitrage Position, rendering these candidate derivative prices invalid as well. So every candidate derivative price other than $V_{\theta}^{(0)}$ is invalid. Now our goal is to {\em establish} $V_{\theta}^{(0)}$ as the derivative price of $D$ by showing that we eliminate the possibility of an arbitrage position in the market involving $D$ if $V_{\theta}^{(0)}$ is indeed the derivative price.

Firstly, note that $V_{\theta}^{(0)}$ can be expressed as the discounted expectation (under $\pi$) of the payoff of $D$ at $t=1$, i.e.,
\begin{multline}
V_{\theta}^{(0)} = \sum_{j=0}^m \theta_j \cdot S_j^{(0)} = \sum_{j=0}^m \theta_j \cdot \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot S_j^{(i)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot \sum_{j=0}^m \theta_j \cdot S_j^{(i)} \\
= \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_D^{(i)}
\label{eq:deriv-disc-exp-price}
\end{multline}

Now consider an {\em arbitrary portfolio} $\beta = (\beta_0, \beta_1, \ldots, \beta_m)$. Define a position $\gamma_D = (\alpha, \beta_0, \beta_1, \ldots, \beta_m)$. Assuming the derivative price $V_D^{(0)}$ is equal to $V_{\theta}^{(0)}$, the Spot Value (at $t=0$) of position $\gamma_D$, denoted $V_{\gamma_D}^{(0)}$, is:
\begin{equation}
V_{\gamma_D}^{(0)} = \alpha \cdot V_{\theta}^{(0)} + \sum_{j=0}^m \beta_j \cdot S_j^{(0)} \label{eq:startpositionval}
\end{equation}
Value of position $\gamma_D$ in random outcome $\omega_i$ (at $t=1$), denoted $V_{\gamma_D}^{(i)}$, is:
\begin{equation}
V_{\gamma_D}^{(i)} = \alpha \cdot V_D^{(i)} + \sum_{j=0}^m \beta_j \cdot S_j^{(i)} \mbox{ for all } i = 1, \ldots, n \label{eq:endpositionval}
\end{equation}
Combining the linearity in Equations \eqref{eq:disc-exp-asset-value}, \eqref{eq:deriv-disc-exp-price}, \eqref{eq:startpositionval} and \eqref{eq:endpositionval}, we get:
\begin{equation}
V_{\gamma_D}^{(0)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_{\gamma_D}^{(i)} \label{eq:positiondiscexp}
\end{equation}

So the position spot value (at $t=0$) is the discounted expectation (under $\pi$) of the position value at $t=1$. For any $\gamma_D$ (containing any arbitrary portfolio $\beta$), with derivative price $V_D^{(0)}$ equal to $V_{\theta}^{(0)}$, if the following two conditions are satisfied:

\begin{itemize}
\item $V_{\gamma_D}^{(i)} \geq 0 \mbox{ for all } i = 1, \ldots,n$
\item There exists an $i \in \{1, \ldots, n\}$ such that $\mu(\omega_i) > 0$ ($\Rightarrow \pi(\omega_i) > 0$) and $V_{\gamma_D}^{(i)} > 0$
\end{itemize}

then:

$$V_{\gamma_D}^{(0)} = \frac 1 {1 + r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_{\gamma_D}^{(i)} > 0$$
This eliminates any arbitrage possibility if $D$ is priced at $V_{\theta}^{(0)}$.

To summarize, we have eliminated all candidate derivative prices other than $V_{\theta}^{(0)}$, and we have established the price $V_{\theta}^{(0)}$ as the correct price of $D$ in the sense that we eliminate the possibility of an arbitrage position involving $D$ if the price of $D$ is $V_{\theta}^{(0)}$.

Finally, we note that with the derivative price $V_D^{(0)} = V_{\theta}^{(0)}$, from Equation \eqref{eq:deriv-disc-exp-price}, we have:

$$V_D^{(0)} = \frac 1 {1+r} \cdot \sum_{i=1}^n \pi(\omega_i) \cdot V_D^{(i)}$$

\end{proof}

Now let us consider the special case of 1 risky asset ($m=1$) and 2 random outcomes ($n=2$), which we will show is a Complete Market. To lighten notation, we drop the subscript 1 on the risky asset price. Without loss of generality, we assume $S^{(1)} < S^{(2)}$. No-arbitrage requires:
$$S^{(1)} \leq (1 + r) \cdot S^{(0)} \leq S^{(2)}$$
Assuming absence of arbitrage and invoking 1st FTAP, there exists a risk-neutral probability measure $\pi$ such that:
$$S^{(0)} = \frac 1 {1 + r} \cdot (\pi(\omega_1) \cdot S^{(1)} + \pi(\omega_2) \cdot S^{(2)})$$
$$\pi(\omega_1) + \pi(\omega_2) = 1$$
With 2 linear equations and 2 variables, this has a straightforward solution, as follows:
This implies:
$$\pi(\omega_1) = \frac {S^{(2)} - (1 + r) \cdot S^{(0)}} {S^{(2)} - S^{(1)}}$$
$$\pi(\omega_2) = \frac {(1 + r) \cdot S^{(0)}  - S^{(1)}} {S^{(2)} - S^{(1)}}$$
Conditions $S^{(1)} < S^{(2)}$ and $S^{(1)} \leq (1 + r) \cdot S^{(0)} \leq S^{(2)}$ ensure that $0 \leq \pi(\omega_1), \pi(\omega_2) \leq 1$. Also note that this is a unique solution for $\pi(\omega_1), \pi(\omega_2)$, which means that the risk-neutral probability measure is unique, implying that this is a complete market.

We can use these probabilities to price a derivative $D$ as:
$$V_D^{(0)} = \frac 1 {1 + r} \cdot (\pi(\omega_1) \cdot V_D^{(1)} + \pi(\omega_2) \cdot V_D^{(2)})$$
Now let us try to form a replicating portfolio $(\theta_0, \theta_1)$ for $D$
$$V_D^{(1)} = \theta_0 \cdot (1 + r) + \theta_1 \cdot S^{(1)}$$
$$V_D^{(2)} = \theta_0 \cdot (1 + r) + \theta_1 \cdot S^{(2)}$$ 
Solving this yields Replicating Portfolio $(\theta_0, \theta_1)$ as follows:
\begin{equation}
\theta_0 = \frac 1 {1 + r} \cdot \frac {V_D^{(1)} \cdot S^{(2)} - V_D^{(2)} \cdot S^{(1)}} {S^{(2)} - S^{(1)}} \mbox{ and } \theta_1 = \frac {V_D^{(2)} - V_D^{(1)}} {S^{(2)} - S^{(1)}}
\label{eq:two-states-hedges-complete-market}
\end{equation}
Note that the derivative price can also be expressed as:
$$V_D^{(0)} = \theta_0 + \theta_1 \cdot S^{(0)}$$ 

### Derivatives Pricing when Market is Incomplete

Theorem \eqref{th:derivatives-pricing-complete} assumed a complete market, but what about an incomplete market? Recall that an incomplete market means some derivatives can't be replicated. Absence of a replicating portfolio for a derivative precludes usual no-arbitrage arguments. The 2nd FTAP says that in an incomplete market, there are multiple risk-neutral probability measures which means there are multiple derivative prices (each consistent with no-arbitrage).

To develop intuition for derivatives pricing when the market is incomplete, let us consider the special case of 1 risky asset ($m=1$) and 3 random outcomes ($n=3$), which we will show is an Incomplete Market. To lighten notation, we drop the subscript 1 on the risky asset price. Without loss of generality, we assume $S^{(1)} < S^{(2)} < S^{(3)}$. No-arbitrage requires:
$$S^{(1)} \leq S^{(0)} \cdot (1 + r) \leq S^{(3)}$$
Assuming absence of arbitrage and invoking the 1st FTAP, there exists a risk-neutral probability measure $\pi$ such that:
$$S^{(0)} = \frac 1 {1 + r} \cdot (\pi(\omega_1) \cdot S^{(1)} + \pi(\omega_2) \cdot S^{(2)} + \pi(\omega_3) \cdot S^{(3)})$$
$$\pi(\omega_1) + \pi(\omega_2) + \pi(\omega_3) = 1$$
So we have 2 equations and 3 variables, which implies there are multiple solutions for $\pi$. Each of these solutions for $\pi$ provides a valid price for a derivative $D$.
$$V_D^{(0)} = \frac 1 {1 + r} \cdot (\pi(\omega_1) \cdot V_D^{(1)} + \pi(\omega_2) \cdot V_D^{(2)} + \pi(\omega_3) \cdot V_D^{(3)})$$
Now let us try to form a replicating portfolio $(\theta_0, \theta_1)$ for $D$
$$V_D^{(1)} = \theta_0 \cdot (1 + r) + \theta_1 \cdot S^{(1)}$$
$$V_D^{(2)} = \theta_0 \cdot (1 + r) + \theta_1 \cdot S^{(2)}$$ 
$$V_D^{(3)} = \theta_0 \cdot (1 + r) + \theta_1 \cdot S^{(3)}$$
3 equations \& 2 variables implies there is no replicating portfolio for *some* $D$. This means this is an Incomplete Market. 

So with multiple risk-neutral probability measures (and consequent, multiple derivative prices), how do we go about determining how much to buy/sell derivatives for?  One approach to handle derivative pricing in an incomplete market is the technique called  *Superhedging*, which provides upper and lower bounds for the derivative price.  The idea of Superhedging is to create a portfolio of fundamental assets whose Value *dominates* the derivative payoff in *all* random outcomes at $t=1$. Superhedging Price is the smallest possible Portfolio Spot ($t=0$) Value among all such Derivative-Payoff-Dominating portfolios. Without getting into too many details of the Superhedging technique (out of scope for this book), we shall simply sketch the outline of this technique for our simple setting.

We note that for our simple setting of discrete-time with a single-period, this is a constrained linear optimization problem:
\begin{equation}
\min_{\theta} \sum_{j=0}^m \theta_j \cdot S_j^{(0)} \mbox{ such that } \sum_{j=0}^m \theta_j \cdot S_j^{(i)} \geq V_D^{(i)} \mbox{ for all } i = 1, \ldots, n \label{eq:superhedging}
\end{equation}
Let $\theta^* = (\theta_0^*, \theta_1^*, \ldots, \theta_m^*)$ be the solution to Equation \eqref{eq:superhedging}. Let $SP$ be the Superhedging Price $\sum_{j=0}^m \theta_j^* \cdot S_j^{(0)}$.

After establishing feasibility, we define the Lagrangian $J(\theta, \lambda)$ as follows:

$$J(\theta, \lambda) = \sum_{j=0}^m \theta_j \cdot S_j^{(0)} + \sum_{i=1}^n \lambda_i \cdot (V_D^{(i)} - \sum_{j=0}^m \theta_j \cdot S_j^{(i)})$$
So there exists $\lambda = (\lambda_1, \ldots, \lambda_n)$ that satisfy the following KKT conditions:
$$\lambda_i \geq 0 \mbox{ for all } i = 1, \ldots, n$$
$$\lambda_i \cdot (V_D^{(i)} - \sum_{j=0}^m \theta_j^* \cdot S_j^{(i)}) = 0 \mbox{ for all } i = 1, \ldots, n \mbox{ (Complementary Slackness)}$$
$$\nabla_{\theta} J(\theta^*, \lambda) = 0 \Rightarrow S_j^{(0)} = \sum_{i=1}^n \lambda_i \cdot S_j^{(i)} \mbox{ for all } j = 0, 1, \ldots, m$$
This implies $\lambda_i = \frac {\pi(\omega_i)} {1 + r}$ for all $i = 1, \ldots, n$ for a risk-neutral probability measure $\pi : \Omega \rightarrow [0,1]$ ($\lambda$ can be thought of as "discounted probabilities").

Define Lagrangian Dual
$$L(\lambda) = \inf_{\theta} J(\theta, \lambda)$$
Then, Superhedging Price
$$SP = \sum_{j=0}^m \theta_j^* \cdot S_j^{(0)} = \sup_{\lambda} L(\lambda) = \sup_{\lambda} \inf_{\theta} J(\theta, \lambda)$$
Complementary Slackness and some linear algebra over the space of risk-neutral probability measures $\pi : \Omega \rightarrow [0,1]$ enables us to argue that:
$$SP = \sup_{\pi} \sum_{i=1}^n \frac {\pi(\omega_i)} {1 + r} \cdot V_D^{(i)}$$

This means the Superhedging Price is the least upper-bound of the discounted expectation of derivative payoff across each of the risk-neutral probability measures in the incomplete market, which is quite an intuitive thing to do amidst multiple risk-neutral probability measures.``

Likewise, the *Subhedging* price $SB$ is defined as:
$$\max_{\theta} \sum_{j=0}^m \theta_j \cdot S_j^{(0)} \mbox{ such that } \sum_{j=0}^m \theta_j \cdot S_j^{(i)} \leq V_D^{(i)} \mbox{ for all } i = 1, \ldots, n$$
Likewise arguments enable us to establish:
$$SB = \inf_{\pi} \sum_{i=1}^n \frac {\pi(\omega_i)} {1+r} \cdot V_D^{(i)}$$

This means the Subhedging Price is the highest lower-bound of the discounted expectation of derivative payoff across each of the risk-neutral probability measures in the incomplete market, which is quite an intuitive thing to do amidst multiple risk-neutral probability measures

So this technique provides an lower bound ($SB$) and an upper bound ($SP$) for the derivative price, meaning:

* A price outside these bounds leads to an arbitrage 
* Valid prices must be established within these bounds

But often these bounds are not tight and so, not useful in practice. 

The alternative approach is to identify hedges that maximize Expected Utility of the combination of the derivative along with it's hedges, for an appropriately chosen market/trader Utility Function (as covered in Chapter [-@sec:utility-theory-chapter]). The Utility function is a specification of reward-versus-risk preference that effectively chooses the risk-neutral probability measure and (hence, Price). 

Consider a concave Utility function $U : \mathbb{R} \rightarrow \mathbb{R}$ applied to the Value in each random outcome $\omega_i, i = 1, \ldots n$, at $t=1$ (eg: $U(x) = \frac {1 - e^{-ax}} {a}$ where $a \in \mathbb{R}$ is the degree of risk-aversion). Let the real-world probabilities be given by $\mu: \Omega \rightarrow [0,1]$. Denote $V_D = (V_D^{(1)}, \ldots, V_D^{(n)})$ as the payoff of Derivative $D$ at $t=1$. Let us say that you buy the derivative $D$ at $t=0$ and will receive the random outcome-contingent payoff $V_D$ at $t=1$. Let $x$ be the candidate derivative price for $D$, which means you will pay a cash quantity of $x$ at $t=0$ for the privilege of receiving the payoff $V_D$ at $t=1$. We refer to the candidate hedge as Portfolio $\theta = (\theta_0, \theta_1, \ldots, \theta_m)$, representing the units held in the fundamental assets.

Note that at $t=0$, the cash quantity $x$ you'd be paying to buy the derivative and the cash quantity you'd be paying to buy the Portfolio $\theta$ should sum to 0 (note: either of these cash quantities can be positive or negative, but they need to sum to 0 since "money can't just appear or disappear"). Formally,

\begin{equation}
x + \sum_{j=0}^m \theta_j \cdot S_j^{(0)} = 0
\label{eq:time0-balance-constraint}
\end{equation}

Our goal is to solve for the appropriate values of $x$ and $\theta$ based on an *Expected Utility* consideration (that we are about to explain). Consider the Utility of the position consisting of derivative $D$ together with portfolio $\theta$ in random outcome $\omega_i$ at $t=1$:
$$U(V_D^{(i)} + \sum_{j=0}^m \theta_j \cdot S_j^{(i)})$$

So, the Expected Utility of this position at $t=1$ is given by:

\begin{equation}
\sum_{i=1}^n \mu(\omega_i) \cdot U(V_D^{(i)} + \sum_{j=0}^m \theta_j \cdot S_j^{(i)})
\label{eq:time1-expected-utility}
\end{equation}

Noting that $S_0^{(0)} = 1, S_0^{(i)} = 1 + r$ for all $i = 1, \ldots, n$,  we can substitute for the value of $\theta_0 = -(x + \sum_{j=1}^m \theta_j \cdot S_j^{(0)})$ (obtained from Equation \eqref{eq:time0-balance-constraint}) in the above Expected Utility expression \eqref{eq:time1-expected-utility}, so as to rewrite this Expected Utility expression in terms of just $(\theta_1, \ldots, \theta_m)$ (call it $\theta_{1:n}$) as:

$$g(V_D, x, \theta_{1:n}) = \sum_{i=1}^n \mu(\omega_i) \cdot U(V_D^{(i)} - (1 + r) \cdot x + \sum_{j=1}^m \theta_j \cdot (S_j^{(i)} - (1 + r) \cdot S_j^{(0)}))$$

We define the *Price* of $D$ as the "breakeven value" $x^*$ such that:
$$\max_{\theta_{1:n}} g(V_D, x^*, \theta_{1:n}) = \max_{\theta_{1:n}} g(0, 0, \theta_{1:n})$$

The core principle here (known as *Expected-Utility-Indifference Pricing*) is that introducing a $t=1$ payoff of $V_D$ together with a derivative price payment of $x^*$ at $t=0$ keeps the Maximum Expected Utility unchanged.

The $(\theta_1^*, \ldots, \theta_m^*)$ that achieve $\max_{\theta_{1:n}} g(V_D, x^*, \theta_{1:n})$ and $\theta_0^* = -(x^*  + \sum_{j=1}^m \theta_j^* \cdot S_j^{(0)})$ are the requisite hedges associated the derivative price $x^*$. Note that the Price of $V_D$ will NOT be the negative of the Price of $-V_D$, hence these prices simply serve as bid prices or ask prices, depending on whether one pays or receives the random outcomes-contingent payoff $V_D$.

To develop some intuition for what this solution looks like, let us now write some code for the case of 1 risky asset (i.e., $m=1$). To make things interesting, we will write code for the case where the risky asset price at $t=1$ (denoted $S$) follows a normal distribution $S \sim \mathcal{N}(\mu, \sigma^2)$. This means we have a continuous (rather than discrete) set of values for the risky asset price at $t=1$. Since there are more than 2 random outcomes at time $t=1$, this is the case of an Incomplete Market. Moreover, we assume the CARA utility function:

$$U(y) = \frac {1 - e^{-a\cdot y}} {a}$$

where $a$ is the CARA coefficient of risk-aversion.

We refer to the units of investment in the risky asset as $\alpha$ and the units of investment in the riskless asset as $\beta$. Let $S_0$ be the spot ($t=0$) value of the risky asset (riskless asset value at $t=0$ is 1). Let $f(S)$ be the payoff of the derivative $D$ at $t=1$. So, the price of derivative $D$ is the breakeven value $x^*$ such that:

\begin{multline}
\max_{\alpha} \mathbb{E}_{S\sim \mathcal{N}(\mu, \sigma^2)}[\frac {1 - e^{-a\cdot (f(S) - (1+r)\cdot x^* + \alpha \cdot (S - (1 + r) \cdot S_0))}} {a}] \\
= \max_{\alpha} \mathbb{E}_{S\sim \mathcal{N}(\mu, \sigma^2)}[\frac {1 - e^{-a \cdot (\alpha \cdot (S - (1 + r) \cdot S_0))}} {a}]
\label{eq:max-exp-utility-eqn}
\end{multline}

The maximizing value of $\alpha$ (call it $\alpha^*$) on the left-hand-side of Equation \eqref{eq:max-exp-utility-eqn} along with $\beta^* = -(x^* + \alpha^* \cdot S_0)$ are the requisite hedges associated with the derivative price $x^*$.

We set up a `@dataclass MaxExpUtility` with data members to represent the risky asset spot price $S_0$ (`risky_spot`), the riskless rate $r$ (`riskless_rate`), mean $\mu$ of $S$ (`risky_mean`), standard deviation $\sigma$ of $S$ (`risky_stdev`), and the payoff function $f(\cdot)$ of the derivative (`payoff_func`).

```python
@dataclass(frozen=True)
class MaxExpUtility:
    risky_spot: float  # risky asset price at t=0
    riskless_rate: float  # riskless asset price grows from 1 to 1+r
    risky_mean: float  # mean of risky asset price at t=1
    risky_stdev: float  # std dev of risky asset price at t=1
    payoff_func: Callable[[float], float]  # derivative payoff at t=1
```

Before we write code to solve the derivatives pricing and hedging problem for an incomplete market, let us write code to solve the problem for a complete market (as this will serve as a good comparison against the incomplete market solution). For a complete market, the risky asset has two random prices at $t=1$: prices $\mu + \sigma$ and $\mu - \sigma$, with probabilities of 0.5 each. As we've seen in Section [-@sec:pricing-complete-market-subsection], we can perfectly replicate a derivative payoff in this complete market situation as it amounts to solving 2 linear equations in 2 unknowns (solution shown in Equation \eqref{eq:two-states-hedges-complete-market}). The requisite hedges units are simply the negatives of the replicating portfolio units. The method `complete_mkt_price_and_hedges` (of the `MaxExpUtility` class) shown below implements this solution, producing a dictionary comprising of the derivative price (`price`) and the hedge units $\alpha$ (`alpha`) and $\beta$ (`beta`). 

```python
def complete_mkt_price_and_hedges(self) -> Mapping[str, float]:
    x = self.risky_mean + self.risky_stdev
    z = self.risky_mean - self.risky_stdev
    v1 = self.payoff_func(x)
    v2 = self.payoff_func(z)
    alpha = (v1 - v2) / (z - x)
    beta = - 1 / (1 + self.riskless_rate) * (v1 + alpha * x)
    price = - (beta + alpha * self.risky_spot)
    return {"price": price, "alpha": alpha, "beta": beta}
```

Next we write a helper method `max_exp_util_for_zero` (to handle the right-hand-side of Equation \eqref{eq:max-exp-utility-eqn}) that calculates the maximum expected utility for the special case of a derivative with payoff equal to 0 in all random outcomes at $t=1$, i.e., it calculates:
$$\max_{\alpha} \mathbb{E}_{S\sim \mathcal{N}(\mu, \sigma^2)}[\frac {1 - e^{-a\cdot (- (1+r)\cdot c + \alpha \cdot (S - (1 + r) \cdot S_0))}} {a}]$$
where $c$ is cash paid at $t=0$ (so, $c = -(\alpha * S_0 + \beta)$).

The method `max_exp_util_for_zero` accepts as input `c: float` (representing the cash $c$ paid at $t=0$) and `risk_aversion_param: float` (representing the CARA coefficient of risk aversion $a$). Refering to Section [-@sec:norm-distrib-mgf-min] in Appendix [-@sec:mgf-appendix], we have a closed-form solution to this maximization problem:

$$\alpha^* = \frac {\mu - (1+r)\cdot S_0} {a \cdot \sigma^2}$$
$$\beta^* = -(c + \alpha \cdot S_0)$$

Substituting $\alpha^*$ in the Expected Utility expression above gives the following maximum value for the Expected Utility for this special case:

$$\frac {1 - e^{-a\cdot (-(1+r)\cdot c + \alpha^* \cdot (\mu - (1+r)\cdot S_0)) + \frac {(a\cdot \alpha^* \cdot \sigma)^2} 2}} a = \frac {1 - e^{a\cdot (1+r) \cdot c - \frac {(\mu - (1+r)\cdot S_0)^2} {2\sigma^2}}} a$$


```python
def max_exp_util_for_zero(
    self,
    c: float,
    risk_aversion_param: float
) -> Mapping[str, float]:
    ra = risk_aversion_param
    er = 1 + self.riskless_rate
    mu = self.risky_mean
    sigma = self.risky_stdev
    s0 = self.risky_spot
    alpha = (mu - s0 * er) / (ra * sigma * sigma)
    beta = - (c + alpha * self.risky_spot)
    max_val = (1 - np.exp(-ra * (-er * c + alpha * (mu - s0 * er))
                          + (ra * alpha * sigma) ** 2 / 2)) / ra
    return {"alpha": alpha, "beta": beta, "max_val": max_val}
```

Next we write a method `max_exp_util` that calculates the maximum expected utility for the general case of a derivative with an arbitrary payoff $f(\cdot)$ at $t=1$ (provided as input `pf: Callable[[float, float]]` below), i.e., it calculates:

$$\max_{\alpha} \mathbb{E}_{S\sim \mathcal{N}(\mu, \sigma^2)}[\frac {1 - e^{-a\cdot (f(S) - (1+r)\cdot c + \alpha \cdot (S - (1 + r) \cdot S_0))}} {a}]$$

Clearly, this has no closed-form solution since $f(\cdot)$ is an arbitary payoff. The method `max_exp_util` uses the `scipy.integrate.quad` function to calculate the expectation as an integral of the CARA utility function of $f(S) - (1+r) \cdot c + \alpha \cdot (S - (1+r) \cdot S_0)$ multiplied by the probability density of $\mathcal{N}(\mu, \sigma^2)$, and then uses the `scipy.optimize.minimize_scalar` function to perform the maximization over values of $\alpha$.

```python
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

def max_exp_util(
    self,
    c: float,
    pf: Callable[[float], float],
    risk_aversion_param: float
) -> Mapping[str, float]:
    sigma2 = self.risky_stdev * self.risky_stdev
    mu = self.risky_mean
    s0 = self.risky_spot
    er = 1 + self.riskless_rate
    factor = 1 / np.sqrt(2 * np.pi * sigma2)

    integral_lb = self.risky_mean - self.risky_stdev * 6
    integral_ub = self.risky_mean + self.risky_stdev * 6

    def eval_expectation(alpha: float, c=c) -> float:

        def integrand(rand: float, alpha=alpha, c=c) -> float:
            payoff = pf(rand) - er * c\
                     + alpha * (rand - er * s0)
            exponent = -(0.5 * (rand - mu) * (rand - mu) / sigma2
                         + risk_aversion_param * payoff)
            return (1 - factor * np.exp(exponent)) / risk_aversion_param

        return -quad(integrand, integral_lb, integral_ub)[0]

    res = minimize_scalar(eval_expectation)
    alpha_star = res["x"]
    max_val = - res["fun"]
    beta_star = - (c + alpha_star * s0)
    return {"alpha": alpha_star, "beta": beta_star, "max_val": max_val}
```

Finally, it's time to put it all together - the method `max_exp_util_price_and_hedge` below calculates the maximizing $x^*$ in Equation \eqref{eq:max-exp-utility-eqn}. First, we call `max_exp_util_for_zero` (with $c$ set to 0) to calculate the right-hand-side of Equation \eqref{eq:max-exp-utility-eqn}. Next, we create a wrapper function `prep_func` around `max_exp_util`, which is provided as input to `scipt.optimize.root_scalar` to solve for $x^*$ in the right-hand-side of Equation \eqref{eq:max-exp-utility-eqn}. Plugging $x^*$ (`opt_price` in the code below) in `max_exp_util` provides the hedges $\alpha^*$  and $\beta^*$ (`alpha` and `beta` in the code below).

```python
from scipy.optimize import root_scalar

def max_exp_util_price_and_hedge(
    self,
    risk_aversion_param: float
) -> Mapping[str, float]:
    meu_for_zero = self.max_exp_util_for_zero(
        0.,
        risk_aversion_param
    )["max_val"]

    def prep_func(pr: float) -> float:
        return self.max_exp_util(
            pr,
            self.payoff_func,
            risk_aversion_param
        )["max_val"] - meu_for_zero

    lb = self.risky_mean - self.risky_stdev * 10
    ub = self.risky_mean + self.risky_stdev * 10
    payoff_vals = [self.payoff_func(x) for x in np.linspace(lb, ub, 1001)]
    lb_payoff = min(payoff_vals)
    ub_payoff = max(payoff_vals)

    opt_price = root_scalar(
        prep_func,
        bracket=[lb_payoff, ub_payoff],
        method="brentq"
    ).root

    hedges = self.max_exp_util(
        opt_price,
        self.payoff_func,
        risk_aversion_param
    )
    alpha = hedges["alpha"]
    beta = hedges["beta"]
    return {"price": opt_price, "alpha": alpha, "beta": beta}
```

The above code for the class `MaxExpUtility` is in the file [rl/chapter8/max_exp_utility.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/chapter8/max_exp_utility.py). As ever, we encourge you to play with various choices of $S_0, r, \mu, \sigma, f$ in creating instances of `MaxExpUtility`, analyze the obtained prices/hedges and plot some graphs to develop intuition on how the results change as a function of the various inputs. 

Running this code for $S_0 = 100, r = 5\%, \mu = 110, \sigma = 25$ when buying a call option (European since we have only one time period) with strike $= 105$, the method `complete_mkt_price_and hedges` gives an option price of 11.43, risky asset hedge units of -0.6 (i.e., we hedge the risk of owning the call option by short-selling 60% of the risky asset) and riskless asset hedge units of 48.57 (i.e., we take the \$60 proceeds of short-sale less the \$11.43 option price payment = \$48.57 of cash and invest in a risk-free account earning 5% interest). As mentioned earlier, this is the perfect hedge if we had a complete market (i.e., two random outcomes). Running this code for the same inputs for an incomplete market (calling the method `max_exp_util_price_and_hedge` for risk-aversion parameter values of $a=0.3, 0.6, 0.9$ gives us the following results:

```
--- Risk Aversion Param = 0.30 ---
{'price': 23.279, 'alpha': -0.473, 'beta': 24.055}
--- Risk Aversion Param = 0.60 ---
{'price': 12.669, 'alpha': -0.487, 'beta': 35.998}
--- Risk Aversion Param = 0.90 ---
{'price': 8.865, 'alpha': -0.491, 'beta': 40.246}
```

We note that the call option price is quite high (23.28) when the risk-aversion is low at $a=0.3$ (relative to the complete market price of 11.43) but the call option price drops to 12.67 and 8.87 for $a=0.6$ and $a=0.9$ respectively. This makes sense since if you are more risk-averse (high $a$), then you'd be more unwilling to take the risk of buying a call option and hence, would want to pay less to buy the call option. Note how the risky asset short-sale is significantly less (~47\% - 49\%) compared the to the risky asset short-sale of 60\% in the case of a complete market. The varying investments in the riskless asset (as a function of the risk-aversion $a$) essentially account for the variation in option prices (as a function of $a$). Figure \ref{fig:buy_call_option_hedges} provides tremendous intuition on how the hedges work for the case of a complete market and for the cases of an incomplete market with the 3 choices of risk-aversion parameters. Note that we have plotted the negatives of the hedge portfolio values at $t=1$ so as to visualize them appropriately relative to the payoff of the call option. Note that the hedge portfolio value is a linear function of the risky asset price at $t=1$. Notice how the slope and intercept of the hedge portfolio value changes for the 3 risk-aversion scenarios and how they compare against the complete market hedge portfolio value.

![Hedges when buying a Call Option \label{fig:buy_call_option_hedges}](./chapter8/buy_call_option_hedges.png "Hedges when buying a Call Option")

Now let us consider the case of selling the same call option. In our code, the only change we make is to make the payoff function `lambda x: - max(x - 105.0, 0)` instead of `lambda x: max(x - 105.0, 0)` to reflect the fact that we are now selling the call option and so, our payoff will be the negative of that of an owner of the call option.

With the same inputs of $S_0 = 100, r = 5\%, \mu = 110, \sigma = 25$, and for the same risk-aversion parameter values of $a=0.3, 0.6, 0.9$, we get the following results:

```
--- Risk Aversion Param = 0.30 ---
{'price': -6.307, 'alpha': 0.527, 'beta': -46.395}
--- Risk Aversion Param = 0.60 ---
{'price': -32.317, 'alpha': 0.518, 'beta': -19.516}
--- Risk Aversion Param = 0.90 ---
{'price': -44.236, 'alpha': 0.517, 'beta': -7.506}
```

We note that the sale price demand for the call option is quite low (6.31) when the risk-aversion is low at $a=0.3$ (relative to the complete market price of 11.43) but the sale price demand for the call option rises sharply to 32.32 and 44.24 for $a=0.6$ and $a=0.9$ respectively. This makes sense since if you are more risk-averse (high $a$), then you'd be more unwilling to take the risk of selling a call option and hence, would want to charge more for the sale of the call option. Note how the risky asset hedge units are less (~52\% - 53\%) compared the to the risky asset hedge units (60\%) in the case of a complete market. The varying riskless borrowing amounts (as a function of the risk-aversion $a$) essentially account for the variation in option prices (as a function of $a$). Figure \ref{fig:sell_call_option_hedges} provides the visual intuition on how the hedges work for the 3 choices of risk-aversion parameters (along with the hedges for the complete market, for reference). 

![Hedges when selling a Call Option \label{fig:sell_call_option_hedges}](./chapter8/sell_call_option_hedges.png "Hedges when selling a Call Option")

Note that each buyer and each seller might have a different level of risk-aversion, meaning each of them would have a different buy price bid/different sale price ask. A transaction can occur between a buyer and a seller (with potentially different risk-aversion levels) if the buyer's bid matches the seller's ask.

### Derivatives Pricing when Market has Arbitrage

Finally, we arrive at the case where the market has arbitrage. This is the case where there is no risk-neutral probability measure and there can be multiple replicating portfolios (which can lead to arbitrage). This lead to an inability to price derivatives. To provide intuition for the case of a market with arbitrage, we consider the special case of 2 risky assets ($m=2$) and 2 random outcomes ($n=2$), which we will show is a Market with Arbitrage. Without loss of generality, we assume $S_1^{(1)} < S_1^{(2)}$ and $S_2^{(1)} < S_2^{(2)}$. Let us try to determine a risk-neutral probability measure $\pi$:
$$S_1^{(0)} = e^{-r} \cdot (\pi(\omega_1) \cdot S_1^{(1)} + \pi(\omega_2) \cdot S_1^{(2)})$$
$$S_2^{(0)} = e^{-r} \cdot (\pi(\omega_1) \cdot S_2^{(1)} + \pi(\omega_2) \cdot S_2^{(2)})$$
$$\pi(\omega_1) + \pi(\omega_2) = 1$$
3 equations and 2 variables implies that there is no risk-neutral probability measure $\pi$. Let's try to form a replicating portfolio $(\theta_0, \theta_1, \theta_2)$ for a derivative $D$:
$$V_D^{(1)} = \theta_0 \cdot e^r + \theta_1 \cdot S_1^{(1)} + \theta_2 \cdot S_2^{(1)}$$
$$V_D^{(2)} = \theta_0 \cdot e^r + \theta_1 \cdot S_1^{(2)} + \theta_2 \cdot S_2^{(2)}$$
2 equations and 3 variables implies that there are multiple replicating portfolios. Each such replicating portfolio yields a price for $D$ as:
$$V_D^{(0)} = \theta_0 + \theta_1 \cdot S_1^{(0)} + \theta_2 \cdot S_2^{(0)}$$
Select two such replicating portfolios with different $V_D^{(0)}$. The combination of one of these replicating portfolios with the negative of the other replicating portfolio is an Arbitrage Portfolio because:

* They cancel off each other's portfolio value in each $t=1$ states
* The combined portfolio value can be made to be negative at $t=0$ (by choosing which replicating portfolio we negate)

So this is a market that admits arbitrage (no risk-neutral probability measure).

## Overview of General Theory
## Classical Pricing and Hedging of Derivatives
## Pricing and Hedging in an Incomplete Market
## Modeling Optimal Hedging as an MDP
## The Principle of Indifference Pricing
## Deep Reinforcement Learning as a practical alternative to traditional approaches
