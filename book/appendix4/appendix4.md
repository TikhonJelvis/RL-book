## The Hamilton-Jacobi-Bellman (HJB) Equation {#sec:hjb-appendix}
\index{Bellman equations!Hamilton-Jacobi-Bellman equation|(}

In this Appendix, we provide a quick coverage of the Hamilton-Jacobi-Bellman (HJB) Equation, which is the continuous-time version of the Bellman Optimality Equation. Although much of this book covers Markov Decision Processes in a discrete-time setting, we do cover some classical Mathematical Finance Stochastic Control formulations in continuous-time. To understand these formulations, one must first understand the HJB Equation, which is the purpose of this Appendix. As is the norm in the Appendices in this book, we will compromise on some of the rigor and emphasize the intuition to develop basic familiarity with HJB. 

### HJB as a Continuous-Time Version of Bellman Optimality Equation

In order to develop the continuous-time setting, we shall consider a (not necessarily time-homogeneous) process where the set of states at time $t$ are denoted as $\mathcal{S}_t$ and the set of allowable actions for each state at time $t$ are denoted as $\mathcal{A}_t$. Since time is continuous, Rewards are represented as a *Reward Rate* function $\mathcal{R}$ 
such that for any state $s_t \in \mathcal{S}_t$ and for any action $a_t \in \mathcal{A}_t$, $\mathcal{R}(t, s_t, a_t) \cdot dt$ is the *Expected Reward* in the time interval $(t, t + dt]$, conditional on state $s_t$ and action $a_t$ (note the functional dependency of $\mathcal{R}$ on $t$ since we will be integrating $\mathcal{R}$ over time). Instead of the discount factor $\gamma$ as in the case of discrete-time MDPs, here we employ a *discount rate* (akin to interest-rate discounting) $\rho \in \mathbb{R}_{\geq 0}$ so that the discount factor over any time interval $(t, t+dt]$ is $e^{-\rho \cdot dt}$.

\index{value function!optimal value function}
We denote the Optimal Value Function as $V^*$ such that the Optimal Value for state $s_t \in \mathcal{S}_t$ at time $t$ is $V^*(t, s_t)$. Note that unlike Section [-@sec:finite-horizon-section] in Chapter [-@sec:dp-chapter] where we denoted the Optimal Value Function as a time-indexed sequence $V^*_t(s_t)$, here we make $t$ an explicit functional argument of $V^*$. This is because in the continuous-time setting, we are interested in the time-differential of the Optimal Value Function.

\index{Bellman equations!Bellman optimality equations}

Now let us write the Bellman Optimality Equation in its continuous-time version, i.e, let us consider the process $V^*$ over the time interval $(t, t+dt]$ as follows:

$$V^*(t, s_t) = \max_{a_t \in \mathcal{A}_t} \{ \mathcal{R}(t, s_t, a_t) \cdot dt + \mathbb{E}_{(t, s_t, a_t)}[e^{-\rho \cdot dt} \cdot V^*(t+dt, s_{t+dt})] \}$$
Multiplying throughout by $e^{-\rho t}$ and re-arranging, we get:
$$\max_{a_t \in \mathcal{A}_t} \{ e^{-\rho t} \cdot \mathcal{R}(t, s_t, a_t) \cdot dt + \mathbb{E}_{(t, s_t, a_t)}[e^{-\rho (t + dt)} \cdot V^*(t + dt, s_{t+dt}) - e^{-\rho t} \cdot V^*(t, s_t)] \} = 0$$
$$\Rightarrow \max_{a_t \in \mathcal{A}_t} \{ e^{-\rho t} \cdot \mathcal{R}(t, s_t, a_t) \cdot dt + \mathbb{E}_{(t, s_t, a_t)}[d\{e^{-\rho  t} \cdot V^*(t, s_t)\}] \} = 0$$
$$\Rightarrow \max_{a_t \in \mathcal{A}_t} \{ e^{-\rho t} \cdot \mathcal{R}(t, s_t, a_t) \cdot dt + \mathbb{E}_{(t, s_t, a_t)}[e^{-\rho t} \cdot (dV^*(t, s_t) - \rho \cdot V^*(t, s_t) \cdot dt)] \} = 0$$
Multiplying throughout by $e^{\rho t}$ and re-arranging, we get:
\begin{equation}
\rho \cdot V^*(t, s_t) \cdot dt = \max_{a_t \in \mathcal{A}_t} \{ \mathbb{E}_{(t, s_t, a_t)}[dV^*(t, s_t)] + \mathcal{R}(t, s_t, a_t) \cdot dt\} \label{eq:hjb}
\end{equation}

\index{Bellman equations!Hamilton-Jacobi-Bellman equation|textbf}

For a finite-horizon problem terminating at time $T$, the above equation is subject to terminal condition:
$$V^*(T, s_T) = \mathcal{T}(s_T)$$
for some terminal reward function $\mathcal{T}(\cdot)$.

Equation \eqref{eq:hjb} is known as the Hamilton-Jacobi-Bellman Equation—the continuous-time analog of the Bellman Optimality Equation. In the literature, it is often written in a more compact form that essentially takes the above form and "divides throughout by dt". This requires a few technical details involving the [stochastic differentiation operator](https://en.wikipedia.org/wiki/Infinitesimal_generator_(stochastic_processes)). To keep things simple, we shall stick to the HJB formulation of Equation \eqref{eq:hjb}.

### HJB with State Transitions as an Ito Process

\index{stochastic process!Ito process}
\index{stochastic process!Brownian motion}

Although we have expressed the HJB Equation for $V^*$, we cannot do anything useful with it unless we know the state transition probabilities (all of which are buried inside the calculation of $\mathbb{E}_{(t, s_t, a_t)}[\cdot]$ in the HJB Equation). In continuous-time, the state transition probabilities are modeled as a stochastic process for states (or of its features). Let us assume that states are real-valued vectors, i.e, state $\bm{s}_t \in \mathbb{R}^n$ at any time $t \geq 0$ and that the transitions for $\bm{s}$ are given by an Ito process, as follows:

$$d\bm{s}_t = \bm{\mu}(t, \bm{s}_t, a_t) \cdot dt + \bm{\sigma}(t, \bm{s}_t, a_t) \cdot d\bm{z}_t$$
where the function $\bm{\mu}$ (drift function) gives an $\mathbb{R}^n$ valued process, the function $\bm{\sigma}$ (dispersion function) gives an $\mathbb{R}^{n \times m}$-valued process and $\bm{z}$ is an $m$-dimensional process consisting of $m$ independent standard Brownian motions.

\index{stochastic process!Ito process}
\index{stochastic calculus!Ito's lemma}

Now we can apply multivariate Ito's Lemma (Equation \eqref{eq:itos-lemma-multi} from Appendix [-@sec:stochasticcalculus-appendix]) for $V^*$ as a function of $t$ and $s_t$ (we lighten notation by writing $\bm{\mu}_t$ and $\bm{\sigma}_t$ instead of $\bm{\mu}(t, \bm{s}_t, a_t)$ and $\bm{\sigma}(t, \bm{s}_t, a_t)$):

$$dV^*(t, \bm{s}_t) = (\pdv{V^*}{t} + (\nabla_{\bm{s}} V^*)^T \cdot \bm{\mu}_t + \frac 1 2 Tr[\bm{\sigma}_t^T \cdot (\Delta_{\bm{s}} V^*) \cdot \bm{\sigma}_t]) \cdot dt + (\nabla_{\bm{s}} V^*)^T \cdot \bm{\sigma}_t \cdot d\bm{z}_t$$

Substituting this expression for $dV^*(t, \bm{s}_t)$ in Equation \eqref{eq:hjb}, noting that
$$\mathbb{E}_{(t, \bm{s}_t, a_t)}[(\nabla_{\bm{s}} V^*)^T \cdot \bm{\sigma}_t \cdot d\bm{z}_t] = 0$$ and dividing throughout by $dt$, we get:

\begin{equation}
\rho \cdot V^*(t, \bm{s}_t) = \max_{a_t \in \mathcal{A}_t} \{ \pdv{V^*}{t} + (\nabla_{\bm{s}} V^*)^T \cdot \bm{\mu}_t + \frac 1 2 Tr[\bm{\sigma}_t^T \cdot (\Delta_{\bm{s}} V^*) \cdot \bm{\sigma}_t] + \mathcal{R}(t, \bm{s}_t, a_t)\} \label{eq:hjb-ito}
\end{equation}

For a finite-horizon problem terminating at time $T$, the above equation is subject to terminal condition:
$$V^*(T, \bm{s}_T) = \mathcal{T}(\bm{s}_T)$$
for some terminal reward function $\mathcal{T}(\cdot)$.
\index{Bellman equations!Hamilton-Jacobi-Bellman equation|)}
