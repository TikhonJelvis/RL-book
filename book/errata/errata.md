## Book Errata {.unnumbered}

The following is the errata for the published book *Foundations of Reinforcement Learning with Applications in Finance* by *Ashwin Rao* and *Tikhon Jelvis*

### Chapter 3 {.unnumbered}

#### Definition of Stationary Distribution of a Markov Process {.unnumbered}

On page 58, the equation in Definition 3.7.1 that defines the Stationary Distribution of a Markov Process is incorrect. The corrected equation is as follows:

$$\pi(s') = \sum_{s \in \mathcal{N}} \pi(s) \cdot \mathcal{P}(s, s') \text{ for all } s' \in \mathcal{N}$$

### Chapter 4 {.unnumbered}

#### Bellman Policy Equation Derivation {.unnumbered}

At the very first line of page 90, we say that Equation (4.3) can be obtained by combining Equation (4.1) and Equation (4.2). This is not correct. Rather, Equation (4.3) can be derived as follows:

Since $Q^{\pi}(s, a) = \mathbb{E}_{\pi, \mathcal{P}_R}[G_t|S_t=s, A_t=a]$, by expanding $G_t$ as $R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots$, we can expand $Q^{\pi}(s,a)$ as:

\begin{equation*}
\begin{split}
& \mathbb{E}_{\mathcal{P}_R}[R_{t+1}|S_t=s,A_t=a] + \gamma \cdot \mathbb{E}_{\pi, \mathcal{P}_R}[R_{t+2}|S_t=s,A_t=a] + \gamma^2 \cdot \mathbb{E}_{\pi, \mathcal{P}_R}[R_{t+3}|S_t=s,A_t=a] \\
& \hspace{8mm} + \ldots \\
& = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{N}} \mathcal{P}(s, a, s') \sum_{a'\in \mathcal{A}} \pi(s',a') \cdot \mathcal{R}(s', a') \\
& \hspace{4mm} + \gamma^2 \cdot \sum_{s' \in \mathcal{N}} \mathcal{P}(s, a', s') \sum_{a'\in \mathcal{A}} \pi(s',a') \sum_{s'' \in \mathcal{N}} \mathcal{P}(s', a'', s'') \sum_{a'' \in \mathcal{A}} \pi(s'',a'') \cdot \mathcal{R}(s'', a'')  \\
& \hspace{4mm} + \ldots \\
& = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{N}} \mathcal{P}(s, a, s') \cdot \mathcal{R}^{\pi}(s') + \gamma^2 \cdot \sum_{s' \in \mathcal{N}} \mathcal{P}(s, a, s') \sum_{s'' \in \mathcal{N}} \mathcal{P}^{\pi}(s', s'') \cdot \mathcal{R}^{\pi}(s'') + \ldots  \\
& = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{N}} \mathcal{P}(s, a, s') \cdot (\mathcal{R}^{\pi}(s') + \gamma \cdot \sum_{s'' \in \mathcal{N}} \mathcal{P}^{\pi}(s', s'') \cdot \mathcal{R}^{\pi}(s'') + \ldots ) \\
& = \mathcal{R}(s,a) + \gamma \cdot \sum_{s'\in \mathcal{N}} \mathcal{P}(s, a, s') \cdot V^{\pi}(s')
\end{split}
\end{equation*}

The last line in the sequence of equations above is obtained by observing Equation (3.1) in Chapter 3, with the MRP Value Function expansion applied to the $\pi$-implied MRP for state $s'$.

### Chapter 6 {.unnumbered}

#### `argmax` code correction {.unnumbered}

On page 144, the body of the code for the method `argmax` (of `class FunctionApprox`) is not correct. The correct code is shown below and can also be found in [rl/function_approximation.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/function_approximation.py). 

```python
    def argmax(self, xs: Iterable[X]) -> X:
        args: Sequence[X] = list(xs)
        return args[np.argmax(self.evaluate(args))]
```

### Chapter 13 {.unnumbered}

On page 374, in the second paragraph of Section 13.7.2, we say that $\bb$ is a linear operator in vector space $\mathbb{R}^n$. This is an incorrect statement. $\bb$ is an affine operator and not a linear operator because there is a vector-shift of $\mathcal{R}^{\pi}$ after multiplying $\bv$ with the matrix $\gamma \mathcal{P}^{\pi}$. This second paragraph of Section 13.7.2 is corrected as follows:

Note that $\bb$ is an [affine transformation](https://en.wikipedia.org/wiki/Affine_transformation) on vectors in $\mathbb{R}^n$. We lighten notation for application of the $\bb$ operator on any vector $\bv \in \mathbb{R}^n$ as simply $\bb \cdot \bv$ (with $\cdot$ conveying the notion of operator application). We've learnt that $\bvpi$ is the fixed point of $\bb$. Therefore, we can write:
$$\bb \cdot \bvpi = \bvpi$$
