## Summary of Notation {.unnumbered}

\begin{longtable}{|p{0.2\linewidth}|p{0.7\linewidth}|}
\hline
$\mathbb{Z}$ & Set of integers \\
\hline
$\mathbb{Z}^+$ & Set of positive integers, i.e., $\{1,2,3,\ldots\}$ \\
\hline
$\mathbb{Z}_{\geq 0}$ & Set of non-negative integers, i.e., $\{0, 1,2,3,\ldots\}$ \\
\hline
$\mathbb{R}$ & Set of real numbers \\
\hline
$\mathbb{R}^+$ & Set of positive real numbers \\
\hline
$\mathbb{R}_{\geq 0}$ & Set of non-negative real numbers \\
\hline
$\log(x)$ & {\em Natural Logarithm} (to the base $e$) of $x$ \\
\hline
$|x|$ & {\em Absolute Value} of $x$ \\
\hline
$sign(x)$ & +1 if $x > 0$, -1 if $x < 0$, 0 if $x=0$ \\
\hline
$[a,b]$ & Set of real numbers that are $\geq a$ and $\leq b$. The notation $x \in [a,b]$ is shorthand for $x \in \mathbb{R}$ and $a \leq x \leq b$ \\
\hline
$[a,b)$ & Set of real numbers that are $\geq a$ and $< b$. The notation $x \in [a,b)$ is shorthand for $x \in \mathbb{R}$ and $a \leq x < b$ \\
\hline
$(a,b]$ & Set of real numbers that are $> a$ and $leq b$. The notation $x \in (a,b]$ is shorthand for $x \in \mathbb{R}$ and $a < x \leq b$ \\
\hline
$\emptyset$ & The Empty Set (Null Set) \\
\hline
$\sum_{i=1}^n a_i$ & Sum of terms $a_1, a_2, \ldots, a_n$ \\
\hline
$\prod_{i=1}^n a_i$ & Product of terms $a_1, a_2, \ldots, a_n$ \\
\hline
$\approx$ & approximately equal to \\ 
\hline
$x \in \mathcal{X}$ & $x$ is an element of the set $\mathcal{X}$ \\
\hline
$x \notin \mathcal{X}$ & $x$ is not an element of the set $\mathcal{X}$ \\
\hline
$\mathcal{X} \cup \mathcal{Y}$ & {\em Union} of the sets $\mathcal{X}$ and $\mathcal{Y}$ \\
\hline
$\mathcal{X} \cap \mathcal{Y}$ & {\em Intersection} of the sets $\mathcal{X}$ and $\mathcal{Y}$ \\
\hline
$\mathcal{X} - \mathcal{Y}$ & {\em Set Difference} of the sets $\mathcal{X}$ and $\mathcal{Y}$, i.e., the set of elements within the set $\mathcal{X}$ that are not elements of the set $\mathcal{Y}$ \\
\hline
$\mathcal{X} \times \mathcal{Y}$ & {\em Cartesian Product} of the sets $\mathcal{X}$ and $\mathcal{Y}$ \\
\hline
$\mathcal{X}^k$ & For a set $\mathcal{X}$ and an integer $k \geq 1$, this refers to the {\em Cartesian Product} $\mathcal{X} \times \mathcal{X} \times \ldots \times \mathcal{X}$ with $k$ occurrences of $\mathcal{X}$ in the Cartesian Product (note: $\mathcal{X}^1 = \mathcal{X}$) \\
\hline
$f: X \rightarrow Y$ & {\em Function} $f$ with {\em Domain} $X$ and {\em Co-domain} $Y$ \\
\hline
$f^k$ & For a function $f$ and an integer $k \geq 0$, this refers to the {\em function composition} of $f$ with itself, repeated $k$ times. So, $f^k(x)$ is the value $f(f(\ldots f(x) \ldots ))$ with $k$ occurrences of $f$ in this function-composition expression (note: $f^1 = f$ and $f^0$ is the identity function) \\
\hline
$f^{-1}$ & {\em Inverse function} of a bijective function $f: \mathcal{X} \rightarrow \mathcal{Y}$, i.e., for all $x \in \mathcal{X}, f^{-1}(f(x)) = x$ and for all $y \in \mathcal{Y}$, $f(f^{-1}(y)) = y$ \\
\hline
$f'(x_0)$ & {\em Derivative} of the function $f: \mathcal{X} \rightarrow \mathbb{R}$ with respect to it's domain variable $x \in \mathcal{X}$, evaluated at $x=x_0$ \\
\hline
$f''(x_0)$ & {\em Second Derivative} of the function $f: \mathcal{X} \rightarrow \mathbb{R}$ with respect to it's domain variable $x \in \mathcal{X}$, evaluated at $x=x_0$ \\
\hline
$\mathbb{P}[X]$ & {\em Probability Density Function} (PDF) of random variable $X$\\
\hline
$\mathbb{P}[X=x]$ & Probability that random variable $X$ takes the value $x$ \\
\hline
$\mathbb{P}[X|Y]$ & {\em Probability Density Function} (PDF) of random variable $X$, conditional on the value of random variable $Y$ (i.e., PDF of $X$ expressed as a function of the values of $Y$) \\
\hline
$\mathbb{P}[X=x|Y=y]$ & Probability that random variable $X$ takes the value $x$, conditional on random variable $Y$ taking the value $y$ \\
\hline
$\mathbb{E}[X]$ & {\em Expected Value} of random variable $X$ \\
\hline
$\mathbb{E}[X|Y]$ & {\em Expected Value} of random variable $X$, conditional on the value of random variable $Y$ (i.e., Expected Value of $X$ expressed as a function of the values of $Y$) \\
\hline
$\mathbb{E}[X|Y=y]$ & {\em Expected Value} of random variable $X$, conditional on random variable $Y$ taking the value $y$ \\
\hline
$x \sim \mathcal{N}(\mu, \sigma^2)$ & Random variable $x$ follows a {\em Normal Distribution} with mean $\mu$ and variance $\sigma^2$ \\
\hline
$x \sim Poisson(\lambda)$ & Random variable $x$ follows a {\em Poisson Distribution} with mean $\lambda$\\
\hline
$f(x;\bm{w})$ & Here $f$ refers to a parameterized function with domain $\mathcal{X}$ ($x \in \mathcal{X}$), $\bm{w}$ refers to the parameters controlling the definition of the function $f$ \\
\hline
$\bm{v}^T$ & {\em Row-vector} with components equal to the components of the {\em Column-vector} $\bm{v}$, i.e., {\em Transpose} of the {\em Column-vector} $\bm{v}$ (by default, we assume vectors are expressed as {\em Column-vectors}) \\
\hline
$\bm{A}^T$ & {\em Transpose} of the {\em matrix} $\bm{A}$ \\
\hline
$|\bm{v}|$ & $L^2$ norm of vector $\bm{v} \in \mathbb{R}^m$, i.e., if $\bm{v} = (v_1, v_2, \ldots, v_m)$, then $|\bm{v}| = \sqrt{v_1^2 + v_2^2 + \ldots + v_m^2}$ \\
\hline
$\bm{A}^{-1}$ & {\em Matrix-Inverse} of the {\em square matrix} $\bm{A}$ \\
\hline
$\bm{A} \cdot \bm{B}$ & {\em Matrix-Multiplication} of matrices $\bm{A}$ and $\bm{B}$ (note: vector notation $\bm{v}$ typically refers to a column-vector, i.e., a matrix with 1 column, and so $\bm{v}^T \cdot \bm{w}$ is simply the {\em inner-product} of same-dimensional vectors $\bm{v}$ and $\bm{w}$) \\
\hline
$\bm{I}_m$ & $m \times m$ {\em Identity Matrix} \\
\hline
$\bm{Diagonal}(\bm{v})$ & $m \times m$ Diagonal Matrix whose elements are the same (also in same order) as the elements of the $m$-dimensional Vector $\bm{v}$ \\
\hline
$dim(\bm{v})$ & Dimension of a vector $\bm{v}$ \\
\hline
$\mathbb{I}_c$ & $\mathbb{I}$ represents the {\em Indicator function} and $\mathbb{I}_c = 1$ if condition $c$ is True, $= 0$ if $c$ is False \\
\hline
$\argmax_{x \in \mathcal{X}} f(x)$ & This refers to the value of $x \in \mathcal{X}$ that maximizes $f(x)$, i.e., $\max_{x \in \mathcal{X}} f(x) = f(\argmax_{x \in \mathcal{X}} f(x))$ \\
\hline
$\nabla_{\bm{w}} f(\bm{w})$ & Gradient of the function $f$ with respect to $\bm{w}$ (note: $\bm{w}$ could be an arbitrary data structure and this gradient is of the same data type as the data type of $\bm{w}$) \\
\hline
$x \leftarrow y$ & Variable $x$ is assigned (or updated to) the value of $y$ \\
\hline
\end{longtable}
