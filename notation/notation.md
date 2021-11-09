# Summary of Notation

\begin{table}
\begin{tabular}{|p{0.2\linewidth}|p{0.7\linewidth}|}
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
$[a,b]$ & Set of real numbers that are $\geq a$ and $\leq b$. The notation $x \in [a,b]$ is shorthand for $x \in \mathbb{R}$ and $a \leq x \leq b$ \\
\hline
$x \in \mathcal{X}$ & $x$ is an element of the set $\mathcal{X}$ \\
\hline
$\mathcal{X} \cup \mathcal{Y}$ & {\em Union} of the sets $\mathcal{X}$ and $\mathcal{Y}$ \\
\hline
$\mathcal{X} \cap \mathcal{Y}$ & {\em Intersection} of the sets $\mathcal{X}$ and $\mathcal{Y}$ \\
\hline
$\mathcal{X} - \mathcal{Y}$ & {\em Set Difference} of the sets $\mathcal{X}$ and $\mathcal{Y}$, i.e., the set of elements within the set $\mathcal{X}$ that are not elements of the set $\mathcal{Y}$ \\
\hline
$\mathcal{X} \times \mathcal{Y}$ & {\em Cartesian Product} of the sets $\mathcal{X}$ and $\mathcal{Y}$ \\
\hline
$f: X \rightarrow Y$ & {\em Function} $f$ with {\em Domain} $X$ and {\em Co-domain} $Y$ \\
\hline
$\mathbb{P}[X]$ & {\em Probability Density Function} (PDF) of random variable $X$\\
\hline
$\mathbb{P}[X=x]$ & Probability that random variable $X$ takes the value $x$ \\
\hline
$\mathbb{P}[X|Y]$ & {\em Probability Density Function} (PDF) of random variable $X$, conditional on the value of random variable $Y$ (i.e., $X$ PDF expressed as a function of the values of $Y$) \\
\hline
$\mathbb{P}[X=x|Y=y]$ & Probability that random variable $X$ takes the value $x$, conditional on random variable $Y$ taking the value $y$ \\
\hline
$\mathbb{E}[X]$ & Expected Value of random variable $X$ \\
\hline
$\mathbb{E}[X|Y]$ & Expected Value of random variable $X$, conditional on the value of random variable $Y$ (i.e., Expected Value of $X$ expressed as a function of the values of $Y$) \\
\hline
$\mathbb{E}[X|Y=y]$ & Expected Value of random variable $X$, conditional on random variable $Y$ taking the value $y$ \\
\hline
$f(x;\bm{w})$ & Here $f$ refers to a parameterized function with domain $\mathcal{X}$ ($x \in \mathcal{X}$), $\bm{w}$ refers to the parameters controlling the definition of the function $f$ \\
\hline
$\bm{A}^T$ & {\em Transpose} of the {\em matrix} $\bm{A}$ \\
\hline
$\bm{v}^T$ & {\em Row-vector} with components equal to the components of the {\em Column-vector} $\bm{v}$, i.e., {\em Transpose} of the {\em (Column-)vector} $\bm{v}$ (by default, we assume vectors are expressed as {\em Column-vectors}) \\
\hline
$\bm{A}^{-1}$ & {\em Matrix-Inverse} of the {\em square matrix} $\bm{A}$ \\
\hline
$\mathbb{I}_c$ & $\mathbb{I}$ represents the {\em Indicator function} and $\mathbb{I}_c = 1$ if condition $c$ is True, $= 0$ if $c$ is False \\
\hline
$\argmax_{x \in \mathcal{X}} f(x)$ & This refers to the value of $x \in \mathcal{X}$ that maximizes $f(x)$, i.e., $\max_{x \in \mathcal{X}} f(x) = f(\argmax_{x \in \mathcal{X}} f(x))$ \\
\hline
\end{tabular}
\end{table}
