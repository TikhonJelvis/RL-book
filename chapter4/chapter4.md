# Dynamic Programming Algorithms

As a reminder, much of this book is about algorithms to solve the MDP Control problem, i.e., to compute the Optimal Value Function (and an associated Optimal Policy). We will also cover algorithms for the MDP Prediction problem, i.e., to compute the Value Function when the agent executes a fixed policy $\pi$ (which, as we know from the previous chapter, is the same as the $\pi$-implied MRP problem). Our typical approach will be to first cover algorithms to solve the Prediction problem before covering algorithms to solve the Prediction problem - not just because Prediction is a key component in solving the Control problem, but also because it helps understand the key aspects of the techniques employed in the Control algorithm in the simpler setting of Prediction.

## Planning versus Learning

In this book, we shall look at Planning and Control from the lens of AI (and we'll specifically use the terminology of AI). We shall distinguish between algorithms that don't have a model of the MDP environment (no access to the $\mathcal{P}_R$ function) versus algorithms that do have a model of the MDP environment (meaning $\mathcal{P}_R$ is available to us either in terms of explicit probability distribution representations or available to us just as a sampling model). The former (algorithms without access to a model) are known as *Learning Algorithms* to reflect the fact that the agent will need to interact with the real-world environment (eg: a robot learning to navigate in an actual forest) and learn the Value Function from typically large streams of data (states visited, actions taken, reward observed) it receives through environment interactions. The latter (algorithms with access to a model) are known as *Planning Algorithms* to reflect the fact that the agent requires no real-world environment interaction and in fact, projects (with the help of the model) probabilistic scenarios of future states/rewards for various choices of actions, and solves for the requisite Value Function with appropriate probabilistic reasoning of the projected outcomes. In both Learning and Planning, the Bellman Equation will be the fundamental concept driving the algorithms but the details of the algorithms will typically make them appear fairly different. We will only focus on Planning algorithms in this chapter, and in fact, will only focus on a subclass of Planning algorithms known as Dynamic Programming. 


## Usage of the term *Dynamic Programming*

Unfortunately, the term Dynamic Programming tends to be used by different fields in somewhat different ways. So it pays to clarify the history and the current usage of the term. The term *Dynamic Programming* was coined by Richard Bellman himself. Here is the rather interesting story told by Bellman about how and why he coined the term.

> "I spent the Fall quarter (of 1950) at RAND. My first task was to find a name for multistage decision processes. An interesting question is, ‘Where did the name, dynamic programming, come from?’ The 1950s were not good years for mathematical research. We had a very interesting gentleman in Washington named Wilson. He was Secretary of Defense, and he actually had a pathological fear and hatred of the word, research. I’m not using the term lightly; I’m using it precisely. His face would suffuse, he would turn red, and he would get violent if people used the term, research, in his presence. You can imagine how he felt, then, about the term, mathematical. The RAND Corporation was employed by the Air Force, and the Air Force had Wilson as its boss, essentially. Hence, I felt I had to do something to shield Wilson and the Air Force from the fact that I was really doing mathematics inside the RAND Corporation. What title, what name, could I choose? In the first place I was interested in planning, in decision making, in thinking. But planning, is not a good word for various reasons. I decided therefore to use the word, ‘programming.’ I wanted to get across the idea that this was dynamic, this was multistage, this was time-varying—I thought, let’s kill two birds with one stone. Let’s take a word that has an absolutely precise meaning, namely dynamic, in the classical physical sense. It also has a very interesting property as an adjective, and that is it’s impossible to use the word, dynamic, in a pejorative sense. Try thinking of some combination that will possibly give it a pejorative meaning. It’s impossible. Thus, I thought dynamic programming was a good name. It was something not even a Congressman could object to. So I used it as an umbrella for my activities."

Bellman had coined the term Dynamic Programming to refer to the general theory of MDPs, together with the techniques to solve MDPs (i.e., to solve the Control problem). So the MDP Bellman Equation was part of this catch-all term *Dynamic Programming*. The core semantic of the term Dynamic Programming was that the Optimal Value Function can be expressed recursively - meaning, to act optimally from a given state, we will need to act optimally from each of the resulting next states (which is the essence of the Bellman Equation). In fact, Bellman used the term "Principle of Optimality" to refer to the idea of "Optimal Substructure", and articulated it as follows:

> PRINCIPLE OF OPTIMALITY. An optimal policy has the property that whatever the initial state and initial decisions are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decisions. 

So, you can see that the term Dynamic Programming was not just an algorithm in its original usage. Crucially, Bellman laid out an iterative algorithm to solve for the Optimal Value Function (i.e., to solve the MDP Control problem). Over the course of the next decade, the term Dynamic Programming got associated with (multiple) algorithms to solve the MDP Control problem. The term Dynamic Programming also got extended to refer to algorithms to solve the MDP Prediction problem. A few decades later, Computer Scientists started refering to the term Dynamic Programming as any algorithm that solves a problem through a recursive formulation as long as the algorithm makes repeated invocations of the solutions to each subproblem (overlapping subproblem structure). A classic such example is the algorithm to compute the Fibonacci sequence by caching the Fibonacci values and re-using those values. The algorithm to calculate the shortest path in a graph is another classic example where each shortest (i.e. optimal) path includes sub-paths that are optimal. However, in this book, we won't use the term Dynamic Programming in this general sense. We will use the term Dynamic Programming to be restricted to algorithms to solve the MDP Prediction and Control problems (even though Bellman originally used it only in the context of Control). Moreover, we will use the term Dynamic Programming in the narrow context of Planning algorithms for problems with the following two specializations:

* The state space is finite, the action space is finite, and the set of pairs of next state and reward (given any pair of current state and action) are also finite.
* We have explicit knowledge of the $\mathcal{P}_R$ model probabilities

This is the setting of the class `FiniteMarkovDecisionProcess` we had covered in the previous chapter. In this setting, Dynamic Programming algorithms solve the Prediction and Control problems *exactly* (meaning the computed Value Function converges to the true Value Function as the algorithm iterations keep increasing). There are variants of Dynamic Programming algorithms known as Asynchronous Dynamic Programming algorithms, Approximate Dynamic Programming algorithms etc. But without such qualifications, when we use just the term Dynamic Programming, we will be refering to the "classical" iterative algorithms (that we will soon describe) for the above-mentioned setting of the `FiniteMarkovDecisionProcess` class to solve MDP Prediction and Control *exactly*. Even though these classical Dynamical Programming algorithms don't scale to large state/action spaces, they are extremely vital to develop one's core understanding of the key concepts in the more advanced algorithms that will enable us to scale (i.e., the Reinforcement Learning algorithms that we shall introduce in later chapters).

## Solving the Value Function as a *Fixed-Point*

We will cover 3 Dynamic Programming algorithms (in this chapter). Each of the 3 algorithms is founded on the Bellman Equations we had covered in the previous chapter. Each of the 3 algorithms is an iterative algorithm where the computed Value Function converges to the true Value Function as the number of iterations approaches infinity. Each of the 3 algorithms is based on the concept of *Fixed-Point* and updating the computed Value Function towards the Fixed-Point (which in this case, is the true Value Function). Fixed-Point is actually a fairly generic and important concept in the broader fields of Pure as well as Applied Mathematics (it's also important in Theoretical Computer Science), and we believe understanding Fixed-Point theory has many benefits beyond the needs of the subject of this book. Of more relevance is the fact that the Fixed-Point view of Dynamic Programming is the best way to understand Dynamic Programming. We shall not only cover the theory of Dynamic Programming through the Fixed-Point perspective, but we shall also implement Dynamic Programming algorithms in our code based on the Fixed-Point concept. So this section will be a short primer on general Fixed-Point Theory (and implementation in code) before we get to the 3 Dynamic Programming algorithms.

\begin{definition}
The Fixed-Point of a function $f: \mathcal{D} \rightarrow \mathcal{D}$ (for some arbitrary set $\mathcal{D}$) is a value $x \in \mathcal{D}$ that satisfies the equation: $x = f(x)$.
\end{definition}

Note that for some functions, there will be multiple fixed-points and for some other functions, a fixed-point won't exist. We will be considering functions which have a unique fixed-point (this will be the case for the Dynamic Programming algorithms).

Let's warm up to the above-defined abstract concept of Fixed-Point with a concrete example. Consider the function $f(x) = \cos(x)$ defined for $x \in \mathbb{R}$ ($x$ in radians, to be clear). So we want to solve for an $x$ such that $x = \cos(x)$. Knowing the frequency and amplitude of cosine, we can see that the cosine curve intersects the line $y=x$ at only one point, which should be somewhere between $0$ and $\frac \pi 2$. But there is no easy way to solve this. Here's an idea: Start with any value $x_0 \in \mathbb{R}$, calculate $x_1 = \cos(x_0)$, then calculate $x_2 = \cos(x_1)$, and so on …, i.e, $x_{i+1}  = \cos(x_i)$ for $i = 0, 1, 2, \ldots$. You will find that $x_i$ and $x_{i+1}$ get closer and closer as $i$ increases, i.e., $|x_{i+1} - x_i| \leq |x_i - x_{i-1}|$ for all $i \geq 1$. So it seems like $\lim_{i\rightarrow \infty} x_i = \lim_{i\rightarrow \infty} \cos(x_{i-1}) = \lim_{i\rightarrow \infty} \cos(x_i)$ which would imply that for large enough $i$, $x_i$ would serve as an approximation to the solution of the equation $x = \cos(x)$. But why does this method of repeated applications of the function $f$ (no matter what $x_0$ we start with) work? Why does it not diverge or oscillate? How quickly does it converge? If there were multiple fixed-points, which fixed-point would it converge to (if at all)? Can we characterize a class of functions $f$ for which this method (repeatedly applying $f$, starting with any arbitrary value of $x_0$) would work (in terms of solving the equation $x = f(x)$)? These are the questions Fixed-Point theory attempts to answer. Can you think of problems you have solved in the past which fall into this method pattern that we've illustrated above for $f(x) = \cos(x)$? It's likely you have because most of the root-finding and optimization methods (including multi-variate solvers) are essentially based on the idea of Fixed-Point. If this doesn't sound convincing, consider the simple Newton method:

For a differential function $g: \mathbb{R} \rightarrow \mathbb{R}$ whose root we want to solve for, the Newton method update rule is:

$$x_{i+1} = x_i - \frac {g(x_i)} {g'(x_i)}$$

Setting $f(x) = x - \frac {g(x)} {g'(x)}$, the update rule is: $$x_{i+1} = f(x_i)$$ and it solves the equation $x = f(x)$ (solves for the fixed-point of $f$), i.e., it solves the equation:

$$x = x - \frac {g(x)} {g'(x)} \Rightarrow g(x) = 0$$ 

Thus, we see that the same pattern as we saw above for $\cos(x)$ (repeated application of a function, starting from any initial value) enables us to solve for the root of $g$.

More broadly, what we are saying is that if we have a function $f: \mathcal{D} \rightarrow \mathcal{D}$ (for some arbitrary set $\mathcal{D}$), under appropriate conditions (that we will state soon), $f(f(\ldots f(x_0)\ldots ))$ converges to a fixed-point of $f$, i.e., to the solution of the equation $x = f(x)$ (no matter what $x_0 \in \mathcal{D}$ we start with). Now we are ready to state this formally. Here's the key Theorem (without proof but with plenty of explanations following the statement of the Theorem).

\begin{theorem}[Banach Fixed-Point Theorem]
Let $\mathcal{D}$ be a non-empty set equipped with a complete metric $d: \mathcal{D} \times \mathcal{D} \rightarrow \mathbb{R}$. Let $f: \mathcal{D} \rightarrow \mathcal{D}$ be such that there exists a $L \in [0, 1)$ such that
$d(f(x_1), f(x_2)) \leq L \cdot d(x_1, x_2)$ for all $x_1, x_2 \in \mathcal{D}$ (this property of $f$ is called a contraction, and we refer to $f$ as a contraction function). Then,
\begin{enumerate}
\item There exists a unique Fixed-Point $x^* \in \mathcal{D}$, i.e.,
$$x^* = f(x^*)$$
\item For any $x_0 \in \mathcal{D}$ and sequence $\{x_i|i=0, 1, 2, \ldots\}$ defined as $x_{i+1} = f(x_i)$ for all $i = 0, 1, 2, \ldots$,
$$\lim_{i\rightarrow \infty} x_i = x^*$$
\item $$d(x^*, x_i) \leq \frac {L^i} {1-L} \cdot d(x_1, x_0)$$
Equivalently,
$$d(x^*, x_{i+1}) \leq \frac {L} {1-L} \cdot d(x_{i+1}, x_i)$$
$$d(x^*, x_{i+1}) \leq L \cdot d(x^*, x_i)$$
\end{enumerate}
\label{th:banach_fixed_point_theorem}
\end{theorem}

Sorry - that was a pretty heavy theorem! We are not going to prove the theorem, but we will be using the theorem. So let's try to understand the theorem in a simple, intuitive manner. First we need to explain the jargon *complete metric*. Let's start with the term *metric*. A metric is simply a function that satisfies the usual "distance" properties (for any $x_1, x_2, x_3 \in \mathcal{D}$):

1. $d(x_1, x_2) = 0 \Leftrightarrow x_1 = x_2$ (meaning two different points will have a distance strictly greater than 0)
2. $d(x_1, x_2) = d(x_2, x_1)$ (meaning distance is directionless)
3. $d(x_1, x_3) \leq d(x_1, x_2) + d(x_2, x_3)$ (meaning the triangle inequality is satisfied)

Any function $d$ that satisfies the above 3 properties qualifies as a metric. The term *complete* is a bit of a technical detail on sequences not escaping the set $\mathcal{D}$ (that's required in the proof). Since we won't be doing the proof and this technical detail is not so important for the intuition, we shall skip the formal definition of *complete*. The set $\mathcal{D}$ together with the function $d$ is called a complete metric space. The key is the contraction function $f$ with the property:
$d(f(x_1), f(x_2)) \leq L \cdot d(x_1, x_2)$ for all $x_1, x_2 \in \mathcal{D}$ 
 
 The theorem basically says that for any contraction function $f$ (with existence of the contraction quantity $L \in [0, 1)$), there is not only a unique fixed-point $x^*$, we also have an algorithm to arrive at $x^*$ by repeated application of $f$ starting with any initial value $x_0 \in \mathcal{D}$:

 $$f(f(\ldots f(x_0) \ldots )) \rightarrow x^*$$

Moreover, the theorem gives us a statement on the speed of convergence relating the distance between $x^*$ and any $x_i$ to the distance between any two successive $x_i$.

This is a powerful theorem. All we need to do is identify the appropriate set $\mathcal{D}$ to work with, identify the appropriate metric $d$ to work with, and ensure that $f$ is indeed a contraction function to solve for the fixed-point of $f$ with this iterative process of applying $f$ repeatedly, starting from any arbitrary starting value of $x_0 \in \mathcal{D}$.

We leave it to you as an exercise to verify that $f(x) = \cos(x)$ is a contraction function in the domain $\mathcal{D} = \mathbb{R}$ with metric $d$ defined as $d(x_1, x_2) = |x_1 - x_2|$. Now we are ready to introduce the world of Dynamic Programming.


## Policy Evaluation Algorithm

Our first Dynamic Programming algorithm is called *Policy Evaluation*. The Policy Evaluation algorithm solves the problem of calculating the Value Function of a Finite MDP evaluated with a fixed policy $\pi$ (i.e., the Prediction problem for finite MDPs). We know that this is equivalent to calculating the Value Function of the $\pi$-implied Finite MRP. To avoid notation confusion, a superscript of $\pi$ for a symbol means it refers to notation for the $\pi$-implied MRP (versus any notation for the MDP). The precise specification of the Prediction problem is as follows:

Let the $\pi$-implied MRP's states be $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ (with $m$ of those states as the non-terminal states $\mathcal{N}$). We are given the $\pi$-implied MRP's transition probability function:

$$\mathcal{P}_R^{\pi}: \mathcal{N} \times \mathcal{S} \times \mathbb{R} \rightarrow [0, 1]$$
in the form of a data structure (since the states are finite and the pairs of next state and reward transitions from each non-terminal state are also finite).

We know from the previous 2 chapters that by extracting (from $\mathcal{P}_R^{\pi}$) the transition probability function $\mathcal{P}^{\pi}: \mathcal{N} \times \mathcal{S} \rightarrow [0, 1]$ of the implicit Markov Process and the reward function $\mathcal{R}^{\pi}: \mathcal{N} \rightarrow \mathbb{R}$, we can perform the following calculation for the Value Function $V^{\pi}: \mathcal{N} \rightarrow \mathbb{R}$ (expressed as a column vector $\bvpi \in \mathbb{R}^m$) to solve this Prediction problem:

$$\bvpi = (\bm{I_m} - \gamma \bm{\mathcal{P}^{\pi}})^{-1} \cdot \bm{\mathcal{R}^{\pi}}$$

where $\bm{I_m}$ is the $m \times m$ identity matrix, $\bm{\mathcal{R}^{\pi}} \in \mathbb{R}^m$ represents $\mathcal{R}^{\pi}$, and $\bm{\mathcal{P}^{\pi}}$ is an $m \times m$ matrix representing $\mathcal{P}^{\pi}$ (rows and columns corresponding to the non-terminal states). However, when $m$ is large, this calculation won't scale. So, we look for a numerical algorithm that would solve (for $\bvpi$) the following MRP Bellman Equation (for a larger number of finite states).

$$\bvpi = \bm{\mathcal{R}^{\pi}} + \gamma \bm{\mathcal{P}^{\pi}} \cdot \bvpi$$

We define the *Bellman Policy Operator* $\bbpi: \mathbb{R}^m \rightarrow \mathbb{R}^m$ as:

$$\bbpi(\bv) = \bm{\mathcal{R}^{\pi}} + \gamma \bm{\mathcal{P}^{\pi}} \cdot \bv \text{ for any vector } \bv \text{ in the vector space } \mathbb{R}^m$$

So, the MRP Bellman Equation can be expressed as:

$$\bvpi = \bbpi(\bvpi)$$

which means $\bvpi \in \mathbb{R}^m$ is the Fixed-Point of the *Bellman Policy Operator* $\bbpi: \mathbb{R}^m \rightarrow \mathbb{R}^m$.

Note that $\bbpi$ is a linear transformation on vectors in $\mathbb{R}^n$ and should be thought of as a generalization of a simple 1-D ($\mathbb{R} \rightarrow \mathbb{R}$) linear transformation $y = a + bx$ where the multiplier $b$ is replaced with the matrix $\gamma \bm{\mathcal{P}^{\pi}}$ and the shift $a$ is replaced with the column vector $\bm{\mathcal{R}^{\pi}}$.


We'd like to come up with a metric for which $\bbpi$ is a contraction function so that we can take advantage of Banach Fixed-Point Theorem and solve this Prediction problem by iterative applications of the Bellman Policy Operator $\bbpi$.  We shall denote the $i^{th}$-dimension of any vector $\bv \in \mathbb{R}^m$ (for all $1 \leq i \leq m$) as $\bv(s_i)$ to represent the Value for state $s_i$.

Our metric $d: \mathbb{R}^m \times \mathbb{R}^m \rightarrow \mathbb{R}$ shall be the $L^{\infty}$ norm defined as:

$$d(\bm{V_1}, \bm{V_2}) = \Vert \bm{V_1} - \bm{V_2} \Vert_{\infty} = \max_{1\leq i \leq m} |(\bm{V_1} - \bm{V_2})(s_i)|$$

$\bbpi$ is a contraction function under $L^{\infty}$ norm because for all $\bm{V_1}, \bm{V_2} \in \mathbb{R}^m$,

$$\max_{1\leq i \leq m} |(\bbpi(\bm{V_1}) - \bbpi(\bm{V_2}))(s_i)| = \gamma \cdot \max_{1\leq i \leq m} |(\bm{\mathcal{P}^{\pi}} \cdot (\bm{V_1} - \bm{V_2}))(s_i)| \leq \max_{1\leq i \leq m} |(\bm{V_1} - \bm{V_2})(s_i)|$$

From Banach Fixed-Point Theorem, this means that $\bbpi$ has a unique Fixed-Point $\bvpi$, and

$$\bbpi(\bbpi(\ldots \bbpi(\bm{V_0}) \ldots )) \rightarrow \bvpi \text{ for all starting Value Functions } \bm{V_0} \in \mathbb{R}^m$$

Expressed as an iterative algorithm (known as the *Policy Evaluation* algorithm),

$$\bm{V_{i+1}} = \bbpi(\bm{V_i}) = \bm{\mathcal{R}^{\pi}} + \gamma \bm{\mathcal{P}^{\pi}} \cdot \bm{V_i} \text{ for all } i = 0, 1, 2, \ldots$$
$$\text{ and  } \lim_{i\rightarrow \infty} \bm{V_i} = \bvpi$$

It pays to emphasize that Banach Fixed-Point Theorem not only assures convergence to the unique solution $V^{\pi}: \mathcal{N} \rightarrow \mathbb{R}$ (no matter what Value Function we start the algorithm with), it also assures a reasonable speed of convergence (dependent on the choice of starting Value Function and the choice of $\gamma$).

