# Dynamic Programming Algorithms

As a reminder, much of this book is about algorithms to solve the MDP Control problem, i.e., to compute the Optimal Value Function (and an associated Optimal Policy). We will also cover algorithms for the MDP Prediction problem, i.e., to compute the Value Function when the agent executes a fixed policy $\pi$ (which, as we know from the previous chapter, is the same as the $\pi$-implied MRP problem). Our typical approach will be to first cover algorithms to solve the Prediction problem before covering algorithms to solve the Control problem - not just because Prediction is a key component in solving the Control problem, but also because it helps understand the key aspects of the techniques employed in the Control algorithm in the simpler setting of Prediction.

## Planning versus Learning

In this book, we shall look at Planning and Control from the lens of AI (and we'll specifically use the terminology of AI). We shall distinguish between algorithms that don't have a model of the MDP environment (no access to the $\mathcal{P}_R$ function) versus algorithms that do have a model of the MDP environment (meaning $\mathcal{P}_R$ is available to us either in terms of explicit probability distribution representations or available to us just as a sampling model). The former (algorithms without access to a model) are known as *Learning Algorithms* to reflect the fact that the agent will need to interact with the real-world environment (eg: a robot learning to navigate in an actual forest) and learn the Value Function from streams of data (states encountered,  actions taken, rewards observed) it receives through environment interactions. The latter (algorithms with access to a model) are known as *Planning Algorithms* to reflect the fact that the agent requires no real-world environment interaction and in fact, projects (with the help of the model) probabilistic scenarios of future states/rewards for various choices of actions, and solves for the requisite Value Function with appropriate probabilistic reasoning of the projected outcomes. In both Learning and Planning, the Bellman Equation will be the fundamental concept driving the algorithms but the details of the algorithms will typically make them appear fairly different. We will only focus on Planning algorithms in this chapter, and in fact, will only focus on a subclass of Planning algorithms known as Dynamic Programming. 


## Usage of the term *Dynamic Programming*

Unfortunately, the term Dynamic Programming tends to be used by different fields in somewhat different ways. So it pays to clarify the history and the current usage of the term. The term *Dynamic Programming* was coined by Richard Bellman himself. Here is the rather interesting story told by Bellman about how and why he coined the term.

> "I spent the Fall quarter (of 1950) at RAND. My first task was to find a name for multistage decision processes. An interesting question is, ‘Where did the name, dynamic programming, come from?’ The 1950s were not good years for mathematical research. We had a very interesting gentleman in Washington named Wilson. He was Secretary of Defense, and he actually had a pathological fear and hatred of the word, research. I’m not using the term lightly; I’m using it precisely. His face would suffuse, he would turn red, and he would get violent if people used the term, research, in his presence. You can imagine how he felt, then, about the term, mathematical. The RAND Corporation was employed by the Air Force, and the Air Force had Wilson as its boss, essentially. Hence, I felt I had to do something to shield Wilson and the Air Force from the fact that I was really doing mathematics inside the RAND Corporation. What title, what name, could I choose? In the first place I was interested in planning, in decision making, in thinking. But planning, is not a good word for various reasons. I decided therefore to use the word, ‘programming.’ I wanted to get across the idea that this was dynamic, this was multistage, this was time-varying—I thought, let’s kill two birds with one stone. Let’s take a word that has an absolutely precise meaning, namely dynamic, in the classical physical sense. It also has a very interesting property as an adjective, and that is it’s impossible to use the word, dynamic, in a pejorative sense. Try thinking of some combination that will possibly give it a pejorative meaning. It’s impossible. Thus, I thought dynamic programming was a good name. It was something not even a Congressman could object to. So I used it as an umbrella for my activities."

Bellman had coined the term Dynamic Programming to refer to the general theory of MDPs, together with the techniques to solve MDPs (i.e., to solve the Control problem). So the MDP Bellman Optimality Equation was part of this catch-all term *Dynamic Programming*. The core semantic of the term Dynamic Programming was that the Optimal Value Function can be expressed recursively - meaning, to act optimally from a given state, we will need to act optimally from each of the resulting next states (which is the essence of the Bellman Optimality Equation). In fact, Bellman used the term "Principle of Optimality" to refer to this idea of "Optimal Substructure", and articulated it as follows:

> PRINCIPLE OF OPTIMALITY. An optimal policy has the property that whatever the initial state and initial decisions are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decisions. 

So, you can see that the term Dynamic Programming was not just an algorithm in its original usage. Crucially, Bellman laid out an iterative algorithm to solve for the Optimal Value Function (i.e., to solve the MDP Control problem). Over the course of the next decade, the term Dynamic Programming got associated with (multiple) algorithms to solve the MDP Control problem. The term Dynamic Programming was extended to also refer to algorithms to solve the MDP Prediction problem. Over the next couple of decades, Computer Scientists started refering to the term Dynamic Programming as any algorithm that solves a problem through a recursive formulation as long as the algorithm makes repeated invocations to the solutions to each subproblem (overlapping subproblem structure). A classic such example is the algorithm to compute the Fibonacci sequence by caching the Fibonacci values and re-using those values during the course of the algorithm execution. The algorithm to calculate the shortest path in a graph is another classic example where each shortest (i.e. optimal) path includes sub-paths that are optimal. However, in this book, we won't use the term Dynamic Programming in this broader sense. We will use the term Dynamic Programming to be restricted to algorithms to solve the MDP Prediction and Control problems (even though Bellman originally used it only in the context of Control). More specifically, we will use the term Dynamic Programming in the narrow context of Planning algorithms for problems with the following two specializations:

* The state space is finite, the action space is finite, and the set of pairs of next state and reward (given any pair of current state and action) are also finite.
* We have explicit knowledge of the model probabilities (either in the form of $\mathcal{P}_R$ or in the form of $\mathcal{P}$ and $\mathcal{R}$ separately).

This is the setting of the class `FiniteMarkovDecisionProcess` we had covered in the previous chapter. In this setting, Dynamic Programming algorithms solve the Prediction and Control problems *exactly* (meaning the computed Value Function converges to the true Value Function as the algorithm iterations keep increasing). There are variants of Dynamic Programming algorithms known as Asynchronous Dynamic Programming algorithms, Approximate Dynamic Programming algorithms etc. But without such qualifications, when we use just the term Dynamic Programming, we will be refering to the "classical" iterative algorithms (that we will soon describe) for the above-mentioned setting of the `FiniteMarkovDecisionProcess` class to solve MDP Prediction and Control *exactly*. Even though these classical Dynamical Programming algorithms don't scale to large state/action spaces, they are extremely vital to develop one's core understanding of the key concepts in the more advanced algorithms that will enable us to scale (i.e., the Reinforcement Learning algorithms that we shall introduce in later chapters).

## Solving the Value Function as a *Fixed-Point*

We cover 3 Dynamic Programming algorithms. Each of the 3 algorithms is founded on the Bellman Equations we had covered in the previous chapter. Each of the 3 algorithms is an iterative algorithm where the computed Value Function converges to the true Value Function as the number of iterations approaches infinity. Each of the 3 algorithms is based on the concept of *Fixed-Point* and updating the computed Value Function towards the Fixed-Point (which in this case, is the true Value Function). Fixed-Point is actually a fairly generic and important concept in the broader fields of Pure as well as Applied Mathematics (also important in Theoretical Computer Science), and we believe understanding Fixed-Point theory has many benefits beyond the needs of the subject of this book. Of more relevance is the fact that the Fixed-Point view of Dynamic Programming is the best way to understand Dynamic Programming. We shall not only cover the theory of Dynamic Programming through the Fixed-Point perspective, but we shall also implement Dynamic Programming algorithms in our code based on the Fixed-Point concept. So this section will be a short primer on general Fixed-Point Theory (and implementation in code) before we get to the 3 Dynamic Programming algorithms.

\begin{definition}
The Fixed-Point of a function $f: \mathcal{D} \rightarrow \mathcal{D}$ (for some arbitrary domain $\mathcal{D}$) is a value $x \in \mathcal{D}$ that satisfies the equation: $x = f(x)$.
\end{definition}

Note that for some functions, there will be multiple fixed-points and for some other functions, a fixed-point won't exist. We will be considering functions which have a unique fixed-point (this will be the case for the Dynamic Programming algorithms).

Let's warm up to the above-defined abstract concept of Fixed-Point with a concrete example. Consider the function $f(x) = \cos(x)$ defined for $x \in \mathbb{R}$ ($x$ in radians, to be clear). So we want to solve for an $x$ such that $x = \cos(x)$. Knowing the frequency and amplitude of cosine, we can see that the cosine curve intersects the line $y=x$ at only one point, which should be somewhere between $0$ and $\frac \pi 2$. But there is no easy way to solve for this point. Here's an idea: Start with any value $x_0 \in \mathbb{R}$, calculate $x_1 = \cos(x_0)$, then calculate $x_2 = \cos(x_1)$, and so on …, i.e, $x_{i+1}  = \cos(x_i)$ for $i = 0, 1, 2, \ldots$. You will find that $x_i$ and $x_{i+1}$ get closer and closer as $i$ increases, i.e., $|x_{i+1} - x_i| \leq |x_i - x_{i-1}|$ for all $i \geq 1$. So it seems like $\lim_{i\rightarrow \infty} x_i = \lim_{i\rightarrow \infty} \cos(x_{i-1}) = \lim_{i\rightarrow \infty} \cos(x_i)$ which would imply that for large enough $i$, $x_i$ would serve as an approximation to the solution of the equation $x = \cos(x)$. But why does this method of repeated applications of the function $f$ (no matter what $x_0$ we start with) work? Why does it not diverge or oscillate? How quickly does it converge? If there were multiple fixed-points, which fixed-point would it converge to (if at all)? Can we characterize a class of functions $f$ for which this method (repeatedly applying $f$, starting with any arbitrary value of $x_0$) would work (in terms of solving the equation $x = f(x)$)? These are the questions Fixed-Point theory attempts to answer. Can you think of problems you have solved in the past which fall into this method pattern that we've illustrated above for $f(x) = \cos(x)$? It's likely you have, because most of the root-finding and optimization methods (including multi-variate solvers) are essentially based on the idea of Fixed-Point. If this doesn't sound convincing, consider the simple Newton method:

For a differential function $g: \mathbb{R} \rightarrow \mathbb{R}$ whose root we want to solve for, the Newton method update rule is:

$$x_{i+1} = x_i - \frac {g(x_i)} {g'(x_i)}$$

Setting $f(x) = x - \frac {g(x)} {g'(x)}$, the update rule is: $$x_{i+1} = f(x_i)$$ and it solves the equation $x = f(x)$ (solves for the fixed-point of $f$), i.e., it solves the equation:

$$x = x - \frac {g(x)} {g'(x)} \Rightarrow g(x) = 0$$ 

Thus, we see the same method pattern as we saw above for $\cos(x)$ (repeated application of a function, starting with any initial value) enables us to solve for the root of $g$.

More broadly, what we are saying is that if we have a function $f: \mathcal{D} \rightarrow \mathcal{D}$ (for some arbitrary domain $\mathcal{D}$), under appropriate conditions (that we will state soon), $f(f(\ldots f(x_0)\ldots ))$ converges to a fixed-point of $f$, i.e., to the solution of the equation $x = f(x)$ (no matter what $x_0 \in \mathcal{D}$ we start with). Now we are ready to state this formally. The statement of the following theorem is quite terse, so we will provide plenty of explanation on how to interpret it and how to use it after stating the theorem (we skip the proof of the theorem).

\begin{theorem}[Banach Fixed-Point Theorem]
Let $\mathcal{D}$ be a non-empty set equipped with a complete metric $d: \mathcal{D} \times \mathcal{D} \rightarrow \mathbb{R}$. Let $f: \mathcal{D} \rightarrow \mathcal{D}$ be such that there exists a $L \in [0, 1)$ such that
$d(f(x_1), f(x_2)) \leq L \cdot d(x_1, x_2)$ for all $x_1, x_2 \in \mathcal{D}$ (this property of $f$ is called a contraction, and we refer to $f$ as a contraction function). Then,
\begin{enumerate}
\item There exists a unique Fixed-Point $x^* \in \mathcal{D}$, i.e.,
$$x^* = f(x^*)$$
\item For any $x_0 \in \mathcal{D}$, and sequence $[x_i|i=0, 1, 2, \ldots]$ defined as $x_{i+1} = f(x_i)$ for all $i = 0, 1, 2, \ldots$,
$$\lim_{i\rightarrow \infty} x_i = x^*$$
\item $$d(x^*, x_i) \leq \frac {L^i} {1-L} \cdot d(x_1, x_0)$$
Equivalently,
$$d(x^*, x_{i+1}) \leq \frac {L} {1-L} \cdot d(x_{i+1}, x_i)$$
$$d(x^*, x_{i+1}) \leq L \cdot d(x^*, x_i)$$
\end{enumerate}
\label{th:banach_fixed_point_theorem}
\end{theorem}

Sorry - that was pretty terse! Let's try to understand the theorem in a simple, intuitive manner. First we need to explain the jargon *complete metric*. Let's start with the term *metric*. A metric is simply a function $d: \mathcal{D} \times \mathcal{D} \rightarrow \mathbb{R}$ that satisfies the usual "distance" properties (for any $x_1, x_2, x_3 \in \mathcal{D}$):

1. $d(x_1, x_2) = 0 \Leftrightarrow x_1 = x_2$ (meaning two different points will have a distance strictly greater than 0)
2. $d(x_1, x_2) = d(x_2, x_1)$ (meaning distance is directionless)
3. $d(x_1, x_3) \leq d(x_1, x_2) + d(x_2, x_3)$ (meaning the triangle inequality is satisfied)

The term *complete* is a bit of a technical detail on sequences not escaping the set $\mathcal{D}$ (that's required in the proof). Since we won't be doing the proof and this technical detail is not so important for the intuition, we shall skip the formal definition of *complete*. A non-empty set $\mathcal{D}$ equipped with the function $d$ (and the technical detail of being *complete*) is known as a complete metric space.

Now we move on to the key concept of *contraction*. A function $f: \mathcal{D} \rightarrow \mathcal{D}$ is said to be a contraction function if two points in $\mathcal{D}$ get closer when they are mapped by $f$ (the statement: $d(f(x_1), f(x_2)) \leq L \cdot d(x_1, x_2)$ for all $x_1, x_2 \in \mathcal{D}$, for some $L \in [0, 1)$).
 
 The theorem basically says that for any contraction function $f$, there is not only a unique fixed-point $x^*$, one can arrive at $x^*$ by repeated application of $f$, starting with any initial value $x_0 \in \mathcal{D}$:

 $$f(f(\ldots f(x_0) \ldots )) \rightarrow x^*$$

 We shall use the notation $f^i: \mathcal{D} \rightarrow \mathcal{D}$ for $i = 0, 1, 2, \ldots$ as follows:

 $$f^{i+1}(x) = f(f^i(x)) \text{ for all } i = 0, 1, 2, \ldots, \text{ for all } x \in \mathcal{D}$$
 $$f^0(x) = x \text{ for all } x \in \mathcal{D}$$

 With this notation, the computation of the fixed-point can be expressed as:

 $$\lim_{i \rightarrow \infty} f^i(x_0) = x^* \text{ for all } x_0 \in \mathcal{D}$$

 The algorithm, in iterative form, is:

 $$x_{i+1} = f(x_i) \text{ for all } i = 0, 2, \ldots$$

We stop the algorithm when $x_i$ and $x_{i+1}$ are close enough based on the distance-metric $d$.

Banach Fixed-Point Theorem also gives us a statement on the speed of convergence relating the distance between $x^*$ and any $x_i$ to the distance between any two successive $x_i$.

This is a powerful theorem. All we need to do is identify the appropriate set $\mathcal{D}$ to work with, identify the appropriate metric $d$ to work with, and ensure that $f$ is indeed a contraction function (with respect to $d$). This enables us to solve for the fixed-point of $f$ with the above-described iterative process of applying $f$ repeatedly, starting with any arbitrary value of $x_0 \in \mathcal{D}$.

We leave it to you as an exercise to verify that $f(x) = \cos(x)$ is a contraction function in the domain $\mathcal{D} = \mathbb{R}$ with metric $d$ defined as $d(x_1, x_2) = |x_1 - x_2|$. Now we are ready to introduce the world of Dynamic Programming.


## Bellman Policy Operator and Policy Evaluation Algorithm

Our first Dynamic Programming algorithm is called *Policy Evaluation*. The Policy Evaluation algorithm solves the problem of calculating the Value Function of a Finite MDP evaluated with a fixed policy $\pi$ (i.e., the Prediction problem for finite MDPs). We know that this is equivalent to calculating the Value Function of the $\pi$-implied Finite MRP. To avoid notation confusion, note that a superscript of $\pi$ for a symbol means it refers to notation for the $\pi$-implied MRP. The precise specification of the Prediction problem is as follows:

Let the states of the MDP (and hence, of the $\pi$-implied MRP) be $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$, and without loss of generality, let $\mathcal{N} = \{s_1, s_2, \ldots, s_m \}$ be the non-terminal states. We are given a fixed policy $\pi: \mathcal{N} \times \mathcal{A} \rightarrow [0, 1]$. We are also given the $\pi$-implied MRP's transition probability function:

$$\mathcal{P}_R^{\pi}: \mathcal{N} \times \mathbb{R} \times \mathcal{S} \rightarrow [0, 1]$$
in the form of a data structure (since the states are finite, and the pairs of next state and reward transitions from each non-terminal state are also finite).

We know from the previous 2 chapters that by extracting (from $\mathcal{P}_R^{\pi}$) the transition probability function $\mathcal{P}^{\pi}: \mathcal{N} \times \mathcal{S} \rightarrow [0, 1]$ of the implicit Markov Process and the reward function $\mathcal{R}^{\pi}: \mathcal{N} \rightarrow \mathbb{R}$, we can perform the following calculation for the Value Function $V^{\pi}: \mathcal{N} \rightarrow \mathbb{R}$ (expressed as a column vector $\bvpi \in \mathbb{R}^m$) to solve this Prediction problem:

$$\bvpi = (\bm{I_m} - \gamma \bm{\mathcal{P}}^{\pi})^{-1} \cdot \bm{\mathcal{R}}^{\pi}$$

where $\bm{I_m}$ is the $m \times m$ identity matrix, column vector $\bm{\mathcal{R}}^{\pi} \in \mathbb{R}^m$ represents $\mathcal{R}^{\pi}$, and $\bm{\mathcal{P}}^{\pi}$ is an $m \times m$ matrix representing $\mathcal{P}^{\pi}$ (rows and columns corresponding to the non-terminal states). However, when $m$ is large, this calculation won't scale. So, we look for a numerical algorithm that would solve (for $\bvpi$) the following MRP Bellman Equation (for a larger number of finite states).

$$\bvpi = \bm{\mathcal{R}}^{\pi} + \gamma \bm{\mathcal{P}}^{\pi} \cdot \bvpi$$

We define the *Bellman Policy Operator* $\bbpi: \mathbb{R}^m \rightarrow \mathbb{R}^m$ as:

\begin{equation}
\bbpi(\bv) = \bm{\mathcal{R}}^{\pi} + \gamma \bm{\mathcal{P}}^{\pi} \cdot \bv \text{ for any vector } \bv \text{ in the vector space } \mathbb{R}^m
\label{eq:bellman_policy_operator}
\end{equation}

So, the MRP Bellman Equation can be expressed as:

$$\bvpi = \bbpi(\bvpi)$$

which means $\bvpi \in \mathbb{R}^m$ is the Fixed-Point of the *Bellman Policy Operator* $\bbpi: \mathbb{R}^m \rightarrow \mathbb{R}^m$.

Note that $\bbpi$ is a linear transformation on vectors in $\mathbb{R}^m$ and should be thought of as a generalization of a simple 1-D ($\mathbb{R} \rightarrow \mathbb{R}$) linear transformation $y = a + bx$ where the multiplier $b$ is replaced with the matrix $\gamma \bm{\mathcal{P}}^{\pi}$ and the shift $a$ is replaced with the column vector $\bm{\mathcal{R}}^{\pi}$.

We'd like to come up with a metric for which $\bbpi$ is a contraction function so we can take advantage of Banach Fixed-Point Theorem and solve this Prediction problem by iterative applications of the Bellman Policy Operator $\bbpi$. For any Value Function $\bv \in \mathbb{R}^m$ (representing $V: \mathcal{N} \rightarrow \mathbb{R}$), we shall express the Value for any state $s\in \mathcal{N}$ as $\bv(s)$.

Our metric $d: \mathbb{R}^m \times \mathbb{R}^m \rightarrow \mathbb{R}$ shall be the $L^{\infty}$ norm defined as:

$$d(\bm{X}, \bm{Y}) = \Vert \bm{X} - \bm{Y} \Vert_{\infty} = \max_{s \in \mathcal{N}} |(\bm{X} - \bm{Y})(s)|$$

$\bbpi$ is a contraction function under $L^{\infty}$ norm because for all $\bm{X}, \bm{Y} \in \mathbb{R}^m$,

$$\max_{s \in \mathcal{N}} |(\bbpi(\bm{X}) - \bbpi(\bm{Y}))(s)| = \gamma \cdot \max_{s \in \mathcal{N}} |(\bm{\mathcal{P}}^{\pi} \cdot (\bm{X} - \bm{Y}))(s)| \leq \gamma \cdot \max_{s \in \mathcal{N}} |(\bm{X} - \bm{Y})(s)|$$

So invoking Banach Fixed-Point Theorem proves the following Theorem:

\begin{theorem}[Policy Evaluation Convergence Theorem]
For a Finite MDP with $|\mathcal{N}| = m$, if $\bvpi \in \mathbb{R}^m$ is the Value Function of the MDP when evaluated with a fixed policy $\pi: \mathcal{N} \times \mathcal{A} \rightarrow [0, 1]$, then $\bvpi$ is the unique Fixed-Point of the Bellman Policy Operator $\bbpi: \mathbb{R}^m \rightarrow \mathbb{R}^m$, and
$$\lim_{i\rightarrow \infty} ({\bbpi})^i(\bm{V_0}) \rightarrow \bvpi \text{ for all starting Value Functions } \bm{V_0} \in \mathbb{R}^m$$
\label{eq:policy_evaluation_convergence_theorem}
\end{theorem}

This gives us the following iterative algorithm (known as the *Policy Evaluation* algorithm for fixed policy $\pi: \mathcal{N} \times \mathcal{A} \rightarrow [0, 1]$):

* Start with any Value Function $\bm{V_0} \in \mathbb{R}^m$
* Iterating over $i = 0, 1, 2, \ldots$, calculate in each iteration:
$$\bm{V_{i+1}} = \bbpi(\bm{V_i}) = \bm{\mathcal{R}}^{\pi} + \gamma \bm{\mathcal{P}}^{\pi} \cdot \bm{V_i}$$

We stop the algorithm when $d(\bm{V_i}, \bm{V_{i+1}}) = \max_{s \in \mathcal{N}} |(\bm{V_i} - \bm{V_{i+1}})(s)|$ is adequately small.

It pays to emphasize that Banach Fixed-Point Theorem not only assures convergence to the unique solution $\bvpi$ (no matter what Value Function $\bm{V_0}$ we start the algorithm with), it also assures a reasonable speed of convergence (dependent on the choice of starting Value Function $\bm{V_0}$ and the choice of $\gamma$).

## Greedy Policy

We had said earlier that we will be presenting 3 Dynamic Programming Algorithms. The first (Policy Evaluation), as we saw in the previous section, solves the MDP Prediction problem. The other two (that will present in the next two sections) solve the MDP Control problem. This section is a stepping stone from *Prediction* to *Control*. In this section, we define a function that is motivated by the idea of *improving a value function/improving a policy* with a "greedy" technique. Formally, the *Greedy Policy Function*

$$G: \mathbb{R}^m \rightarrow (\mathcal{N} \rightarrow \mathcal{A})$$

interpreted as a function mapping a Value Function $\bv$ (represented as a vector) to a deterministic policy $\pi_D': \mathcal{N} \rightarrow \mathcal{A}$, is defined as:

\begin{equation}
G(\bv)(s) = \pi_D'(s) = \argmax_{a\in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bv(s')\} \text{ for all } s\in \mathcal{N}
\label{eq:greedy_policy_function}
\end{equation}

The word "Greedy" is a reference to the term "Greedy Algorithm", which means an algorithm that takes heuristic steps guided by locally-optimal choices in the hope of moving towards a global optimum. Here, the reference to *Greedy Policy* means if we have a policy $\pi$ and its corresponding Value Function $\bvpi$ (obtained say using Policy Evaluation algorithm), then applying the Greedy Policy function $G$ on $\bvpi$ gives us a deterministic policy $\pi_D': \mathcal{N} \rightarrow \mathcal{A}$ that is hopefully "better" than $\pi$ in the sense that $\bm{V}^{\pi_D'}$ is "greater" than $\bvpi$. We shall now make this statement precise and show how to use the *Greedy Policy Function* to perform *Policy Improvement*.

## Policy Improvement

Terms such a "better" or "improvement" refer to either Value Functions or to Policies (in the latter case, to Value Functions of an MDP evaluated with the policies). So what does it mean to say a Value Function $\bm{X}$ is "better" than a Value Function $\bm{Y}$? Here's the answer:

\begin{definition}[Value Function Comparison]
We say $\bm{X} \geq \bm{Y}$ for Value Functions $\bm{X}, \bm{Y} \in \mathbb{R}^m$ if and only if:
$$\bm{X}(s) \geq \bm{Y}(s) \text{ for all } s \in \mathcal{N}$$
\end{definition}

So whenever you hear terms like "Better Value Function" or "Improved Value Function", you should interpret it to mean that the Value Function is *no worse for each of the states* (versus the Value Function it's being compared to).

So then, what about the claim of $\pi_D' = G(\bvpi)$ being "better" than $\pi$? The following theorem provides the clarification:

\begin{theorem}[Policy Improvement Theorem]
For a finite MDP, for any policy $\pi$,
$$\bm{V}^{\pi_D'} = \bm{V}^{G(\bvpi)} \geq \bvpi$$
\label{th:policy_improvement_theorem}
\end{theorem}

\begin{proof}
We start by noting that applying the Bellman Policy Operator $\bm{B}^{\pi_D'}$ repeatedly, starting with the Value Function $\bvpi$, will converge to the Value Function $\bm{V}^{\pi_D'}$. Formally,

$$\lim_{i\rightarrow \infty} (\bm{B}^{\pi_D'})^i(\bvpi) = \bm{V}^{\pi_D'}$$

So the proof is complete if we prove that:

$$(\bm{B}^{\pi_D'})^{i+1}(\bvpi) \geq (\bm{B}^{\pi_D'})^i(\bvpi) \text{ for all } i = 0, 1, 2, \ldots$$

which means we get an increasing tower of Value Functions $[(\bm{B}^{\pi_D'})^i(\bvpi)|i = 0, 1, 2, \ldots]$ with repeated applications of $\bm{B}^{\pi_D'}$ starting with the Value Function $\bvpi$. 

Let us prove this by induction. The base case (for $i=0$) of the induction is to prove that:

$$\bm{B}^{\pi_D'}(\bvpi) \geq \bvpi$$

Note that:

$$\bm{B}^{\pi_D'}(\bvpi)(s) = \mathcal{R}(s,\pi_D'(s)) + \gamma \sum_{s'\in \mathcal{S}} \mathcal{P}(s,\pi_D'(s),s') \cdot \bvpi(s') \text{ for all } s \in \mathcal{N}$$

From Equation \eqref{eq:greedy_policy_function}, we know that for each $s \in \mathcal{N}$, $\pi_D'(s) = G(\bvpi)(s)$ is the action that maximizes $\{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bvpi(s')\}$. Therefore,
$$\bm{B}^{\pi_D'}(\bvpi)(s) = \max_{a \in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bvpi(s')\} = \max_{a \in \mathcal{A}} Q^{\pi}(s,a) \text{ for all } s \in \mathcal{N}$$
Let's compare this equation against the Bellman Policy Equation for $\pi$ (below):
$$\bvpi(s) = \sum_{a \in \mathcal{A}} \pi(s, a) \cdot Q^{\pi}(s, a) \text{ for all } s \in \mathcal{N}$$
We see that $\bvpi(s)$ is a weighted average of $Q^{\pi}(s,a)$ (with weights equal to probabilities $\pi(s,a)$ over choices of $a$) while $\bm{B}^{\pi_D'}(\bvpi)(s)$ is the maximum (over choices of $a$) of $Q^{\pi}(s,a)$. Therefore,
$$\bm{B}^{\pi_D'}(\bvpi) \geq \bvpi$$

This establishes the base case of the proof by induction. Now to complete the proof, all we have to do is to prove:

$$\text{If } (\bm{B}^{\pi_D'})^{i+1}(\bvpi) \geq (\bm{B}^{\pi_D'})^i(\bvpi), \text{ then } (\bm{B}^{\pi_D'})^{i+2}(\bvpi) \geq (\bm{B}^{\pi_D'})^{i+1}(\bvpi) \text{ for all } i = 0, 1, 2, \ldots$$

Since $(\bm{B}^{\pi_D'})^{i+1}(\bvpi) = \bm{B}^{\pi_D'}((\bm{B}^{\pi_D'})^i(\bvpi))$, from the definition of Bellman Policy Operator (Equation \eqref{eq:bellman_policy_operator}), we can write the following two equations:

$$(\bm{B}^{\pi_D'})^{i+2}(\bvpi)(s) = \mathcal{R}(s,\pi_D'(s)) + \gamma \sum_{s'\in \mathcal{S}} \mathcal{P}(s,\pi_D'(s),s') \cdot (\bm{B}^{\pi_D'})^{i+1}(\bvpi)(s') \text{ for all } s \in \mathcal{N}$$
$$(\bm{B}^{\pi_D'})^{i+1}(\bvpi)(s) = \mathcal{R}(s,\pi_D'(s)) + \gamma \sum_{s'\in \mathcal{S}} \mathcal{P}(s,\pi_D'(s),s') \cdot (\bm{B}^{\pi_D'})^i(\bvpi)(s') \text{ for all } s \in \mathcal{N}$$
Subtracting each side of the second equation from the first equation yields:

$$(\bm{B}^{\pi_D'})^{i+2}(\bvpi)(s) - (\bm{B}^{\pi_D'})^{i+1}(s)$$
$$= \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s, \pi_D'(s), s') \cdot ((\bm{B}^{\pi_D'})^{i+1}(\bvpi)(s') - (\bm{B}^{\pi_D'})^i(\bvpi)(s'))$$
for all $s \in \mathcal{N}$

Since $\gamma \mathcal{P}(s,\pi_D'(s),s')$ consists of all non-negative values and since the induction step assumes $(\bm{B}^{\pi_D'})^{i+1}(\bvpi)(s') \geq (\bm{B}^{\pi_D'})^i(\bvpi)(s')$ for all $s' \in \mathcal{S}$, the right-hand-side of this equation is non-negative,  meaning the left-hand-side of this equation is non-negative, i.e., 
$$(\bm{B}^{\pi_D'})^{i+2}(\bvpi)(s) \geq (\bm{B}^{\pi_D'})^{i+1}(\bvpi)(s) \text{ for all } s \in \mathcal{N}$$

This completes the proof by induction.
\end{proof}

The way to understand the above proof is to think in terms of how each stage of further application of $\bm{B}^{\pi_D'}$ improves the Value Function. Stage 0 is when you have the Value Function $\bvpi$ where we execute the policy $\pi$ throughout the MDP. Stage 1 is when you have the Value Function $\bm{B}^{\pi_D'}(\bvpi)$ where from each state $s$, we execute the policy $\pi_D'$ for the first time step following $s$ and then execute the policy $\pi$ for all further time steps. This has the effect of improving the Value Function from Stage 0 ($\bvpi$) to Stage 1 ($\bm{B}^{\pi_D'}(\bvpi)$). Stage 2 is when you have the Value Function $(\bm{B}^{\pi_D'})^2(\bvpi)$ where from each state $s$, we execute the policy $\pi_D'$ for the first two time steps following $s$ and then execute the policy $\pi$ for all further time steps. This has the effect of improving the Value Function from Stage 1 ($\bm{B}^{\pi_D'}(\bvpi)$) to Stage 2 ($(\bm{B}^{\pi_D'})^2(\bvpi)$). And so on … each stage applies policy $\pi_D'$ instead of policy $\pi$ for one extra time step, which has the effect of improving the Value Function. Note that "improve" means $\geq$ (really means that the Value Function doesn't get worse for *any* of the states). These stages are simply the iterations of the Policy Evaluation algorithm (using policy $\pi_D'$) with starting Value Function $\bvpi$, building an increasing tower of Value Functions $[(\bm{B}^{\pi_D'})^i(\bvpi)|i = 0, 1, 2, \ldots]$ that get closer and closer until they converge to the Value Function $\bm{V}^{\pi_D'}$ that is $\geq \bvpi$ (hence, the term *Policy Improvement*).

The Policy Improvement Theorem yields our first Dynamic Programming algorithm to solve the MDP Control problem - known as *Policy Iteration*

## Policy Iteration Algorithm

The Policy Improvement algorithm above showed us how to start with the Value Function $\bvpi$ (for a policy $\pi$), perform a greedy policy improvement to create a policy $\pi_D' = G(\bvpi)$, and then perform Policy Evaluation (with policy $\pi_D'$) with starting Value Function $\bvpi$, resulting in the Value Function $\bm{V}^{\pi_D'}$ that is an improvement over the Value Function $\bvpi$ we started with. Now note that we can do the same process again to go from $\pi_D'$ and $\bm{V}^{\pi_D'}$ to an improved policy $\pi_D''$ and associated improved Value Function $\bm{V}^{\pi_D''}$. And we can keep going in this way to create further improved policies and associated Value Functions, until there is no further improvement. This methodology of performing Policy Improvement together with Policy Evaluation using the improved policy, in an iterative manner, is known as the Policy Iteration algorithm (shown below).

* Start with any Value Function $\bm{V_0} \in \mathbb{R}^m$
* Iterating over $j = 0, 1, 2, \ldots$, calculate in each iteration:
$$\text{ Deterministic Policy } \pi_{j+1} = G(\bm{V_j})$$
$$\text{ Value Function } \bm{V_{j+1}} = \lim_{i\rightarrow \infty} (\bm{B}^{\pi_{j+1}})^i(\bm{V_j})$$

We end these iterations (over $j$) when $\bm{V_{j+1}}$ is essentially the same as $\bm{V_j}$, i.e., when $\max_{s \in \mathcal{N}}|\bm{V_{j+1}}(s) - \bm{V_j}(s)|$ is close to 0. When this happens, the following equation should hold:
$$\bm{V_j} = (\bm{B}^{G(\bm{V_j})})^i(\bm{V_j}) = \bm{V_{j+1}} \text{ for all } i = 0, 1, 2, \ldots$$
In particular, this equation should hold for $i = 1$:

$$\bm{V_j}(s) = \bm{B}^{G(\bm{V_j})}(\bm{V_j})(s) = \mathcal{R}(s, G(\bm{V_j})(s)) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s, G(\bm{V_j})(s), s') \cdot \bm{V_j}(s') \text{ for all } s \in \mathcal{N}$$
From Equation \eqref{eq:greedy_policy_function}, we know that for each $s \in \mathcal{N}$, $\pi_{j+1}(s) = G(\bm{V_j})(s)$ is the action that maximizes $\{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{V_j}(s')\}$. Therefore,
$$\bm{V_j}(s) = \max_{a \in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{V_j}(s')\} \text{ for all  } s \in \mathcal{N}$$ 

But this in fact is the MDP Bellman Optimality Equation which would mean that $\bm{V_j} = \bvs$, i.e., when $V_j$ is close enough to $V_{j+1}$, Policy Iteration would have converged to the Optimal Value Function. The associated deterministic policy at the convergence of the Policy Iteration algorithm ($\pi_j: \mathcal{N} \rightarrow \mathcal{A}$) is an Optimal Policy because $\bm{V}^{\pi_j} = \bm{V_j} = \bvs$, meaning that evaluating the MDP with the deterministic policy $\pi_j$ achieves the Optimal Value Function. This means Policy Iteration algorithm solves the MDP Control problem. This proves the following Theorem:

\begin{theorem}[Policy Iteration Convergence Theorem]
For a Finite MDP with $|\mathcal{N}| = m$, Policy Iteration algorithm converges to the Optimal Value Function $\bvs \in \mathbb{R}^m$ along with a Deterministic Optimal Policy $\pi_D^*: \mathcal{N} \rightarrow \mathcal{A}$, no matter which Value Function $\bm{V_0} \in \mathbb{R}^m$ we start the algorithm with.
\label{eq:policy_iteration_convergence_theorem}
\end{theorem}

## Bellman Optimality Operator and Value Iteration Algorithm

By making a small tweak to the definition of Greedy Policy Function in Equation \eqref{eq:greedy_policy_function} (changing the $\argmax$ to $\max$), we define the *Bellman Optimality Operator*

$$\bbs: \mathbb{R}^m \rightarrow \mathbb{R}^m$$

as the following (non-linear) transformation of a vector (representing a Value Function) in the vector space $\mathbb{R}^m$

\begin{equation}
\bbs(\bv)(s) = \max_{a\in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bv(s')\} \text{ for all } s \in \mathcal{N}
\label{eq:bellman_optimality_operator}
\end{equation}

For each $s\in \mathcal{N}$, the action $a\in \mathcal{A}$ that produces the maximization in \eqref{eq:bellman_optimality_operator} is the action prescribed by the deterministic policy $\pi_D$ in \eqref{eq:greedy_policy_function}. Therefore, if we apply the Bellman Policy Operator on any Value Function $\bv \in \mathbb{R}^m$ using the Greedy Policy $G(\bv)$, it should be identical to applying the Bellman Optimality Operator.

\begin{equation}
\bm{B}^{G(\bv)}(\bv) = \bbs(\bv) \text{ for all } \bv \in \mathbb{R}^m
\label{eq:greedy_improvement_optimality_operator}
\end{equation}

In particular, it's interesting to observe that by specializing $\bv$ to be the Value Function $\bvpi$ for a policy $\pi$, we get:
$$\bm{B}^{G(\bvpi)}(\bvpi) = \bbs(\bvpi)$$
which is a succinct representation of the first stage of Policy Evaluation with an improved policy $G(\bvpi)$ (note how all three of Bellman Policy Operator, the Bellman Optimality Operator and Greedy Policy Function come together in this equation).

Much like how the Bellman Policy Operator $\bbpi$ was motivated by the MDP Bellman Policy Equation (equivalently, the MRP Bellman Equation), Bellman Optimality Operator $\bbs$ is motivated by the MDP Bellman Optimality Equation (expressed below):

$$\bvs(s) = \max_{a \in \mathcal{A}} \{ \mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bvs(s') \} \text{ for all } s \in \mathcal{N}$$

Note that the MDP Bellman Optimality Equation can be expressed as:

$$\bvs = \bbs(\bvs)$$

which means $\bvs \in \mathbb{R}^m$ is the Fixed-Point of the Bellman Optimality Operator $\bbs: \mathbb{R}^m \rightarrow \mathbb{R}^m$.

Much like how we proved that $\bbpi$ is a contraction function, we want to prove that $\bbs$ is a contraction function (under $L^{\infty}$ norm) so we can take advantage of Banach Fixed-Point Theorem and solve the Control problem by iterative applications of the Bellman Optimality Operator $\bbs$. So we need to prove that for all $\bm{X}, \bm{Y} \in \mathbb{R}^m$,

$$\max_{s \in \mathcal{N}} |(\bbs(\bm{X}) - \bbs(\bm{Y}))(s)| \leq \gamma \cdot \max_{s \in \mathcal{N}} |(\bm{X} - \bm{Y})(s)|$$

This proof is a bit harder than the proof we did for $\bbpi$. Here we'd need to utilize two key properties of $\bbs$.

1. Monotonicity Property, i.e, for all $\bm{X}, \bm{Y} \in \mathbb{R}^m$,
$$\text{ If } \bm{X}(s) \geq \bm{Y}(s) \text{ for all } s \in \mathcal{N}, \text{ then } \bbs(\bm{X})(s) \geq \bbs(\bm{Y})(s) \text{ for all } s \in \mathcal{N}$$
Observe that for each state $s \in \mathcal{N}$ and each action $a \in \mathcal{A}$,
$$\{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{X}(s')\} - \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{Y}(s')\}$$
$$ = \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot (\bm{X}(s') - \bm{Y}(s')) \geq 0$$
Therefore for each state $s \in \mathcal{N}$,
$$\bbs(\bm{X})(s) - \bbs(\bm{Y})(s)$$
$$= \max_{a\in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{X}(s')\} - \max_{a \in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{Y}(s')\} \geq 0$$

2. Constant Shift Property, i.e., for all $\bm{X} \in \mathbb{R}^m$, $c \in \mathbb{R}$,
$$\bbs(\bm{X} + c)(s) = \bbs(\bm{X})(s) + \gamma c \text{ for all } s \in \mathcal{N}$$
In the above statement, adding a constant ($\in \mathbb{R}$) to a Value Function ($\in \mathbb{R}^m$) adds the constant point-wise to all states of the Value Function (to all dimensions of the vector representing the Value Function). In other words, a constant $\in \mathbb{R}$ might as well be treated as a Value Function with the same (constant) value for all states. Therefore,

$$\bbs(\bm{X}+c)(s) = \max_{a \in \mathcal{A}} \{ \mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot (\bm{X}(s) + c) \}$$
$$ = \max_{a \in \mathcal{A}} \{ \mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{X}(s) \} + \gamma c = \bbs(\bm{X}) + \gamma c$$

With these two properties of $\bbs$ in place, let's prove that $\bbs$ is a contraction function. For given $\bm{X}, \bm{Y} \in \mathbb{R}^m$, assume:
$$\max_{s \in \mathcal{N}} |(\bm{X} - \bm{Y})(s)| = c$$
We can rewrite this as:
$$\bm{X}(s) - c \leq \bm{Y}(s) \leq \bm{X}(s) + c \text{ for all } s \in \mathcal{N}$$
Since $\bbs$ has the monotonicity property, we can apply $\bbs$ throughout the above double-inequality.
$$\bbs(\bm{X} - c)(s) \leq \bbs(\bm{Y})(s) \leq \bbs(\bm{X} + c)(s) \text{ for all } s \in \mathcal{N}$$
Since $\bbs$ has the constant shift property,
$$\bbs(\bm{X})(s) - \gamma c \leq \bbs(\bm{Y})(s) \leq \bbs(\bm{X})(s) + \gamma c \text{ for all } s \in \mathcal{N}$$
In other words,
$$\max_{s \in \mathcal{N}} |(\bbs(\bm{X}) - \bbs(\bm{Y}))(s)| \leq \gamma c  = \gamma \cdot \max_{s\in \mathcal{N}} |(\bm{X} - \bm{Y})(s)|$$


So invoking Banach Fixed-Point Theorem proves the following Theorem:

\begin{theorem}[Value Iteration Convergence Theorem]
For a Finite MDP with $|\mathcal{N}| = m$, if $\bvs \in \mathbb{R}^m$ is the Optimal Value Function, then $\bvs$ is the unique Fixed-Point of the Bellman Optimality Operator $\bbs: \mathbb{R}^m \rightarrow \mathbb{R}^m$, and
$$\lim_{i\rightarrow \infty} (\bbs)^i(\bm{V_0}) \rightarrow \bvs \text{ for all starting Value Functions } \bm{V_0} \in \mathbb{R}^m$$
\label{eq:policy_evaluation_convergence_theorem}
\end{theorem}

This gives us the following iterative algorithm (known as the *Value Iteration* algorithm):

* Start with any Value Function $\bm{V_0} \in \mathbb{R}^m$
* Iterating over $i = 0, 1, 2, \ldots$, calculate in each iteration:
$$\bm{V_{i+1}}(s) = \bbs(\bm{V_i})(s) = \max_{a\in \mathcal{A}} \{\mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \cdot \bm{V_i}(s')\} \text{ for all } s \in \mathcal{N}$$

We stop the algorithm when $d(\bm{V_i}, \bm{V_{i+1}}) = \max_{s \in \mathcal{N}} |(\bm{V_i} - \bm{V_{i+1}})(s)|$ is adequately small.

It pays to emphasize that Banach Fixed-Point Theorem not only assures convergence to the unique solution $\bvs$ (no matter what Value Function $\bm{V_0}$ we start the algorithm with), it also assures a reasonable speed of convergence (dependent on the choice of starting Value Function $\bm{V_0}$ and the choice of $\gamma$).

## Optimal Policy from Optimal Value Function

Note that the Policy Iteration algorithm produces a policy together with a Value Function in each iteration. So, in the end, when we converge to the Optimal Value Function $\bm{V_j} = \bvs$ in iteration $j$, the Policy Iteration algorithm has a deterministic policy $\pi_j$ associated with $\bm{V_j}$ such that:
$$\bm{V_j} = \bm{V}^{\pi_j} = \bvs$$
and we refer to $\pi_j$ as the Optimal Policy $\pi^*$, one that yields the Optimal Value Function $\bvs$, i.e.,
$$\bm{V}^{\pi^*} = \bvs$$

But Value Iteration has no such policy associated with it since the entire algorithm is devoid of a policy representation and operates only with Value Functions. So now the question is: when Value Iteration converges to the Optimal Value Function 
$\bm{V_i} = \bvs$ in iteration $i$, how do we get hold of an Optimal Policy $\pi^*$ such that:
$$\bm{V}^{\pi^*} = \bm{V_i} = \bvs$$

The answer lies in the Greedy Policy function $G$. Consider $G(\bvs)$. Equation \eqref{eq:greedy_improvement_optimality_operator} told us that:
$$\bm{B}^{G(\bv)}(\bv) = \bbs(\bv) \text{ for all } \bv \in \mathbb{R}^m$$
Specializing $\bv$ to be $\bvs$, we get:
$$\bm{B}^{G(\bvs)}(\bvs) = \bbs(\bvs)$$ 
But we know that $\bvs$ is the Fixed-Point of the Bellman Optimality Operator $\bbs$, i.e., $\bbs(\bvs) = \bvs$. Therefore,
$$\bm{B}^{G(\bvs)}(\bvs) = \bvs$$ 
The above equation says $\bvs$ is the Fixed-Point of the Bellman Policy Operator $\bm{B}^{G(\bvs)}$. However, we know that $\bm{B}^{G(\bvs)}$ has a unique Fixed-Point equal to $\bm{V}^{G(\bvs)}$. Therefore,
$$\bm{V}^{G(\bvs)} = \bvs$$
This says that evaluating the MDP with the deterministic greedy policy $G(\bvs)$ (policy created from the Optimal Value Function $\bvs$ using the Greedy Policy Function $G$) in fact achieves the Optimal Value Function $\bvs$. In other words, $G(\bvs)$ is the (Deterministic) Optimal Policy $\pi^*$ we've been seeking.

## Generalized Policy Iteration

## Aysnchronous Dynamic Programming

## Finite-Horizon Dynamic Programming: Backward Induction

## Extensions to Non-Finite Cases

Much of the theory we covered for Dynamic Programming also applies to non-finite cases. In this section, we point out the specifics that extends to non-finite cases and the specifics that don't. In particular, note that each Dynamic algorithm makes a sweep through states (looping over all states) which is obviously non-possible. 

Running time of the 3 Dynamic Programming Algorithms.


### Approximate Dynamic Programming

### Reinforcement Learning

