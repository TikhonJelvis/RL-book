## Overview {#sec:overview .unnumbered}

### Learning Reinforcement Learning {#sec:learning-rl}

Reinforcement Learning (RL) is emerging as a viable and powerful technique for solving a variety of complex business problems across industries that involve Sequential Optimal Decisioning under Uncertainty. Although RL is classified as a branch of Machine Learning (ML), it tends to be viewed and treated quite differently from other branches of ML (Supervised and Unsupervised Learning). Indeed, **RL seems to hold the key to unlocking the promise of AI** – machines that adapt their decisions to vagaries in observed information, while continuously steering towards the optimal outcome. It’s penetration in high-profile problems like self-driving cars, robotics and strategy games points to a future where RL algorithms will have decisioning abilities far superior to humans.

But when it comes getting educated in RL, there seems to be a reluctance to jump right in because RL seems to have acquired a reputation of being mysterious and exotic. We often hear even technical people claim that RL involves “advanced math” and “complicated engineering”, and so there seems to be a psychological barrier to entry. While real-world RL algorithms and implementations do get fairly elaborate and complicated in overcoming the proverbial last-mile of business problems, the foundations of RL can actually be learnt without heavy technical machinery. **The core purpose of this book is to demystify RL by finding a balance between depth of understanding and keeping technical content basic**. So now we list the key features of this book which enable this balance:

* Focus on the foundational theory underpinning RL. Our treatment of this theory is based on undergraduate-level Probability, Optimization, Statistics and Linear Algebra. **We emphasize rigorous but simple mathematical notations and formulations in developing the theory**, and encourage you to write out the equations rather than just reading from the book. Occasionally, we invoke some advanced mathematics (eg: Stochastic Calculus) but the majority of the book is based on easily understandable mathematics. In particular, two basic theory concepts - Bellman Optimality Equation and Generalized Policy Iteration - are emphasized throughout the book as they form the basis of pretty much everything we do in RL, even in the most advanced algorithms.
* Parallel to the mathematical rigor, we bring the concepts to life with simple examples and informal descriptions to help you develop an intuitive understanding of the mathematical concepts. **We drive towards creating appropriate mental models to visualize the concepts**. Often, this involves turning mathematical abstractions into physical examples (emphasizing visual intuition). So we go back and forth between rigor and intuition, between abstractions and visuals, so as to blend them nicely and get the best of both worlds.
* Each time you learn a new mathematical concept or algorithm, we ask you to write small pieces of code (in Python) that implements the concept/algorithm. As an example, if you just learnt a surprising theorem, we’d ask you to write a simulator to simply verify the statement of the theorem. We emphasize this approach not just to bolster the theoretical and intuitive understanding with a hands-on experience, but also because there is a strong emotional effect of seeing expected results emanating from one’s code, which in turn promotes long-term retention of the concepts. Most importantly, we avoid messy and complicated ML/RL/BigData tools/packages and stick to bare-bones Python/numpy as these unnecessary tools/packages are huge blockages to core understanding. We believe **coding-from-scratch is the correct approach to truly understand the concepts/algorithms**. 
* Lastly, it is important to work with examples that are A) simplified versions of real-world problems in a business domain rich with applications, B) adequately comprehensible without prior business-domain knowledge, C) intellectually interesting and D) sufficiently marketable to employers. We’ve chosen Financial Trading applications. For each financial problem, we first cover the traditional approaches (including solutions from landmark papers) and then cast the problems in ways that can be solved with RL. **We have made considerable effort to make this book self-contained in terms of the financial knowledge required to navigate these problems**.

### What you'll learn from this Book

Here is what you will specifically learn and gain from the book:

* You will learn about the simple but powerful theory of Markov Decision Processes (MDPs) – a framework for Sequential Optimal Decisioning under Uncertainty. You will firmly understand the power of Bellman Equations, which is at the heart of all Dynamic Programming as well as all RL algorithms.
* You will master Dynamic Programming (DP) Algorithms, which are a class of (in the language of AI) Planning Algorithms.  You will learn about Policy Iteration, Value Iteration, Backward Induction, Approximate Dynamic Programming and the all-important concept of Generalized Policy Iteration which lies at the heart of all DP as well as all RL algorithms.
* You will gain a solid understanding of a variety of Reinforcement Learning (RL) Algorithms, starting with the basic algorithms like SARSA and Q-Learning and moving on to several important algorithms that work well in practice, including Gradient Temporal Difference, Deep Q-Network, Least-Squares Policy Iteration, Policy Gradient, Monte-Carlo Tree Search. You will learn about how to gain advantages in these algorithms with bootstrapping, off-policy learning and deep-neural-networks-based function approximation. You will also learn how to balance exploration and exploitation with Multi-Armed Bandits techniques like Upper Confidence Bounds, Thompson Sampling, Gradient Bandits and Information State-Space algorithms.
* You will exercise with plenty of “from-scratch” Python implementations of models and algorithms. Throughout the book, we emphasize healthy Python programming practices including interface design, type annotations, functional programming and inheritance-based polymorphism (always ensuring that the programming principles reflect the mathematical principles). The larger take-away from this book will be a rare (and high-in-demand) ability to blend Applied Mathematics concepts with Software Design paradigms.
* You will go deep into important Financial Trading problems, including:

  - (Dynamic) Asset-Allocation to maximize Utility of Consumption
  - Pricing and Hedging of Derivatives in an Incomplete Market
  - Optimal Exercise/Stopping of Path-Dependent American Options
  - Optimal Trade Order Execution (managing Price Impact)
  - Optimal Market-Making (Bid/Ask managing Inventory Risk)
 
* We treat each of the above problems as MDPs (i.e., Optimal Decisioning formulations), first going over classical/analytical solutions to these problems, then introducing real-world frictions/considerations, and tackling with DP and/or RL.
* As a bonus, we throw in a few applications beyond Finance, including a couple from Supply-Chain and Clearance Pricing in a Retail business. 
* We implement a wide range of Algorithms and develop various models in [this git code base](https://github.com/TikhonJelvis/RL-book) that we refer to and explain in detail throughout the book. This code base not only provides detailed clarity on the algorithms/models, but also serves to educate on healthy programming patterns suitable not just for RL, but more generally for any Applied Mathematics work.
* In summary, this book blends Theory/Mathematics, Programming/Algorithms and Real-World Financial Nuances while always keeping things simple and intuitive.

### Expected Background to read this Book

There is no short-cut to learning Reinforcement Learning or learning the Financial Applications content. You will need to allocate at least 50 hours of effort to learn this material (assuming you have no prior background in these topics). Also, although we have kept the Mathematics, Programming and Financial content fairly basic, this topic is only for technically-inclined readers. Below we outline the technical preparation that is required to follow the material covered in this book.

* Experience with (but not necessarily expertise in) Python is expected and a good deal of comfort with numpy is required. The nature of Python programming we do is mainly numerical algorithms. You don’t need to be a professional software developer/engineer but you need to have a healthy interest in learning Python best practices associated with mathematical modeling, algorithms development and numerical programming (we teach these best practices in this book). We don't use any of the popular (but messy and complicated) Big Data/Machine Learning libraries such as Pandas, PySpark, scikit, Tensorflow, PyTorch, OpenCV, NLTK etc. (all you need to know is numpy).
* Familiarity with git and use of an Integrated Development Environment (IDE), eg: Pycharm or Emacs (with Python plugins), is recommended, but not required.
* Familiarity with LaTeX for writing equations is recommended, but not required (other typesetting tools, or even hand-written math is fine, but LaTeX is a skill that is very valuable if you’d like a future in the general domain of Applied Mathematics).
* You need to be strong in undergraduate-level Probability as it is the most important foundation underpinning RL. 
* You will also need to have some preparation in undergraduate-level Numerical Optimization, Statistics, Linear Algebra.
* No background in Finance is required, but a strong appetite for Mathematical Finance is required.

### Decluttering the Jargon linked to Reinforcement Learning

Machine Learning has exploded in the past decade or so, and Reinforcement Learning (treated as a branch of Machine Learning and hence, a branch of A.I.) has surfaced to the mainstream in both academia and in the industry. It is important to understand what Reinforcement Learning aims to solve, rather than the more opaque view of RL as a technique to learn from data. RL aims to solve problems that involve making *Sequential Optimal Decisions under Uncertainty*. Let us break down this jargon so as to develop an intuitive (and high-level) understanding of the features pertaining to the problems RL solves.

Firstly, let us understand the term *Uncertainty*. This means the problems under consideration involve random variables that evolve over time. The technical term for this is *Stochastic Processes*. We will cover this in detail later in this book, but for now, it's important to recognize that evolution of random variables over time is very common in nature (eg: weather) and in business (eg: customer demand or stock prices), but modeling and navigating such random evolutions can be enormously challenging. 

The next term is *Optimal Decisions*, which refers to the technical term *Optimization*. This means there is a well-defined quantity to be maximized (the "goal"). The quantity to be maximized might be financial (like investment value  or business profitability), or it could be a safety or speed metric (such as health of customers or time to travel), or something more complicated like a blend of multiple objectives rolled into a single objective.

The next term is *Sequential*, which refers to the fact that as we move forward in time, the relevant random variables' values evolve, and the optimal decisions have to be adjusted to the "changing circumstances". Due to this non-static nature of the optimal decisions, the term *Dynamic Decisions* is often used in the literature covering this subject. 

Putting together the three notions of (Uncertainty/Stochastic, Optimization, Sequential/Dynamic Decisions), these problems (that RL tackles) have the common feature that one needs to *overpower the uncertainty by persistent steering towards the goal*. This brings us to the term *Control* (in references to *persistent steering*). These problems are often (aptly) characterized by the technical term *Stochastic Control*. So you see that there is indeed a lot of jargon here. All of this jargon will become amply clear after the first few chapters in this book where we develop mathematical formalism to understand these concepts precisely (and also write plenty of code to internalized these concepts). For now, we just wanted to familiarize you with the range of jargon linked to Reinforcement Learning.

This jargon overload is due to the confluence of terms from Control Theory (emerging from Engineering disciplines), from Operations Research, and from Artificial Intelligence (emerging from Computer Science). For simplicity, we prefer to refer to the class of problems RL aims to solve as *Stochastic Control* problems. Reinforcement Learning is a class of algorithms that are used to solve Stochastic Control problems. We should point out here that there are other disciplines (beyond Control Theory, Operations Research and Artificial Intelligence) with a rich history of developing theory and techniques within the general space of Stochastic Control. Figure \ref{fig:many_faces_of_rl} (a popular image on the internet) illustrates the many faces of Stochastic Control, which has recently been refered to as "The many faces of Reinforcement Learning".

![Many Faces of Reinforcement Learning \label{fig:many_faces_of_rl}](./chapter0/ManyFacesOfRL.png "Many Faces of Reinforcement Learning")

It is also important to recognize that Reinforcement Learning is considered to be a branch of Machine Learning. While there is no crisp definition for *Machine Learning* (ML), ML generally refers to the broad set of techniques to infer mathematical models/functions by acquiring ("learning") knowledge of patterns and properties in the presented data. In this regard, Reinforcement Learning does fit this definition. However, unlike the other branches of ML (Supervised Learning and Unsupervised Learning), Reinforcement Learning is a lot more ambitious - it not only learns the patterns and properties of the presented data (internally building a model of the data), it also learns about the appropriate behaviors to be exercised (appropriate decisions to be made) so as to drive towards the optimization objective. It is sometimes said that Supervised Learning and Unsupervised learning are about "minimization" (i.e., they minimize the fitting error of a model to the presented data), while Reinforcement Learning is about "maximization" (i.e., RL also identifies the suitable decisions to be made to maximize a well-defined objective). Figure \ref{fig:ml_branches} depicts the in-vogue classification of Machine Learning.

![Branches of Machine Learning \label{fig:ml_branches}](./chapter0/BranchesofML.jpg "Branches of Machine Learning")

More importantly, the class of problems RL aims to solve can be described with a simple yet powerful mathematical framework known as *Markov Decision Processes* (abbreviated as MDPs). We have an entire chapter dedicated to deep coverage of MDPs, but we provide a quick high-level introduction to MDPs in the next section.

### Introduction to the Markov Decision Process (MDP) framework

![The MDP Framework \label{fig:mdp_framework}](./chapter0/MDP.png "Agent-Environment Interaction in a MDP")

The framework of a Markov Decision Process is depicted in Figure \ref{fig:mdp_framework}. As the Figure indicates, the *Agent* and the *Environment* interact in a time-sequenced loop. The term *Agent* refers to an algorithm (AI algorithm) and the term *Environment* refers to an abstract entity that serves up uncertain outcomes to the Agent. It is important to note that the Environment is indeed abstract in this framework and can be used to model all kinds of real-world situations such as the financial market serving up random stock prices or customers of a company serving up random demand or a chess opponent serving up random moves (from the perspective of the Agent), or really anything at all you can imagine that serves up something random at each time step (it is up to us to model an Environment appropriately to fit the MDP framework).

As the Figure indicates, at each time step $t$, the Agent observes an abstract piece of information (which we call *State*) and a numerical (real number) quantity that we call *Reward*. Note that the concept of *State* is indeed completely abstract in this framework and we can model *State* to be any data type, as complex or elaborate as we'd like. This flexibility in modeling *State* permits us to model all kinds of real-world situations as an MDP. Upon observing a *State* and a *Reward* at time step $t$, the *Agent* responds by taking an *Action*. Again, the concept of *Action* is completely abstract and is meant to represent an activity performed by an AI algorithm. It could be a purchase or sale of a stock responding to market stock price movements, or it could be movement of inventory from a warehouse to a store in response to large sales at the store, or it could be a chess move in response to the opponent's chess move (opponent is *Environment*), or really anything at all you can imagine that responds to observations (*State* and *Reward*) served by the *Environment*.

Upon receiving an *Action* from the *Agent* at time step $t$, the *Environment* responds (with time ticking over to $t+1$) by serving up the next time step's random *State* and random *Reward*. A technical detail (that we shall explain in detail later) is that the *State* is assumed to have the *Markov Property*, which means:

* The next *State/Reward* depends only on Current *State* (for a given *Action*).
* The current *State* encapsulates all relevant information from the history of the interaction between the *Agent* and the *Environment*.
* The current *State* is a sufficient statistic of the future (for a given *Action*).

The goal of the *Agent* at any point in time is to maximize the *Expected Sum* of all future *Reward*s by controlling (at each time step) the *Action* as a function of the observed *State* (at that time step). This function from a *State* to *Action* at any time step is known as the *Policy* function. So we say that the agent's job is exercise control by determining the optimal *Policy* function. Hence, this is a dynamic (i.e., time-sequenced) control system under uncertainty. If the above description was too terse, don't worry - we will explain all of this in great detail in the coming chapters. For now, we just wanted to provide a quick flavor for what the MDP framework looks like. Now we sketch the above description with some (terse) mathematical notation to provide a bit more of the overview of the MDP framework. The following notation is for discrete time steps (continuous time steps notation is analogous, but technically more complicated to describe here):

We denote time steps as $t = 1, 2, 3, \ldots$. Markov State at time $t$ is denoted as $S_t \in \mathcal{S}$ where $\mathcal{S}$ is refered to as the *State Space*. Action at time $t$ is denoted as $A_t \in \mathcal{A}$ where $\mathcal{A}$ is refered to as the *Action Space*. Reward at time $t$ is denoted as $R_t \in \mathbb{R}$ (representing the numerical feedback served by the Environment, along with the State, at each time step $t$). 

We represent the transition probabilities from one time step to the next with the following notation:

$$p(s',r|s,a) = \mathbb{P}[S_{t+1}=s',R_{t+1}=r|S_t=s,A_t=a]$$

$\gamma \in [0,1]$ is known as the discount factor used to discount Rewards when accumulating Rewards, as follows:

$$\text{Return } G_t = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots$$

The discount factor $\gamma$ allows us to model situations where a future reward is less desirable than a current reward of the same quantity.

The goal is to find a *Policy* $\pi: \mathcal{S} \rightarrow \mathcal{A}$ that maximizes  $\mathbb{E}[G_t|S_t = s]$ for all $s \in \mathcal{S}$. In subsequent chapters, we clarify that the MDP framework actually considers more general policies than described here - policies that are stochastic, i.e., functions that take as input a state and output a probability distribution of actions (rather than a single action). However, for ease of understanding of the core concepts, in this chapter, we stick to deterministic policies $\pi: \mathcal{S} \rightarrow \mathcal{A}$.

The intuition here is that the two entities *Agent* and *Environment* interact in a time-sequenced loop wherein the *Environment* serves up next states and rewards based on the transition probability function $p$ and the *Agent* exerts control over the vagaries of $p$ by exercising the policy $\pi$ in a way that optimizes the Expected "accumulated rewards" (i.e., Expected Return) from any state.

![Baby Learning MDP \label{fig:baby_mdp}](./chapter0/BabyMDP.jpg "Baby Learning MDP")

It's worth pointing out that the MDP framework is inspired by how babies (*Agent*) learn to perform tasks (i.e., take *Action*s) in response to the random activities and events (*State*s and *Reward*s) they observe as being served up from the world (*Environment*) around them. Figure \ref{fig:baby_mdp} illustrates this - at the bottom of the Figure (labeled "World", i.e., *Environment*), we have a room in a house with a vase atop a bookcase. At the top of the Figure is a baby (learning *Agent*) on the other side of the room who wants to make her way to the bookcase, reach for the vase, and topple it - doing this efficiently (i.e., in quick time and quietly) would mean a large *Reward* for the baby. At each time step, the baby finds herself in a certain posture (eg: lying on the floor, or sitting up, or trying to walk etc.) and observes various visuals around the room - her posture and her visuals would constitute the *State* for the baby at each time step. The baby's *Action*s are various options of physical movements to try to get to the other side of the room (assume the baby is still learning how to walk). The baby tries one physical movement, but is unable to move forward with that movement. That would mean a negative *Reward* - the baby quickly learns that this movement is probably not a good idea. Then she tries a different movement, perhaps trying to stand on her feet and start walking. She makes a couple of good steps forward (positive *Reward*s), but then falls down and hurts herself (that would be a big negative *Reward*). So by trial and error, the baby learns about the consequences of different movements (different actions). Eventually, the baby learns that by holding on to the couch, she can walk across, and then when she reaches the bookcase, she learns (again by trial and error) a technique to climb the bookcase that is quick yet quiet (so she doesn't raise her mom's attention). This means the baby learns of the optimal policy (best actions for each of the states she finds herself in) after essentially what is a "trial and error" method of learning what works and what doesn't. This example is essentially generalized in the MDP framework, and the baby's "trial and error" way of learning is essentially a special case of the general technique of Reinforcement Learning.

### Real-world problems that fit the MDP framework

As you might imagine by now, all kinds of problems in nature and in business (and indeed, in our personal lives) can be modeled as Markov Decision Processes. Here is a sample of such problems:

* Self-driving vehicle (Actions constitute speed/steering to optimize safety/time).
* Game of Chess (Actions constitute moves of the pieces to optimize chances of winning the game).
* Complex Logistical Operations, such as those in a Warehouse (Actions constitute inventory movements to optimize throughput/time).
* Making a humanoid robot walk/run on a difficult terrain (Actions are walking movements to optimize time to destination).
* Manage an investment portfolio (Actions are trades to optimize long-term investment gains).
* Optimal decisions during a football game (Actions are strategic game calls to optimize chances of winning the game).
* Strategy to win an election (Actions constitute political decisions to optimize chances of winning the election).

![Self-driving Car MDP \label{fig:car_mdp}](./chapter0/CarMDP.jpg "Self-driving Car MDP")

Figure \ref{fig:car_mdp} illustrates the MDP for a self-driving car. At the top of the figure is the *Agent* (the car's driving algorithm) and at the bottom of the figure is the *Environment* (constituting everything the car faces when driving - other vehicles, traffic signals, road conditions, weather etc.). The *State* consists of the car's location, velocity, and all of the information picked up by the car's sensors/cameras. The *Action* consists of the steering, acceleration and brake. The *Reward* would be a combination of metrics on ride comfort and safety, as well as the negative of each time step (because maximizing the accumulated Reward would then amount to minimizing time taken to reach the destination).

### The inherent difficulty in solving MDPs

"Solving" an MDP refers to identifying the optimal policy with an algorithm. This section paints an intuitive picture of why solving a general MDP is fundamentally a hard problem. Often, the challenge is simply that the *State Space* is very large (involving many variables) or complex (elaborate data structure), and hence, is computationally intractable. Likewise, sometimes the *Action Space* can be quite large or complex. 

But the main reason for why solving an MDP is inherently difficult is the fact that there is no direct feedback on what the "correct" Action is for a given State. What we mean by that is that unlike a supervised learning framework, the MDP framework doesn't give us anything other than a *Reward* feedback to indicate if an Action is the *right one* or not. A large Reward might encourage the Agent, but it's not clear if one just got lucky with the large Reward or if there could be an even larger Reward if the Agent tries the Action again. The linkage between Actions and Rewards is further complicated by the fact that there is time-sequenced complexity in an MDP, meaning an Action can influence future States, which in turn influences future Actions. Consequently, we sometimes find that Actions can have delayed consequences, i.e., the Rewards for a good Action might come after many time steps (eg: in a game of Chess, a brilliant move leads to a win after several further moves). 

The other problem one encounters in real-world situations is that the Agent often doesn't know the *Model* of the *Environment*. By *Model*, we are refering to the probabilities of state-transitions and rewards, i.e., the function $p$ we defined above. This means the Agent has to simultaneously learn the Model (from the real-world data stream) and solve for the Optimal Policy. 

Lastly, when there are many actions, the Agent needs to try them all to check if there are some hidden gems (great actions that haven't been tried yet), which in turn means one could end up wasting effort on "duds" (bad actions). So the agent has to find the balance between *exploitation* (retrying actions which have yielded good rewards so far) and *exploration* (trying actions that have either not been tried enough or not been tried at all).

All of this seems to indicate that we don't have much hope in solving MDPs in a reliable and efficient manner. But it turns out that with some clever mathematics, we can indeed make some good inroads. We outline the core idea of this "clever mathematics" in the next section.

### Value Function, Bellman Equation, Dynamic Programming and RL

Perhaps the most important concept we want to highlight in this entire book is the idea of a *Value Function* and how
we can compute it in an efficient manner with either Planning or Learning algorithms. The Value Function $V^{\pi}: \mathcal{S} \rightarrow \mathbb{R}$ for a given policy $\pi$ is defined as:

$$V^{\pi}(s) = \mathbb{E}[G_t|S_t = s] \text{ for all } s \in \mathcal{S}$$

The intuitive way to understand Value Function is that it tells us how much "accumulated future reward" (i.e., *Return*) we expect to obtain from a given state. The randomness under the expectation comes from the uncertain future states and rewards the Agent is going to see upon taking future actions prescribed by the policy $\pi$. The key in evaluating the Value Function for a given policy is that it can be expressed recursively, in terms of the Value Function for the next time step's states. In other words, 
$$V^{\pi}(s) = \sum_{s',r} p(s',r|s,\pi(s)) \cdot (r + \gamma \cdot V^{\pi}(s')) \text{ for all } s \in \mathcal{S}$$
This equation says that when the Agent is in a given state $s$, it takes an action $a=\pi(s)$, then sees a random next state $s'$ and a random reward $r$, so $V^{\pi}(s)$ can be broken into the expectation of $r$ (first step's expected reward) and the remainder of the future expected accumulated rewards (which can be written in terms of the expectation of $V^{\pi}(s')$). We won't get into the details of how to solve this recursive formulation in this chapter (will cover this in great detail in future chapters), but it's important for you to recognize for now that this recursive formulation is the key to evaluating the Value Function for a given policy.

However, evaluating the Value Function for a given policy is not the end goal - it is simply a means to the end goal of evaluating the *Optimal Value Function* (from which we obtain the *Optimal Policy*). The Optimal Value Function $V^*: \mathcal{S} \rightarrow \mathbb{R}$ is defined as:
$$V^*(s) = \max_{\pi} V^{\pi}(s) \text{ for all } s \in \mathcal{S}$$

The good news is that even the Optimal Value Function can be expressed recursively, as follows:

$$V^*(s) = \max_{a} \sum_{s',r} p(s',r|s,a) \cdot (r + \gamma \cdot V^*(s')) \text{ for all } s \in \mathcal{S}$$

Furthermore, we can prove that there exists an Optimal Policy $\pi^*$ achieving $V^*(s)$ for all $s \in \mathcal{S}$ (the proof is constructive, which gives a simple method to obtain the function $\pi^*$ from the function $V^*$)$. Specifically, this means that the Value Function obtained by following the optimal policy $\pi^*$ is the same as the Optimal Value Function $V^*$, i.e.,

$$V^{\pi^*}(s) = V^*(s) \text{ for all } s \in \mathcal{S}$$

There is a bit of terminology here to get familiar with. The problem of calculating $V^{\pi}(s)$ (Value Function for a give policy) is known as the *Prediction* problem (since this amounts of statistical estimation of the expected returns from any given state when following a policy $\pi$). The problem of calculating the Optimal Value Function $V^*$ (and hence, Optimal Policy $\pi^*$), is known as the *Control* problem (since this requires steering of the policy such that we obtain the maximum expected return from any state). Solving the Prediction problem is typically a stepping stone towards solving the (harder) problem of Control. These recursive equations for $V^{\pi}$ and $V^*$ are known as the (famous) Bellman Equations (which you will hear a lot about in future chapters). In a continuous-time formulation, the Bellman Equation is refered to as the famous *Hamilton-Jacobi-Bellman (HJB)* equation.

The algorithms to solve the prediction and control problems based on Bellman equations are broadly classified as:

* Dynamic Programming, a class of (in the language of A.I.) *Planning* algorithms.
* Reinforcement Learning, a class of (in the language of A.I.) *Learning* algorithms.

Now let's talk a bit about the difference between Dynamic Programming and Reinforcement Learning algorithms. Dynamic Programming algorithms (which we cover a lot of in this book) assume that the agent knows of the transition probabilities $p$ and the algorithm takes advantage of the knowledge of those probabilities (leveraging the Bellman Equation to efficiently calculate the Value Function). Dynamic Programming algorithms are considered to be *Planning* and not *Learning* (in the language of A.I.) because the algorithm doesn't need to interact with the Environment and doesn't need to learn from the (states, rewards) data stream coming from the Environment. Rather, armed with the transition probabilities, the algorithm can reason about future probabilistic outcomes and perform the requisite optimization calculation to infer the Optimal Policy. So it *plans* it's path to success, rather than *learning* about how to succeed.

However, in typical real-word situations, one doesn't really know of the transition probabilities $p$. This is the realm of Reinforcement Learning (RL). RL algorithms interact with the Environment, learn with each new (state, reward) pair received from the Environment, and incrementally figure out the Optimal Value Function (with the "trial and error" approach that we outlined earlier). However, note that the Environment interaction could be *real* interaction or *simulated* interaction. In the latter case, we do have a model of transition probabilities but the structure of the model is so complex that we only have access to samples of the next state and reward (rather than an explicit representation of the probabilities). This is known as a *Sampling Model* of the Environment. With access to such a sampling model of the environment, we can employ the same RL algorithm that we would have used when interacting with a real environment (eg: a robot learning on an actual terrain versus a simulated terrain). In fact, most RL algorithms in practice learn from simulated models of the environment. As we explained earlier, RL is essentially a "trial and error" learning approach and hence, is quite laborious and fundamentally inefficient. The progress in RL is coming from more efficient ways of learning the Optimal Value Function, and better ways of approximating the Optimal Value Function. One of the key challenges for RL in the future is to identify better ways of finding the balance between "exploration" and "exploitation" of actions. In any case, one of the key reasons RL has started doing well lately is due to assistance it has obtained from Deep Learning (typically Deep Neural Networks are used to approximate the Value Function and/or to approximate the Value Function). RL with such deep learning approximations is now known by the catchy modern term *Deep RL*.

We believe the current promise of A.I. is dependent on the success of Deep RL. The next decade will be exciting as RL research will yield improved algorithms and it's pairing with Deep Learning will hopefully enable us to solve fairly complex real-world stochastic control problems. 


### Outline of Chapters

#### Section I: Planning Algorithms (Dynamic Programming)

##### Chapter 0: Overview

* Learning Reinforcement Learning
* What you'll learn from this book
* Expected Background to read this book
* Overview of Chapters
* Decluttering the jargon in Optimal Decisioning under Uncertainty
* Introduction to the Markov Decision Process (MDP) framework
* Examples of important real-world problems that fit the MDP framework
* The inherent difficulty in solving MDPs
* Overview of Value Function, Bellman Equation, Dynamic Programming and Reinforcement Learning
* Why is RL useful/interesting to learn about?
* Many faces of RL and where it lies in the classification of ML
* Outline of chapters in the book
* Overview of the 5 Financial Applications
 
This is an introductory/overview chapter familiarizing the reader with the general space of optimal decisioning under uncertainty, the framework used to tackle such problems, and a high-level overview of the techniques/algorithms (Dynamic Programming and Reinforcement Learning). The reader will also get an overview of the range of business problems that can be targeted and the intuitive difficulty in tackling these problems. In this chapter, the reader will be familiarized with the flow of chapters and will be introduced to the 5 important Financial applications that will be covered in this book. This chapter starts with an articulation of the approach we take in this book to explain RL, blending Mathematics/Theory, Intuition, Programming, and Applications.

##### Chapter 1: Best Practices in Python for Applied Mathematics

* Type Annotations
* List and Dict Comprehensions
* Functional Programming – lambdas and higher-order functions
* Class inheritance and Abstract Base Classes
* Generics Programming
* Generators
 
Given this book’s emphasis on “learning by programming”, this chapter will familiarize the reader with some core techniques in Python that will be very important in implementing models and algorithms. To be clear, this chapter is not a full Python tutorial – the reader is expected to have some background in Python already. It is a tutorial of some key techniques and practices in Python (that many readers might not be accustomed to) that are highly relevant to programming in the broader area of Applied Mathematics. As part of this chapter, the reader will do some exercises with these techniques/practices in Python and will emerge prepared to learn the material of this book with appropriate Python paradigms.

##### Chapter 2: Markov Processes

* The concept of State in a Process
* Understanding Markov Property from Stock Price Examples
* Formal Definitions for Markov Processes (MP)
* Stock Price Examples modeled as Markov Processes
* Finite Markov Processes
* Simple Inventory Example
* Stationary Distribution of a Markov Process
* Formalism of Markov Reward Processes (MRP)
* Simple Inventory Example as a Markov Reward Process
* Finite Markov Reward Processes
* Simple Inventory Example as a Finite Markov Reward Process
* Value Function of a Markov Reward Process

This chapter covers the foundational topics required to understand Markov Decision Processes - The Markov property, Markov Processes (sometimes refered to as Markov Chains), Markov Reward Processes, the concept of Value Function of a Markov Reward Process, and the Bellman Equation for a Markov Reward Process. These concepts are motivated with examples of stock prices and with a simple inventory example that serves first as a Markov Process and then as a Markov Reward Process. There will be a significant amount of programming in this chapter to develop comfort with these concepts.

##### Chapter 3: Markov Decision Processes

* Simple Inventory Example: How much to Order?
* The Difficulty of Sequential Decisioning under Uncertainty
* Formal Definition of a Markov Decision Process (MDP)
* Policy
* [Markov Decision Process, Policy] := Markov Reward Process
* Simple Inventory Example with Unlimited Capacity
* Finite Markov Decision Processes
* Simple Inventory Example as a Finite Markov Decision Process
* MDP Value Function for a Fixed Policy
* Optimal Value Function and Optimal Policies
* Variants and Extensions of MDPs

This chapter lays the foundational theory underpinning RL – the framework for representing problems dealing with optimal decisioning under uncertainty (Markov Decision Process). The reader will learn about the relationship between Markov Decision Processes and Markov Reward Processes, the Value Function (key ingredient in RL algorithms) and the Bellman Equations (a powerful property, which enables solving complex decisioning problems). There will be a considerable amount of programming exercises in this chapter. The heavy investment in this theory together with hands-on programming will put the reader in a highly advantaged position to learn the following chapters in a very clear and speedy manner.

##### Chapter 4: Dynamic Programming Algorithms

* Planning versus Learning
* Usage of the term **Dynamic Programming** (DP)
* Solving the Value Function as a **Fixed-Point**
* Bellman Policy Operator and Policy Evaluation Algorithm
* Greedy Policy
* Policy Improvement
* Policy Iteration Algorithm
* Bellman Optimality Operator and Value Iteration Algorithm
* Optimal Policy from Optimal Value Function
* Revisiting the Simple Inventory Example
* Generalized Policy Iteration
* Asynchronous Dynamic Programming
* Finite-Horizon Dynamic Programming: Backward Induction
* Dynamic Pricing For End-of-Life/End-of-Season of a Product
* Extensions to Non-Tabular Algorithms

This chapter covers the Planning technique of Dynamic Programming (DP), which is an important class of foundational algorithms that can be an alternative to RL if the business problem is not too large or too complex. Also, learning these algorithms will provide important foundations to the reader to be able to understand subsequent RL algorithms more deeply. The reader will learn about several important DP algorithms and at the end of the chapter, the reader will learn about why DP gets difficult in practice which draws the reader to the motivation behind RL. Again, we will do plenty of DP algorithms, which are quick to implement and will aid considerably in internalizing the concepts. Finally, we emphasize a special algorithm - Backward Induction - for solving finite-horizon Markov Decision Processes, which is the setting for the financial applications we cover in this book.

##### Chapter 5: Function Approximation and Approximate Dynamic Programming

* Function Approximation
* Linear Function Approximation
* Neural Network Function Approximation
* Tabular as a form of **FunctionApprox**
* Approximate Policy Evaluation
* Approximate Value Iteration
* Finite-Horizon Approximate Policy Evaluation
* Finite-Horizon Approximate Value Iteration
* Finite-Horizon Approximate Q-Value Iteration
* How to Construct the States Distribution

The Dynamic Programming algorithms covered in the previous chapter suffer from the two so-called curses: Curse of Dimensionality and Curse of Modeling. These curses can be cured with a combination of sampling and function approximation. The next section covers the sampling cure (using Reinforcement Learning). This chapter covers the topic of function approximation and shows how an intermediate cure - Approximate Dynamic Programming (function approximation without sampling) - is quite viable and can be suitable for some problems.

#### Section II: Modeling Financial Applications

##### Chapter 6: Utility Theory

* Introduction to the Concept of Utility
* A Simple Financial Example
* The shape of the Utility function
* Calculating the Risk-Premium
* Constant Absolute Risk-Aversion (CARA)
* A Portfolio Application of CARA
* Constant Relative Risk-Aversion (CRRA)
* A Portfolio Application of CRRA

Having learnt DP algorithms and before learning RL algorithms, the reader will learn about the 5 financial applications. For each of the financial applications, we will go over the core financial background and concepts within these applications, then learn how to solve them with DP, then introduce real-world considerations and finally explain why tackling those considerations requires RL. Later, we will revisit these financial problems when we cover RL algorithms (in the 2nd half of the book). But before we get into each of the 5 financial applications, readers need to get familiar with the key economics concept of risk-aversion that applies to 4 of the 5 financial applications. So this chapter is dedicated to risk-aversion and the related concepts of risk-premium and Utility functions. As ever, we will write plenty of code in this chapter to understand these concepts thoroughly.

##### Chapter 7: Dynamic Asset Allocation and Consumption

* Optimization of Personal Finance
* Merton’s Portfolio Problem and Solution
* Developing Intuition for the Solution to Merton's Portfolio Problem
* A Discrete-Time Asset-Allocation Example
* Porting to Real-World

This chapter cover the first financial application – dynamic asset allocation and consumption. This problem is best understood in the context of Merton’s landmark paper in 1969 where he stated and solved this problem. This chapter is mainly focused on the mathematical derivation of Merton’s solution of this problem with Dynamic Programming. Through this derivation, the reader will learn about the broader result called Hamilton-Jacobi-Bellman (HJB) equation, which will re-appear in the Market-Making problem in Chapter 12. As ever, we will do some programming exercises. In particular, the reader will learn how to solve Dynamic Asset Allocation in a simple setting with Backward Induction (a DP algorithm we learnt in Chapter 4). Finally, the reader will learn about Dynamic Asset Allocation in practice, where Merton’s frictionless setting is replaced by several real-world frictions, which will require us to move to RL. We will revisit this problem after coverage of RL algorithms in the second half of the book.

##### Chapter 8: Derivatives Pricing and Hedging

* A Brief Introduction to Derivatives
* Notation for the Single-Period Simple Setting
* Portfolios, Arbitrage and Risk-Neutral Probability Measure
* First Fundamental Theorem of Asset Pricing (1st FTAP)
* Second Fundamental Theorem of Asset Pricing (2nd FTAP)
* Derivatives Pricing in Single-Period Setting
* Overview of Multi-Period/Continuous-Time Theory
* Pricing/Hedging in an Incomplete Market Cast as an MDP
* Optimal Exercise of American Options Cast as an MDP

This chapter covers the most important topic in Mathematical Finance: Pricing and Hedging of Derivatives. Full and rigorous coverage of derivatives pricing and hedging is a fairly elaborate and advanced topic, and beyond the scope of this book. But we have provided a way to understand the theory in this chapter by considering a very simple setting - that of a single-period with discrete outcomes and no provision for rebalancing of the hedges, that is typical in the general theory. Following the coverage of the foundational theory, we cover the problem of optimal pricing/hedging of derivatives in an incomplete market and the problem of optimal exercise of American Options (both problems are modeled as MDPs). In this chapter, you will learn about some highly important financial foundations such as the concepts of arbitrage, replication, market completeness, and the all-important risk-neutral measure. You will learn the proofs of the two fundamental theorems of asset pricing in this simple setting. We also provide an overview of the general theory (beyond this simple setting). Next you will learn about how to price/hedge derivatives incorporating real-world frictions by modeling this problem as an MDP. We will revisit this problem after we learn the RL algorithms because the only practical way of solving this problem in the real-world is by designing a market-calibrated simulator and then employing RL algorithms on this simulator. In the final section of this chapter, you will learn how to model optimal stopping as an MDP. You will learn how to use Backward Induction (a DP algorithm we learnt in Chapter 4) to solve this problem when the state-space is not too big. We will revisit this problem after we have covered RL algorithms. By the end of this chapter, the reader would have developed significant expertise in pricing and hedging complex derivatives, a skill that is in high demand in the finance industry.

##### Chapter 9:  Order-Book Trading Algorithms

* Basics of Limit Order Books (LOB)
* Price Impact and LOB Dynamics
* Problem Statement and MDP Formulation
* Simple Linear Impact Model with no Risk-Aversion
* Incorporating Risk-Aversion and Real-World Considerations
* Market-Making on a Limit Order Book
* Problem Statement and MDP Formulation
* Avellaneda-Stoikov Continuous-time Formulation
* Solving the Avellaneda-Stoikov formulation
* Real-world Market-Making

This chapter introduces the reader to the world of Algorithmic Trading. However, current Algorithms tend to be rules-based and heuristic. Algorithmic Trading is transforming into Machine Learning-based Algorithms. The natural extension is to automate not just forecasting but also decisioning, the realm of Reinforcement Learning. In this chapter, the reader is first introduced to the mechanics of trade order placements (market orders and limit orders), and then introduced to a very important real-world problem – how to submit a large-sized market order by splitting and timing the splits optimally in order to overcome “price impact” and gain maximum proceeds. The reader will learn about the classical methods based on Dynamic Programming. Then the reader will learn about market frictions and the need to tackle them with RL. We will return to this problem in the second half of the book after we have learnt some RL algorithms. In the second half of this chapter, we cover the Algorithmic-Trading twin of the Optimal Execution problem – that of a market-maker having to submit dynamically-changing bid and ask limit orders so she can make maximum gains. The reader will learn about how market-makers (a big and thriving industry) operate. Then the reader will learn about how to formulate this problem as an MDP. We will do a thorough coverage of the classical Dynamic Programming solution by Avellaneda and Stoikov, together with some programming exercises to absorb those concepts. Finally, the reader is exposed to the real-world nuances of this problem, and hence, the need to tackle with a market-calibrated simulator and RL.  We will return to this problem in the second half of the book after we have learnt some RL algorithms.

#### Section III: Reinforcement Learning Algorithms

##### Chapter 10: Tabular Monte-Carlo & Temporal Difference for Prediction

* Tabular Monte-Carlo (MC) Policy Evaluation
* Tabular Temporal-Difference (TD) Policy Evaluation
* Estimate adjustments in a soccer game (MC versus TD)
* Bias-Variances Tradeoff and other contrasts between MC and TD
* Some simple examples
* Wide versus sample backups, Deep versus shallow backups
* TD(lambda)
* Eligibility Traces

This chapter starts a new phase in this book, our entry into the world of RL algorithms. To understand the basics of RL, in this chapter, I restrict the RL problem to a very simple one – one where the state space is small and manageable as a table enumeration (known as tabular RL) and one where we only have to calculate the Value Function for a Fixed Policy (this problem is known as the Prediction problem, versus the optimization problem which is known as the Control problem). So this chapter is about RL for Tabular Prediction. This restriction is important because it makes it much easier to understand the core concepts of Monte-Carlo (MC) and Temporal-Difference (TD) in this simplified setting. The remaining chapters will build upon this chapter by adding more complexity and more nuances, while retaining much of the key core concepts developed in this chapter. As ever, the reader will learn by coding plenty of MC and TD algorithms from scratch.

##### Chapter 11: Tabular Monte-Carlo & Temporal Difference for Control

* Making MC work with GPI: Action-Value Function and Epsilon-Greedy
* GLIE Monte-Carlo Control
* SARSA
* Off-Policy Learning: Importance Sampling
* Off-Policy Learning: Q-Learning

This chapter makes the natural extension from Prediction to Control, while remaining in the tabular setting. The investments made in the previous chapter will bear fruit here as important algorithms such as SARSA and Q-learning can now be learnt with enormous clarity. In this chapter, the reader will implement both SARSA and Q-Learning from scratch in Python. This chapter also introduces the reader to a very important concept for the future success of RL in the real-world: off-policy learning (Q-Learning is the simplest off-policy learning algorithm and it has had good success in various applications).

##### Chapter 12: Monte-Carlo and Temporal Difference with Function Approximation

* Value Function Approximation
* A quick review of Deep Neural Networks and Stochastic Gradient Descent
* Linear Value Function Approximation
* MC and TD Prediction with Value Function Approximation
* MC and TD Control with Action-Value Function Approximation
* Convergence of Prediction and Control Algorithms
 
In this chapter, the reader will learn how to scale the RL algorithms learnt in chapters 11 and 12 to real-world situations where the state space is large and consequently require approximate representations of the Value Function, eg: using Deep Neural Networks. The reader will gain an appreciation of how RL and Neural Networks technology are complementary and together they can surmount real-world challenges. The reader will also learn about intricate choices of how much to bootstrap, and whether to do on-policy or off-policy learning when doing function approximation.

##### Chapter 13: Batch RL

* Experience Replay
* Deep Q-Networks (DQN)
* Least-Squares MC and TD
* Least-Squares Policy Iteration (LSPI) 

In this chapter, the reader will learn about a different class of RL algorithms that are data-efficient. These batch algorithms such as Experience Replay, Deep Q-networks and Least Squares Policy Iteration are state-of-the-art and have had a lot of success in practice. The reader will also explore these algorithms in the context of the Financial Applications that were previously covered.

##### Chapter 14: Value Function Geometry and Gradient TD

* Motivation for Value Function Geometry
* Bellman Operator and Projection Operator
* Value Function Vectors of interest
* Residual Gradient Algorithm
* Naïve Residual Gradient Algorithm
* Gradient TD Algorithm

This chapter is heavier on Theory relative to other chapters. The motivation for this chapter is to develop deeper insights into the core mathematics underpinning RL algorithms (back to the basics of Bellman Equation). Understanding Value Function Geometry will place the reader in a highly advantaged situation in terms of truly understanding what is it that makes some Algorithms succeed in certain situations and other than don’t. This chapter also shows the reader how to break out of the so-called Deadly Traid (when bootstrapping, function approximation and off-policy are employed together, RL algorithms tend to fail). The state-of-the-art Gradient TD Algorithm resist the deadly triad and we dive deep into its inner workings to understand how and why.

##### Chapter 15: Policy Gradient Algorithms

* Policy Improvement with Gradient Ascent
* Expected-Returns Objective and Policy Gradient Theorem (PGT)
* Simple parametric representations for Policies
* Proof of Policy Gradient Theorem
* The REINFORCE Algorithm
* Variance Reduction
* Compatible Function Approximation Theorem
* Natural Policy Gradient
* Deterministic Policy Gradient
* Evolutionary Strategies

In this chapter, the reader will be exposed to yet another class of RL algorithms that are based on improving the policy using the gradient of the policy function approximation (rather than the usual policy improvement based on argmax). When action spaces are large or continuous, Policy Gradient tends to be the only option and so, this chapter is useful to overcome many real-world situations (including many financial applications) where the action space is indeed large. The reader will learn about the mathematical proof of the elegant Policy Gradient Theorem and implement a couple of Policy Gradient Algorithms from scratch. The reader will learn about state-of-the-art Actor-Critic methods. Lastly, the reader will also learn about Evolutionary Strategies, an algorithm that looks quite similar to Policy Gradient Algorithms, but is actually not classified as an RL Algorithm. However, learning about Evolutionary Strategies is important because some real-world applications, including Financial Applications can indeed be tackled well with Evolutionary Strategies.

##### Chapter 16: Learning versus Planning

* Model-based RL
* Learning a Model
* Sample-based Planning
* Planning with Inaccurate Models
* Integrating Learning and Planning
* Simulation-based Search
* Monte-Carlo Tree Search
* Adaptive Multistage Sampling

This chapter brings the various pieces of Planning and Learning concepts learnt in this book together. The reader will learn that in practice, one needs to be creative about blending planning and learning concepts (a technique known as Model-based RL). In practice, many problems are tackled using Model-based RL. The reader will also get familiar with an algorithm (Monte Carlo Tree Search) that was highly popularized when it solved the Game of GO, a problem that was thought to be unsurmountable by present AI technology.

##### Chapter 17: Exploration versus Exploitation

* The Multi-Armed Bandit (MAB) Problem
* Regret Analysis
* Simple Algorithms for MAB
* Lai-Robbins lower bound
* Hoeffding’s Inequality and Upper-Confidence Bound (UCB) Algorithm
* Probability Matching and Thompson Sampling
* Gradient Bandit Algorithms
* Information State-Space Algorithms
* Contextual Bandits
* Exploration versus Exploitation in MDPs

This chapter enables the reader to deep-dive into the topic of balancing exploration and exploitation, a topic of great importance in RL algorithms. Exploration versus Exploitation is best understood in the simpler setting of the Multi-Armed Bandit (MAB) problem. The reader will learn about various state-of-the-art MAB algorithms, implement them in Python, and draw various graphs to understand how they perform versus each other in various problem settings. The reader will then be exposed to Contextual Bandits which are a popular technique in optimal choices of Advertisement placements. Finally, the reader will learn how to apply the MAB algorithms within RL.

##### Chapter 18: RL in Real-World Finance: Reality versus Hype, Present versus Future

This concluding chapter will enable the reader to put the entire book’s content in perspective relative to the current state of the financial industry, the practical challenges in adoption of RL, and some guidance on how to go about building an end-to-end system for financial applications based on RL. The reader will be guided on reality versus hype in the current “AI First” landscape. The reader will also gain a perspective of where RL stands today and what the future holds.

#### Appendices

1. Moment-Generating Function and it's Applications
2. Portfolio Theory
3. Introduction To and Overview Of Stochastic Calculus Basics
4. The Hamilton-Jacobi-Bellman (HJB) Equation
5. Black-Scholes Equation and it's Solution for Call/Put Options
