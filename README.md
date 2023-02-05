# Foundations of Reinforcement Learning

A textbook teaching foundational ideas in reinforcement learning with examples in finance. Written by [Ashwin Rao][ashwin] and [Tikhon Jelvis][tikhon].

> Reinforcement Learning (RL) is emerging as a viable and powerful technique for solving a variety of complex business problems across industries that involve Sequential Optimal Decisioning under Uncertainty. Although RL is classified as a branch of Machine Learning (ML), it tends to be viewed and treated quite differently from other branches of ML (Supervised and Unsupervised Learning). Indeed, **RL seems to hold the key to unlocking the promise of AI** – machines that adapt their decisions to vagaries in observed information, while continuously steering towards the optimal outcome. It’s penetration in high-profile problems like self-driving cars, robotics and strategy games points to a future where RL algorithms will have decisioning abilities far superior to humans.

[ashwin]: https://stanford.edu/~ashlearn
[tikhon]: https://jelv.is

# Getting the Book

The first edition of the book is now available from Routledge. You can:

  * Buy a copy from [Routledge][buy-routledge] or [Amazon][buy-amazon]
  * Download a [PDF of the manuscript][recent-pdf]
  * Compile the latest version of the manuscript [from this repo][compile-pdf]
  
You can also download [errata for the print version][errata].

[buy-routledge]: https://www.routledge.com/Foundations-of-Reinforcement-Learning-with-Applications-in-Finance/Rao-Jelvis/p/book/9781032124124
[buy-amazon]: https://www.amazon.com/Foundations-Reinforcement-Learning-Applications-Finance/dp/1032124121
[recent-pdf]: https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf
[errata]: https://stanford.edu/~ashlearn/RLForFinanceBook/errata.pdf
[compile-pdf]: book/README.md

# Getting the Code

The [`rl`](./rl) directory in this repo contains the code used in this book, with a simple framework for reinforcement learning as well as fleshed out examples for each chapter.

## Working with Python and venv

The Python code for the book requires a few additional libraries. We can manage our Python dependencies with a venv.

First, create a venv:

```
> nix-shell
[nix-shell:~/Documents/RL-book]$ python -m venv .venv
```

Then, each time you're working on this project, make sure to *activate* the venv:

```
> source .venv/bin/activate
```

Once the venv is activated, you should see a `(.venv)` in your shell prompt:

```
(.venv) RL-book:RL-book>
```

Now you can use `pip` to install the needed dependencies inside the venv:

```
(.venv) RL-book:RL-book>pip install -r requirements.txt
```

If you want additional libraries, you can install them explicitly:

```
(.venv) RL-book:RL-book> pip install matplotlib
```
