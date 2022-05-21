# Foundations of Reinforcement Learning

An in-development book teaching foundational ideas in reinforcement learning with examples in finance.

> Reinforcement Learning (RL) is emerging as a viable and powerful technique for solving a variety of complex business problems across industries that involve Sequential Optimal Decisioning under Uncertainty. Although RL is classified as a branch of Machine Learning (ML), it tends to be viewed and treated quite differently from other branches of ML (Supervised and Unsupervised Learning). Indeed, **RL seems to hold the key to unlocking the promise of AI** – machines that adapt their decisions to vagaries in observed information, while continuously steering towards the optimal outcome. It’s penetration in high-profile problems like self-driving cars, robotics and strategy games points to a future where RL algorithms will have decisioning abilities far superior to humans.

The book and its codebase are under active development. A [recent PDF version][recent-pdf] is available online, but the only way to get the most up-to-date PDF is to generate it yourself following the instructions in this README.

[recent-pdf]: https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf

# RL Book Setup

Basic setup for working with Pandoc and TeX.

## Installation

To work on the book, you need to [install Nix][install].

### Set up the environment

On macOS, first install the XCode command-line tools:

```
xcode-select --install
```

the install Nix with:


```
sh <(curl https://nixos.org/nix/install) --darwin-use-unencrypted-nix-store-volume
```

On Linux, install Nix with:

```
curl -L https://nixos.org/nix/install | sh
```

### Nix Shells

Once you have Nix installed, run `nix-shell` to get access to Pandoc, LaTeX and all the other tools you need. The first time you run `nix-shell` will take a while to finish as it downloads and installs all the packages you need.

[install]: https://nixos.org/download.html

## Generating PDFs

Once inside the Nix shell, you'll have access to Pandoc and you'll be able to generate PDFs with XeTeX. The `to-pdf` script can do this for a single chapter in the [`book`][book] directory:

```
[nix-shell:~/Documents/RL-book]$ bin/to-pdf chapter0
Converting book/chapter0/chapter0.md to book/chapter0/chapter0.pdf
```

[book]: ./book

You can also generate the entire book to a file called `book.pdf`:

```
[nix-shell:~/Documents/RL-book]$ bin/to-pdf
Combining
book/chapter0/chapter0.md
book/chapter2/chapter2.md
book/chapter3/chapter3.md
book/chapter4/chapter4.md
book/chapter5/chapter5.md
into book.pdf
```

Note that this can take a little while (10–20 seconds for chapters 0–5).

### Index

We can generate an index with the `--index` flag:

``` shell
bin/to-pdf chapter1 --index
```

This requires running `xelatex` twice, so it'll take longer to generate the PDF. Index terms [are defined within the text with the `\index` command][index-command].

[index-command]: INDEXING.md

### Cross-references

We can define labels for chapters and headings:

```
# Overview {#sec:overview}

## Learning Reinforcement Learning {#sec:learning-rl}
```

Because of limitations with the system I'm using for managing internal references, labels for sections *and* chapters always have to start with `sec:`.

Once you have defined a label for a section or chapter, you can reference its number as follows:

```
Take a look at Chapter [-@sec:mdp].
```

> Take a look at Chapter 3.

For sections, you can also use:

```
Take a look at [@sec:learning-rl].
```

> Take a look at sec. 1.

(The `[-@sec:foo]` syntax drops the "sec. " text.)

For references across chapters to render correctly, you have to compile the entire book PDF (following the instructions above).

## Working with Python and venv

We can manage our Python dependencies with a venv.

First, create a venv *from inside a Nix shell*:

```
> nix-shell
[nix-shell:~/Documents/RL-book]$ python -m venv .venv
```

Then, each time you're working on this project, make sure to *activate* the venv:

```
> source .venv/bin/activate
```

(This can now be done even outside a Nix shell.)

Once the venv is activated, you should see a `(.venv)` in your shell prompt:

```
(.venv) RL-book:RL-book>
```

Now you can use `pip` to install dependencies inside the venv:

```
(.venv) RL-book:RL-book> pip install matplotlib
```

To make this reproducible, we can save the libraries to a `requirements.txt` file:

```
(.venv) RL-book:RL-book>pip freeze > requirements.txt
```

Then, when somebody is starting, they can install every Python package they need using:

```
(.venv) RL-book:RL-book>pip install -r requirements.txt
```
